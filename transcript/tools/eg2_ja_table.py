#!/usr/bin/env python3
"""Generate Japanese 768D word embeddings with EmbeddingGemma 2 (Colab-safe).

- Loads only the text tower, avoiding unnecessary image/audio model weights.
- Never runs float16 inference. Uses BF16 only on supported CUDA GPUs, FP32 otherwise.
- Shrinks inference batches after CUDA OOM; preserves completed output on failure.
- Emits the underlying Python traceback and a focused diagnosis on error.
"""
import argparse
import gc
import os
from pathlib import Path
import sys
import traceback

import numpy as np

PROMPT = 'task: search result | query: '


def choose_dtype(torch, device, dtype_name):
    if dtype_name not in ('auto','fp32','bf16'):
        raise ValueError(f'unsupported dtype: {dtype_name}')
    if dtype_name == 'bf16' and device != 'cuda':
        raise ValueError('BF16 mode requires a CUDA GPU; use --dtype fp32 on CPU')
    if dtype_name == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support BF16; use --dtype fp32')
    if dtype_name == 'bf16' or (dtype_name == 'auto' and device == 'cuda' and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float32


def force_plain_text(model):
    """Use the text processor rather than a chat template.

    sentence-transformers 6.x may advertise a `message` modality even if
    EmbeddingGemma2Processor has no chat template.  In that case ordinary
    strings are silently promoted to messages before `.encode()`, raising
    `Cannot use apply_chat_template because this processor does not have a
    chat template`. Removing the message path keeps the original text tower,
    original prompt and pooling unchanged.
    """
    try:
        module = model[0]
        routes = module.modality_config
    except (AttributeError, IndexError, TypeError, KeyError) as e:
        raise RuntimeError('Sentence Transformers text input module is unavailable') from e
    if 'text' not in routes:
        raise RuntimeError(f'Sentence Transformers model has no plain text modality (available: {list(routes)})')
    if 'message' in routes:
        routes.pop('message')
        print('Disabled chat-template routing; using direct text embeddings.',flush=True)
    return model


def is_cuda_oom(exc):
    s = str(exc).lower()
    return ('out of memory' in s or 'cuda error: out of memory' in s) and 'cuda' in s


def generate(model_dir, vocab_file, out_file, batch=32, device='cuda', dtype_name='auto', loader=None):
    import torch
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. In Colab: Runtime > Change runtime type > GPU')
    if device not in ('cpu','cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if batch < 1:
        raise ValueError('--batch must be >= 1')

    model_dir, vocab_file, out_file = Path(model_dir), Path(vocab_file), Path(out_file)
    if not model_dir.is_dir():
        raise FileNotFoundError(f'model directory not found: {model_dir}')
    if not (model_dir / 'config.json').exists():
        raise FileNotFoundError(f'config.json is missing from {model_dir} (rerun the model download cell)')
    if not list(model_dir.rglob('*.safetensors')):
        raise FileNotFoundError(f'*.safetensors weights missing from {model_dir} (rerun the download cell)')
    if not vocab_file.is_file():
        raise FileNotFoundError(f'vocabulary file not found: {vocab_file}')
    vocab = vocab_file.read_text(encoding='utf-8').splitlines()
    if not vocab or any(not w or w != w.strip() for w in vocab):
        raise ValueError('the vocabulary must contain non-empty, trimmed words (one per line)')
    if len(vocab) != len(set(vocab)):
        raise ValueError('the vocabulary contains duplicate entries')

    dtype = choose_dtype(torch, device, dtype_name)
    print(f'Vectorizing {len(vocab)} Japanese tokens; device={device}, compute_dtype={dtype}, batch={batch}',flush=True)
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f'GPU={props.name}, VRAM={props.total_memory/2**30:.1f} GiB; BF16 supported={torch.cuda.is_bf16_supported()}',flush=True)

    if loader is None:
        from sentence_transformers import SentenceTransformer
        loader = SentenceTransformer
    # Match the project's original text embedding prompt and full 768-dimensional output.
    # IMPORTANT: fp16 is unsafe for this model even if the GPU supports it.
    model = loader(str(model_dir), device=device,
                   model_kwargs={'dtype': dtype},
                   config_kwargs={'vision_config': None, 'audio_config': None})

    force_plain_text(model)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    partial = out_file.with_name(out_file.stem + '.partial.npy')
    if partial.exists():
        partial.unlink()   # never mistake a truncated run for a complete word table
    table = None
    try:
        table = np.lib.format.open_memmap(str(partial), mode='w+', dtype=np.float16, shape=(len(vocab), 768))
        offset, working_batch = 0, batch
        while offset < len(vocab):
            count = min(working_batch, len(vocab) - offset)
            try:
                v = model.encode([PROMPT+(w[1:] if w.startswith('▁') else w) for w in vocab[offset:offset+count]],
                                 batch_size=count, convert_to_numpy=True,
                                 show_progress_bar=False)
                v = np.asarray(v, dtype=np.float32)
                if v.shape != (count, 768):
                    raise RuntimeError(f'EmbeddingGemma produced {v.shape}, expected ({count}, 768)')
                norms = np.linalg.norm(v, axis=1, keepdims=True)
                if not np.isfinite(v).all() or not np.isfinite(norms).all() or np.any(norms <= 1e-8):
                    raise RuntimeError('EmbeddingGemma produced NaN/Inf/zero vectors; ensure BF16 or FP32 (never FP16)')
                table[offset:offset+count] = (v / norms).astype(np.float16)
                offset += count
                if offset == count or offset == len(vocab) or offset % max(50, working_batch*25) < count:
                    print(f'embeddings {offset}/{len(vocab)}, current batch={working_batch}', flush=True)
            except RuntimeError as exc:
                if device != 'cuda' or not is_cuda_oom(exc) or working_batch == 1:
                    raise
                old = working_batch
                working_batch = max(1, working_batch // 2)
                del exc
                gc.collect()
                torch.cuda.empty_cache()
                print(f'CUDA OOM: reducing batch {old} -> {working_batch} and retrying token {offset}', flush=True)
        table.flush()
        del table
        table = None
        # Sanity check the output before exposing it as a complete table.
        shape = np.load(partial, mmap_mode='r').shape
        if shape != (len(vocab), 768):
            raise RuntimeError(f'invalid table shape on disk: {shape}')
        os.replace(partial, out_file)
        print(f'word_table.npy finished: {out_file} ({out_file.stat().st_size/2**20:.1f} MiB)',flush=True)
    except BaseException:
        if table is not None:
            del table
        partial.unlink(missing_ok=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True)
    p.add_argument('--vocab',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    p.add_argument('--dtype',choices=('auto','bf16','fp32'),default='auto')
    args = p.parse_args()
    try:
        generate(args.model_dir,args.vocab,args.out,args.batch,args.device,args.dtype)
    except Exception as exc:
        print('\n== EmbeddingGemma 2 embedding generation failed ==',file=sys.stderr)
        traceback.print_exc()
        msg=str(exc).lower()
        if 'cuda' in msg and 'out of memory' in msg:
            hint='CUDA memory exhausted. Restart runtime or try --batch 2, --dtype fp32.'
        elif any(s in msg for s in ('not found','no such file','missing','does not appear to have')):
            hint='Download is missing files: rerun the model download cell, and check Hugging Face access.'
        elif 'bf16' in msg or 'float16' in msg or 'nan' in msg:
            hint='Use --dtype fp32 (never FP16); rerun on supported CUDA hardware.'
        elif 'chat template' in msg or 'apply_chat_template' in msg:
            hint='The current sentence-transformers version is still routing as chat. Ensure the Stage 37 eg2_ja_table.py is installed and has force_plain_text().'
        elif '401' in msg or '403' in msg or 'gated' in msg:
            hint='Accept the model license / sign in to Hugging Face, then download again.'
        else:
            hint='Use the traceback ABOVE this line; the surrounding CalledProcessError alone is insufficient.'
        print('Possible action: '+hint,file=sys.stderr,flush=True)
        sys.exit(1)

if __name__=='__main__':
    main()
