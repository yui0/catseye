#!/usr/bin/env python3
"""Cache EmbeddingGemma 2 audio soft tokens for Japanese/English/mixed CTC.

Input JSONL rows: {"id":"unique_ascii_id","audio":"path.wav","text":"transcript"}.
Output: one float16 [time,512] .npy per utterance and index.jsonl.

- Loads Google EmbeddingGemma 2 with only the audio path needed at inference.
- Uses BF16 on BF16-capable GPUs, otherwise FP32 (never FP16).
- Checks corrupt/partial cached files; writes completed arrays atomically.
- Emits tracebacks and actionable hints, including offending audio/manifest.
- Can inspect dataset and probe with --limit 1 before a long cache job.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from eg2_ja_vocab import load_rows


def read_audio(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    wav, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    if wav.ndim != 1 or not np.isfinite(wav).all():
        raise ValueError(f'invalid waveform in {path}; shape={wav.shape}')
    return wav


def pick_dtype(torch, device, dtype):
    if dtype not in ('auto', 'fp32', 'bf16'):
        raise ValueError('dtype must be auto, fp32 or bf16')
    if dtype == 'bf16' and not (device == 'cuda' and torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 requires a compatible CUDA GPU; select --dtype fp32')
    if dtype == 'bf16' or (dtype == 'auto' and device == 'cuda' and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float32


def load_encoder(model_dir, device, dtype='auto'):
    import torch
    from transformers import EmbeddingGemma2Config, EmbeddingGemma2Model, Gemma4AudioFeatureExtractor
    model_dir = Path(model_dir)
    if not (model_dir / 'config.json').is_file():
        raise FileNotFoundError(f'Missing {model_dir}/config.json. Rerun Colab step ⑤.')
    if not (model_dir / 'preprocessor_config.json').is_file():
        raise FileNotFoundError(f'Missing {model_dir}/preprocessor_config.json. Rerun Colab step ⑤.')
    if not list(model_dir.glob('*.safetensors')):
        raise FileNotFoundError(f'No *.safetensors in {model_dir}. Download step ⑤ may be incomplete.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; choose Colab GPU runtime or --device cpu')
    compute_dtype = pick_dtype(torch, device, dtype)
    print(f'Loading audio encoder: device={device} dtype={compute_dtype} attention=eager', flush=True)
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f'GPU={props.name}; VRAM={props.total_memory / 2**30:.1f} GiB', flush=True)
    fe = Gemma4AudioFeatureExtractor.from_pretrained(str(model_dir))
    config = EmbeddingGemma2Config.from_pretrained(str(model_dir))
    # Discard the large vision tower. The text tower is required temporarily by
    # EmbeddingGemma2Model's constructor but is unused by get_audio_features.
    config.vision_config = None
    # Gemma4AudioModel cannot use Torch SDPA in current Transformers.
    # 'eager' must apply to both the parent and its nested audio tower.
    if config.audio_config is not None:
        config.audio_config._attn_implementation = 'eager'
    model = EmbeddingGemma2Model.from_pretrained(
        str(model_dir), config=config, dtype=compute_dtype,
        attn_implementation='eager',
    )
    if model.audio_tower is None or model.embed_audio is None:
        raise RuntimeError('Model did not load audio_tower and embed_audio')
    model.language_model = None  # free text memory before long audio inference
    model = model.to(device).eval()
    print('Audio encoder ready', flush=True)
    return model, fe


def soft_tokens(model, fe, wav, device):
    import torch
    f = fe([wav], return_tensors='np', max_length=None, truncation=False)
    x = torch.from_numpy(np.asarray(f['input_features'], dtype=np.float32)).to(device)
    mask = torch.from_numpy(np.asarray(f['input_features_mask'], dtype=np.bool_)).to(device)
    # Keep features consistent with model dtype: FP32 on T4, BF16 on newer GPUs.
    x = x.to(dtype=next(model.audio_tower.parameters()).dtype)
    with torch.inference_mode():
        y = model.get_audio_features(x, mask, return_dict=True)
        soft = y.pooler_output[0][y.attention_mask[0]].float().cpu().numpy()
    # Two stride-2 conv layers; valid mel frame count comes from fe's own mask.
    n_valid = int(mask[0].sum().item())
    expected = (n_valid + 3) // 4
    if soft.shape != (expected, 512) or not np.isfinite(soft).all():
        raise RuntimeError(f'invalid soft tokens: got {soft.shape}, expected {(expected,512)} '
                           f'from {n_valid} valid mel frames')
    return soft.astype(np.float16)


def validate_manifest(manifest, limit=0):
    manifest = Path(manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f'Cannot open manifest {manifest}')
    rows = list(load_rows(manifest))
    if limit:
        rows = rows[:limit]
    valid = []
    seen = set()
    for i, r in enumerate(rows):
        uid = str(r.get('id', i))
        if not uid.isascii() or not uid.replace('-', '').replace('_', '').isalnum():
            raise ValueError(f'{manifest} row {i+1}: invalid id={uid!r}; ASCII letters/digits/_/- only')
        if uid in seen:
            raise ValueError(f'{manifest} row {i+1}: duplicate id={uid!r}; would overwrite previous audio')
        seen.add(uid)
        audio = r.get('audio')
        if not isinstance(audio, str) or not audio:
            raise ValueError(f'{manifest} row {i+1}, id={uid}: missing audio path')
        source = Path(audio)
        if not source.is_absolute():
            source = manifest.resolve().parent / source
        if not source.is_file():
            raise FileNotFoundError(f'{manifest} row {i+1}, id={uid}: audio not found: {source}')
        valid.append((uid, source, r['text']))
    if not valid:
        raise ValueError(f'Empty manifest: {manifest}')
    print(f'Manifest preflight OK: {manifest}, items={len(valid)}, first={valid[0][1]}',flush=True)
    return valid


def valid_cached_array(path):
    try:
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        return a.ndim == 2 and a.shape[1] == 512 and a.shape[0] >= 1 and a.dtype == np.float16
    except (OSError, ValueError, EOFError):
        return False


def save_atomic(path, array):
    # np.save('name.tmp') would silently append .npy, so use a file handle.
    tmp = path.with_name(path.name + '.partial')
    try:
        with tmp.open('wb') as f:
            np.save(f, array, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def build_cache(manifest, model_dir, out, device='cuda', limit=0, dtype='auto'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    valid = validate_manifest(manifest, limit)
    model = fe = None
    result = []
    for ix, (uid, source, txt) in enumerate(valid):
        cache_file = out / (uid + '.npy')
        try:
            if not cache_file.exists() or not valid_cached_array(cache_file):
                if cache_file.exists():
                    print(f'Rebuilding incomplete cache: {cache_file}', flush=True)
                wav = read_audio(source)
                if not 320 <= len(wav) <= 30 * 16000:
                    print(f'SKIP {uid}: {len(wav)/16000:.2f}s outside 0.02-30s',flush=True)
                    continue
                if model is None:
                    model, fe = load_encoder(model_dir, device, dtype=dtype)
                save_atomic(cache_file, soft_tokens(model, fe, wav, device))
            result.append({'id': uid, 'soft': str(cache_file.resolve()), 'text': txt})
            if ix == 0 or (ix+1) % 20 == 0 or ix+1 == len(valid):
                print(f'Cached {ix+1}/{len(valid)} | {uid} | {cache_file}', flush=True)
        except Exception:
            print(f'\nFAILED at {ix+1}/{len(valid)} id={uid} audio={source}', file=sys.stderr, flush=True)
            raise
    if not result:
        raise RuntimeError('No valid audio was cached; inspect durations and paths in manifest.')
    target = out / 'index.jsonl'
    tmp = out / 'index.jsonl.partial'
    try:
        with tmp.open('w', encoding='utf8') as f:
            for r in result:
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    print(f'Saved {target} ({len(result)} utterances)', flush=True)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='First N utterances only, e.g. --limit 1 for diagnosis')
    p.add_argument('--device', choices=('cuda','cpu','auto'), default='auto')
    p.add_argument('--dtype', choices=('auto','fp32','bf16'), default='auto')
    a = p.parse_args()
    if a.limit < 0:
        p.error('--limit must be nonnegative')
    try:
        if a.device == 'auto':
            import torch
            a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        build_cache(a.manifest, a.model_dir, a.out, a.device, a.limit, a.dtype)
    except Exception as exc:
        print('\n=== EG2 audio caching failed (root traceback) ===',file=sys.stderr)
        traceback.print_exc()
        msg = str(exc).lower()
        if 'out of memory' in msg or 'cuda out of memory' in msg:
            hint = 'CUDA VRAM exhausted. Restart Colab runtime; use an L4/A100 GPU, or rerun with --device cpu (slower).'
        elif any(v in msg for v in ('no such file','audio not found', 'not found','missing')):
            hint = 'Check step ④ dataset paths and step ⑤ model checkpoint files.'
        elif 'sdpa' in msg or 'eager' in msg or 'attention' in msg:
            hint = 'Gemma4AudioModel does not support SDPA here: use attn_implementation="eager" in eg2_ja_cache.py, then re-run Colab cell ③ and ⑧.'
        elif 'soft tokens' in msg:
            hint = 'The encoder output length/dimension does not match expectations; inspect transformers version and audio duration.'
        else:
            hint = 'See traceback and FAILED audio/id above. If another cell only shows CalledProcessError, open the saved cache log.'
        print('ACTION:',hint,file=sys.stderr,flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
