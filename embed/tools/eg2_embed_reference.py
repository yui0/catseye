#!/usr/bin/env python3
"""Reference EmbeddingGemma 2 embeddings (text + image) with per-layer dumps for the C port.

Runs google/embeddinggemma-2 in float32 through sentence-transformers and hooks the stages the C
implementation has to reproduce. Everything is written as float32 .npy so the C tests can compare
layer by layer; the final unit vectors are also compared against the official encode() path.

  vision : patches -> patch_embedder -> 16 layers -> pooler (3x3 avg * sqrt(768)) -> embed_vision(512)
  text   : tokens -> *sqrt(512) -> PLE -> 24 layers -> norm -> Linear 512->768 -> mean pool -> L2 norm
  image  : [BOS, boi, 512-d soft tokens x N, eoi, EOS] through the same text backbone

Usage:
  python eg2_embed_reference.py --model-dir eg2_hf --out-dir ref \\
      --text "task: search result | query: a dog" --text "title: none | text: a cat" \\
      --image photo1.jpg --image photo2.png
Output: <out>/index.json, <out>/text_N/*.npy, <out>/image_N/*.npy
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


def to_np(t):
    return t.detach().float().cpu().numpy().astype('<f4')


class Recorder:
    """Collects tensors produced by hooks into a dict, keyed by stage name."""

    def __init__(self, core):
        self.store = {}
        lm, vt = core.language_model, core.vision_tower
        self.hook(vt.patch_embedder, 'vis_patch_embed')
        for i, layer in enumerate(vt.encoder.layers):
            self.hook(layer, f'vis_layer_{i:02d}')
        self.hook(core.embed_vision, 'vis_soft_tokens', pre='vis_pooled')
        self.hook(lm.ple, 'text_ple')
        for i, layer in enumerate(lm.layers):
            self.hook(layer, f'text_layer_{i:02d}')
        self.hook(lm.norm, 'text_norm')
        self.hook(lm.embedding_projection, 'text_projected')
        lm.register_forward_pre_hook(self._merged_embeds, with_kwargs=True)

    def hook(self, module, name, pre=None):
        def post(mod, args, kwargs, out):
            out = out[0] if isinstance(out, tuple) else out
            self.store[name] = to_np(out)
            if pre:  # module input; the multimodal embedder is called with a keyword argument
                inp = args[0] if args else kwargs.get('inputs_embeds')
                if inp is not None:
                    self.store[pre] = to_np(inp)
        module.register_forward_hook(post, with_kwargs=True)

    def _merged_embeds(self, _, args, kwargs):
        if kwargs.get('inputs_embeds') is not None:
            self.store['text_inputs_embeds'] = to_np(kwargs['inputs_embeds'])


def save(dirpath, arrays):
    dirpath.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(dirpath / f'{name}.npy', np.ascontiguousarray(arr))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--text', action='append', default=[])
    p.add_argument('--image', action='append', default=[], type=Path)
    p.add_argument('--official-images', action='store_true', help='also compare images with SentenceTransformer.encode')
    p.add_argument('--max-soft', type=int, default=280, help='soft tokens per image: 70, 140, 280, 560 or 1120')
    a = p.parse_args()

    model = SentenceTransformer(str(a.model_dir), device='cpu', model_kwargs={'dtype': torch.float32},
                                config_kwargs={'audio_config': None})
    tr = model[0]
    tr.processor.image_processor.max_soft_tokens = a.max_soft
    core = tr.model
    items = [('text', i, t) for i, t in enumerate(a.text)] + [('image', i, f) for i, f in enumerate(a.image)]
    # Official encode() first: with the recording hooks installed it becomes extremely slow.
    official = {}
    for kind, i, src in items:
        if kind == 'text':
            official[(kind, i)] = model.encode([src], convert_to_numpy=True)[0]
        elif a.official_images:  # encode() of images can stall in a worker pool; the manual path below is exact
            official[(kind, i)] = model.encode({'image': Image.open(src).convert('RGB')}, convert_to_numpy=True)
    # Image preprocessing leaves torch at one thread (a worker-pool side effect): restore it for the forwards.
    torch.set_num_threads(os.cpu_count() or 1)
    rec = Recorder(core)
    index = []
    for kind, i, src in items:
        feats = tr.preprocess([src]) if kind == 'text' else tr.preprocess([{'image': Image.open(src).convert('RGB')}])
        rec.store.clear()
        with torch.no_grad():
            out = core(**{k: v for k, v in feats.items() if hasattr(v, 'shape')})
        hidden = out.last_hidden_state                       # [1, T, 768] per-token projected states
        mask = feats['attention_mask'][..., None].float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)         # mean pooling
        emb = torch.nn.functional.normalize(pooled, dim=-1)[0]
        ref = official.get((kind, i))
        cos = None if ref is None else float(torch.dot(emb, torch.from_numpy(np.asarray(ref).reshape(-1)).float()))
        arrays = dict(rec.store)
        arrays['input_ids'] = feats['input_ids'][0].numpy().astype('<i4')
        arrays['embedding'] = to_np(emb)
        if kind == 'image':
            pos = feats['image_position_ids'][0]
            real = (pos != -1).all(-1)
            # C works on the real patches only; padding is fully masked in the reference.
            arrays['pixel_values'] = to_np(feats['pixel_values'][0][real])
            arrays['patch_positions'] = pos[real].numpy().astype('<i4')
            for k in [k for k in arrays if k.startswith('vis_layer_') or k == 'vis_patch_embed']:
                arrays[k] = arrays[k][0][real.numpy()]
            arrays['vis_pooled'] = arrays['vis_pooled'][0] if arrays['vis_pooled'].ndim == 3 else arrays['vis_pooled']
        for k in [k for k in arrays if k.startswith('text_') and arrays[k].ndim == 3]:
            arrays[k] = arrays[k][0]
        name = f'{kind}_{i}'
        save(a.out_dir / name, arrays)
        index.append({'name': name, 'kind': kind, 'source': str(src), 'tokens': int(feats['input_ids'].shape[1]),
                      'cos_vs_official': cos})
        print(f'{name}: {feats["input_ids"].shape[1]} tokens, cos vs official encode() = {cos}', flush=True)
    (a.out_dir / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
