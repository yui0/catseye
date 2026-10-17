#!/usr/bin/env python3
"""Offline one-shot converter: public EG2 Transcribe ReaderA (.pt) -> native C EG2R.
No PyTorch needed at inference. Only works for arch='a' 512/8/2048 reader
from https://github.com/john-rocky/eg2-transcribe reader/train/model.py.
Output is 24-byte EG2R header + contiguous little-endian f32 + logit scale.
This script does not download, train, or redistribute any pretrained weights.
"""
import argparse
import math
from pathlib import Path
import struct
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint', type=Path, help='reader.pt checkpoint, from the public Reader repository')
    ap.add_argument('output', type=Path, help='output reader.eg2r')
    args = ap.parse_args()
    try:
        import torch
        import numpy as np
    except ImportError as e:
        ap.error(f'conversion-time dependency missing: {e}; install torch and numpy (not required at C runtime)')
    data = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    for name in ('state_dict', 'model', 'reader'):
        if isinstance(data, dict) and name in data and isinstance(data[name], dict):
            data = data[name]
    if not isinstance(data, dict):
        ap.error('expected a torch state_dict')
    if 'body.ln_in.weight' not in data and 'module.body.ln_in.weight' in data:
        data = {k[7:] if k.startswith('module.') else k: v for k, v in data.items()}
    layer_ids = sorted({int(k.split('.')[2]) for k in data if k.startswith('body.blocks.') and k.split('.')[2].isdigit()})
    if layer_ids != list(range(len(layer_ids))) or not 1 <= len(layer_ids) <= 12:
        ap.error(f'unsupported or non-contiguous reader block IDs: {layer_ids}')
    layers = len(layer_ids)
    entries = []

    def add(name, shape):
        if name not in data:
            raise ValueError(f'missing tensor: {name}')
        x = data[name]
        if tuple(x.shape) != tuple(shape):
            raise ValueError(f'{name}: shape {tuple(x.shape)} != expected {shape}')
        v = x.detach().to(torch.float32).cpu().contiguous().numpy().astype('<f4', copy=False)
        if not np.isfinite(v).all():
            raise ValueError(f'{name} contains NaN/Inf')
        entries.append(v)

    add('body.ln_in.weight', (512,))
    add('body.ln_in.bias', (512,))
    add('body.inp.weight', (512, 512))
    add('body.inp.bias', (512,))
    for idx in layer_ids:
        p = f'body.blocks.{idx}.'
        for name, shape in [
            ('ln1.weight', (512,)), ('ln1.bias', (512,)),
            ('qkv.weight', (1536, 512)), ('qkv.bias', (1536,)),
            ('proj.weight', (512, 512)), ('proj.bias', (512,)),
            ('ln2.weight', (512,)), ('ln2.bias', (512,)),
            ('ff1.weight', (2048, 512)), ('ff1.bias', (2048,)),
            ('ff2.weight', (512, 2048)), ('ff2.bias', (512,)),
        ]:
            add(p + name, shape)
    add('body.ln_out.weight', (512,))
    add('body.ln_out.bias', (512,))
    add('body.out.weight', (768, 512))
    add('body.out.bias', (768,))
    add('blank.weight', (1, 768))
    add('blank.bias', (1,))
    if 'log_scale' not in data or data['log_scale'].numel() != 1:
        raise ValueError('missing scalar log_scale')
    scale = math.exp(float(data['log_scale'].item()))
    if not math.isfinite(scale) or not 0 < scale <= 1e6:
        raise ValueError('invalid logit scale')
    expected = ((512*2 + 512*512 + 512) + layers*(
        512*2 + 1536*512 + 1536 + 512*512 + 512 +
        512*2 + 2048*512 + 2048 + 512*2048 + 512) +
        (512*2 + 768*512 + 768) + (768+1))
    nfloat = sum(a.size for a in entries)
    if nfloat != expected:
        raise ValueError(f'internal count mismatch: {nfloat} expected {expected}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as out:
        out.write(struct.pack('<4s5I', b'EG2R', 1, layers, 512, 2048, nfloat))
        for tensor in entries:
            out.write(tensor.tobytes(order='C'))
        out.write(struct.pack('<f', scale))
    print(f'EG2R v1: {layers} transformer layers, {nfloat:,} weights, scale={scale:.7g}, {args.output.stat().st_size:,} bytes')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        sys.exit(f'conversion failed: {exc}')
