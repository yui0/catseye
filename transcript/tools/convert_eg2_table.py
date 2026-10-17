#!/usr/bin/env python3
"""Offline streaming fp16 .npy word embeddings -> native row-wise INT8 EG2Q.
The source should be the matching Reader checkpoint's word_table.npy
(200000 x 768 normalized float16). Does not depend on torch, runtime only C.
"""
import argparse
from pathlib import Path
import struct
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        import numpy as np
    except ImportError as ex:
        parser.error(f'converter needs numpy: {ex}')
    table = np.load(args.input, mmap_mode='r', allow_pickle=False)
    if len(table.shape) != 2 or table.shape[1] != 768 or not 0 < table.shape[0] <= 1000000:
        parser.error(f'expected [N,768] array, got {table.shape}')
    if not np.issubdtype(table.dtype, np.floating):
        parser.error(f'expected fp16/fp32 embeddings, got {table.dtype}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open('wb') as out:
            out.write(struct.pack('<4s3I', b'EG2Q', 1, int(table.shape[0]), 768))
            for offset in range(0, table.shape[0], 512):
                rows = np.asarray(table[offset:offset+512], dtype=np.float32)
                if not np.isfinite(rows).all():
                    raise ValueError(f'NaN/Inf around row {offset}')
                mx = np.max(np.abs(rows), axis=1, keepdims=True)
                if np.any(mx < 1e-20):
                    raise ValueError(f'zero row near {offset}')
                quant = np.clip(np.rint(rows / mx * 127), -127, 127).astype(np.int8)
                out.write(quant.tobytes(order='C'))
    except Exception:
        args.output.unlink(missing_ok=True)
        raise
    print(f'EG2Q: {table.shape[0]:,} words, {args.output.stat().st_size:,} bytes')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as e:
        sys.exit(f'conversion failed: {e}')
