#!/usr/bin/env python3
"""Stage 32: offline structural preflight for native EmbeddingGemma2 ASR files.

No ML/runtime dependencies. This checks exact binary headers/sizes and word ids,
NOT model weight provenance, numerical parity or ASR accuracy.

  python check_eg2_native.py --model-dir models \
      --reader models/reader.eg2r --table models/word_table.eg2q \
      --vocab models/reader_vocab.txt --require-mel
"""
import argparse
import math
from pathlib import Path
import struct
import sys
from convert_eg2_conformer import SPECS


def read_header(path, fmt, magic, expected_size=None):
    if not path.is_file():
        raise ValueError(f'missing {path}')
    n = struct.calcsize(fmt)
    with path.open('rb') as f:
        b = f.read(n)
    if len(b) != n:
        raise ValueError(f'truncated header: {path}')
    values = struct.unpack(fmt, b)
    if values[0] != magic:
        raise ValueError(f'wrong file type {path}: expected {magic!r}')
    if values[1] != 1:
        raise ValueError(f'unsupported version {values[1]}: {path}')
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f'{path}: expected {expected_size:,} bytes, got {path.stat().st_size:,}')
    return values


def check(args):
    count = 0
    if not args.reader_only:
        stem = args.model_dir / 'subsampler.eg2c'
        subcount = 128*9 + 128 + 32*128*9 + 32 + 1024*1024
        hdr = read_header(stem, '<4s4I', b'EG2C', 20+4*subcount)
        if hdr[2:] != (128,32,1024):
            raise ValueError('unexpected subsampler shape')
        print(f'OK subsampler: {stem}');count += 1
        layer_count = sum(math.prod(shape) for _,shape in SPECS)
        for i in range(12):
            path = args.model_dir / f'layer_{i:02}.eg2l'
            hdr = read_header(path,'<4s6I',b'EG2L',28+4*(layer_count+40))
            if hdr[2:] != (1024,4096,8,5,10):
                raise ValueError(f'unexpected layer shape {path}')
            count += 1
        print(f'OK 12 complete EG2L layers ({layer_count:,} weights + 40 clip scalars each)')
        path = args.model_dir / 'projection.eg2p'
        size = 20 + 4*(1536*1024+1536+512*1536)
        hdr = read_header(path,'<4s4I',b'EG2P',size)
        if hdr[2:] != (1024,1536,512):
            raise ValueError('unexpected final projection shape')
        print(f'OK projection: {path}');count += 1
        mel = args.model_dir / 'mel_filters.bin'
        if mel.exists():
            if not mel.is_file() or mel.stat().st_size != 257*128*8:
                raise ValueError(f'bad mel bank size: {mel}')
            print(f'OK reference mel bank: {mel}');count += 1
        elif args.require_mel:
            raise ValueError(f'missing exact mel filter bank {mel}')
        else:
            print('WARN no mel_filters.bin: native fallback is not guaranteed to match official frontend')

    path = args.reader
    if path:
        h = read_header(path,'<4s5I',b'EG2R')
        layers,d,ff,nfloat = h[2:]
        if d != 512 or ff != 2048 or layers<1 or layers>12:
            raise ValueError(f'unsupported reader dimensions {h[2:]}')
        expected = (2*512+512*512+512) + layers*(
            2*512+3*512*512+3*512+512*512+512+
            2*512+2048*512+2048+512*2048+512) + (
            2*512+768*512+768) + (768+1)
        if nfloat != expected or path.stat().st_size != 24+4*nfloat+4:
            raise ValueError(f'bad reader length or parameter count ({path})')
        with path.open('rb') as f:
            f.seek(-4,2);scale=struct.unpack('<f',f.read(4))[0]
        if not math.isfinite(scale) or not 0<scale<=1e6:
            raise ValueError('bad reader score scale')
        print(f'OK reader: {layers} layers, score scale {scale:.5g}');count+=1
    if args.table:
        h=read_header(args.table,'<4s3I',b'EG2Q')
        rows,width=h[2:]
        if not 0<rows<=1000000 or width!=768 or args.table.stat().st_size!=16+rows*768:
            raise ValueError('bad INT8 word table length or shape')
        print(f'OK word-table: {rows:,} words');count+=1
        if args.vocab:
            with args.vocab.open('rb') as f:
                n=0
                for line in f:
                    if not line.strip() or len(line)>4095:
                        raise ValueError(f'empty or oversized vocab line {n+1}')
                    n+=1
            if n!=rows:
                raise ValueError(f'vocabulary size ({n}) != word table rows ({rows})')
            print(f'OK vocabulary: {n:,} entries');count+=1
    return count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=Path('models'))
    p.add_argument('--reader',type=Path)
    p.add_argument('--table',type=Path)
    p.add_argument('--vocab',type=Path)
    p.add_argument('--require-mel',action='store_true')
    p.add_argument('--reader-only',action='store_true',help='skip full audio encoder files')
    a=p.parse_args()
    if bool(a.table)!=bool(a.vocab):
        p.error('--table and --vocab must be supplied together')
    if a.reader_only and not a.reader and not a.table:
        p.error('--reader-only requires --reader or --table')
    try:
        n=check(a)
    except (OSError,ValueError,struct.error) as exc:
        sys.exit(f'FAIL: {exc}')
    print(f'PASS {n} structural items; ASR parity/accuracy not established')

if __name__=='__main__':
    main()
