#!/usr/bin/env python3
"""Optional OFFLINE converter: EmbeddingGemma 2 audio encoder soft_tokens.npy -> EG2T.
Source can be shape [T,512] or [1,T,512], T between 1 and 750.
No Python required at runtime; this only converts your own encoder output.
"""
import argparse
from pathlib import Path
import struct
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    try:
        import numpy as np
    except ImportError as ex:
        parser.error(f'conversion-time NumPy required: {ex}')
    a=np.load(args.input,allow_pickle=False)
    if a.ndim==3 and a.shape[0]==1:
        a=a[0]
    if a.ndim!=2 or a.shape[1]!=512 or not 1<=a.shape[0]<=750:
        parser.error(f'expected [T,512] with 1<=T<=750; got {a.shape}')
    a=np.asarray(a,dtype='<f4',order='C')
    if not np.isfinite(a).all():
        parser.error('contains non-finite values')
    with args.output.open('wb') as f:
        f.write(struct.pack('<4sII',b'EG2T',1,len(a)))
        f.write(a.tobytes(order='C'))
    print(f'EG2T: {a.shape[0]} soft tokens, {args.output.stat().st_size} bytes')

if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError) as e:
        sys.exit(f'conversion failed: {e}')
