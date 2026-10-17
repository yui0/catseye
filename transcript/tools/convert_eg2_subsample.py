#!/usr/bin/env python3
"""One-time dependency-free converter for Gemma4 audio subsampler safetensors.

Usage: python convert_eg2_subsample.py /models/embeddinggemma2/ subsample.eg2c
Accepts one .safetensors or a directory with shards; accepts BF16/F16/F32.
EG2C includes only conv0 / conv1 / input projection, not the audio tower.
"""
import json
import math
import mmap
import pathlib
import struct
import sys
from contextlib import ExitStack

SPECS = (
    ("layer0.conv.weight", (128, 1, 3, 3)),
    ("layer0.norm.weight", (128,)),
    ("layer1.conv.weight", (32, 128, 3, 3)),
    ("layer1.norm.weight", (32,)),
    ("input_proj_linear.weight", (1024, 1024)),
)


def to_floats(data, dtype):
    if dtype == "F32":
        if len(data) % 4:
            raise ValueError("F32 size misaligned")
        for (v,) in struct.iter_unpack("<f", data):
            yield v
    elif dtype in ("F16", "BF16"):
        if len(data) % 2:
            raise ValueError("F16/BF16 size misaligned")
        for (bits,) in struct.iter_unpack("<H", data):
            if dtype == "F16":
                yield struct.unpack("<e", struct.pack("<H", bits))[0]
            else:
                yield struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    else:
        raise ValueError(f"unsupported dtype: {dtype}")



def write_fp32(out, raw, dtype, name='tensor', chunk_elements=1 << 19):
    """Stream BF16/F16/F32 safetensors to little-endian FP32 with bounded memory.

    NumPy is optional: use vectorized conversion for Colab, retain the original
    pure-Python fallback. Checks every chunk for NaN/Inf before writing.
    """
    elem_size = 4 if dtype == 'F32' else 2 if dtype in ('F16', 'BF16') else 0
    if not elem_size or len(raw) % elem_size:
        raise ValueError(f'{name}: invalid {dtype} tensor byte length')
    try:
        import numpy as np
    except ImportError:
        count = 0
        for value in to_floats(raw, dtype):
            if not math.isfinite(value):
                raise ValueError(f'nonfinite weights in {name}')
            out.write(struct.pack('<f', value))
            count += 1
        return count
    count = 0
    stride = chunk_elements * elem_size
    for offset in range(0, len(raw), stride):
        section = raw[offset:offset+stride]
        if dtype == 'F32':
            f32 = np.frombuffer(section, dtype='<f4')
        elif dtype == 'F16':
            f32 = np.frombuffer(section, dtype='<f2').astype('<f4')
        else:
            # Convert BF16 bits without rounding through float16.
            bits = np.frombuffer(section, dtype='<u2').astype('<u4') << 16
            f32 = bits.view('<f4')
        if not np.isfinite(f32).all():
            raise ValueError(f'nonfinite weights in {name}')
        out.write(f32.tobytes())
        count += f32.size
    return count


def convert(src, dst):
    folder = pathlib.Path(src)
    paths = sorted(folder.glob('*.safetensors')) if folder.is_dir() else [folder]
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError("no .safetensors shards found")
    found = {}
    with ExitStack() as stack:
        for path in paths:
            f = stack.enter_context(path.open('rb'))
            mm = stack.enter_context(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
            if len(mm) < 8:
                raise ValueError(f"bad safetensors header: {path}")
            (length,) = struct.unpack_from('<Q', mm, 0)
            if length > 100_000_000 or length + 8 > len(mm):
                raise ValueError(f"bad safetensors metadata size: {path}")
            meta = json.loads(mm[8:8+length]);base = 8 + length
            for name, info in meta.items():
                if name == '__metadata__':
                    continue
                # Reject ambiguously similar tensors, do not silently guess.
                matches = [s for s, _ in SPECS if name.endswith('.'+s) or name == s]
                if not matches or 'audio_tower' not in name:
                    continue
                suffix = matches[0]
                if suffix in found:
                    raise ValueError(f"multiple tensors match {suffix}")
                shape = tuple(info.get('shape', []))
                expected = next(shape for s, shape in SPECS if s == suffix)
                if shape != expected:
                    raise ValueError(f"{name} shape {shape}, expected {expected}")
                begin, end = info['data_offsets']
                if begin < 0 or end < begin or base+end > len(mm):
                    raise ValueError(f"invalid offsets: {name}")
                dtype=info['dtype']
                element_size = 4 if dtype == 'F32' else 2 if dtype in ('BF16', 'F16') else 0
                if end - begin != math.prod(shape) * element_size:
                    raise ValueError(f"invalid size or dtype: {name}")
                found[suffix] = (mm,base+begin,base+end,dtype,name)
        missing=[s for s, _ in SPECS if s not in found]
        if missing:
            raise ValueError('missing audio tower weights: '+', '.join(missing))
        dst = pathlib.Path(dst)
        tmp = dst.with_suffix(dst.suffix+'.tmp')
        try:
            with tmp.open('wb') as out:
                out.write(struct.pack('<4s4I',b'EG2C',1,128,32,1024))
                for suffix, shape in SPECS:
                    mm,begin,end,dtype,name = found[suffix]
                    count=write_fp32(out,memoryview(mm)[begin:end],dtype,name)
                    if count != math.prod(shape):
                        raise ValueError(f"bad tensor length: {name}")
                    print(f"  {name}: {shape} {dtype} -> F32")
            tmp.replace(dst)
        finally:
            if tmp.exists():
                tmp.unlink()
    print(f'Wrote {dst} ({dst.stat().st_size} bytes). Subsampler only; not full encoder.')

if __name__=='__main__':
    if len(sys.argv)!=3:
        print(__doc__);raise SystemExit(2)
    try:
        convert(sys.argv[1],sys.argv[2])
    except (ValueError, OSError) as e:
        print('converter:',e,file=sys.stderr);raise SystemExit(1)
