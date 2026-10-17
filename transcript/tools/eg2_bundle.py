#!/usr/bin/env python3
"""EG2PKG1 format: header(16) + entries(80 each) + raw, uncompressed components.
This is a container, NOT additional compression or quantization.
"""
import argparse
from pathlib import Path
import struct
import os
import re
import zipfile

MAGIC = b'EG2PKG1\0'
HEADER = struct.Struct('<8sII')
ENTRY = struct.Struct('<64sQQ')
ENCODER = ['subsampler.eg2c', *[f'layer_{i:02}.eg2l' for i in range(12)],
           'projection.eg2p', 'mel_filters.bin']
LANGUAGE = ['reader.eg2r', 'word_table.eg2q', 'reader_vocab.txt']
LIMIT = 64


def _components(path):
    with Path(path).open('rb') as f:
        hdr = f.read(HEADER.size)
        if len(hdr) != HEADER.size:
            raise ValueError('Truncated EG2 package header')
        magic, count, reserved = HEADER.unpack(hdr)
        if magic != MAGIC or not 1 <= count <= LIMIT or reserved:
            raise ValueError('Invalid EG2 package header')
        size = f.seek(0, os.SEEK_END)
        f.seek(HEADER.size)
        names, blocks = set(), []
        floor = HEADER.size + count*ENTRY.size
        for _ in range(count):
            data = f.read(ENTRY.size)
            if len(data) != ENTRY.size:
                raise ValueError('Truncated EG2 package index')
            name, pos, length = ENTRY.unpack(data)
            name = name.split(b'\0', 1)[0]
            if not name or len(name) > 63 or b'/' in name or b'\\' in name:
                raise ValueError('Invalid package component name')
            name = name.decode('ascii')
            if name in names or not length or pos < floor or pos > size or length > size-pos:
                raise ValueError('Invalid package component range / duplicate')
            names.add(name)
            blocks.append((name, pos, length))
        ordered = sorted(blocks, key=lambda b:b[1])
        for (_, pos, length), (_, nextpos, _) in zip(ordered, ordered[1:]):
            if pos + length > nextpos:
                raise ValueError('Overlapping package components')
        return blocks


def pack(folder, names, output, force=False):
    folder, output = Path(folder), Path(output)
    if output.exists() and not force:
        raise FileExistsError(f'{output}: use --force to replace')
    paths = [folder / n for n in names]
    for p in paths:
        if not p.is_file() or not p.stat().st_size:
            raise ValueError(f'Missing or empty model file: {p}')
    offset = HEADER.size + len(paths)*ENTRY.size
    blocks = []
    for n,p in zip(names,paths):
        offset = (offset + 7) & ~7
        blocks.append((n, offset, p.stat().st_size))
        offset += p.stat().st_size
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + '.partial')
    try:
        with tmp.open('wb') as f:
            f.write(HEADER.pack(MAGIC, len(paths), 0))
            for n,pos,length in blocks:
                f.write(ENTRY.pack(n.encode('ascii'),pos,length))
            for (_,pos,length),p in zip(blocks,paths):
                f.write(b'\0' * (pos-f.tell()))
                with p.open('rb') as source:
                    while chunk := source.read(4*1024*1024):
                        f.write(chunk)
                if f.tell() != pos + length:
                    raise IOError(f'{p} changed during packing')
        if _components(tmp) != blocks:
            raise ValueError('Package verification failed')
        os.replace(tmp, output)
    finally:
        if tmp.exists(): tmp.unlink()
    print(f'{output}: {len(blocks)} components, {output.stat().st_size:,} bytes')
    return output


def pack_zip(archive, mode, out, force=False):
    """Repack a legacy model ZIP directly, with no giant extracted staging tree."""
    out = Path(out)
    archive = Path(archive)
    if mode not in ('full', 'en', 'ja'):
        raise ValueError(f'Unknown mode: {mode}')
    tasks = ([('encoder.eg2m', ENCODER), ('en.eg2m', LANGUAGE)] if mode == 'full'
             else [(f'{mode}.eg2m', LANGUAGE)])
    with zipfile.ZipFile(archive) as zin:
        by_base = {}
        for info in zin.infolist():
            if info.is_dir(): continue
            base=Path(info.filename).name
            if base in set(ENCODER + LANGUAGE):
                by_base.setdefault(base, []).append(info)
        out.mkdir(parents=True, exist_ok=True)
        for filename, components in tasks:
            if (out/filename).exists() and not force:
                raise FileExistsError(f'{out/filename}: use --force to replace')
            for n in components:
                if len(by_base.get(n, [])) != 1 or by_base[n][0].file_size < 1:
                    raise ValueError(f'Missing, empty or ambiguous ZIP component: {n}')
            offset = HEADER.size + len(components)*ENTRY.size
            offsets = []
            for name in components:
                offset = (offset + 7) & ~7
                info = by_base[name][0]
                offsets.append((name, offset, info.file_size))
                offset += info.file_size
            output=out/filename
            tmp=output.with_name(output.name+'.partial')
            try:
                with tmp.open('wb') as f:
                    f.write(HEADER.pack(MAGIC,len(components),0))
                    for name,pos,length in offsets:
                        f.write(ENTRY.pack(name.encode('ascii'),pos,length))
                    for name,pos,length in offsets:
                        f.write(b'\0'*(pos-f.tell()))
                        with zin.open(by_base[name][0]) as src:
                            while data:=src.read(4*1024*1024):
                                f.write(data)
                        if f.tell()!=pos+length:
                            raise IOError(f'ZIP component size changed: {name}')
                if _components(tmp)!=offsets:
                    raise ValueError('Invalid package index')
                os.replace(tmp,output)
            finally:
                if tmp.exists():tmp.unlink()
            print(f'{output}: {len(components)} components, {output.stat().st_size:,} bytes')


# ---------------------------------------------------------------------------
# EG2L v2: float16 storage for the large Conformer matrices (--to-fp16).
# The C loader widens them to float32, so inference arithmetic is unchanged;
# only the file (and read bandwidth) halves.  Norm gains, the depthwise
# kernel and the clip bounds stay float32.  Tensor order must match
# cs_eg2_layer_tensor_count / cs_eg2_layer_tensor_half in cats_rnnt.h.
# ---------------------------------------------------------------------------
LAYER_NAME = re.compile(r'layer_\d\d\.eg2l')
LAYER_HEADER = struct.Struct('<4s6I')      # magic, version, d, ff, heads, kernel, clips
HALF_TENSORS = {1, 2, 5, 6, 7, 8, 9, 13, 16, 18, 19}


def _layer_counts(d, ff, heads, kernel):
    dd = d * d
    return [d, ff*d, d*ff, d, d, dd, dd, dd, dd, dd, d//heads, d, d,
            2*dd, d*kernel, d, dd, d, ff*d, d*ff, d, d]


def _layer_header(f, pos, length):
    f.seek(pos)
    raw = f.read(LAYER_HEADER.size)
    magic, ver, d, ff, heads, kernel, clips = LAYER_HEADER.unpack(raw)
    if magic != b'EG2L' or ver != 1 or not (d and ff and heads and kernel) or d % heads:
        raise ValueError('Not an EG2L v1 layer (already converted?)')
    counts = _layer_counts(d, ff, heads, kernel)
    if LAYER_HEADER.size + sum(counts)*4 + clips*16 != length:
        raise ValueError('EG2L layer size does not match its header')
    return (d, ff, heads, kernel, clips), counts


def _layer_fp16_size(f, pos, length):
    (_, _, _, _, clips), counts = _layer_header(f, pos, length)
    return (LAYER_HEADER.size + clips*16 +
            sum(c*(2 if i in HALF_TENSORS else 4) for i, c in enumerate(counts)))


def _emit_layer_fp16(f, pos, length, out, stats):
    import numpy as np
    (d, ff, heads, kernel, clips), counts = _layer_header(f, pos, length)
    out.write(LAYER_HEADER.pack(b'EG2L', 2, d, ff, heads, kernel, clips))
    f.seek(pos + LAYER_HEADER.size)
    for i, count in enumerate(counts):
        a = np.fromfile(f, dtype='<f4', count=count)
        if a.size != count or not np.isfinite(a).all():
            raise ValueError('Layer tensor %d is truncated or not finite' % i)
        if i in HALF_TENSORS:
            h = a.astype('<f2')
            if not np.isfinite(h).all():
                raise ValueError('Layer tensor %d overflows float16' % i)
            scale = float(np.abs(a).max()) or 1.0
            stats['rel_err'] = max(stats['rel_err'],
                                   float(np.abs(h.astype('<f4') - a).max()) / scale)
            out.write(h.tobytes())
        else:
            out.write(a.tobytes())
    out.write(f.read(clips*16))


def convert_fp16(src, out_dir, force=False):
    """Rewrite an .eg2m package with every EG2L v1 layer converted to v2."""
    src, out_dir = Path(src), Path(out_dir)
    dst = out_dir / src.name
    if dst.exists() and dst.resolve() == src.resolve():
        raise ValueError('--out must be a different directory from the source package')
    if dst.exists() and not force:
        raise FileExistsError(f'{dst}: use --force to replace')
    blocks = _components(src)
    stats = {'rel_err': 0.0}
    offset = HEADER.size + len(blocks)*ENTRY.size
    plan = []
    with src.open('rb') as f:
        for name, pos, length in blocks:
            offset = (offset + 7) & ~7
            convert = bool(LAYER_NAME.fullmatch(name))
            new_len = _layer_fp16_size(f, pos, length) if convert else length
            plan.append((name, pos, length, offset, new_len, convert))
            offset += new_len
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.partial')
    try:
        with src.open('rb') as f, tmp.open('wb') as out:
            out.write(HEADER.pack(MAGIC, len(plan), 0))
            for name, _, _, new_pos, new_len, _ in plan:
                out.write(ENTRY.pack(name.encode('ascii'), new_pos, new_len))
            for name, pos, length, new_pos, new_len, convert in plan:
                out.write(b'\0' * (new_pos - out.tell()))
                if convert:
                    _emit_layer_fp16(f, pos, length, out, stats)
                else:
                    f.seek(pos)
                    left = length
                    while left:
                        chunk = f.read(min(left, 4*1024*1024))
                        if not chunk:
                            raise IOError(f'{src}: truncated component {name}')
                        out.write(chunk)
                        left -= len(chunk)
                if out.tell() != new_pos + new_len:
                    raise IOError(f'{name}: unexpected converted size')
        if _components(tmp) != [(n, p, l) for n, _, _, p, l, _ in plan]:
            raise ValueError('Package verification failed')
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    n = sum(1 for p in plan if p[5])
    print(f'{dst}: {len(plan)} components ({n} layers -> float16), '
          f'{src.stat().st_size:,} -> {dst.stat().st_size:,} bytes, '
          f'max relative weight error {stats["rel_err"]:.2e}')


def main(argv=None):
    a = argparse.ArgumentParser(description='Pack native EG2 components as a single .eg2m file')
    a.add_argument('--encoder-dir', type=Path)
    a.add_argument('--en-dir', type=Path)
    a.add_argument('--ja-dir', type=Path)
    a.add_argument('--out', type=Path, default=Path('models'))
    a.add_argument('--force', action='store_true')
    a.add_argument('--verify', type=Path, help='verify/list an existing .eg2m')
    a.add_argument('--to-fp16', type=Path, metavar='PKG',
                   help='rewrite an encoder .eg2m with float16 layer matrices into --out (about half the size)')
    a.add_argument('--from-zip', type=Path, help='legacy Stage34/35 ZIP; no manual extraction')
    a.add_argument('--source', choices=['full','en','ja'], default='full', help='ZIP contents (full English model, English Reader, or Japanese Reader)')
    args = a.parse_args(argv)
    if args.verify:
        for name,pos,length in _components(args.verify):
            print(f'{name:24} offset={pos:12} bytes={length}')
        return
    if args.to_fp16:
        convert_fp16(args.to_fp16, args.out, args.force)
        return
    if args.from_zip:
        pack_zip(args.from_zip,args.source,args.out,args.force)
        return
    if not (args.encoder_dir or args.en_dir or args.ja_dir):
        a.error('Specify --encoder-dir, --en-dir and/or --ja-dir')
    for d, filename, components in ((args.encoder_dir,'encoder.eg2m',ENCODER),
                                    (args.en_dir,'en.eg2m',LANGUAGE),
                                    (args.ja_dir,'ja.eg2m',LANGUAGE)):
        if d is not None:
            pack(d, components, args.out/filename, args.force)


if __name__=='__main__': main()
