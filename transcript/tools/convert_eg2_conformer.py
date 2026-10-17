#!/usr/bin/env python3
"""Export the 12 Gemma4/EmbeddingGemma2 audio Conformer layers to native EG2L.

    python convert_eg2_conformer.py /path/to/model-safetensors /path/to/layers/
    python convert_eg2_conformer.py /path/to/model-safetensors /path/to/layers/ --layer 0

Python 3 standard library only. Supports BF16, F16 and F32 tensors and
clipped-linear input/output extrema. The output is large (~1.2 GB in F32).
The output is NOT the full encoder: conv stem and output projection are separate.
"""
import argparse
import json
import math
import mmap
import pathlib
import struct
from contextlib import ExitStack
from convert_eg2_subsample import to_floats, write_fp32

D, FF, HEADS, KERNEL = 1024, 4096, 8, 5
SPECS = [
 ('feed_forward1.pre_layer_norm.weight', (D,)),
 ('feed_forward1.ffw_layer_1.linear.weight', (FF,D)),
 ('feed_forward1.ffw_layer_2.linear.weight', (D,FF)),
 ('feed_forward1.post_layer_norm.weight', (D,)),
 ('norm_pre_attn.weight', (D,)),
 ('self_attn.q_proj.linear.weight', (D,D)),
 ('self_attn.k_proj.linear.weight', (D,D)),
 ('self_attn.v_proj.linear.weight', (D,D)),
 ('self_attn.post.linear.weight', (D,D)),
 ('self_attn.relative_k_proj.weight', (D,D)),
 ('self_attn.per_dim_scale', (D//HEADS,)),
 ('norm_post_attn.weight', (D,)),
 ('lconv1d.pre_layer_norm.weight', (D,)),
 ('lconv1d.linear_start.linear.weight', (2*D,D)),
 ('lconv1d.depthwise_conv1d.weight', (D,1,KERNEL)),
 ('lconv1d.conv_norm.weight', (D,)),
 ('lconv1d.linear_end.linear.weight', (D,D)),
 ('feed_forward2.pre_layer_norm.weight', (D,)),
 ('feed_forward2.ffw_layer_1.linear.weight', (FF,D)),
 ('feed_forward2.ffw_layer_2.linear.weight', (D,FF)),
 ('feed_forward2.post_layer_norm.weight', (D,)),
 ('norm_out.weight', (D,)),
]
CLIPPED = [
 'feed_forward1.ffw_layer_1', 'feed_forward1.ffw_layer_2',
 'self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.post',
 'lconv1d.linear_start','lconv1d.linear_end',
 'feed_forward2.ffw_layer_1','feed_forward2.ffw_layer_2',
]


def gather(paths, stack):
    entries = {}
    for path in paths:
        f = stack.enter_context(path.open('rb'))
        mm = stack.enter_context(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        if len(mm) < 8:
            raise ValueError(f'Invalid safetensors file {path}')
        n = struct.unpack_from('<Q', mm)[0]
        if n > 100_000_000 or n + 8 > len(mm):
            raise ValueError(f'Invalid header length in {path}')
        metadata = json.loads(mm[8:8+n]);base = n+8
        for name, v in metadata.items():
            if name == '__metadata__':
                continue
            if name in entries:
                raise ValueError(f'Duplicate tensor {name}')
            begin,end=v['data_offsets']
            shape=v['shape']; dtype=v['dtype']; size=4 if dtype=='F32' else 2 if dtype in ('BF16','F16') else 0
            if not size or begin<0 or end<begin or base+end>len(mm) or end-begin!=math.prod(shape)*size:
                # Ignore unsupported dtype only for non-audio tensors in the same checkpoint.
                if 'audio_tower.layers.' in name:
                    raise ValueError(f'Invalid dtype, shape or offsets: {name}')
                continue
            if '.audio_tower.layers.' in name or name.startswith('audio_tower.layers.'):
                entries[name]=(mm,base+begin,base+end,dtype,tuple(shape))
    return entries


def extract(entries,full_name,expected):
    names=[name for name in entries if name==full_name or name.endswith('.'+full_name)]
    if len(names)!=1:
        raise ValueError(f'Expected exactly one tensor {full_name}; got {len(names)}')
    name=names[0];mm,start,end,dtype,shape=entries[name]
    # PyTorch Conv1D uses [C,1,K]; MLX exports [C,K,1].
    if shape!=expected and not (full_name.endswith('depthwise_conv1d.weight') and shape==(D,KERNEL,1)):
        raise ValueError(f'{name}: expected {expected}, got {shape}')
    return mm,start,end,dtype


def write_tensor(out,entries,name,shape):
    mm,start,end,dtype=extract(entries,name,shape)
    n=write_fp32(out,memoryview(mm)[start:end],dtype,name)
    if n!=math.prod(shape):
        raise ValueError(f'Wrong element count {name}')


def write_clips(out,entries,base):
    for module in CLIPPED:
        vals=[]
        for suffix in ('input_min','input_max','output_min','output_max'):
            name=base+module+'.'+suffix
            candidates=[x for x in entries if x==name or x.endswith('.'+name)]
            if candidates:
                mm,a,b,dtype=extract(entries,name,())
                values=list(to_floats(memoryview(mm)[a:b],dtype))
                if len(values)!=1 or math.isnan(values[0]):
                    raise ValueError(f'Invalid clip value {name}')
                vals.extend(values)
            else:
                # When not trained with clipped linears, leave bound unbounded.
                vals.append(float('-inf') if suffix.endswith('_min') else float('inf'))
        if vals[0]>vals[1] or vals[2]>vals[3]:
            raise ValueError(f'Invalid clip interval in {base+module}')
        out.write(struct.pack('<4f',*vals))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=pathlib.Path)
    p.add_argument('output',type=pathlib.Path)
    p.add_argument('--layer',type=int,choices=range(12),default=None)
    a=p.parse_args()
    paths=sorted(a.checkpoint.glob('*.safetensors')) if a.checkpoint.is_dir() else [a.checkpoint]
    if not paths or any(not f.is_file() for f in paths):
        p.error('no safetensors found')
    a.output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        entries=gather(paths,stack)
        layers=range(12) if a.layer is None else [a.layer]
        for i in layers:
            base=f'audio_tower.layers.{i}.'
            outfile=a.output/f'layer_{i:02}.eg2l'
            tmp=outfile.with_suffix('.eg2l.tmp')
            try:
                with tmp.open('wb') as out:
                    out.write(struct.pack('<4s6I',b'EG2L',1,D,FF,HEADS,KERNEL,10))
                    for suffix,shape in SPECS:
                        write_tensor(out,entries,base+suffix,shape)
                    write_clips(out,entries,base)
                tmp.replace(outfile)
                print(f'Layer {i:02d} -> {outfile} ({outfile.stat().st_size:,} bytes)')
            finally:
                tmp.unlink(missing_ok=True)
    print('Converted audio Conformer weights only; model-owned Gemma license still applies.')

if __name__=='__main__':
    main()
