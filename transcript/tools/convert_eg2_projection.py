#!/usr/bin/env python3
"""Convert original Gemma4 EmbeddingGemma2 audio tower final projection to EG2P.

Usage: python convert_eg2_projection.py /path/to/safetensors_or_dir/ projection.eg2p
This script requires only Python standard library; not needed at inference time.
"""
import sys,struct,mmap,json,math,pathlib
from contextlib import ExitStack
from convert_eg2_subsample import write_fp32
SPECS=[('audio_tower.output_proj.weight',(1536,1024)),
       ('audio_tower.output_proj.bias',(1536,)),
       ('embed_audio.embedding_projection.weight',(512,1536))]
def main(src,dst):
  path=pathlib.Path(src)
  paths=sorted(path.glob('*.safetensors')) if path.is_dir() else [path]
  if not paths or any(not p.is_file() for p in paths):raise ValueError('No safetensors')
  found={}
  with ExitStack() as stack:
    for file in paths:
      f=stack.enter_context(file.open('rb'))
      mm=stack.enter_context(mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ))
      if len(mm)<8:raise ValueError('File too small')
      n=struct.unpack_from('<Q',mm,0)[0]
      if n>100000000 or n+8>len(mm):raise ValueError('Bad metadata length')
      meta=json.loads(mm[8:8+n]);base=n+8
      for name,inf in meta.items():
        if name=='__metadata__':continue
        for suffix,shape in SPECS:
          if name==suffix or name.endswith('.'+suffix):
            if suffix in found:raise ValueError(f'Duplicate {suffix}')
            if tuple(inf['shape'])!=shape:raise ValueError(f'Wrong shape {name}')
            dtype=inf['dtype'];element=4 if dtype=='F32' else 2 if dtype in ('F16','BF16') else 0
            a,b=inf['data_offsets']
            if not element or a<0 or b<a or base+b>len(mm) or b-a!=math.prod(shape)*element:
                raise ValueError(f'Invalid tensor {name}')
            found[suffix]=(mm,base+a,base+b,dtype)
    for name,_ in SPECS:
      if name not in found:raise ValueError(f'Missing {name}')
    dst=pathlib.Path(dst);tmp=dst.with_suffix(dst.suffix+'.tmp')
    try:
      with tmp.open('wb') as out:
        out.write(struct.pack('<4s4I',b'EG2P',1,1024,1536,512))
        for name,shape in SPECS:
          mm,a,b,dtype=found[name]
          cnt=write_fp32(out,memoryview(mm)[a:b],dtype,name)
          if cnt!=math.prod(shape):raise ValueError('Incorrect element count')
      tmp.replace(dst)
    finally:tmp.unlink(missing_ok=True)
  print(f'Wrote {dst} ({dst.stat().st_size:,} bytes)')
if __name__=='__main__':
  if len(sys.argv)!=3:print(__doc__);sys.exit(2)
  try:main(*sys.argv[1:])
  except (ValueError,OSError) as e:print(e,file=sys.stderr);sys.exit(1)
