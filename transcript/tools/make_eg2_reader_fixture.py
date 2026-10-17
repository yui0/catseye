#!/usr/bin/env python3
"""Create a reproducible, UNTRAINED ReaderA 1-layer fixture for C/Torch parity.
Conversion uses convert_eg2_reader.py; all tensor values are synthetic.
"""
from pathlib import Path
import math
import struct
import subprocess
import sys
import torch
import torch.nn.functional as F

R=Path(__file__).resolve().parent
FOLDER=R.parent/'fixtures'
FOLDER.mkdir(exist_ok=True)

def get(shape, name):
    n=math.prod(shape)
    seq=torch.arange(n, dtype=torch.int64)
    vals=((seq*7+sum(map(ord,name)))%17-8).to(torch.float32)
    if name.endswith('.weight') and ('ln' in name):
        v=1+vals*.012
    elif name.endswith('.weight') and 'blank' in name:
        v=vals*.0002
    elif name.endswith('.weight'):
        v=vals*.0006
    else:
        v=vals*.002
    return v.reshape(shape)

s={}
for name,shape in [('body.ln_in.weight',(512,)),('body.ln_in.bias',(512,)),
 ('body.inp.weight',(512,512)),('body.inp.bias',(512,)),
 ('body.ln_out.weight',(512,)),('body.ln_out.bias',(512,)),
 ('body.out.weight',(768,512)),('body.out.bias',(768,)),
 ('blank.weight',(1,768)),('blank.bias',(1,))]:s[name]=get(shape,name)
for i in range(1):
 for name,shape in [('ln1.weight',(512,)),('ln1.bias',(512,)),
 ('qkv.weight',(1536,512)),('qkv.bias',(1536,)),
 ('proj.weight',(512,512)),('proj.bias',(512,)),
 ('ln2.weight',(512,)),('ln2.bias',(512,)),
 ('ff1.weight',(2048,512)),('ff1.bias',(2048,)),
 ('ff2.weight',(512,2048)),('ff2.bias',(512,))]:
    full=f'body.blocks.{i}.{name}';s[full]=get(shape,full)
s['log_scale']=torch.tensor(math.log(20.0))
# The synthetic fixture is trained on nothing and must not be confused with real weights.
ckpt=FOLDER/'eg2_mock_reader_stage28.pt'
torch.save(s,ckpt)
subprocess.run([sys.executable,str(R/'convert_eg2_reader.py'),str(ckpt),str(FOLDER/'eg2_mock_reader_stage28.eg2r')],check=True)
ckpt.unlink()

T=4
x=torch.tensor([[0.06*math.sin((t+1)*(j+1)*.08)+.03*math.cos(j*.047) for j in range(512)] for t in range(T)],dtype=torch.float32)
with (FOLDER/'eg2_mock_reader_stage28.eg2t').open('wb') as fp:
 fp.write(struct.pack('<4sII',b'EG2T',1,T));fp.write(x.numpy().astype('<f4').tobytes())

def layernorm(v,p):
 return F.layer_norm(v,(v.shape[-1],),s[p+'.weight'],s[p+'.bias'],1e-5)
def linear(v,p):
 return F.linear(v,s[p+'.weight'],s[p+'.bias'])
with torch.no_grad():
 h=linear(layernorm(x,'body.ln_in'),'body.inp')
 for i in range(1):
    p=f'body.blocks.{i}'
    qkv=linear(layernorm(h,p+'.ln1'),p+'.qkv').view(T,3,8,64).permute(1,2,0,3)
    q,k,v=qkv[0],qkv[1],qkv[2]
    freqs=torch.arange(T,dtype=torch.float32)[:,None] * (10000**(-torch.arange(0,64,2,dtype=torch.float32)/64))[None,:]
    angles=torch.cat([freqs,freqs],dim=-1)
    c=angles.cos()[None,:,:];si=angles.sin()[None,:,:]
    def rotate(q): return torch.cat([-q[...,32:],q[...,:32]],dim=-1)
    q=q*c+rotate(q)*si;k=k*c+rotate(k)*si
    attn=F.scaled_dot_product_attention(q,k,v)
    h=h+linear(attn.permute(1,0,2).reshape(T,512),p+'.proj')
    h=h+linear(F.gelu(linear(layernorm(h,p+'.ln2'),p+'.ff1')),p+'.ff2')
 out=linear(layernorm(h,'body.ln_out'),'body.out')
 blank=linear(out,'blank').flatten()
 proj=F.normalize(out,dim=-1)
with (FOLDER/'eg2_mock_reader_stage28.expected').open('wb') as fp:
 fp.write(proj.numpy().astype('<f4').tobytes());fp.write(blank.numpy().astype('<f4').tobytes())
print('Reader 1-layer synthetic reference: projection shape',tuple(proj.shape),'blank',blank.tolist())
