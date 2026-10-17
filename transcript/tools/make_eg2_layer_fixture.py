#!/usr/bin/env python3
"""Build small synthetic EG2L test fixtures and PyTorch reference output.
Conversion/inference do NOT depend on PyTorch; only fixture regeneration does.
"""
import math, pathlib, struct
import torch
import torch.nn.functional as F

D=8; FF=32; HEADS=2; HD=D//HEADS; KERNEL=5; T=18
root=pathlib.Path(__file__).resolve().parent.parent/'fixtures';root.mkdir(exist_ok=True)
torch.manual_seed(13030)
shapes=[(D,),(FF,D),(D,FF),(D,),(D,), (D,D),(D,D),(D,D),(D,D),(D,D),(HD,), (D,), (D,), (2*D,D), (D,KERNEL), (D,), (D,D), (D,), (FF,D),(D,FF),(D,), (D,)]
weights=[]
for i,shape in enumerate(shapes):
    z=torch.randn(shape,dtype=torch.float32)*0.04
    if len(shape)==1 and shape[0]==D: z=1+z
    weights.append(z)
# All clipping values are real and tested; allowing input and output to saturate.
clips=torch.tensor([[-.2,.21,-.3,.31]]*10,dtype=torch.float32)

# float32 model file
path=root/'eg2_mock_layer_stage30.eg2l'
with path.open('wb') as f:
    f.write(struct.pack('<4s6I',b'EG2L',1,D,FF,HEADS,KERNEL,10))
    for w in weights:f.write(w.contiguous().numpy().astype('<f4').tobytes())
    f.write(clips.numpy().astype('<f4').tobytes())

X=torch.randn(T,D,dtype=torch.float32)*0.12
X.tofile=str(X.dtype)
(root/'eg2_mock_layer_stage30.input').write_bytes(X.numpy().astype('<f4').tobytes())

def clipped_linear(x,w,clip):
    c=clip.tolist()
    return F.linear(torch.clamp(x,c[0],c[1]),w).clamp(c[2],c[3])
def rms(x,weight):
    return x*torch.pow(x.float().square().mean(-1,keepdim=True)+1e-6,-0.5)*weight

def ff(x,base,c):
    h=rms(x,weights[base]);h=clipped_linear(h,weights[base+1],clips[c]);h=F.silu(h)
    h=clipped_linear(h,weights[base+2],clips[c+1]);h=rms(h,weights[base+3]);return x+.5*h

def attn(x,valid):
    y=rms(x,weights[4]);q=clipped_linear(y,weights[5],clips[2]).reshape(T,HEADS,HD)
    k=clipped_linear(y,weights[6],clips[3]).reshape(T,HEADS,HD)
    v=clipped_linear(y,weights[7],clips[4]).reshape(T,HEADS,HD)
    q=q*(HD**-0.5)/math.log(2)*F.softplus(weights[10]).reshape(1,1,HD)
    k=k*(math.log1p(math.e)/math.log(2))
    inv=torch.exp(torch.arange(D//2)*(-math.log(10000)/(D//2-1)))
    times=torch.arange(12,-1,-1,dtype=torch.float32)[:,None]*inv[None,:]
    pos=torch.cat([times.sin(),times.cos()],dim=1)
    r=F.linear(pos,weights[9]).reshape(13,HEADS,HD)
    out=torch.zeros_like(v)
    # Exactly PyTorch blocked matrix-ac/matrix-bd with _rel_shift.
    for b in range((T+11)//12):
        a=torch.zeros(12,HEADS,HD)
        ky=torch.zeros(24,HEADS,HD)
        val=torch.zeros(24,HEADS,HD)
        for qi in range(12):
            t=b*12+qi
            if t<T:a[qi]=q[t]
        for ki in range(24):
            t=b*12-12+ki
            if 0<=t<T:ky[ki]=k[t];val[ki]=v[t]
        ac=torch.einsum('qhd,khd->hqk',a,ky)
        bd=torch.einsum('qhd,phd->hqp',a,r)
        bd=F.pad(bd,(0,12)).reshape(HEADS,-1)[:,:12*24].reshape(HEADS,12,24)
        logits=50*(ac+bd).div(50).tanh()
        for qi in range(12):
            t=b*12+qi
            for ki in range(24):
                kt=b*12-12+ki
                if kt<0 or kt>=valid or kt>t or t-kt>=12:logits[:,qi,ki]=-1e9
        sm=F.softmax(logits,dim=-1)
        outblock=torch.einsum('hqk,khd->qhd',sm,val)
        for qi in range(12):
            if b*12+qi<T:out[b*12+qi]=outblock[qi]
    y=clipped_linear(out.reshape(T,D),weights[8],clips[5]);return x+rms(y,weights[11])

def conv(x):
    y=rms(x,weights[12]);y=clipped_linear(y,weights[13],clips[6]);y=F.glu(y,dim=-1)
    y=F.conv1d(F.pad(y.T[None,:,:],(KERNEL-1,0)),weights[14].reshape(D,1,KERNEL),groups=D)[0].T
    y=rms(y,weights[15]);y=F.silu(y)
    y=clipped_linear(y,weights[16],clips[7]);return x+y

for valid,tag in ((T,'full'),(T-3,'masked')):
    y=ff(X,0,0)
    y=attn(y,valid)
    y=conv(y)
    y=ff(y,17,8)
    y=rms(y,weights[21])
    (root/f'eg2_mock_layer_stage30.{tag}.expected').write_bytes(y.numpy().astype('<f4').tobytes())
    print(tag,'range',round(float(y.min()),6),round(float(y.max()),6))
print('wrote',path.name, path.stat().st_size)
