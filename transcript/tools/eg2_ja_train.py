#!/usr/bin/env python3
"""Fine-tune EG2 ReaderA for Japanese CTC from frozen audio soft tokens.

Japanese classes are a newly generated vocabulary/table. English Reader v2
weights are INITIALIZATION ONLY and cannot transcribe Japanese without training.
Produces reader_ja.pt directly compatible with convert_eg2_reader.py (EG2R).
"""
import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from eg2_ja_vocab import normalize,tokenize,load_rows


def rotate_half(x):
    a,b=x.chunk(2,dim=-1)
    return torch.cat([-b,a],dim=-1)


def rope(T,device,head_dim=64):
    inv=10000.0**(-torch.arange(0,head_dim,2,device=device,dtype=torch.float32)/head_dim)
    angle=torch.arange(T,device=device,dtype=torch.float32)[:,None]*inv[None,:]
    angle=torch.cat((angle,angle),dim=-1)
    return angle.cos()[None,None],angle.sin()[None,None]


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(512)
        self.qkv=nn.Linear(512,1536)
        self.proj=nn.Linear(512,512)
        self.ln2=nn.LayerNorm(512)
        self.ff1=nn.Linear(512,2048)
        self.ff2=nn.Linear(2048,512)

    def forward(self,x,mask,cos,sin):
        B,T,_=x.shape
        q,k,v=self.qkv(self.ln1(x)).view(B,T,3,8,64).permute(2,0,3,1,4)
        q=q*cos+rotate_half(q)*sin
        k=k*cos+rotate_half(k)*sin
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        x=x+self.proj(a.transpose(1,2).reshape(B,T,512))
        return x+self.ff2(F.gelu(self.ff1(self.ln2(x))))


class ReaderBody(nn.Module):
    def __init__(self,layers=6):
        super().__init__()
        self.ln_in=nn.LayerNorm(512)
        self.inp=nn.Linear(512,512)
        self.blocks=nn.ModuleList(Block() for _ in range(layers))
        self.ln_out=nn.LayerNorm(512)
        self.out=nn.Linear(512,768)

    def forward(self,x,lengths):
        T=x.shape[1]
        mask=(torch.arange(T,device=x.device)[None,:]<lengths[:,None])[:,None,None,:]
        cos,sin=rope(T,x.device)
        h=self.inp(self.ln_in(x))
        for blk in self.blocks:h=blk(h,mask,cos,sin)
        return self.out(self.ln_out(h))


class Reader(nn.Module):
    def __init__(self,layers=6,n_vocab=10000):
        super().__init__()
        self.body=ReaderBody(layers)
        self.blank=nn.Linear(768,1)
        nn.init.constant_(self.blank.bias,math.log(n_vocab))
        self.log_scale=nn.Parameter(torch.tensor(math.log(20.0)))

    def forward(self,x,lengths):
        v=self.body(x,lengths)
        return F.normalize(v,dim=-1),self.blank(v).squeeze(-1)


def load_initial(model,path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    for name in ('state_dict','model','reader'):
        if isinstance(data,dict) and isinstance(data.get(name),dict):data=data[name]
    if not isinstance(data,dict):raise ValueError('expected flat state_dict')
    model.load_state_dict(data,strict=True)


def load_examples(path,vocab):
    indices={t:i for i,t in enumerate(vocab)}
    out=[];oov=0;infeasible=0
    for r in load_rows(path):
        source=Path(r['soft']);source=source if source.is_absolute() else Path(path).resolve().parent/source
        x=np.load(source,mmap_mode='r',allow_pickle=False)
        if x.ndim!=2 or x.shape[1]!=512 or not 0<x.shape[0]<=750:
            raise ValueError(f'invalid soft tokens {source}: {x.shape}')
        pieces=tokenize(r['text'],vocab)
        if not pieces:continue
        if any(t not in indices for t in pieces):oov+=1;continue
        y=np.array([indices[t] for t in pieces],dtype=np.int64)
        if len(y)+int(np.sum(y[1:]==y[:-1]))>len(x):infeasible+=1;continue
        out.append((source,len(x),y,normalize(r['text'])))
    print(f'{path}: valid {len(out)}, OOV {oov}, CTC-infeasible {infeasible}')
    if not out:raise ValueError('no usable examples')
    return out


def batchify(rows):
    size=len(rows)
    X=[torch.from_numpy(np.asarray(np.load(r[0]),dtype=np.float32).copy()) for r in rows]
    sizes=torch.tensor([len(x) for x in X],dtype=torch.long)
    maxlen=max(sizes)
    x=torch.zeros(size,int(maxlen),512,dtype=torch.float32)
    for i,z in enumerate(X):x[i,:len(z)]=z
    y=[torch.from_numpy(r[2]) for r in rows]
    lengths=torch.tensor([len(k) for k in y],dtype=torch.long)
    return x,sizes,torch.cat(y),lengths


def ctc_objective(model,table,x,lengths,targets,tgt_lens):
    h,blank=model(x,lengths)
    logits=torch.cat((h@table.T*model.log_scale.exp().clamp(1,500),blank[:,:,None]),dim=-1)
    logp=F.log_softmax(logits.float(),dim=-1).transpose(0,1).contiguous()
    loss=F.ctc_loss(logp,targets.to('cpu'),lengths.to('cpu'),tgt_lens.to('cpu'),
                    blank=int(table.shape[0]),reduction='mean',zero_infinity=False)
    return loss,logits


def collapse(indices,blank):
    previous=-1;out=[]
    for i in indices:
        if i!=previous and i!=blank:out.append(i)
        previous=i
    return out


def levenshtein(a,b):
    prev=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        row=[i]
        for j,d in enumerate(b,1):row.append(min(prev[j]+1,row[-1]+1,prev[j-1]+(c!=d)))
        prev=row
    return prev[-1]


@torch.inference_mode()
def evaluate(model,table,rows,vocab,device,max_eval=100):
    model.eval()
    err=0;count=0;examples=[]
    for item in rows[:max_eval]:
        x,l,_,_=batchify([item]);h,blank=model(x.to(device),l.to(device))
        logits=h[0]@table.T*model.log_scale.exp().clamp(1,500)
        labels=torch.where(blank[0]>logits.max(-1).values,table.shape[0],logits.argmax(-1)).tolist()
        pred=''.join(vocab[i] for i in collapse(labels,table.shape[0]))
        gold=item[3]
        err+=levenshtein(pred,gold);count+=len(gold)
        if len(examples)<3:examples.append((gold,pred))
    return err/max(1,count),examples


def train(args):
    torch.manual_seed(args.seed);random.seed(args.seed)
    device=torch.device(args.device)
    vocab=Path(args.vocab).read_text(encoding='utf8').splitlines()
    table=np.load(args.table,mmap_mode='r',allow_pickle=False)
    if table.shape!=(len(vocab),768):raise ValueError('vocab and word table do not match')
    tensor=torch.tensor(np.array(table,dtype=np.float32),device=device)
    if not torch.isfinite(tensor).all():raise ValueError('non-finite word table')
    tensor=F.normalize(tensor,dim=-1)
    train_rows=load_examples(args.train,vocab)
    dev_rows=load_examples(args.dev,vocab) if args.dev else []
    model=Reader(args.layers,len(vocab))
    if args.init_reader:load_initial(model,args.init_reader)
    model.to(device)
    # The English checkpoint's blank bias is calibrated for 200k English
    # classes; the Japanese N-way normalizer is substantially different.
    # Match its median log-mass before Japanese fine-tuning, as the original
    # Reader training recipe does when starting a new vocabulary.
    if args.init_reader:
        with torch.no_grad():
            x,l,_,_=batchify(train_rows[:min(args.batch,len(train_rows))])
            h,_=model(x.to(device),l.to(device))
            valid=torch.arange(h.shape[1],device=device)[None,:]<l.to(device)[:,None]
            logits=h[valid]@tensor.T*model.log_scale.exp().clamp(1,500)
            median=torch.logsumexp(logits,dim=-1).median()
            if not torch.isfinite(median):raise ValueError('invalid blank calibration')
            model.blank.weight.zero_()
            model.blank.bias.fill_(float(median))
            print(f'calibrated Japanese blank prior to {float(median):.2f}',flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    args.out.mkdir(parents=True,exist_ok=True)
    best=float('inf')
    for epoch in range(args.epochs):
        model.train();random.shuffle(train_rows);losses=[]
        for start in range(0,len(train_rows),args.batch):
            x,l,y,tl=batchify(train_rows[start:start+args.batch])
            loss,_=ctc_objective(model,tensor,x.to(device),l.to(device),y,tl)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            losses.append(loss.item())
            if args.steps and len(losses)>=args.steps:break
        metric=float(np.mean(losses))
        if dev_rows:
            cer,ex=evaluate(model,tensor,dev_rows,vocab,device,args.max_eval)
            print(f'epoch {epoch+1} loss {metric:.4f} dev_CER {cer:.4f} samples {ex[:1]}',flush=True)
        else:
            cer=metric
            print(f'epoch {epoch+1} loss {metric:.4f} (no dev; this is NOT a CER)',flush=True)
        torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},args.out/'reader_ja_last.pt')
        if cer<best:
            best=cer;torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},args.out/'reader_ja.pt')
    (args.out/'training_info.json').write_text(json.dumps({
        'trained':True,'dev_CER':best if dev_rows else None,
        'epochs':args.epochs,'train_size':len(train_rows),'dev_size':len(dev_rows),
        'source_reader':str(args.init_reader) if args.init_reader else None,
        'vocab_size':len(vocab),'status':'dev CER is a validation value, not benchmark accuracy'
    },ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('saved',args.out/'reader_ja.pt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--dev',type=Path)
    p.add_argument('--vocab',type=Path,required=True)
    p.add_argument('--table',type=Path,required=True)
    p.add_argument('--init-reader',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--layers',type=int,default=6)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--steps',type=int,default=0,help='smoke-test limit per epoch')
    p.add_argument('--max-eval',type=int,default=50)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if not 1<=a.layers<=12 or a.epochs<1 or a.batch<1:raise ValueError('bad hyperparameters')
    train(a)

if __name__=='__main__':main()
