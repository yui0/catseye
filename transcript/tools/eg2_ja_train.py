#!/usr/bin/env python3
"""Fine-tune EG2 ReaderA for Japanese CTC from frozen audio soft tokens.

Japanese classes are a newly generated vocabulary/table. English Reader v2
weights are INITIALIZATION ONLY and cannot transcribe Japanese without training.
Produces reader_ja.pt directly compatible with convert_eg2_reader.py (EG2R).
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from eg2_ja_vocab import load_rows, normalize as japanese_normalize, tokenize as japanese_tokenize
from eg2_mix_vocab import normalize as mixed_normalize, tokenize as mixed_tokenize


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
    """Load an English Reader backbone, never silently discard its trained layers.

    A foreign checkpoint may include a different CTC/word head (English vocabulary
    differs from the shared ja/en table) or a module./model. namespace prefix.
    Require the ENTIRE backbone to match in shape, and only allow head mismatch.
    """
    print(f'Loading English Reader initialization: {path}',flush=True)
    data=torch.load(path,map_location='cpu',weights_only=True)
    for _ in range(4):
        if not isinstance(data,dict):break
        found=next((data[k] for k in ('state_dict','model_state_dict','model','reader')
                    if isinstance(data.get(k),dict)),None)
        if found is None:break
        data=found
    if not isinstance(data,dict):raise ValueError('Reader checkpoint does not contain a state_dict')
    reference=model.state_dict()
    expected=set(reference)
    for prefix in ('module.','model.','reader.'):
        if not any(k in expected for k in data) and any(k.startswith(prefix) for k in data):
            data={k[len(prefix):] if k.startswith(prefix) else k:v for k,v in data.items()}
    body_keys=[k for k in reference if k.startswith('body.')]
    incompatible=[f'{k}: checkpoint={tuple(data[k].shape) if k in data and hasattr(data[k],"shape") else "absent"}, expected={tuple(reference[k].shape)}'
                  for k in body_keys if k not in data or not hasattr(data[k],'shape') or tuple(data[k].shape)!=tuple(reference[k].shape)]
    if incompatible:
        raise ValueError('English Reader backbone incompatible (first 8):\n  '+
                         '\n  '.join(incompatible[:8])+ '\nCheck --layers and pretrained checkpoint architecture, or disable INITIALIZE_FROM_ENGLISH.')
    compatible={k:v for k,v in data.items() if k in reference and
                hasattr(v,'shape') and tuple(v.shape)==tuple(reference[k].shape)}
    result=model.load_state_dict(compatible,strict=False)
    unexpected=set(data)-expected
    if unexpected:
        print(f'Note: ignored {len(unexpected)} non-reader keys (e.g. {list(sorted(unexpected))[:3]})',flush=True)
    print(f'Initialized {len(body_keys)}/{len(body_keys)} backbone tensors; '
          f'head compatible {len(compatible)-len(body_keys)}, '
          f'missing {len(result.missing_keys)} (new head initialized separately).',flush=True)


def load_examples(path,vocab,normalize=japanese_normalize,tokenize=japanese_tokenize,allow_empty=False):
    indices={t:i for i,t in enumerate(vocab)}
    out=[];oov=0;infeasible=0;first_oov=None
    for r in load_rows(path):
        source=Path(r['soft']);source=source if source.is_absolute() else Path(path).resolve().parent/source
        x=np.load(source,mmap_mode='r',allow_pickle=False)
        if x.ndim!=2 or x.shape[1]!=512 or not 0<x.shape[0]<=750:
            raise ValueError(f'invalid soft tokens {source}: {x.shape}')
        try:
            pieces=tokenize(r['text'],vocab)
        except ValueError as exc:
            # A validation utterance can contain a kanji absent from the train
            # vocab. Treat this as OOV, not an unrecoverable exception.
            oov+=1
            if first_oov is None:first_oov=(r.get('id','unknown'),str(exc))
            continue
        if not pieces:continue
        if any(t not in indices for t in pieces):
            oov+=1
            if first_oov is None:first_oov=(r.get('id','unknown'),'unknown token')
            continue
        y=np.array([indices[t] for t in pieces],dtype=np.int64)
        if len(y)+int(np.sum(y[1:]==y[:-1]))>len(x):infeasible+=1;continue
        out.append((source,len(x),y,normalize(r['text']),str(r.get('id',''))))
    print(f'{path}: valid {len(out)}, OOV {oov}, CTC-infeasible {infeasible}',flush=True)
    if first_oov:print(f'  first OOV skipped: id={first_oov[0]} reason={first_oov[1]}',flush=True)
    if not out and not allow_empty:raise ValueError('no usable examples: check vocabulary vs cached transcripts')
    if not out and allow_empty:print('WARNING: no dev examples usable; CER unavailable.',flush=True)
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


def ctc_objective(model,table,x,lengths,targets,tgt_lens,return_features=False):
    h,blank=model(x,lengths)
    logits=torch.cat((h@table.T*model.log_scale.exp().clamp(1,500),blank[:,:,None]),dim=-1)
    logp=F.log_softmax(logits.float(),dim=-1).transpose(0,1).contiguous()
    loss=F.ctc_loss(logp,targets.to('cpu'),lengths.to('cpu'),tgt_lens.to('cpu'),
                    blank=int(table.shape[0]),reduction='mean',zero_infinity=False)
    return (loss,logits,h) if return_features else (loss,logits)


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


def language_of(row):
    """Classify acoustic transcript for separate dev metrics, without relying on manifest order."""
    s = row[3]
    ja = any('\u3040' <= c <= '\u30ff' or '\u3400' <= c <= '\u9fff' for c in s)
    en = any('a' <= c <= 'z' for c in s)
    return 'mixed' if ja and en else 'ja' if ja else 'en' if en else 'other'


@torch.inference_mode()
def evaluate(model,table,rows,vocab,device,max_eval=100):
    """Evaluate a bounded, language-stratified dev set (max_eval per language).

    Old notebook passed --max-eval 30, while dev manifest was ordered JA then EN.
    Taking rows[:30] meant English was never checked before selecting best model.
    """
    model.eval()
    by_group=defaultdict(list)
    for item in rows:
        by_group[language_of(item)].append(item)
    errors={}; samples=[]; totals={}
    for group, group_rows in sorted(by_group.items()):
        # Deterministic evenly spread examples rather than the first 30 rows.
        rng=random.Random(719)
        selected=(rng.sample(group_rows,max_eval) if max_eval>0 and len(group_rows)>max_eval else group_rows)
        wrong=total=0
        for item in selected:
            x,l,_,_=batchify([item]);h,blank=model(x.to(device),l.to(device))
            logits=h[0]@table.T*model.log_scale.exp().clamp(1,500)
            labels=torch.where(blank[0]>logits.max(-1).values,table.shape[0],logits.argmax(-1)).tolist()
            pred=''.join(vocab[i] for i in collapse(labels,table.shape[0]))
            gold=item[3]
            wrong+=levenshtein(pred,gold); total+=len(gold)
            if len(samples)<6: samples.append({'language':group,'reference':gold,'hypothesis':pred})
        errors[group]=wrong/max(1,total)
        totals[group]=len(selected)
    # Macro over languages: never let Japanese dominate by manifest order or size.
    macro=sum(errors.values())/len(errors) if errors else float('inf')
    return macro,samples,errors,totals


def train(args):
    torch.manual_seed(args.seed);random.seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected but not available; switch Colab runtime to T4 GPU or use --device cpu')
    device=torch.device(args.device)
    print(f'CTC training device={device} torch={torch.__version__} rows={args.train}',flush=True)
    if device.type=='cuda':print(f'GPU={torch.cuda.get_device_name(device)} allocated={torch.cuda.memory_allocated(device)/2**20:.0f}MiB',flush=True)
    vocab=Path(args.vocab).read_text(encoding='utf8').splitlines()
    table=np.load(args.table,mmap_mode='r',allow_pickle=False)
    if table.shape!=(len(vocab),768):raise ValueError('vocab and word table do not match')
    tensor=torch.tensor(np.array(table,dtype=np.float32),device=device)
    if not torch.isfinite(tensor).all():raise ValueError('non-finite word table')
    tensor=F.normalize(tensor,dim=-1)
    norm,tok=(mixed_normalize,mixed_tokenize) if args.mixed else (japanese_normalize,japanese_tokenize)
    train_rows=load_examples(args.train,vocab,norm,tok)
    dev_rows=load_examples(args.dev,vocab,norm,tok,allow_empty=True) if args.dev else []
    model=Reader(args.layers,len(vocab))
    if args.init_reader:load_initial(model,args.init_reader)
    if args.resume_reader:
        state=torch.load(args.resume_reader,map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
        print('Resuming previous joint Reader checkpoint:',args.resume_reader,flush=True)
    model.to(device)
    # Teacher is the original EN backbone, frozen while we fine-tune the joint Reader.
    # Without this, a few hundred English clips can destroy useful pretrained English alignment.
    teacher=None
    if args.english_distill>0:
        if not args.init_reader:
            raise ValueError('--english-distill requires --init-reader pointing to original English checkpoint')
        teacher=Reader(args.layers,len(vocab))
        load_initial(teacher,args.init_reader)
        teacher.to(device).eval().requires_grad_(False)
        print(f'English Reader preservation: distillation weight={args.english_distill}',flush=True)
    if args.init_reader and not args.resume_reader:
        # The old English blank bias was calibrated for a ~200k-word dictionary.
        with torch.no_grad():
            x,l,_,_=batchify(train_rows[:min(args.batch,len(train_rows))])
            h,_=model(x.to(device),l.to(device))
            valid=torch.arange(h.shape[1],device=device)[None,:]<l.to(device)[:,None]
            logits=h[valid]@tensor.T*model.log_scale.exp().clamp(1,500)
            median=torch.logsumexp(logits,dim=-1).median()
            if not torch.isfinite(median):raise ValueError('invalid blank calibration')
            model.blank.weight.zero_()
            model.blank.bias.fill_(float(median))
            print(f'calibrated Reader blank prior to {float(median):.2f}',flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    args.out.mkdir(parents=True,exist_ok=True)
    num_batches=math.ceil(len(train_rows)/args.batch)
    steps_epoch=min(args.steps,num_batches) if args.steps else num_batches
    all_steps=max(1,args.epochs*steps_epoch)
    warmup=min(args.warmup_steps,max(0,all_steps//10))
    def schedule(step):
        if warmup and step<warmup:return max(0.05,(step+1)/warmup)
        p=max(0,min(1,(step-warmup)/max(1,all_steps-warmup)))
        return args.lr_floor + (1-args.lr_floor)*0.5*(1+math.cos(math.pi*p))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    best=float('inf'); history=[]; stale=0
    groups=defaultdict(list)
    for item in train_rows:groups[language_of(item)].append(item)
    print('Train labels by language:',{k:len(v) for k,v in groups.items()},flush=True)
    if len(groups.get('mixed',[]))==0:
        print('WARNING: zero mixed-language training clips: within-utterance code switching is not learned.',flush=True)
    for epoch in range(args.epochs):
        model.train()
        # Rather than concatenating [all JA, all EN], mix languages for every epoch.
        # Keep one visit per example to prevent inflating the effective dataset silently.
        shuffled=[]
        for g in groups.values():
            random.shuffle(g)
            shuffled.extend(g)
        random.shuffle(shuffled)
        losses=[];distill_losses=[]
        for start in range(0,len(shuffled),args.batch):
            batch=shuffled[start:start+args.batch]
            x,l,y,tl=batchify(batch)
            x=x.to(device); l=l.to(device)
            loss,_,features=ctc_objective(model,tensor,x,l,y,tl,return_features=True)
            if teacher is not None:
                english_indices=[i for i,r in enumerate(batch) if language_of(r)=='en']
                if english_indices:
                    subset=torch.tensor(english_indices,device=device,dtype=torch.long)
                    with torch.no_grad():ref,_=teacher(x.index_select(0,subset),l.index_select(0,subset))
                    actual=features.index_select(0,subset)
                    lens=l.index_select(0,subset)
                    mask=torch.arange(x.shape[1],device=device)[None,:]<lens[:,None]
                    penalty=((1.0-(actual*ref).sum(-1))*mask).sum()/mask.sum().clamp_min(1)
                    loss=loss+args.english_distill*penalty
                    distill_losses.append(float(penalty.detach()))
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();scheduler.step()
            losses.append(loss.item())
            if len(losses)==1 or len(losses)%100==0:
                print(f'epoch={epoch+1} step={len(losses)}/{steps_epoch} loss={losses[-1]:.4f} lr={scheduler.get_last_lr()[0]:.2e}',flush=True)
            if args.steps and len(losses)>=args.steps:break
        metric=float(np.mean(losses))
        if dev_rows:
            cer,examples,metrics,totals=evaluate(model,tensor,dev_rows,vocab,device,args.max_eval)
            print(f'epoch {epoch+1} train_loss={metric:.4f} dev_macro_CER={cer:.4f} '
                  f'by_language={metrics} counts={totals}',flush=True)
            for example in examples[:3]:print(' sample',example,flush=True)
        else:
            cer=metric;metrics={};totals={}
            print(f'epoch {epoch+1} loss {metric:.4f} (no dev; not a CER)',flush=True)
        history.append({'epoch':epoch+1,'loss':metric,'dev_macro_CER':cer if dev_rows else None,
                        'dev_CER_by_language':metrics,'dev_samples':totals,
                        'english_distill_loss':float(np.mean(distill_losses)) if distill_losses else None})
        torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},args.out/'reader_ja_last.pt')
        if cer < best:
            best=cer;stale=0
            torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},args.out/'reader_ja.pt')
            print(f'NEW BEST: macro CER={best:.4f}',flush=True)
        else:
            stale+=1
            if args.patience and stale>=args.patience:
                print(f'Early stopping after {stale} epochs without dev macro-CER improvement.',flush=True)
                break
    (args.out/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (args.out/'training_info.json').write_text(json.dumps({
        'trained':True,'mixed':args.mixed,'dev_macro_CER':best if dev_rows else None,'dev_CER':best if dev_rows else None,
        'epochs':args.epochs,'train_size':len(train_rows),'dev_size':len(dev_rows),
        'source_reader':str(args.init_reader) if args.init_reader else None,
        'vocab_size':len(vocab),'status':'dev CER is a validation value, not benchmark accuracy',
        'english_distill':args.english_distill,'learning_rate':args.lr,
        'vocab_sha256':hashlib.sha256(Path(args.vocab).read_bytes()).hexdigest(),
        'table_sha256':hashlib.sha256(Path(args.table).read_bytes()).hexdigest()
    },ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('saved',args.out/'reader_ja.pt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--dev',type=Path)
    p.add_argument('--vocab',type=Path,required=True)
    p.add_argument('--mixed',action='store_true',help='English+Japanese joint vocabulary, ▁ marks Latin word boundaries')
    p.add_argument('--table',type=Path,required=True)
    p.add_argument('--init-reader',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--layers',type=int,default=6)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--steps',type=int,default=0,help='smoke-test limit per epoch')
    p.add_argument('--max-eval',type=int,default=50,help='Maximum validation utterances PER LANGUAGE, 0 for all')
    p.add_argument('--lr',type=float,default=1.2e-5)
    p.add_argument('--warmup-steps',type=int,default=100)
    p.add_argument('--lr-floor',type=float,default=0.15)
    p.add_argument('--english-distill',type=float,default=0.0,help='Keep pretrained English acoustic alignment via frozen teacher')
    p.add_argument('--patience',type=int,default=4,help='Early stop on macro-CER; 0 disables')
    p.add_argument('--resume-reader',type=Path,help='Continue joint Reader with IDENTICAL vocab and table')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if not 1<=a.layers<=12 or a.epochs<1 or a.batch<1 or a.lr<=0 or a.warmup_steps<0 or not 0<=a.lr_floor<=1 or a.english_distill<0 or a.patience<0:
        raise ValueError('bad hyperparameters')
    try:
        train(a)
    except Exception:
        print('EG2 Reader CTC training failed (root traceback):',file=sys.stderr,flush=True)
        traceback.print_exc()
        raise SystemExit(1)

if __name__=='__main__':main()
