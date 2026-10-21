#!/usr/bin/env python3
"""Train frozen-English-reader + small Japanese adapter for native catseye C.

Original EN model (reader.pt and 200k EG2Q table) is NEVER optimized or changed.
prepare: precompute English frame embeddings/blank and full English vocabulary
         logsumexp for CTC (reused every epoch; requires CUDA preferably).
train:   joint CTC with exact English 200k-class denominator, JA classes,
         and one shared original EN blank; gate BCE on monolingual examples.
export:  ja_adapter.eg2a EG2PKG1 bundle. No external inference dependencies.

Usage: python eg2_ja_adapter.py prepare --train cache/train/index.jsonl --dev cache/dev/index.jsonl --en-package models/en.eg2m --reader-pt reader.pt --out-dir adapter_cache
       python eg2_ja_adapter.py train --train cache/train/index.jsonl --dev cache/dev/index.jsonl --cache-dir adapter_cache --en-package models/en.eg2m --ja-table word_table.npy --ja-vocab reader_vocab.txt --out-dir trained_adapter --device cuda
       python eg2_ja_adapter.py export --checkpoint trained_adapter/adapter.pt --ja-table word_table.npy --ja-vocab reader_vocab.txt --out models/ja_adapter.eg2a
"""
import argparse
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import random
import re
import struct
import sys
import traceback
import unicodedata

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from eg2_ja_train import Reader, load_initial, levenshtein
from eg2_bundle import pack, _components

INPUT=1280
OUT=768
RANK_MAX=256
KERNEL_MAX=31


def pkg_entry(pkg,name):
    pkg=Path(pkg)
    if pkg.is_dir():
        path=pkg/name
        return path,0,path.stat().st_size
    for component,start,length in _components(pkg):
        if component==name:
            return pkg,start,length
    raise FileNotFoundError(f'{name} missing inside {pkg}')


def read_vocab(pkg):
    path,offset,length=pkg_entry(pkg,'reader_vocab.txt')
    with path.open('rb') as f:
        f.seek(offset)
        lines=f.read(length).decode('utf8').splitlines()
    if not lines or len(set(lines))!=len(lines) or any(not line for line in lines):
        raise ValueError('English vocabulary invalid or has duplicates')
    return lines


def q8_table_view(package):
    path,off,size=pkg_entry(package,'word_table.eg2q')
    with path.open('rb') as f:
        f.seek(off)
        hdr=f.read(16)
    if len(hdr)!=16:
        raise ValueError('invalid EG2Q header')
    magic,version,n,dim=struct.unpack('<4s3I',hdr)
    if magic!=b'EG2Q' or version!=1 or dim!=768 or not (0<n<=1000000) or size!=16+n*768:
        raise ValueError('invalid EG2Q table header/length')
    return np.memmap(path,mode='r',dtype=np.int8,shape=(n,768),offset=off+16)


def english_table_float(table,device):
    # Original Q8 table row-normalized EXACTLY like C's cosine lookup.
    # Retains the full 200k English vocabulary; never modifies/re-embeds it.
    arr=np.asarray(table,dtype=np.float32)
    denom=np.linalg.norm(arr,axis=1,keepdims=True)
    if np.any(denom<=0):raise ValueError('zero English embedding row')
    return torch.from_numpy(arr/denom).to(device)


def verify_original_reader(reader_pt,en_package):
    """Fail closed if .pt used for cache differs from frozen EN model in C."""
    import subprocess
    import tempfile
    native_path,native_offset,native_size=pkg_entry(en_package,'reader.eg2r')
    with tempfile.TemporaryDirectory() as d:
        output=Path(d)/'english.eg2r'
        command=[sys.executable,str(Path(__file__).with_name('convert_eg2_reader.py')),
                 str(reader_pt),str(output)]
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
        if output.stat().st_size!=native_size:
            raise ValueError('English checkpoint and native en.eg2m Reader size differ. Do not combine unrelated English weights.')
        with output.open('rb') as a,native_path.open('rb') as b:
            b.seek(native_offset)
            while (chunk:=a.read(4*1024*1024)):
                if b.read(len(chunk))!=chunk:
                    raise ValueError('English reader.pt != en.eg2m/reader.eg2r. Use matching English weights to avoid accuracy loss.')
    print('Frozen English reader.pt / native EN weights: byte-identical',flush=True)


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        while (part:=f.read(4*1024*1024)):
            digest.update(part)
    return digest.hexdigest()


def english_cache_metadata(package,checkpoint):
    model_path=Path(package)
    if not model_path.is_file():
        # Alternative unpacked model directories are also supported.
        components=['reader.eg2r','word_table.eg2q','reader_vocab.txt']
        package_checksum={p:sha256_file(model_path/p) for p in components}
    else:
        package_checksum=sha256_file(model_path)
    return {'english_package_sha256':package_checksum,
            'reader_pt_sha256':sha256_file(checkpoint),
            'format_version':1}


def validate_english_cache(cache_dir,package,reader_pt,create=False):
    cache_dir=Path(cache_dir)
    cache_dir.mkdir(parents=True,exist_ok=True)
    metadata=english_cache_metadata(package,reader_pt)
    target=cache_dir/'english_source.json'
    if target.exists():
        existing=json.loads(target.read_text(encoding='utf8'))
        if existing!=metadata:
            raise ValueError('English model changed since feature cache generation. '
                             'Use a NEW --out-dir / --cache-dir; EN features are not interchangeable.')
    elif create:
        target.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
    else:
        raise ValueError(f'Missing {target}. Run prepare with this English model first.')


def cached_english_path(cache_dir,soft_path):
    # content identifier uses canonical soft path + file metadata; safe across train/dev.
    p=Path(soft_path)
    st=p.stat()
    name=hashlib.sha256((str(p.resolve())+':'+str(st.st_size)+':'+str(st.st_mtime_ns)).encode()).hexdigest()[:28]
    return Path(cache_dir)/f'{name}.npz'


def rows_from_manifest(path):
    rows=[]
    with Path(path).open(encoding='utf8') as f:
        for lineno,line in enumerate(f,1):
            if line.strip():
                data=json.loads(line)
                soft=Path(data['soft'])
                if not soft.is_absolute():soft=Path(path).resolve().parent/soft
                rows.append({'id':str(data.get('id',f'row_{lineno}')),
                             'text':data['text'], 'soft':str(soft.resolve())})
    if not rows:raise ValueError(f'empty index: {path}')
    return rows


def read_reader_checkpoint(path):
    # Pretrained architecture inferred from state dict. No reading of English vocab tables here.
    data=torch.load(path,map_location='cpu',weights_only=True)
    if isinstance(data,dict):
        for key in ('state_dict','model_state_dict','model','reader'):
            if isinstance(data.get(key),dict):data=data[key];break
    while all(k.startswith(('module.','model.','reader.')) for k in data):
        prefix=next((p for p in ('module.','model.','reader.') if all(k.startswith(p) for k in data)),None)
        if not prefix:break
        data={k[len(prefix):]:v for k,v in data.items()}
    layers=sorted({int(k.split('.')[2]) for k in data if k.startswith('body.blocks.') and k.split('.')[2].isdigit()})
    if layers!=list(range(len(layers))) or not 1<=len(layers)<=12:
        raise ValueError('invalid pretrained Reader layer configuration')
    model=Reader(len(layers),200000)
    load_initial(model,path)
    # Require exact original blank and scale. Skipping those would silently corrupt EN CTC.
    required=('blank.weight','blank.bias','log_scale')
    if any(k not in data or tuple(data[k].shape)!=tuple(model.state_dict()[k].shape) for k in required):
        raise ValueError('checkpoint missing original English blank / log_scale')
    model.eval().requires_grad_(False)
    return model


@torch.no_grad()
def english_lse(h,table,scale,chunk=8192):
    # English logsumexp computed in chunks => stable and exact (float32),
    # never allocate [T,200000] scores. h is already normalized.
    lse=torch.full((h.shape[0],),-float('inf'),device=h.device)
    best=torch.full((h.shape[0],),-float('inf'),device=h.device)
    best_idx=torch.zeros((h.shape[0],),device=h.device,dtype=torch.int64)
    for start in range(0,table.shape[0],chunk):
        scores=h @ table[start:start+chunk].T * scale
        lse=torch.logaddexp(lse,torch.logsumexp(scores,dim=-1))
        local,indices=scores.max(dim=-1)
        higher=local>best
        best_idx=torch.where(higher,indices+start,best_idx)
        best=torch.maximum(best,local)
    return lse,best,best_idx


def prepare(args):
    device=torch.device(args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    verify_original_reader(args.reader_pt,args.en_package)
    en_vocab=read_vocab(args.en_package)
    raw=q8_table_view(args.en_package)
    if len(en_vocab)!=raw.shape[0]:raise ValueError('English Q8 table and vocabulary mismatch')
    en_table=english_table_float(raw,device)
    reader=read_reader_checkpoint(args.reader_pt).to(device)
    cache_dir=Path(args.out_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    validate_english_cache(cache_dir,args.en_package,args.reader_pt,create=True)
    print(f'EN frozen: vocab={len(en_vocab)} reader_layers={reader.body.blocks.__len__()} device={device} scale={reader.log_scale.exp().item():.4f}',flush=True)
    seen=set()
    for index in (args.train,args.dev):
        if not index:continue
        rows=rows_from_manifest(index)
        for pos,r in enumerate(rows,1):
            target=cached_english_path(cache_dir,r['soft'])
            if target in seen:continue
            seen.add(target)
            if target.exists():
                with np.load(target,allow_pickle=False) as existing:
                    if {'h','blank','en_lse','en_topval','en_topidx'} <= set(existing.files):continue
            soft=np.asarray(np.load(r['soft'],mmap_mode='r',allow_pickle=False),dtype=np.float32)
            if soft.ndim!=2 or soft.shape[1]!=512 or not 0<len(soft)<=750:raise ValueError(f'bad soft tokens {r["soft"]}: {soft.shape}')
            with torch.inference_mode():
                x=torch.from_numpy(soft.copy()).to(device)[None,:,:]
                lengths=torch.tensor([len(soft)],device=device)
                h,blank=reader(x,lengths)
                en=h[0].float()
                lse,topval,topidx=english_lse(en,en_table,reader.log_scale.exp())
                arr={'h':en.cpu().numpy().astype('<f4'),
                     'blank':blank[0].cpu().numpy().astype('<f4'),
                     'en_lse':lse.cpu().numpy().astype('<f4'),
                     'en_topval':topval.cpu().numpy().astype('<f4'),
                     'en_topidx':topidx.cpu().numpy().astype('<i4')}
            tmp=target.with_suffix('.partial.npz')
            np.savez_compressed(tmp,**arr)
            tmp.replace(target)
            if pos==1 or pos%50==0 or pos==len(rows):
                print(f'EN cache {pos}/{len(rows)} for {index}',flush=True)
    print('English reader weights/table unmodified. Cached features in',cache_dir)


def norm_text(text):
    return unicodedata.normalize('NFKC',text).casefold()


def build_piece_index(vocab):
    first=defaultdict(list)
    for i,tok in enumerate(vocab):
        if tok and not any(x in tok for x in ('\n','\r')):first[tok[0]].append((tok,i))
    for v in first.values():v.sort(key=lambda a:-len(a[0]))
    return first


def tokenize_joint(text,en_vocab,ja_vocab):
    """Mixed utterance -> original EN word IDs and appended JA token IDs.
    Unknown English words are a clear error, not secretly mapped to JA ASCII.
    Entire training utterance skipped on OOV; use more audio/vocab coverage.
    Japanese punctuation and white space ignored (like original ja training).
    """
    original_en=len(en_vocab)
    english={t.casefold():i for i,t in enumerate(en_vocab)}
    jap=build_piece_index(ja_vocab)
    text=norm_text(text)
    tokens=[]
    i=0
    while i<len(text):
        ch=text[i]
        if ch.isspace() or unicodedata.category(ch)[0] in 'PS' or unicodedata.category(ch)=='Sm':
            i+=1;continue
        if ch.isascii() and ch.isalnum():
            j=i+1
            while j<len(text) and text[j].isascii() and text[j].isalnum():j+=1
            term=text[i:j]
            term_ids=(term,'▁'+term)
            en_id=next((english[p] for p in term_ids if p in english),None)
            if en_id is None:raise ValueError(f'English OOV: {term!r}')
            tokens.append(en_id)
            i=j;continue
        match=next(((word,j) for word,j in jap.get(ch,[]) if text.startswith(word,i)),None)
        if match is None:raise ValueError(f'Japanese OOV U+{ord(ch):04X} ({ch!r})')
        word,ja_id=match
        tokens.append(original_en+ja_id)
        i+=len(word)
    if not tokens:raise ValueError('no labels')
    return tokens


def classify(text):
    s=norm_text(text)
    en=any(c.isascii() and c.isalpha() for c in s)
    ja=any('\u3040'<=c<='\u30ff' or '\u3400'<=c<='\u9fff' for c in s)
    return 'mixed' if en and ja else 'en' if en else 'ja' if ja else 'other'


class Ema:
    """Exponential moving average of trainable tensors; swap() toggles live/EMA weights."""
    def __init__(self,params,decay):
        self.decay=decay
        self.shadow=[p.detach().clone() for p in params]

    @torch.no_grad()
    def update(self,params):
        for s,p in zip(self.shadow,params):s.mul_(self.decay).add_(p.detach(),alpha=1-self.decay)

    @torch.no_grad()
    def swap(self,params):
        for s,p in zip(self.shadow,params):
            live=p.detach().clone();p.copy_(s);s.copy_(live)


def warmup_cosine(step,warmup,total,floor=0.05):
    if step<warmup:return (step+1)/max(1,warmup)
    progress=min(1.0,(step-warmup)/max(1,total-warmup))
    return floor+(1-floor)*0.5*(1+math.cos(math.pi*progress))


class JapaneseAdapter(nn.Module):
    def __init__(self,rank=64,n_japanese=4000,kernel=1):
        super().__init__()
        if kernel<1 or kernel%2==0 or kernel>KERNEL_MAX:raise ValueError('kernel must be odd and <=31')
        self.kernel=kernel
        self.w1=nn.Linear(INPUT,rank)
        # Depthwise temporal mixing in the rank space (EG2A v2). Centre tap 1 = per-frame v1.
        self.dw=nn.Conv1d(rank,rank,kernel,padding=kernel//2,groups=rank,bias=False) if kernel>1 else None
        if self.dw is not None:
            nn.init.zeros_(self.dw.weight)
            with torch.no_grad():self.dw.weight[:,0,kernel//2]=1.0
        self.w2=nn.Linear(rank,OUT)
        self.gate=nn.Linear(rank,1)
        self.ja_log_scale=nn.Parameter(torch.tensor(math.log(20.0)))
        nn.init.normal_(self.w1.weight,std=0.015)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias,-2.0)  # English-first at start

    def forward(self,x,h_en):
        z=torch.cat((x,h_en),dim=-1)
        if self.dw is None:
            feat=torch.tanh(self.w1(z))
        else:
            # Bias is added after the conv so zero padding stays exactly zero, as in C.
            u=F.linear(z,self.w1.weight).transpose(1,2)
            feat=torch.tanh(self.dw(u).transpose(1,2)+self.w1.bias)
        ja=F.normalize(h_en+self.w2(feat),dim=-1)
        gate=self.gate(feat).squeeze(-1).sigmoid().clamp(1e-7,1-1e-7)
        return ja,gate


def load_training_rows(index,cache_dir,en_vocab,ja_vocab,limit=0):
    valid=[]; reasons=defaultdict(int)
    for row in rows_from_manifest(index):
        try:
            targets=tokenize_joint(row['text'],en_vocab,ja_vocab)
            arr=np.load(row['soft'],mmap_mode='r',allow_pickle=False)
            if arr.ndim!=2 or arr.shape[1]!=512 or not 0<len(arr)<=750:raise ValueError('invalid soft shape')
            if len(targets)+sum(a==b for a,b in zip(targets,targets[1:]))>len(arr):
                raise ValueError('CTC infeasible')
            cache=cached_english_path(cache_dir,row['soft'])
            if not cache.exists():raise FileNotFoundError(f'run prepare first: {cache}')
            valid.append((row,targets,cache,classify(row['text'])))
            if limit and len(valid)>=limit:break
        except ValueError as e:
            reasons[str(e)[:80]]+=1
    if reasons:print('Skipped invalid transcripts:',dict(sorted(reasons.items(),key=lambda a:-a[1])[:10]),flush=True)
    if not valid:raise ValueError(f'No usable examples in {index}')
    print('Valid',index,len(valid),'languages',dict((g,sum(v[3]==g for v in valid)) for g in ('en','ja','mixed')))
    return valid


def build_batch(rows,device):
    b=len(rows)
    records=[]
    for row,targets,cache,language in rows:
        with np.load(cache,allow_pickle=False) as frozen:
            h=np.asarray(frozen['h'],dtype=np.float32)
            blank=np.asarray(frozen['blank'],dtype=np.float32)
            lse=np.asarray(frozen['en_lse'],dtype=np.float32)
        x=np.asarray(np.load(row['soft'],mmap_mode='r'),dtype=np.float32)
        if len(x)!=len(h) or len(x)!=len(blank) or len(x)!=len(lse):raise ValueError(f'EN cache mismatch for {row["id"]}')
        records.append((x,h,blank,lse,targets,language))
    lens=torch.tensor([len(r[0]) for r in records],dtype=torch.long)
    maxlen=int(lens.max())
    xx=torch.zeros((b,maxlen,512),dtype=torch.float32)
    h=torch.zeros((b,maxlen,768),dtype=torch.float32)
    bb=torch.zeros((b,maxlen),dtype=torch.float32)
    ee=torch.zeros((b,maxlen),dtype=torch.float32)
    for k,(x,hh,blank,lse,_,_) in enumerate(records):
        n=len(x)
        xx[k,:n]=torch.from_numpy(x.copy())
        h[k,:n]=torch.from_numpy(hh.copy())
        bb[k,:n]=torch.from_numpy(blank.copy())
        ee[k,:n]=torch.from_numpy(lse.copy())
    return xx.to(device),h.to(device),bb.to(device),ee.to(device),lens,records


def objective(model,ja_table,en_raw,orig_en_scale,batch,device,
              gate_weight=0.2,compute_decode=False):
    x,h,blank,en_lse,lens,records=build_batch(batch,device)
    ja,gate=model(x,h)
    ja_table=F.normalize(ja_table,dim=-1)
    ja_logits=(ja @ ja_table.T)*model.ja_log_scale.exp().clamp(1,300)
    # Full 200k EN softmax normalization precomputed from frozen English Reader.
    en_logprior=torch.log1p(-gate)
    ja_logprior=torch.log(gate)
    ja_logits=ja_logits+ja_logprior[...,None]
    en_lse=en_lse+en_logprior
    # English target classes in THIS batch: avoid allocating [T,200k].
    en_ids=sorted({i for rec in records for i in rec[4] if i<en_raw.shape[0]})
    if en_ids:
        en_embeddings=torch.tensor(np.asarray(en_raw[en_ids],dtype=np.float32),device=device)
        en_embeddings=F.normalize(en_embeddings,dim=-1)
        en_logits=h @ en_embeddings.T * orig_en_scale + en_logprior[...,None]
    else:en_logits=ja_logits[:,:,:0]
    combined_lse=torch.logaddexp(torch.logaddexp(en_lse,torch.logsumexp(ja_logits,dim=-1)),blank)
    selected=torch.cat((en_logits,ja_logits,blank[...,None]),dim=-1)
    logp=selected-combined_lse[...,None]
    mapping={num:i for i,num in enumerate(en_ids)}
    num_en=len(en_ids)
    n_jap=ja_table.shape[0]
    labels=[];targets_len=[]
    for rec in records:
        t=[mapping[a] if a<en_raw.shape[0] else num_en+(a-en_raw.shape[0]) for a in rec[4]]
        labels+=t
        targets_len.append(len(t))
    # blank is last compact class, but probability is from original fixed blank.
    ctc=F.ctc_loss(logp.transpose(0,1).contiguous(),torch.tensor(labels,dtype=torch.long),
                   lens,torch.tensor(targets_len,dtype=torch.long),
                   blank=num_en+n_jap,reduction='mean',zero_infinity=False)
    # Auxiliary language targets only on monolingual clips; mixed = unsupervised.
    gate_losses=[]
    for i,rec in enumerate(records):
        if rec[5]=='en':
            gate_losses.append(-torch.log1p(-gate[i,:lens[i]]).mean())
        elif rec[5]=='ja':
            gate_losses.append(-torch.log(gate[i,:lens[i]]).mean())
    loss=ctc+(gate_weight*torch.stack(gate_losses).mean() if gate_losses else 0)
    if not torch.isfinite(loss):raise RuntimeError(f'non-finite CTC loss (ctc={ctc.item():.5g})')
    if not compute_decode:return loss,{'ctc':float(ctc.detach()),'gate':float(gate.detach().mean())}
    predictions=[]
    for i,rec in enumerate(records):
        ids=selected[i,:lens[i]].argmax(dim=-1).tolist()
        # Greedy predictions over compact output; CTC collapse
        pred=[];prev=None
        for label in ids:
            if label!=prev and label!=num_en+n_jap:pred.append(label)
            prev=label
        pred_glob=[en_ids[k] if k<num_en else en_raw.shape[0]+(k-num_en) for k in pred]
        predictions.append(pred_glob)
    return loss,{'ctc':float(ctc.detach()),'gate':float(gate.detach().mean())},predictions


def ctc_tokens(ids,blank):
    out=[];prev=-1
    for token in ids:
        if token!=blank and token!=prev:out.append(token)
        prev=token
    return out


def predicted_words(ids,en_vocab,ja_vocab):
    n=len(en_vocab)
    text=''
    for token in ids:
        if token<n:
            word=en_vocab[token]
            if word.startswith('▁'):word=word[1:]
            if text:text+=' '
            text+=word
        else:
            word=ja_vocab[token-n]
            if word.startswith('▁'):
                if text:text+=' '
                word=word[1:]
            text+=word
    return text


def evaluate_native_gate(model,ja_table,dev_rows,en_vocab,ja_vocab,device,margin=0.08,
                         max_each=40):
    """Actual greedy CTC logic (top1 EN/JA + original blank) including gate.
    Validation uses cached EN frame scores; no re-run of full English table.
    """
    groups=defaultdict(list)
    for row in dev_rows:groups[row[3]].append(row)
    results={}
    model.eval()
    with torch.inference_mode():
        jt=F.normalize(ja_table,dim=-1)
        for lang,group in sorted(groups.items()):
            chosen=group if not max_each or len(group)<=max_each else random.Random(410).sample(group,max_each)
            err=total=0
            samples=[]
            for r in chosen:
                xx,hh,bb,_,lens,_=build_batch([r],device)
                with np.load(r[2],allow_pickle=False) as c:
                    en_idx=c['en_topidx'].astype(np.int64).tolist()
                    en_val=torch.tensor(c['en_topval'],device=device)
                ja,gate=model(xx,hh)
                jap=((ja[0]@jt.T)*model.ja_log_scale.exp().clamp(1,300))
                japval,japidx=jap.max(dim=-1)
                n=int(lens[0]);choice=[]
                for t in range(n):
                    g=float(gate[0,t]);en_score=float(en_val[t]);ja_score=float(japval[t]);blank=float(bb[0,t]);
                    if g<0.5:
                        word=int(en_idx[t]);top=en_score
                    else:
                        a=en_score+math.log1p(-g)
                        b=ja_score+math.log(g)-margin
                        word=len(en_vocab)+int(japidx[t]) if b>a else int(en_idx[t])
                        top=max(a,b)
                    choice.append(word if top>=blank else len(en_vocab)+len(ja_vocab))
                tokens=ctc_tokens(choice,len(en_vocab)+len(ja_vocab))
                pred=predicted_words(tokens,en_vocab,ja_vocab)
                truth=r[0]['text']
                if lang=='en':
                    tok=lambda s:re.findall(r'[a-z0-9]+',norm_text(s))
                elif lang=='ja':
                    from eg2_adapter_vocab import only_ja
                    tok=lambda s:list(only_ja(s))
                else:
                    # Spaces/punctuation are ignored for language-mixed CER.
                    tok=lambda s:list(''.join(c for c in norm_text(s)
                           if c.isalnum() or ('\u3040'<=c<='\u30ff')))
                gold=tok(truth);hyp=tok(pred)
                err+=levenshtein(gold,hyp);total+=len(gold)
                if len(samples)<2:samples.append((truth,pred))
            results[lang]={'error_rate':err/max(1,total),'count':len(chosen),'samples':samples}
    return results


def english_baseline(dev_rows,en_vocab,max_each=40):
    group=[r for r in dev_rows if r[3]=='en']
    selected=group if not max_each or len(group)<=max_each else random.Random(410).sample(group,max_each)
    err=total=0
    for row,_,cache,_ in selected:
        with np.load(cache,allow_pickle=False) as c:
            logits=c['en_topval'];blank=c['blank'];idx=c['en_topidx']
            labels=[int(i) if s>=b else len(en_vocab) for i,s,b in zip(idx,logits,blank)]
        pred=' '.join(en_vocab[i].removeprefix('▁') for i in ctc_tokens(labels,len(en_vocab)))
        tok=lambda s:re.findall('[a-z0-9]+',norm_text(s))
        ref=tok(row['text']);hyp=tok(pred)
        err+=levenshtein(ref,hyp);total+=len(ref)
    return err/max(1,total) if selected else None


def train(args):
    device=torch.device(args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    verify_original_reader(args.reader_pt,args.en_package)
    validate_english_cache(args.cache_dir,args.en_package,args.reader_pt)
    en_vocab=read_vocab(args.en_package)
    en_raw=q8_table_view(args.en_package)
    if len(en_vocab)!=len(en_raw):raise ValueError('English en.eg2m table/vocab mismatched')
    ja_vocab=Path(args.ja_vocab).read_text(encoding='utf8').splitlines()
    if not ja_vocab or len(set(ja_vocab))!=len(ja_vocab):raise ValueError('Japanese vocabulary bad')
    if any(all(ch.isascii() for ch in w) for w in ja_vocab):
        print('WARNING: Japanese vocab contains ASCII words; prefer JA-only vocab (English already has 200k).',flush=True)
    table=np.load(args.ja_table,allow_pickle=False)
    if table.shape!=(len(ja_vocab),768):raise ValueError('Japanese table dimensions != vocabulary')
    tensor=nn.Parameter(F.normalize(torch.tensor(table.astype(np.float32),device=device),dim=-1),requires_grad=args.train_ja_table)
    data_train=load_training_rows(args.train,args.cache_dir,en_vocab,ja_vocab,args.limit)
    data_dev=load_training_rows(args.dev,args.cache_dir,en_vocab,ja_vocab,args.limit_dev) if args.dev else []
    model=JapaneseAdapter(rank=args.rank,n_japanese=len(ja_vocab),kernel=args.kernel).to(device)
    original=read_reader_checkpoint(args.reader_pt)
    en_scale=float(original.log_scale.exp())
    params=list(model.parameters())+([tensor] if args.train_ja_table else [])
    groups=[{'params':list(model.parameters()),'lr':args.lr,'weight_decay':0.001}]
    if args.train_ja_table:
        # Rows are re-normalized in objective/export, so decay is pointless; use a gentler lr
        # so Japanese word vectors adapt to the encoder without drifting from their embeddings.
        groups.append({'params':[tensor],'lr':args.lr_table,'weight_decay':0.0})
    optimizer=torch.optim.AdamW(groups)
    steps_per_epoch=math.ceil(math.ceil(len(data_train)/args.batch)/args.accum)
    total_steps=steps_per_epoch*args.epochs
    scheduler=torch.optim.lr_scheduler.LambdaLR(
        optimizer,lambda s:warmup_cosine(s,min(args.warmup,total_steps//5),total_steps))
    ema=Ema(params,args.ema_decay) if args.ema_decay>0 else None
    output=Path(args.out_dir);output.mkdir(parents=True,exist_ok=True)
    config={'rank':args.rank,'kernel':args.kernel,'en_vocab_size':len(en_vocab),'ja_vocab_size':len(ja_vocab),
            'en_scale':en_scale,'en_package':str(args.en_package),'reader_pt':str(args.reader_pt),
            'train_ja_table':args.train_ja_table}
    baseline=english_baseline(data_dev,en_vocab,args.max_eval_per_language) if data_dev else None
    print('EN original baseline WER:',baseline,flush=True)
    best=float('inf');stale=0;history=[];accepted=False
    for epoch in range(1,args.epochs+1):
        random.shuffle(data_train);model.train();total=[]
        optimizer.zero_grad(set_to_none=True)
        starts=list(range(0,len(data_train),args.batch))
        for n,idx in enumerate(starts,1):
            batch=data_train[idx:idx+args.batch]
            loss,metrics=objective(model,tensor,en_raw,en_scale,batch,device,gate_weight=args.gate_weight)
            (loss/args.accum).backward()
            if n%args.accum==0 or n==len(starts):
                torch.nn.utils.clip_grad_norm_(params,1.0)
                optimizer.step();scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema:ema.update(params)
            total.append(float(loss.detach()))
            if len(total)==1 or len(total)%100==0:
                print(f'epoch {epoch}/{args.epochs}, batch {len(total)} loss={total[-1]:.4f} gate={metrics["gate"]:.3f} lr={scheduler.get_last_lr()[0]:.2e}',flush=True)
        # Evaluate and checkpoint the EMA weights; restore live weights before the next epoch.
        if ema:ema.swap(params)
        model.eval();sample=[];group=defaultdict(list)
        with torch.no_grad():
            for i in range(0,len(data_dev),args.batch):
                batch=data_dev[i:i+args.batch]
                loss,_=objective(model,tensor,en_raw,en_scale,batch,device,gate_weight=args.gate_weight)
                sample.append(float(loss))
                for r in batch:group[r[3]].append(r)
        metric=float(np.mean(sample)) if sample else float(np.mean(total))
        scores=evaluate_native_gate(model,tensor,data_dev,en_vocab,ja_vocab,device,
                                    max_each=args.max_eval_per_language) if data_dev else {}
        en_wer=scores.get('en',{}).get('error_rate')
        guard=(baseline is None or en_wer is None or
               en_wer<=baseline+args.en_wer_tolerance)
        ja_cer=scores.get('ja',{}).get('error_rate')
        mixed_cer=scores.get('mixed',{}).get('error_rate')
        # JA/mixed errors determine best, with EN regression guard.
        selection=(ja_cer if ja_cer is not None else metric)
        if mixed_cer is not None:selection=(selection+mixed_cer)/2
        record={'epoch':epoch,'train_loss':float(np.mean(total)),'dev_loss':metric,
                'dev_languages':{k:len(v) for k,v in group.items()},
                'actual_greedy_metrics':scores,'en_baseline_wer':baseline,
                'english_guard_pass':guard, 'selection_metric':selection}
        history.append(record)
        print('epoch summary:',record,flush=True)
        checkpoint={'adapter':{k:v.cpu() for k,v in model.state_dict().items()},
                    'ja_table':F.normalize(tensor.detach(),dim=-1).cpu().half(),
                    'config':config}
        torch.save(checkpoint,output/'adapter_last.pt')
        if ema:ema.swap(params)
        if guard and selection<best:
            best=selection;stale=0;accepted=True
            torch.save(checkpoint,output/'adapter.pt')
            print(f'Accepted checkpoint: EN WER={en_wer}, JA CER={ja_cer}, mixed CER={mixed_cer}',flush=True)
        else:
            stale+=1
            if args.patience and stale>=args.patience:
                print('early stopping (no dev loss improvement)',flush=True);break
    (output/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2)+'\n')
    if not accepted:
        raise RuntimeError('NO checkpoint meets frozen-English WER protection. adapter_last.pt is for diagnosis only; do not deploy.')
    print('Best model saved:',output/'adapter.pt')


def export(args):
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=checkpoint['config']
    rank=config['rank']
    if not 1<=rank<=RANK_MAX:raise ValueError('unsupported rank')
    kernel=config.get('kernel',1)
    model=JapaneseAdapter(rank,config['ja_vocab_size'],kernel)
    model.load_state_dict(checkpoint['adapter'],strict=True)
    stage=Path(args.out).with_suffix('.staging')
    stage.mkdir(parents=True,exist_ok=True)
    adapter_file=stage/'adapter.eg2a'
    names=['w1.weight','w1.bias','w2.weight','w2.bias','gate.weight','gate.bias']
    arrays=[model.state_dict()[k].detach().numpy().astype('<f4').ravel() for k in names]
    if kernel>1:  # v2: dw[R,K] after gate_b; Conv1d weight is [R,1,K]
        arrays.append(model.dw.weight.detach().numpy().astype('<f4').reshape(rank,kernel).ravel())
    count=sum(len(a) for a in arrays)
    with adapter_file.open('wb') as f:
        f.write(struct.pack('<4s5I',b'EG2A',2 if kernel>1 else 1,rank,INPUT,OUT,count))
        if kernel>1:f.write(struct.pack('<I',kernel))
        for a in arrays:f.write(a.tobytes())
        f.write(struct.pack('<f',math.exp(float(model.ja_log_scale.detach()))))
    vocab=Path(args.ja_vocab).read_text(encoding='utf8').splitlines()
    if len(vocab)!=config['ja_vocab_size']:raise ValueError('checkpoint Japanese vocab size mismatch')
    (stage/'reader_vocab.txt').write_text('\n'.join(vocab)+'\n',encoding='utf8')
    table=checkpoint['ja_table'].float().numpy().astype(np.float32)
    if table.shape!=(len(vocab),768):raise ValueError('checkpoint Japanese table mismatch')
    from convert_eg2_table import main as table_main
    tmp_table=stage/'word_table.npy'
    np.save(tmp_table,table.astype(np.float16))
    old_argv=sys.argv
    try:
        sys.argv=['convert_eg2_table',str(tmp_table),str(stage/'word_table.eg2q')]
        table_main()
    finally:sys.argv=old_argv
    tmp_table.unlink()
    pack(stage,['adapter.eg2a','word_table.eg2q','reader_vocab.txt'],Path(args.out),force=args.force)
    print('EN weights and 200k-word table are NOT included or modified.')
    print('Use: ./cats -a eg2-native-asr -m models -L adapter -w speech.wav')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare',help='precompute frozen English Reader scores, no training')
    prep.add_argument('--train',type=Path,required=True)
    prep.add_argument('--dev',type=Path)
    prep.add_argument('--en-package',type=Path,required=True)
    prep.add_argument('--reader-pt',type=Path,required=True)
    prep.add_argument('--out-dir',type=Path,required=True)
    prep.add_argument('--device',default='auto',choices=('auto','cpu','cuda'))
    a=sub.add_parser('train',help='joint CTC, train only JA Adapter/gate')
    a.add_argument('--train',type=Path,required=True)
    a.add_argument('--dev',type=Path)
    a.add_argument('--cache-dir',type=Path,required=True)
    a.add_argument('--en-package',type=Path,required=True)
    a.add_argument('--reader-pt',type=Path,required=True)
    a.add_argument('--ja-vocab',type=Path,required=True)
    a.add_argument('--ja-table',type=Path,required=True)
    a.add_argument('--out-dir',type=Path,required=True)
    a.add_argument('--epochs',type=int,default=10)
    a.add_argument('--patience',type=int,default=3)
    a.add_argument('--batch',type=int,default=2)
    a.add_argument('--rank',type=int,default=64)
    a.add_argument('--kernel',type=int,default=7,help='temporal conv taps (odd, 1=per-frame v1); 7 = ~280 ms context')
    a.add_argument('--lr',type=float,default=0.0003)
    a.add_argument('--gate-weight',type=float,default=0.15)
    a.add_argument('--train-ja-table',action=argparse.BooleanOptionalAction,default=True,
                   help='fine-tune the Japanese word vectors (default on; --no-train-ja-table freezes them)')
    a.add_argument('--lr-table',type=float,default=1e-4)
    a.add_argument('--accum',type=int,default=8,help='gradient accumulation steps (effective batch = batch*accum)')
    a.add_argument('--warmup',type=int,default=200,help='LR warmup optimizer steps')
    a.add_argument('--ema-decay',type=float,default=0.999,help='0 disables EMA')
    a.add_argument('--limit',type=int,default=0)
    a.add_argument('--limit-dev',type=int,default=0)
    a.add_argument('--max-eval-per-language',type=int,default=100)
    a.add_argument('--en-wer-tolerance',type=float,default=0.01,help='Max absolute English WER regression versus original en.eg2m')
    a.add_argument('--device',default='auto',choices=('auto','cpu','cuda'))
    a.add_argument('--seed',type=int,default=42)
    e=sub.add_parser('export',help='export small JA EG2A package for native C')
    e.add_argument('--checkpoint',type=Path,required=True)
    e.add_argument('--ja-vocab',type=Path,required=True)
    e.add_argument('--out',type=Path,required=True)
    e.add_argument('--force',action='store_true')
    args=p.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='train':
        if args.kernel<1 or args.kernel%2==0 or args.kernel>KERNEL_MAX:p.error('--kernel must be odd, 1..31')
        if (not 1<=args.rank<=256 or args.epochs<1 or args.batch<1 or args.lr<=0 or args.accum<1
                or args.lr_table<=0 or not 0<=args.ema_decay<1):
            p.error('invalid training hyperparameters')
        train(args)
    else:export(args)

if __name__=='__main__':
    try:main()
    except Exception:
        print('EG2 JA Adapter failed (root traceback):',file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
