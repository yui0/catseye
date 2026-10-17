#!/usr/bin/env python3
"""Cache audio soft tokens of a Japanese JSONL speech corpus using a FROZEN encoder.

Rows: {"id":"safe_unique_id", "audio":"/path/to/audio.wav", "text":"こんにちは"}
Writes NPY float16 [T,512] and JSONL {id, soft, text}. Supports resumability.
Works with the installed google/embeddinggemma-2 HuggingFace checkpoint.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from eg2_ja_vocab import load_rows


def read_audio(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    wav,sr=sf.read(str(path),dtype='float32',always_2d=False)
    if wav.ndim==2:wav=wav.mean(axis=1)
    if sr!=16000:
        g=gcd(sr,16000)
        wav=resample_poly(wav,16000//g,sr//g).astype(np.float32)
    return np.ascontiguousarray(wav,dtype=np.float32)


def load_encoder(model_dir,device):
    import torch
    from transformers import EmbeddingGemma2Model, Gemma4AudioFeatureExtractor
    fe=Gemma4AudioFeatureExtractor.from_pretrained(str(model_dir))
    model=EmbeddingGemma2Model.from_pretrained(str(model_dir),dtype=torch.float32,vision_config=None)
    model=model.to(device).eval()
    impl=getattr(model.config.audio_config,'_attn_implementation',None)
    if impl=='eager':raise RuntimeError('transformers 5.19.0 eager attention mask is incompatible; use SDPA')
    return model,fe


def soft_tokens(model,fe,wav,device):
    import torch
    f=fe([wav],return_tensors='np',max_length=None,truncation=False)
    x=torch.from_numpy(np.asarray(f['input_features'],dtype=np.float32)).to(device)
    mask=torch.from_numpy(np.asarray(f['input_features_mask'],dtype=bool)).to(device)
    with torch.inference_mode():
        y=model.get_audio_features(x,mask,return_dict=True)
        soft=y.pooler_output[0][y.attention_mask[0]].float().cpu().numpy()
    expected=((max(0,(len(wav)-161)//160+1)+3)//4)
    if soft.shape!=(expected,512) or not np.isfinite(soft).all():
        raise RuntimeError(f'invalid audio tokens {soft.shape}, expected {(expected,512)}')
    return soft.astype(np.float16)


def build_cache(manifest,model_dir,out,device='cuda',limit=0):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=list(load_rows(manifest))
    if limit:rows=rows[:limit]
    valid=[]
    for i,r in enumerate(rows):
        uid=str(r.get('id',i))
        if not uid.isascii() or not uid.replace('-','').replace('_','').isalnum():
            raise ValueError('id must be ASCII alphanumeric, - or _')
        source=Path(r['audio']);source=source if source.is_absolute() else Path(manifest).resolve().parent/source
        if not source.exists():raise FileNotFoundError(source)
        valid.append((uid,source,r['text']))
    model=fe=None
    result=[]
    for ix,(uid,source,txt) in enumerate(valid):
        cache_file=out/(uid+'.npy')
        if not cache_file.exists():
            if model is None:model,fe=load_encoder(model_dir,device)
            wav=read_audio(source)
            if not 320<=len(wav)<=30*16000:
                print(f'SKIP {uid}: audio duration outside 0.02-30 s');continue
            np.save(cache_file,soft_tokens(model,fe,wav,device))
        a=np.load(cache_file,mmap_mode='r')
        if a.ndim!=2 or a.shape[1]!=512 or a.shape[0]<1:raise ValueError('invalid cache '+str(cache_file))
        result.append({'id':uid,'soft':str(cache_file.resolve()),'text':txt})
        if ix%20==0:print(f'cache {ix+1}/{len(valid)}',flush=True)
    target=out/'index.jsonl'
    with target.open('w',encoding='utf8') as f:
        for r in result:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('saved',target,'items',len(result))
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--device',default='cuda')
    a=p.parse_args();build_cache(a.manifest,a.model_dir,a.out,a.device,a.limit)

if __name__=='__main__':main()
