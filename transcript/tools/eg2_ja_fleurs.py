#!/usr/bin/env python3
"""Fetch a bounded Japanese FLEURS CC-BY-4.0 speech sample for a Colab demo.

Produces train.jsonl/dev.jsonl: {id,audio,text}. Training audio data is NOT
included in the distribution. --limit 0 uses whole selected dataset.
"""
import argparse
import io
import json
from pathlib import Path
import numpy as np


def decode_audio(obj):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    if 'array' in obj:
        data,sr=np.asarray(obj['array'],dtype=np.float32),obj['sampling_rate']
    elif 'bytes' in obj and obj['bytes']:
        data,sr=sf.read(io.BytesIO(obj['bytes']),dtype='float32',always_2d=False)
    elif obj.get('path') and Path(obj['path']).exists():
        data,sr=sf.read(obj['path'],dtype='float32',always_2d=False)
    else:
        raise RuntimeError('FLEURS Audio returned neither decoded data nor bytes/path')
    if data.ndim==2:data=data.mean(axis=1)
    if sr!=16000:
        g=gcd(sr,16000);data=resample_poly(data,16000//g,sr//g).astype('float32')
    return data


def fetch(out,train_limit=100,dev_limit=30):
    from datasets import load_dataset,Audio
    import soundfile as sf
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results={}
    for split,limit in [('train',train_limit),('dev',dev_limit)]:
        remote='validation' if split=='dev' else 'train'
        source=load_dataset('google/fleurs','ja_jp',split=remote,streaming=True)
        source=source.cast_column('audio',Audio(decode=False))
        audio_dir=out/split;audio_dir.mkdir(exist_ok=True)
        meta=out/(split+'.jsonl');written=0
        with meta.open('w',encoding='utf8') as fd:
            for k,row in enumerate(source):
                if limit and written>=limit:break
                uid=f'{split}_{k:06d}';text=(row.get('transcription') or '').strip()
                if not text:continue
                wav=decode_audio(row['audio'])
                if not 320<=len(wav)<=480000:continue
                p=audio_dir/(uid+'.wav')
                sf.write(p,wav,16000,subtype='PCM_16')
                fd.write(json.dumps({'id':uid,'audio':str(p.resolve()),'text':text},ensure_ascii=False)+'\n')
                written+=1
                if written%50==0:print(f'{split}: {written} examples',flush=True)
        if not written:raise RuntimeError('no usable FLEURS samples downloaded')
        results[split]=(str(meta),written)
        print(split,written,meta,flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train-limit',type=int,default=100)
    p.add_argument('--dev-limit',type=int,default=30)
    a=p.parse_args();fetch(a.out,a.train_limit,a.dev_limit)

if __name__=='__main__':main()
