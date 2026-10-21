#!/usr/bin/env python3
"""Make synthetic JA/EN turn-taking clips from split-specific WAV files.

This is inexpensive extra acoustic supervision for a shared CTC Reader, NOT a
replacement for natural in-sentence code-switch recordings. Never mix train
with dev or test: this command processes exactly one JSONL split at a time.
"""
import argparse
import json
from pathlib import Path
import random


def augmented_rows(manifest, out_manifest, pair_count=400, seed=19, gap_ms=100):
    import numpy as np
    import soundfile as sf

    manifest, out_manifest = Path(manifest), Path(out_manifest)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not 0 <= gap_ms <= 600 or pair_count < 0:
        raise ValueError('pair_count must be >=0, gap_ms in 0..600')
    rows = [json.loads(v) for v in manifest.read_text(encoding='utf-8').splitlines() if v.strip()]
    japanese, english = [], []
    for r in rows:
        text = r.get('text','')
        if not isinstance(text,str) or not text.strip():continue
        # FLEURS prefix; other datasets may provide {language: 'ja'|'en'}.
        language = r.get('language', '')
        name = str(r.get('id','')).lower()
        if language.startswith('ja') or name.startswith('ja_jp_'):
            japanese.append(r)
        elif language.startswith('en') or name.startswith('en_us_'):
            english.append(r)
    if pair_count and (not japanese or not english):
        raise ValueError('Need both JA and EN rows with language field or ja_jp_/en_us_ ID prefixes')
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    clips = out_manifest.with_suffix('')
    clips.mkdir(parents=True,exist_ok=True)
    rng = random.Random(seed)
    rng.shuffle(japanese); rng.shuffle(english)
    result = list(rows)
    created = 0
    # Pair clips cyclically, shuffle their order so that both switch directions occur.
    # Reuse at most a few times; skip combined clips longer than 30 s.
    for i in range(pair_count * 2):
        if created>=pair_count:break
        ja = japanese[i % len(japanese)]
        en = english[i % len(english)]
        a = ja['audio']; b = en['audio']
        pa = Path(a); pb = Path(b)
        if not pa.is_absolute():pa = manifest.resolve().parent/pa
        if not pb.is_absolute():pb = manifest.resolve().parent/pb
        x, sr1 = sf.read(pa, dtype='float32', always_2d=False)
        y, sr2 = sf.read(pb, dtype='float32', always_2d=False)
        if x.ndim==2:x=x.mean(axis=1)
        if y.ndim==2:y=y.mean(axis=1)
        if sr1!=16000 or sr2!=16000:
            raise ValueError('Synthetic code-switch expects 16 kHz input WAV; run through FLEURS/normalizer first')
        gap = np.zeros(int(gap_ms*16),dtype=np.float32)
        if len(x)+len(y)+len(gap)>30*16000:continue
        if i%2:
            x,y=y,x
            text=str(en['text']).strip()+' '+str(ja['text']).strip()
        else:
            text=str(ja['text']).strip()+' '+str(en['text']).strip()
        uid=f'mix_pair_{seed}_{created:06d}'
        dst=clips/(uid+'.wav')
        sf.write(dst,np.concatenate((x,gap,y)),16000,subtype='PCM_16')
        result.append({'id':uid,'audio':str(dst.resolve()),'text':text,'language':'mixed',
                       'synthetic':True})
        created+=1
    if pair_count and not created:raise ValueError('All source pair durations exceed 30 seconds')
    with out_manifest.open('w',encoding='utf-8') as f:
        for r in result:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f'Synthetic code-switch: {created} new clips, {len(rows)} originals, -> {out_manifest}',flush=True)
    return created


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=400)
    p.add_argument('--seed',type=int,default=19)
    p.add_argument('--gap-ms',type=int,default=100)
    a=p.parse_args()
    augmented_rows(a.input,a.output,a.pairs,a.seed,a.gap_ms)

if __name__=='__main__':main()
