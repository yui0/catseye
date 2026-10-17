#!/usr/bin/env python3
"""Build Japanese CTC lexicon from UTF-8 JSONL (id,audio,text or id,soft,text).

Supports SudachiPy word tokens + Unicode character fallback. Vocabulary entries are
line indexed, identical class IDs in training, word_table.npy and native EG2Q.
No model training occurs here. Punctuation is dropped for acoustic supervision.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import unicodedata


def normalize(s):
    s=unicodedata.normalize('NFKC',s).strip()
    return ''.join(c.lower() if c.isascii() else c for c in s if not (c.isspace() or
           unicodedata.category(c)[0] in 'PC' or unicodedata.category(c)=='Sm'))


def chars(s):
    return list(normalize(s))


def segment(text, tok=None):
    s=normalize(text)
    if not tok:return list(s)
    return [t.surface() for t in tok.tokenize(s) if t.surface()]


def sudachi():
    try:
        from sudachipy import dictionary, tokenizer
        return dictionary.Dictionary().create(mode=tokenizer.Tokenizer.SplitMode.C)
    except ImportError as exc:
        raise RuntimeError('word mode needs pip install sudachipy sudachidict_core') from exc


def build(rows, mode='word', max_vocab=12000, min_count=2):
    if max_vocab<1:raise ValueError('max_vocab must be positive')
    tok=sudachi() if mode=='word' else None
    wc,cc=Counter(),Counter()
    for r in rows:
        s=normalize(r['text']);cc.update(s)
        if tok:wc.update(segment(s,tok))
    if not cc:raise ValueError('empty training corpus')
    if len(cc)>max_vocab:raise ValueError('character inventory exceeds max_vocab; increase limit')
    alphabet=sorted(cc)  # Include all code points of the training labels to eliminate train OOV.
    remain=max_vocab-len(alphabet)
    selected=[w for w,_ in sorted(wc.items(),key=lambda kv:(-kv[1],kv[0]))
                if len(w)>1 and wc[w]>=min_count and w not in cc][:remain]
    return alphabet+selected


def tokenize(text, vocab):
    """Deterministic longest-match segmentation, fallback to known characters."""
    s=normalize(text)
    lengths={}
    for w in vocab:
        if w:
            lengths.setdefault(w[0],[]).append(w)
    for ws in lengths.values():ws.sort(key=lambda w:(-len(w),w))
    out=[];i=0
    while i<len(s):
        t=next((w for w in lengths.get(s[i],()) if s.startswith(w,i)),None)
        if t is None:raise ValueError(f'no vocabulary entry for U+{ord(s[i]):04X} at position {i}')
        out.append(t);i+=len(t)
    return out


def load_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for no,line in enumerate(f,1):
            if not line.strip():continue
            r=json.loads(line)
            if not isinstance(r.get('text'),str):raise ValueError(f'{path}:{no}: missing text')
            yield r


def save_vocab(vocab,path):
    if len(set(vocab))!=len(vocab) or any(not w or '\n' in w or '\r' in w for w in vocab):
        raise ValueError('invalid vocabulary')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(vocab)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True,help='JSONL with {text, id, audio/soft}')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=('word','character'),default='word')
    p.add_argument('--max-vocab',type=int,default=12000)
    p.add_argument('--min-count',type=int,default=2)
    args=p.parse_args()
    rows=list(load_rows(args.train))
    vocab=build(rows,args.mode,args.max_vocab,args.min_count)
    save_vocab(vocab,args.out)
    lengths=[len(tokenize(r['text'],vocab)) for r in rows]
    print(f'Japanese vocab {len(vocab)} classes ({args.mode}); {len(rows)} labels, mean {sum(lengths)/len(lengths):.1f} tokens per utterance')

if __name__=='__main__':main()
