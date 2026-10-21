#!/usr/bin/env python3
"""Create JA-ONLY CTC vocabulary; original English 200k words stay in en.eg2m.
Every kanji/kana observed in training has a single-character fallback.
Optional common Japanese word tokens improve compression/decoding.
"""
import argparse
from collections import Counter
from pathlib import Path
import unicodedata
from eg2_ja_vocab import load_rows,save_vocab


def only_ja(text):
    text=unicodedata.normalize('NFKC',text)
    return ''.join(c for c in text if (('\u3040'<=c<='\u30ff') or ('\u3400'<=c<='\u9fff') or '\u3005'==c))


def build(rows,max_vocab=8000,min_count=3,words=False):
    char_counts,words_count=Counter(),Counter()
    tokenizer=None
    if words:
        try:
            from sudachipy import dictionary,tokenizer as sudachi_tokenizer
            tokenizer=dictionary.Dictionary().create(mode=sudachi_tokenizer.Tokenizer.SplitMode.C)
        except ImportError:
            raise RuntimeError('Word mode requires pip install sudachipy sudachidict_core')
    for r in rows:
        cleaned=only_ja(r['text'])
        char_counts.update(cleaned)
        if tokenizer and cleaned:
            words_count.update(t.surface() for t in tokenizer.tokenize(cleaned) if len(t.surface())>1)
    chars=sorted(char_counts)
    if not chars:raise ValueError('No Japanese characters found in dataset')
    if len(chars)>max_vocab:raise ValueError(f'{len(chars)} Japanese characters exceed max_vocab={max_vocab}')
    words=[w for w,n in sorted(words_count.items(),key=lambda a:(-a[1],a[0])) if n>=min_count and w not in char_counts]
    return chars+words[:max_vocab-len(chars)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--max-vocab',type=int,default=8000)
    p.add_argument('--min-count',type=int,default=3)
    p.add_argument('--words',action='store_true')
    a=p.parse_args()
    vocab=build(load_rows(a.train),a.max_vocab,a.min_count,a.words)
    save_vocab(vocab,a.out)
    print(f'Japanese-only tokens: {len(vocab)}; created {a.out}')

if __name__=='__main__':main()
