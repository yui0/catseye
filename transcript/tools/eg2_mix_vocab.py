#!/usr/bin/env python3
"""Build a *single* Japanese + English CTC vocabulary, preserving English boundaries.

Native C's Japanese-mode decoder joins arbitrary pieces directly. The UTF-8
SentencePiece-style U+2581 prefix means 'insert a space unless at the start'.
Normalizing English words to ▂-prefixed tokens (actual prefix: U+2581) makes
code-switching possible without changing the CTC ABI or an external tokenizer.
Both the table-generation and training scripts must use this exact vocabulary.
"""
import argparse
from collections import Counter
import re
import unicodedata
from pathlib import Path
from eg2_ja_vocab import load_rows, save_vocab

BOUNDARY = '▁'


def normalize(text):
    """e.g. '今日は AI modelです' -> '今日は▁ai▁modelです'."""
    src = unicodedata.normalize('NFKC', text).lower()
    out = []
    pending = False
    prev_latin = False
    for ch in src:
        cat = unicodedata.category(ch)
        if ch.isspace() or cat[0] in 'PS' or cat == 'Sm' or ch == BOUNDARY:
            pending = True
            continue
        current_latin = ch.isascii() and ch.isalnum()
        # Prefix all English runs. Preserve explicit gaps before Japanese
        # following English ("hello 日本語"), not Japanese morpheme gaps.
        if (current_latin and (not prev_latin or pending)) or (
                not current_latin and pending and prev_latin):
            out.append(BOUNDARY)
        out.append(ch)
        prev_latin = current_latin
        pending = False
    return ''.join(out)


def build(rows, max_vocab=12000, min_count=2, mode='word'):
    if max_vocab < 1 or min_count < 1 or mode not in ('word', 'character'):
        raise ValueError('invalid parameters')
    cc = Counter()
    wc = Counter()
    for r in rows:
        normalized = normalize(r['text'])
        # Every character must have a fallback. The start of a Latin span is
        # a *prefixed* single character, not a bare ▁ on its own.
        for i, ch in enumerate(normalized):
            if ch == BOUNDARY:
                if i + 1 < len(normalized):
                    cc[BOUNDARY + normalized[i+1]] += 1
            elif i == 0 or normalized[i-1] != BOUNDARY:
                cc[ch] += 1
        if mode == 'word':
            for match in re.finditer(r'▁[a-z0-9]+', normalized):
                wc[match.group()] += 1
            # Japanese words may optionally be added by Sudachi if available.
            try:
                from eg2_ja_vocab import sudachi
                if not hasattr(build, '_sudachi'):
                    build._sudachi = sudachi()
                tok = build._sudachi
                for block in re.split(r'▁[a-z0-9]+', normalized):
                    for t in tok.tokenize(block):
                        if len(t.surface()) > 1: wc[t.surface()] += 1
            except (ImportError, RuntimeError):
                # Word-mode remains useful for English without Sudachi.
                pass
    if not cc:
        raise ValueError('no training labels')
    alphabet = sorted(cc)
    if len(alphabet) > max_vocab:
        raise ValueError('character alphabet larger than max_vocab')
    selected = [w for w,c in sorted(wc.items(), key=lambda v:(-v[1],v[0]))
                if c >= min_count and w not in cc]
    return alphabet + selected[:max_vocab-len(alphabet)]


def tokenize(text, vocab):
    s = normalize(text)
    first = {}
    for word in vocab:
        if word: first.setdefault(word[0], []).append(word)
    for words in first.values(): words.sort(key=lambda w:(-len(w),w))
    pieces = []
    at = 0
    while at < len(s):
        candidates = first.get(s[at], ())
        word = next((w for w in candidates if s.startswith(w, at)), None)
        if word is None:
            raise ValueError(f'vocab cannot encode {s[at:at+12]!r} at {at}')
        pieces.append(word)
        at += len(word)
    return pieces


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-vocab', type=int, default=12000)
    p.add_argument('--min-count', type=int, default=2)
    p.add_argument('--mode', choices=['word','character'], default='word')
    a = p.parse_args()
    rows = list(load_rows(a.train))
    vocab = build(rows,a.max_vocab,a.min_count,a.mode)
    for r in rows:
        if ''.join(tokenize(r['text'],vocab)) != normalize(r['text']):
            raise ValueError(f'round-trip mismatch {r["text"]!r}')
    save_vocab(vocab,a.out)
    print(f'mixed en/ja vocabulary: {len(vocab)} classes, {len(rows)} examples; saved {a.out}')

if __name__ == '__main__': main()
