#!/usr/bin/env python3
"""Convert the EmbeddingGemma 2 tokenizer.json (BPE, byte fallback) to a compact binary for C.

Layout (little endian):  "EG2T", u32 n_vocab, u32 n_merges,
  n_vocab  x { u16 length, utf-8 bytes }                  token id = index
  n_merges x { u32 left_id, u32 right_id, u32 result_id } merge rank = index (lower merges first)

Usage: python eg2_embed_tokenizer.py MODEL_DIR/tokenizer.json OUT.eg2t
"""
import json
import struct
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    t = json.load(open(src, encoding='utf8'))
    model = t['model']
    if model['type'] != 'BPE' or not model.get('byte_fallback'):
        raise SystemExit('expected a byte-fallback BPE tokenizer')
    vocab = model['vocab']
    inv = [None] * len(vocab)
    for tok, i in vocab.items():
        inv[i] = tok
    merges = model['merges']
    with open(dst, 'wb') as f:
        f.write(b'EG2T' + struct.pack('<II', len(inv), len(merges)))
        for tok in inv:
            b = tok.encode('utf8')
            f.write(struct.pack('<H', len(b)) + b)
        skipped = 0
        for a, b in (m if isinstance(m, list) else m.split(' ') for m in merges):
            r = vocab.get(a + b)
            if r is None:  # cannot happen for a consistent BPE; keep rank alignment anyway
                skipped += 1
                f.write(struct.pack('<III', vocab[a], vocab[b], 0xffffffff))
            else:
                f.write(struct.pack('<III', vocab[a], vocab[b], r))
    print(f'{len(inv)} tokens, {len(merges)} merges ({skipped} without result) -> {dst}')


if __name__ == '__main__':
    main()
