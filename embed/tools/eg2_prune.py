#!/usr/bin/env python3
"""Shrink google/embeddinggemma-2 for small deployments (browser / WASM): prune the vocabulary and the
vision position table, drop the audio tower. Output: a bf16 model.safetensors plus a matching .eg2t tokenizer;
quantize the result with `eg2e -m OUT_DIR quantize Q_DIR/model.safetensors`.

  vocabulary : keeps ids < 260 (specials, <0xXX> bytes), the image marker tokens, every single-character
               token of the allowed scripts, and the results of the first --ascii / --cjk BPE merges
               (merge rank ~ frequency). Other words fall back to shorter pieces or <0xXX> bytes.
               Tokens are renumbered densely; <|image>, <image|>, <|image|> go to tensor `eg2.special`.
  position   : [2][10240][768] -> [2][--pos][768] (the C code clamps larger coordinates).

Usage: python eg2_prune.py MODEL_DIR OUT_DIR [--ascii 20000] [--cjk 12000] [--pos 512]
Needs: pip install safetensors torch
"""
import argparse
import json
import os
import struct

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def jis(c):
    try:
        c.encode('shift_jis')
        return True
    except UnicodeEncodeError:
        return False


def allowed(tok):
    for c in tok.replace('▁', ' '):
        o = ord(c)
        if o < 128 or 0x3000 <= o <= 0x30ff or 0xff00 <= o <= 0xffef or 0x2010 <= o <= 0x206f:
            continue
        if 0x4e00 <= o <= 0x9fff and jis(c):
            continue
        return False
    return True


def is_ascii(tok):
    return all(ord(c) < 128 or c == '▁' for c in tok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--ascii', type=int, default=20000, help='merges kept among ASCII-only tokens')
    ap.add_argument('--cjk', type=int, default=12000, help='merges kept among tokens with Japanese characters')
    ap.add_argument('--pos', type=int, default=512, help='rows kept per axis of the vision position table')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    tj = json.load(open(os.path.join(a.model_dir, 'tokenizer.json'), encoding='utf8'))['model']
    vocab, merges = tj['vocab'], tj['merges']
    inv = {i: t for t, i in vocab.items()}
    special = [vocab['<|image>'], vocab['<image|>'], vocab['<|image|>']]

    keep = set(range(260)) | set(special)
    keep |= {i for t, i in vocab.items() if len(t) == 1 and allowed(t)}
    na = nc = 0
    for m in merges:
        l, r = m if isinstance(m, list) else m.split(' ')
        res = vocab.get(l + r)
        if res is None or res in keep or not allowed(inv[res]):
            continue
        if is_ascii(inv[res]):
            if na < a.ascii:
                keep.add(res)
                na += 1
        elif nc < a.cjk:
            keep.add(res)
            nc += 1
        if na >= a.ascii and nc >= a.cjk:
            break
    old_ids = sorted(keep)
    new_of = {o: n for n, o in enumerate(old_ids)}
    print(f'vocabulary {len(vocab)} -> {len(old_ids)} ({na} ascii + {nc} japanese merges)')

    # tokenizer: keep merges whose parts and result all survive (ranks keep their relative order)
    out_merges = []
    for m in merges:
        l, r = m if isinstance(m, list) else m.split(' ')
        res = vocab.get(l + r)
        if res is not None and vocab[l] in new_of and vocab[r] in new_of and res in new_of:
            out_merges.append((new_of[vocab[l]], new_of[vocab[r]], new_of[res]))
    with open(os.path.join(a.out_dir, 'tokenizer.eg2t'), 'wb') as f:
        f.write(b'EG2T' + struct.pack('<II', len(old_ids), len(out_merges)))
        for o in old_ids:
            b = inv[o].encode('utf8')
            f.write(struct.pack('<H', len(b)) + b)
        for l, r, res in out_merges:
            f.write(struct.pack('<III', l, r, res))
    print(f'merges {len(merges)} -> {len(out_merges)}')

    tensors = {}
    with safe_open(os.path.join(a.model_dir, 'model.safetensors'), 'pt') as sf:
        for k in sf.keys():
            if k.startswith(('audio_tower.', 'embed_audio.')):
                continue
            t = sf.get_tensor(k)
            if k == 'language_model.embed_tokens.weight':
                t = t[torch.tensor(old_ids)].contiguous()
            elif k == 'vision_tower.patch_embedder.position_embedding_table':
                t = t[:, :a.pos].contiguous()
            tensors[k] = t
    tensors['eg2.special'] = torch.tensor([float(new_of[s]) for s in special])
    save_file(tensors, os.path.join(a.out_dir, 'model.safetensors'), metadata={'format': 'pt'})
    print('wrote', a.out_dir, '(boi, eoi, image ids =', [new_of[s] for s in special], ')')


if __name__ == '__main__':
    main()
