#!/usr/bin/env python3
"""Compare two eg2e models (e.g. fp32 and Q4) on a folder of images.

Both models must have indexed the same files with `eg2e -C CACHE index FILE...`. The script reports
  * image -> image: how often the nearest neighbours agree between the two models,
  * near-duplicate pairs (cosine above a threshold),
  * optional text queries (needs the eg2e binary, the tokenizer and both model directories),
and can write contact sheets (query + best matches) to look at the results.

Usage: eg2_eval_folder.py --cache-a cache_fp32.bin --cache-b cache_q4.bin [--eg2e ./eg2e --tok eg2.eg2t
       --model-a eg2_hf --model-b eg2_q4 --queries queries.txt] [--sheets out_dir]
"""
import argparse
import os
import struct
import subprocess
import sys

import numpy as np


def load_cache(path):
    vecs = {}
    with open(path, 'rb') as f:
        assert f.read(4) == b'EG2C'
        n, = struct.unpack('<i', f.read(4))
        for _ in range(n):
            ln, = struct.unpack('<i', f.read(4))
            name = f.read(ln).decode('utf8')
            f.read(8 + 8 + 4)  # size, mtime, soft
            vecs[name] = np.frombuffer(f.read(768 * 4), dtype='<f4').copy()
    return vecs


def topk(sim, k, exclude=None):
    s = sim.copy()
    if exclude is not None:
        s[exclude] = -9
    idx = np.argpartition(-s, k)[:k]
    return idx[np.argsort(-s[idx])]


def text_vec(eg2e, model, tok, query):
    out = subprocess.run([eg2e, '-m', model, '-k', tok, 'vec', '-t', query], capture_output=True, text=True, check=True).stdout
    return np.array(out.split(), dtype=np.float32)


def sheet(paths, title, out, tile=220):
    from PIL import Image, ImageDraw
    ims = []
    for p in paths:
        try:
            im = Image.open(p).convert('RGB')
            im.thumbnail((tile, tile))
        except Exception:
            im = Image.new('RGB', (tile, tile), (60, 60, 60))
        ims.append(im)
    canvas = Image.new('RGB', (tile * len(ims), tile + 18), (20, 20, 20))
    for i, im in enumerate(ims):
        canvas.paste(im, (i * tile, 18))
    ImageDraw.Draw(canvas).text((4, 3), title[:120], fill=(255, 255, 0))
    canvas.save(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-a', required=True)
    ap.add_argument('--cache-b', required=True)
    ap.add_argument('--eg2e')
    ap.add_argument('--tok')
    ap.add_argument('--model-a')
    ap.add_argument('--model-b')
    ap.add_argument('--queries', help='text file, one query per line')
    ap.add_argument('--sheets', help='directory for contact sheets')
    ap.add_argument('--dup', type=float, default=0.97)
    ap.add_argument('--first', type=int, default=0, help='use only the first N images (sorted by name)')
    a = ap.parse_args()

    A, B = load_cache(a.cache_a), load_cache(a.cache_b)
    names = sorted(set(A) & set(B))
    if a.first:
        names = names[:a.first]
    X = np.stack([A[n] for n in names]); Y = np.stack([B[n] for n in names])
    N = len(names)
    print(f'{N} images in both caches (A only {len(set(A) - set(B))}, B only {len(set(B) - set(A))})')
    print('cosine(A_i, B_i) per image: mean %.5f  min %.5f' % ((X * Y).sum(1).mean(), (X * Y).sum(1).min()))

    SA, SB = X @ X.T, Y @ Y.T
    k = 10
    ov, top1, rho = [], [], []
    for i in range(N):
        ta, tb = topk(SA[i], k, i), topk(SB[i], k, i)
        ov.append(len(set(ta) & set(tb)) / k)
        top1.append(ta[0] == tb[0])
        ra, rb = np.argsort(np.argsort(-SA[i])), np.argsort(np.argsort(-SB[i]))
        rho.append(np.corrcoef(ra, rb)[0, 1])
    print(f'image->image: top-{k} overlap {np.mean(ov):.3f}, top-1 agree {np.mean(top1):.3f}, rank correlation {np.mean(rho):.4f}')

    iu = np.triu_indices(N, 1)
    for label, S in (('A', SA), ('B', SB)):
        print(f'near-duplicate pairs (cos >= {a.dup}) in {label}: {int((S[iu] >= a.dup).sum())}')
    pa = set(zip(*[x[(SA[iu] >= a.dup)] for x in iu])); pb = set(zip(*[x[(SB[iu] >= a.dup)] for x in iu]))
    print(f'  pairs found by both {len(pa & pb)}, only A {len(pa - pb)}, only B {len(pb - pa)}')

    if a.sheets:
        os.makedirs(a.sheets, exist_ok=True)
        rng = np.random.default_rng(3)
        for q in rng.choice(N, 4, replace=False):
            ta = topk(SA[q], 5, q)
            sheet([names[q]] + [names[j] for j in ta], f'image query {os.path.basename(names[q])}: fp32 top5', f'{a.sheets}/img_{q}_A.png')
            tb = topk(SB[q], 5, q)
            sheet([names[q]] + [names[j] for j in tb], f'image query {os.path.basename(names[q])}: Q4 top5', f'{a.sheets}/img_{q}_B.png')

    if a.queries:
        qa, qb = [], []
        for n, line in enumerate([l.strip() for l in open(a.queries, encoding='utf8') if l.strip()]):
            va, vb = text_vec(a.eg2e, a.model_a, a.tok, line), text_vec(a.eg2e, a.model_b, a.tok, line)
            sa, sb = X @ va, Y @ vb
            ta, tb = topk(sa, 10), topk(sb, 10)
            print(f'\n"{line}"  top-10 overlap {len(set(ta) & set(tb)) / 10:.1f}, top-1 {"same" if ta[0] == tb[0] else "different"},'
                  f' best score A {sa[ta[0]]:.3f} / B {sb[tb[0]]:.3f}')
            for j in ta[:3]:
                print(f'    A {sa[j]:.3f} {os.path.basename(names[j])}')
            if a.sheets:
                sheet([names[j] for j in ta[:6]], f'"{line}" fp32 top6', f'{a.sheets}/text_{n:02d}_A.png')
                sheet([names[j] for j in tb[:6]], f'"{line}" Q4 top6', f'{a.sheets}/text_{n:02d}_B.png')


if __name__ == '__main__':
    main()
