#!/usr/bin/env python3
"""Restore FLEURS Japanese+English audio caches for frozen-English JA adapter.

Never alters or regenerates the original en.eg2m. Prefer already cached
soft-token arrays and metadata, otherwise download the dataset/model and
rebuild the missing cache. Suitable for restarted Colab runtimes.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback


def index_ok(index: Path, *, check_arrays=True):
    """Verify both index.jsonl and every referenced soft-token array."""
    import numpy as np
    if not index.is_file():
        return False, f'index missing: {index}'
    try:
        rows = [json.loads(s) for s in index.read_text(encoding='utf8').splitlines() if s.strip()]
        if not rows:
            return False, f'empty index: {index}'
        seen = set()
        for n, row in enumerate(rows, 1):
            ident = row.get('id')
            if not isinstance(ident, str) or ident in seen:
                return False, f'invalid/duplicate id at row {n}: {ident}'
            seen.add(ident)
            if not isinstance(row.get('text'), str) or not row['text'].strip():
                return False, f'invalid transcript at row {n}'
            p = Path(row['soft'])
            if not p.is_absolute():
                p = index.parent / p
            if not p.is_file():
                return False, f'missing soft token: {p}'
            if check_arrays:
                a = np.load(p, mmap_mode='r', allow_pickle=False)
                if a.ndim != 2 or a.shape[1] != 512 or a.shape[0] < 1 or a.dtype != np.float16:
                    return False, f'invalid soft token: {p}, shape={a.shape}, dtype={a.dtype}'
    except (KeyError, ValueError, OSError, TypeError, EOFError, json.JSONDecodeError) as exc:
        return False, f'invalid index {index}: {exc}'
    return True, f'{len(rows)} utterances'


def manifest_ok(manifest: Path):
    """Validate pre-existing manifests and WAV paths to prevent stale Colab paths."""
    if not manifest.is_file():
        return False
    try:
        rows = [json.loads(s) for s in manifest.read_text(encoding='utf8').splitlines() if s.strip()]
        if not rows:
            return False
        for row in rows:
            source = Path(row['audio'])
            if not source.is_absolute():
                source = manifest.parent / source
            if not source.is_file() or not str(row.get('text', '')).strip():
                return False
        return True
    except (ValueError, OSError, KeyError, TypeError):
        return False


def merge_fleurs(data: Path, split: str):
    path = data / (split + '.jsonl')
    count = 0
    with path.open('w', encoding='utf8') as dest:
        for language in ('ja_jp', 'en_us'):
            src = data / language / f'{split}.jsonl'
            if not manifest_ok(src):
                raise RuntimeError(f'Invalid FLEURS manifest or missing audio: {src}')
            for line in src.read_text(encoding='utf8').splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row['id'] = language + '_' + row['id']
                row['language'] = language
                dest.write(json.dumps(row, ensure_ascii=False) + '\n')
                count += 1
    if not count:
        raise RuntimeError('Dataset merge produced no utterances')
    print(f'Merged {split} dataset: {count} samples ({path})', flush=True)
    return path


def ensure_manifests(base: Path, train_limit: int, dev_limit: int, *, runner=subprocess.run):
    data, tools = base / 'dataset', base / 'tools'
    data.mkdir(parents=True, exist_ok=True)
    outputs = {split: data / f'{split}.jsonl' for split in ('train', 'dev')}
    if all(manifest_ok(path) for path in outputs.values()):
        print('Reusing existing WAV manifests:', *(str(p) for p in outputs.values()), flush=True)
        return outputs
    print('Dataset manifests or audio are missing; fetching FLEURS JA/EN.', flush=True)
    for lang in ('ja_jp', 'en_us'):
        train_file = data / lang / 'train.jsonl'
        dev_file = data / lang / 'dev.jsonl'
        if manifest_ok(train_file) and manifest_ok(dev_file):
            print(f'Already have FLEURS {lang}; reuse.', flush=True)
            continue
        runner([sys.executable, '-u', str(tools / 'eg2_ja_fleurs.py'),
                '--out', str(data / lang), '--language', lang,
                '--train-limit', str(train_limit), '--dev-limit', str(dev_limit)], check=True)
    return {split: merge_fleurs(data, split) for split in ('train', 'dev')}


def model_looks_downloaded(model: Path):
    return (model / 'config.json').is_file() and (model / 'preprocessor_config.json').is_file() and bool(list(model.glob('*.safetensors')))


def ensure_model(model: Path, *, downloader=None):
    if model_looks_downloaded(model):
        print('Reusing downloaded EmbeddingGemma 2 audio encoder:', model, flush=True)
        return
    print('Downloading Google EmbeddingGemma 2 to regenerate audio soft tokens:', model, flush=True)
    if downloader is None:
        from huggingface_hub import snapshot_download
        downloader = snapshot_download
    model.mkdir(parents=True, exist_ok=True)
    downloader(repo_id='google/embeddinggemma-2', local_dir=str(model),
               allow_patterns=['*.json', '*.safetensors', '*.model', 'tokenizer*', 'merges.txt', 'vocab.*'])
    if not model_looks_downloaded(model):
        raise RuntimeError(f'EmbeddingGemma 2 download incomplete: {model}')


def recover(base: Path, train_limit=500, dev_limit=60, device='auto', dtype='auto',
            *, runner=subprocess.run, downloader=None):
    base = Path(base)
    cache = base / 'cache'
    paths = {split: cache / split / 'index.jsonl' for split in ('train', 'dev')}
    usable = {}
    for split, index in paths.items():
        ok, msg = index_ok(index)
        usable[split] = ok
        print(f'{split} cache: {"OK" if ok else "missing/invalid"} - {msg}', flush=True)
    if all(usable.values()):
        print('Both audio caches are valid. No downloads or recomputation necessary.', flush=True)
        return paths

    manifests = ensure_manifests(base, train_limit, dev_limit, runner=runner)
    ensure_model(base / 'source_encoder', downloader=downloader)
    for split, index in paths.items():
        if usable[split]:
            continue
        out = index.parent
        out.mkdir(parents=True, exist_ok=True)
        print(f'Generating {split} audio soft tokens. Already valid .npy files will be reused.', flush=True)
        runner([sys.executable, '-u', str(base / 'tools' / 'eg2_ja_cache.py'),
                '--manifest', str(manifests[split]), '--model-dir', str(base / 'source_encoder'),
                '--out', str(out), '--device', device, '--dtype', dtype], check=True)
        ok, msg = index_ok(index)
        if not ok:
            raise RuntimeError(f'Regenerated {split} cache is invalid: {msg}')
        print(f'{split} cache rebuilt successfully: {msg}', flush=True)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('/content/eg2_mix'))
    p.add_argument('--train-limit', type=int, default=500,
                   help='FLEURS items per language (0=all) only if cache is missing')
    p.add_argument('--dev-limit', type=int, default=60,
                   help='FLEURS validation items per language (0=all)')
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--dtype', choices=('auto', 'bf16', 'fp32'), default='auto')
    args = p.parse_args()
    if args.train_limit < 0 or args.dev_limit < 0:
        p.error('train/dev limits cannot be negative')
    try:
        recover(args.base, args.train_limit, args.dev_limit, args.device, args.dtype)
    except Exception:
        print('=== Adapter cache recovery failed ===', file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)

if __name__ == '__main__':
    main()
