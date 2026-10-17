#!/usr/bin/env python3
"""Download and convert EmbeddingGemma 2 + independent English Reader for native C.

Examples:
  python prepare_eg2_native.py --list-models
  python prepare_eg2_native.py --download --preset official-v2 --out models
  python prepare_eg2_native.py --download --preset official-v1 --download-only
  python prepare_eg2_native.py --download --preset unsloth-v2 --out models
  python prepare_eg2_native.py --encoder /path/to/model.safetensors \\
    --reader /path/to/reader.pt --table /path/to/word_table.npy \\
    --vocab /path/to/vocab.txt --mel /path/to/mel_filters.bin --out models

Only downloads require huggingface_hub. The C inference runtime does NOT.
Conversion of Reader and word table also requires PyTorch/NumPy.
The independently trained Reader is NOT the original Mac app's private Reader.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from check_eg2_native import check
from eg2_bundle import pack, ENCODER, LANGUAGE


ENC_OFFICIAL = 'google/embeddinggemma-2'
ENC_UNSLOTH = 'unsloth/embeddinggemma-2'
READER = 'mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader'
ASSETS = 'mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI'
# Reader v1 is preserved on an historical HF revision; v2 follows main.
PRESETS = {
    'official-v2': dict(encoder=ENC_OFFICIAL, encoder_rev='main', reader_rev='main',
                        info='Recommended: Google original encoder + Reader v2 (6 layers)'),
    'official-v1': dict(encoder=ENC_OFFICIAL, encoder_rev='main', reader_rev='1af9c5f5',
                        info='Historical Reader v1 (4 layers, lower accuracy)'),
    'unsloth-v2': dict(encoder=ENC_UNSLOTH, encoder_rev='main', reader_rev='main',
                       info='Alternative community-hosted encoder + Reader v2; parity unverified'),
}

FILES = {
    'encoder': ('encoder', 'model.safetensors'),
    'reader': ('reader', 'reader.pt'),
    'table': ('reader', 'word_table.npy'),
    'vocab': ('reader', 'vocab.txt'),
    'mel': ('assets', 'mel_filters.bin'),
}


def describe():
    print('Downloadable model combinations (English only):')
    for name, p in PRESETS.items():
        print(f'  {name:14} {p["info"]}')
        print(f'     encoder: https://huggingface.co/{p["encoder"]}/tree/{p["encoder_rev"]}')
        print(f'     reader:  https://huggingface.co/{READER}/tree/{p["reader_rev"]}')
        print(f'     mel:     https://huggingface.co/{ASSETS}/tree/main')
    print('No TensorFlow, Sherpa-ONNX or other new C inference dependency is added.')
    print('Requires `pip install huggingface_hub` for download; `pip install torch numpy` for conversion.')
    print('CoreAI/LiteRT executable bundles and quantized MLX models are not native C inputs.')


def download_sources(args, paths):
    """Download individually, not entire repositories with multi-GB Core AI bundles."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError('Download requires: python -m pip install huggingface_hub') from e
    p = PRESETS[args.preset]
    source_dir = args.download_dir or args.out / '_source'
    source_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        'encoder': (p['encoder'], p['encoder_rev']),
        'reader': (READER, p['reader_rev']),
        'table': (READER, p['reader_rev']),
        'vocab': (READER, p['reader_rev']),
        'mel': (ASSETS, 'main'),
    }
    manifest = {'preset': args.preset, 'files': {}}
    # Download smaller files first. Discover missing assets before retrieving a large encoder.
    for key in ('vocab', 'mel', 'reader', 'table', 'encoder'):
        if paths.get(key) is not None:
            continue  # Explicit source path: preserve user's input.
        repo_id, revision = specs[key]
        directory, filename = FILES[key]
        local = source_dir / directory
        print(f'download {repo_id}@{revision}/{filename}', flush=True)
        path = Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                    revision=revision, local_dir=str(local)))
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'Downloaded file missing/empty: {path}')
        paths[key] = path
        manifest['files'][key] = {'repo': repo_id, 'revision': revision,
                                  'filename': filename, 'local': str(path)}
    (source_dir / 'source_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return paths


def outputs_in(directory):
    fs = [directory / x for x in ('subsampler.eg2c', 'projection.eg2p',
          'reader.eg2r', 'word_table.eg2q', 'reader_vocab.txt', 'mel_filters.bin')]
    fs += [directory / f'layer_{i:02}.eg2l' for i in range(12)]
    return fs


def prepare(args):
    paths = {name: getattr(args, name) for name in FILES}
    missing = [x for x, p in paths.items() if p is None]
    if missing and not args.download:
        raise ValueError('Missing local input paths: ' + ', '.join(missing) +
                         '; use --download to fetch them automatically')
    if not args.download_only and not args.force and not args.dry_run:
        existing = [p for p in outputs_in(args.out) if p.exists()]
        if existing:
            raise ValueError(f'Output exists: {existing[0]}; use --force to overwrite converted files')
    spec = PRESETS[args.preset]
    source_dir = args.download_dir or args.out / '_source'
    if args.dry_run:
        if args.download:
            for name in ('vocab', 'mel', 'reader', 'table', 'encoder'):
                if paths[name] is None:
                    repo, rev = ((spec['encoder'], spec['encoder_rev']) if name == 'encoder' else
                                 (READER, spec['reader_rev']) if name in ('reader', 'table', 'vocab')
                                 else (ASSETS, 'main'))
                    folder, filename = FILES[name]
                    paths[name] = source_dir / folder / filename
                    print(f'WOULD DOWNLOAD {repo}@{rev}/{filename} -> {paths[name]}')
        if args.download_only:
            return
    elif args.download:
        download_sources(args, paths)
    if args.download_only:
        print('Download complete. Pass --download without --download-only to convert.')
        return
    if not args.dry_run:
        for name, path in paths.items():
            if not path or not path.is_file():
                raise ValueError(f'Missing {name} source: {path}')
        args.out.mkdir(parents=True, exist_ok=True)

    def run(script, *params):
        cmd = [sys.executable, str(Path(__file__).with_name(script)), *(str(i) for i in params)]
        print(' '.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    run('convert_eg2_subsample.py', paths['encoder'], args.out / 'subsampler.eg2c')
    run('convert_eg2_conformer.py', paths['encoder'], args.out)
    run('convert_eg2_projection.py', paths['encoder'], args.out / 'projection.eg2p')
    run('convert_eg2_reader.py', paths['reader'], args.out / 'reader.eg2r')
    run('convert_eg2_table.py', paths['table'], args.out / 'word_table.eg2q')
    for key, filename in (('vocab', 'reader_vocab.txt'), ('mel', 'mel_filters.bin')):
        dst = args.out / filename
        print(f'copy {paths[key]} -> {dst}', flush=True)
        if not args.dry_run:
            if paths[key].resolve() != dst.resolve():
                shutil.copyfile(paths[key], dst)
    if not args.dry_run:
        from argparse import Namespace
        n = check(Namespace(model_dir=args.out, reader=args.out/'reader.eg2r',
            table=args.out/'word_table.eg2q', vocab=args.out/'reader_vocab.txt',
            require_mel=True, reader_only=False))
        print(f'PASS: {n} structural files validated; actual ASR accuracy has not been verified')
        if not args.no_bundle:
            pack(args.out, ENCODER, args.out/'encoder.eg2m', args.force)
            pack(args.out, LANGUAGE, args.out/'en.eg2m', args.force)
            print('Bundled model files: encoder.eg2m + en.eg2m')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list-models', action='store_true', help='show recommended, legacy and alternative sources')
    ap.add_argument('--download', action='store_true', help='fetch missing sources from Hugging Face')
    ap.add_argument('--download-only', action='store_true', help='fetch files but skip C format conversion')
    ap.add_argument('--preset', choices=tuple(PRESETS), default='official-v2')
    ap.add_argument('--download-dir', type=Path, help='location for original weights (default OUT/_source)')
    for name in FILES:
        ap.add_argument('--'+name, type=Path, help='local source override for '+name)
    ap.add_argument('--out', type=Path, default=Path('models'))
    ap.add_argument('--force', action='store_true', help='overwrite already converted model output')
    ap.add_argument('--no-bundle', action='store_true', help='keep only legacy individual components')
    ap.add_argument('--dry-run', action='store_true', help='show exact downloads/commands without running')
    args = ap.parse_args(argv)
    if args.list_models:
        describe()
        return
    if args.download_only:
        args.download = True
    prepare(args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(f'model preparation failed: {exc}')
