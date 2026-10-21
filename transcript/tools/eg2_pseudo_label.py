#!/usr/bin/env python3
"""Build pseudo-labelled Japanese / mixed training data from unlabeled speech (sea-asr style).

Two steps; the output manifest is the usual {id,audio,text} JSONL that eg2_ja_cache.py reads.

  label   teacher #1 (faster-whisper) transcribes long audio, segments are merged into
          clips of <= --max-sec and written as 16 kHz WAV. Segments with low confidence
          are dropped. Writes clips.jsonl.
  filter  keep only clips whose text AGREES with a second, independent ASR (any system that
          wrote {"id","text"} JSONL for the clips) within --max-cer, and whose text is
          usable by the adapter trainer (English words in the 200k EN vocab, Japanese
          characters in the JA vocab). Writes the final train manifest.

Never pseudo-label the dev set: keep dev human-labelled (FLEURS etc.) so WER/CER and the
English regression guard stay trustworthy. Teacher mistakes are inherited by the student;
agreement filtering reduces, but does not remove, them.

Usage:
  python eg2_pseudo_label.py label  --audio-dir raw/ --out-dir pseudo --language ja
  # run another ASR over pseudo/clips/*.wav -> hyp2.jsonl ({"id":..,"text":..})
  python eg2_pseudo_label.py filter --manifest pseudo/clips.jsonl --hyp hyp2.jsonl \\
         --out pseudo/train.jsonl --en-vocab EN/reader_vocab.txt --ja-vocab reader_vocab.txt
"""
import argparse
import json
from pathlib import Path
import sys
import unicodedata

AUDIO_EXT = ('.wav', '.flac', '.mp3', '.m4a', '.ogg', '.opus')


def norm(text):
    """Comparison form: NFKC, case-folded, letters/digits/kana/kanji only."""
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    return edit_distance(ref, hyp) / max(1, len(ref))


def merge_segments(segments, max_sec, max_gap=1.0):
    """segments: [(start,end,text)] -> [(start,end,text)] chunks of <= max_sec.

    Consecutive segments are joined only when the gap is small, so that a chunk never
    spans a long silence. A single segment longer than max_sec is dropped (cannot be cut
    without word timestamps)."""
    chunks, cur = [], None
    for start, end, text in segments:
        if end - start > max_sec:
            if cur:
                chunks.append(cur)
            cur = None
            continue
        if cur and start - cur[1] <= max_gap and end - cur[0] <= max_sec:
            cur = (cur[0], end, cur[2] + text)
        else:
            if cur:
                chunks.append(cur)
            cur = (start, end, text)
    if cur:
        chunks.append(cur)
    return chunks


def keep_segment(seg, min_logprob, max_no_speech, max_compress):
    """Whisper confidence gate: seg = dict(avg_logprob,no_speech_prob,compression_ratio)."""
    return (seg['avg_logprob'] >= min_logprob and seg['no_speech_prob'] <= max_no_speech
            and seg['compression_ratio'] <= max_compress)


def read_audio(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    import numpy as np
    wav, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype('float32')
    return np.ascontiguousarray(wav, dtype='float32')


def label(args):
    import soundfile as sf
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError('label needs: pip install faster-whisper') from exc
    out = Path(args.out_dir)
    clips_dir = out / 'clips'
    clips_dir.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    files = sorted(p for p in Path(args.audio_dir).rglob('*') if p.suffix.lower() in AUDIO_EXT)
    if args.limit:
        files = files[:args.limit]
    written = dropped = 0
    with (out / 'clips.jsonl').open('w', encoding='utf8') as manifest:
        for fi, path in enumerate(files, 1):
            wav = read_audio(path)
            raw, _ = model.transcribe(wav, language=args.language, vad_filter=True,
                                      condition_on_previous_text=False, beam_size=args.beam)
            segments = []
            for s in raw:
                info = {'avg_logprob': s.avg_logprob, 'no_speech_prob': s.no_speech_prob,
                        'compression_ratio': s.compression_ratio}
                text = s.text.strip()
                if text and keep_segment(info, args.min_logprob, args.max_no_speech, args.max_compress):
                    segments.append((s.start, s.end, text))
                else:
                    dropped += 1
            for k, (a, b, text) in enumerate(merge_segments(segments, args.max_sec)):
                if b - a < args.min_sec:
                    dropped += 1
                    continue
                uid = f'{args.prefix}{path.stem}_{k:04d}'
                clip = clips_dir / f'{uid}.wav'
                sf.write(clip, wav[int(a * 16000):int(b * 16000)], 16000, subtype='PCM_16')
                manifest.write(json.dumps({'id': uid, 'audio': str(clip.resolve()), 'text': text,
                                           'language': args.language}, ensure_ascii=False) + '\n')
                written += 1
            print(f'[{fi}/{len(files)}] {path.name}: {written} clips so far, {dropped} segments dropped', flush=True)
    print(f'wrote {written} clips -> {out / "clips.jsonl"}')


def load_jsonl(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def usable_by_trainer(text, tokenize, en_vocab, ja_vocab):
    try:
        tokenize(text, en_vocab, ja_vocab)
        return None
    except ValueError as exc:
        return str(exc)[:40]


def filter_clips(rows, hyps, max_cer, tokenize=None, en_vocab=None, ja_vocab=None):
    """Return (kept_rows, reason_counts)."""
    kept, reasons = [], {}

    def reject(why):
        reasons[why] = reasons.get(why, 0) + 1

    for r in rows:
        other = hyps.get(r['id'])
        if other is None:
            reject('no second hypothesis')
            continue
        c = cer(r['text'], other)
        if c > max_cer:
            reject('teachers disagree')
            continue
        if tokenize:
            why = usable_by_trainer(r['text'], tokenize, en_vocab, ja_vocab)
            if why:
                reject('trainer: ' + ' '.join(why.replace(':', '').split()[:2]))
                continue
        kept.append({**r, 'teacher_cer': round(c, 4)})
    return kept, reasons


def filter_cmd(args):
    rows = load_jsonl(args.manifest)
    hyps = {h['id']: h['text'] for h in load_jsonl(args.hyp)}
    tokenize = en_vocab = ja_vocab = None
    if args.en_vocab and args.ja_vocab:
        sys.path.insert(0, str(Path(__file__).parent))
        from eg2_ja_adapter import tokenize_joint as tokenize
        en_vocab = Path(args.en_vocab).read_text(encoding='utf8').splitlines()
        ja_vocab = Path(args.ja_vocab).read_text(encoding='utf8').splitlines()
    elif args.en_vocab or args.ja_vocab:
        raise ValueError('give both --en-vocab and --ja-vocab (or neither)')
    kept, reasons = filter_clips(rows, hyps, args.max_cer, tokenize, en_vocab, ja_vocab)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf8') as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'kept {len(kept)}/{len(rows)} clips -> {out}; rejected: {reasons}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('label', help='teacher #1: transcribe + cut into <=30 s clips')
    a.add_argument('--audio-dir', type=Path, required=True)
    a.add_argument('--out-dir', type=Path, required=True)
    a.add_argument('--language', default='ja')
    a.add_argument('--prefix', default='pl_')
    a.add_argument('--model', default='large-v3')
    a.add_argument('--device', default='auto')
    a.add_argument('--compute-type', default='float16')
    a.add_argument('--beam', type=int, default=5)
    a.add_argument('--max-sec', type=float, default=25.0, help='clip length cap (encoder window is 30 s)')
    a.add_argument('--min-sec', type=float, default=1.0)
    a.add_argument('--min-logprob', type=float, default=-0.6)
    a.add_argument('--max-no-speech', type=float, default=0.3)
    a.add_argument('--max-compress', type=float, default=2.4)
    a.add_argument('--limit', type=int, default=0, help='first N audio files only (0=all)')
    b = sub.add_parser('filter', help='keep clips on which a second ASR agrees')
    b.add_argument('--manifest', type=Path, required=True)
    b.add_argument('--hyp', type=Path, required=True)
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--max-cer', type=float, default=0.10)
    b.add_argument('--en-vocab', type=Path, help='reader_vocab.txt of en.eg2m')
    b.add_argument('--ja-vocab', type=Path, help='Japanese-only reader_vocab.txt')
    args = p.parse_args()
    if args.command == 'label':
        label(args)
    else:
        if not 0 <= args.max_cer <= 1:
            p.error('--max-cer must be in [0,1]')
        filter_cmd(args)


if __name__ == '__main__':
    main()
