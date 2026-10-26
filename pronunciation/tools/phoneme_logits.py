#!/usr/bin/env python3
"""Run a CTC phoneme recognizer on a WAV and write what pcheck needs.

  phoneme_logits.py --wav in.wav --out in.cpost --vocab vocab.txt [--model ID]

Default model: facebook/wav2vec2-xlsr-53-espeak-cv-ft (IPA phonemes, multilingual,
covers English and Portuguese). Needs: pip install transformers torch numpy
Writes vocab.txt (one symbol per line, line number = id, id 0 = blank) and
.cpost = "CPST", u32 T, u32 V, T*V float32 logits.
"""
import argparse, struct, subprocess, sys, wave
import numpy as np

def read_wav(path, rate=16000):
    try:
        with wave.open(path, "rb") as w:
            n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(n)
        if sw != 2:
            raise ValueError("not 16-bit")
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        x = x.reshape(-1, ch).mean(1)
    except Exception:  # fall back to ffmpeg for other formats
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"],
                             check=True, capture_output=True).stdout
        return np.frombuffer(raw, dtype="<f4").copy()
    if sr != rate:
        t = np.arange(int(len(x) * rate / sr)) * (sr / rate)
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True); ap.add_argument("--out", required=True); ap.add_argument("--vocab", required=True)
    ap.add_argument("--model", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
    fe = Wav2Vec2FeatureExtractor.from_pretrained(a.model)
    model = AutoModelForCTC.from_pretrained(a.model).eval()
    tok = Wav2Vec2CTCTokenizer.from_pretrained(a.model)
    inv = {i: s for s, i in tok.get_vocab().items()}
    x = None
    V = None
    x = read_wav(a.wav)
    inp = fe(x, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inp.input_values).logits[0].numpy().astype("<f4")
    V = logits.shape[1]  # the tokenizer may carry extra added tokens beyond the model's output size
    with open(a.vocab, "w", encoding="utf-8") as f:
        for i in range(V):
            f.write(inv[i] + "\n")
    with open(a.out, "wb") as f:
        f.write(b"CPST" + struct.pack("<II", logits.shape[0], logits.shape[1]))
        f.write(logits.tobytes())

if __name__ == "__main__":
    main()
