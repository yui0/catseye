#!/usr/bin/env python3
"""PyTorch reference for the C Kokoro engine: runs KModel on a phoneme string and dumps intermediates.

  kokoro_ref.py --config models/kokoro_config.json --pth models/kokoro-v1_0.pth --voice models/af_heart.pt \
                --phonemes "hɛlˈO wˈɜɹld" --out /tmp/kref [--speed 1]

Randomness (harmonic phase offsets, noise) is zeroed so the C engine's deterministic mode can match it.
Needs: pip install --no-deps kokoro loguru; torch, transformers, scipy, numpy.
"""
import argparse, importlib.util, json, os, sys, types
import numpy as np
import torch

def load_kokoro():
    spec = importlib.util.find_spec("kokoro")
    pkg = types.ModuleType("kokoro"); pkg.__path__ = list(spec.submodule_search_locations)
    sys.modules["kokoro"] = pkg   # avoid kokoro/__init__ (pulls misaki)
    from kokoro.model import KModel
    return KModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--pth", required=True); ap.add_argument("--voice", required=True)
    ap.add_argument("--phonemes", required=True); ap.add_argument("--out", required=True); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--noise", action="store_true", help="keep the stochastic parts")
    a = ap.parse_args()
    KModel = load_kokoro()
    m = KModel(config=a.config, model=a.pth).eval()
    if not a.noise:
        torch.rand = lambda *s, **k: torch.zeros(*s, **{kk: v for kk, v in k.items() if kk == "device"})
        torch.randn_like = lambda t, **k: torch.zeros_like(t)
    pack = torch.load(a.voice, map_location="cpu", weights_only=True)
    ids = [i for i in (m.vocab.get(p) for p in a.phonemes) if i is not None]
    ref_s = pack[len(ids) - 1]  # [1,256]
    os.makedirs(a.out, exist_ok=True)
    def save(name, t):
        t = t.detach().float().cpu().numpy(); t.tofile(f"{a.out}/{name}.bin")
        meta[name] = list(t.shape)
    meta = {"ids": [0] + ids + [0]}
    with torch.no_grad():
        input_ids = torch.LongTensor([[0, *ids, 0]])
        input_lengths = torch.LongTensor([input_ids.shape[-1]])
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(1, -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        bert_dur = m.bert(input_ids, attention_mask=(~text_mask).int()); save("bert", bert_dur)
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2); save("d_en", d_en)
        s = ref_s[:, 128:]
        d = m.predictor.text_encoder(d_en, s, input_lengths, text_mask); save("d", d)
        x, _ = m.predictor.lstm(d); save("dur_lstm", x)
        duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(axis=-1) / a.speed; save("duration", duration)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze(); meta["pred_dur"] = pred_dur.tolist()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        aln = torch.zeros((input_ids.shape[1], indices.shape[0])); aln[indices, torch.arange(indices.shape[0])] = 1
        aln = aln.unsqueeze(0)
        en = d.transpose(-1, -2) @ aln; save("en", en)
        F0, N = m.predictor.F0Ntrain(en, s); save("F0", F0); save("N", N)
        t_en = m.text_encoder(input_ids, input_lengths, text_mask); save("t_en", t_en)
        asr = t_en @ aln; save("asr", asr)
        audio = m.decoder(asr, F0, N, ref_s[:, :128]).squeeze(); save("audio", audio)
    json.dump(meta, open(f"{a.out}/meta.json", "w"))
    print("ids", len(meta["ids"]), "frames", int(pred_dur.sum()), "samples", audio.numel())

if __name__ == "__main__":
    main()
