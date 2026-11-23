#!/usr/bin/env python3
"""Checks inkling_quantize's expert decoding against an independent PyTorch implementation of the Hugging Face
Inkling2LoopDequantizeExperts formulas (NVFP4: E2M1 nibbles, even element low, E4M3 block scales of 16, factor
amax / outer_scale_denominator; MXFP8: E4M3 values, E8M0 scale per 32) plus the gate/up de-interleave.
usage: check_quant.py HF_DIR QUANTIZED_F16_FILE   (quantize the tiny model with -b f16 first)"""
import json, struct, sys
import numpy as np, torch
from safetensors import safe_open

hf, qf = sys.argv[1], sys.argv[2]
f = open(qf, "rb"); n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
raw = np.memmap(qf, dtype=np.uint8, mode="r")
def qt(name):
    v = hdr[name]; b, e = v["data_offsets"]; assert v["dtype"] == "F16", v["dtype"]
    return np.frombuffer(raw[base + b: base + e], np.float16).astype(np.float32).reshape(v["shape"])
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
def dequant(w, s, amax, den):
    if amax is not None:
        lo, hi = E2M1[(w & 15).long()], E2M1[(w >> 4).long()]
        v = torch.stack([lo, hi], -1).flatten(-2)                                # [E,rows,cols]
        v = v.view(*s.shape, 16) * s.float()[..., None]
        return (v * (amax.float() / den.float())[:, None, None, None]).flatten(-2)
    return (w.float().view(*s.shape, 32) * torch.exp2(s.float() - 127)[..., None]).flatten(-2)
worst = 0.0
cfg = json.load(open(hf + "/config.json"))["text_config"]; I = cfg["intermediate_size"]; NL = cfg["num_hidden_layers"]
with safe_open(hf + "/model-00001-of-00001.safetensors", "pt") as st:
    for l in range(NL):
        for w in ("w13", "w2"):
            b = f"model.llm.layers.{l}.mlp.experts.{w}_weight"
            names = st.keys()
            amax = st.get_tensor(b + "_amax") if b + "_amax" in names else None
            den = st.get_tensor(b + "_outer_scale_denominator") if amax is not None else None
            d = dequant(st.get_tensor(b), st.get_tensor(b + "_scale"), amax, den)          # [E, rows, cols]
            if w == "w13": d = torch.cat([d[:, 0::2], d[:, 1::2]], 1)                      # gate = even rows, up = odd rows
            got = qt(f"blk.{l}.{w}").reshape(d.shape)
            err = np.abs(got - d.numpy()).max() / (np.abs(d.numpy()).max() + 1e-9); worst = max(worst, err)
            print(f"layer {l} {w:3s} {'NVFP4' if amax is not None else 'MXFP8'} max rel err {err:.2e}")
print("worst", worst); sys.exit(0 if worst < 2e-3 else 1)
