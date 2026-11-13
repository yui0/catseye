#!/usr/bin/env python3
"""Compares ./plamo with the official Hugging Face implementation (tokenizer ids and greedy translations).

    python3 -I tests/check_reference.py HF_DIR [PLAMO_MODEL_DIR]

HF_DIR holds the original checkpoint (needs torch, transformers, numba). PLAMO_MODEL_DIR defaults to HF_DIR;
pass a quantized directory to see how much the translation drifts from the BF16 reference.
"""
import subprocess, sys, os

hf = sys.argv[1]
mdir = sys.argv[2] if len(sys.argv) > 2 else hf
exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plamo")

CASES = [
    ("ja", "en", None, "今日はとても良い天気ですね。散歩に行きましょう。"),
    ("en", "ja", None, "The quick brown fox jumps over the lazy dog."),
    ("en", "ja", "polite", "Please send me the report by tomorrow morning."),
    ("ja", "zh-hans", None, "来週の会議は10時から始まります。\n資料を事前に読んでおいてください。"),
    ("ja", "en", None, "吾輩は猫である。名前はまだ無い。    どこで生れたかとんと見当がつかぬ。ｗｗｗｗｗ"),
    (None, "fr", None, "Machine translation has improved a lot in recent years."),
]

def run(args):
    return subprocess.run([exe] + args, capture_output=True, text=True, check=True).stdout

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(hf, trust_remote_code=True, dtype=torch.bfloat16)
model.eval()

bad = 0
for src, dst, form, text in CASES:
    kw = dict(target_lang=dst)
    if src: kw["source_lang"] = src
    if form: kw["form"] = form
    enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, return_dict=True, return_tensors="pt", **kw)
    ref_ids = enc["input_ids"][0].tolist()
    args = ["-m", mdir, "-t", dst, "-p", text]
    if src: args += ["-s", src]
    if form: args += ["--" + form]
    got_ids = list(map(int, run(args + ["--dump-ids"]).split()))
    ok_ids = got_ids == ref_ids
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=256, eos_token_id=[1, 16], pad_token_id=3)
    ref = tok.decode(out[0, len(ref_ids):], skip_special_tokens=True).strip()
    got = run(args).strip()
    print(f"[{src or 'auto'}->{dst}] tokenizer {'OK' if ok_ids else 'MISMATCH'}; translation {'same' if ref == got else 'DIFFERENT'}")
    if not ok_ids:
        print("  ref:", ref_ids, "\n  got:", got_ids)
    if ref != got:
        print("  ref:", ref, "\n  got:", got)
    bad += (not ok_ids) + (ref != got)
sys.exit(1 if bad else 0)
