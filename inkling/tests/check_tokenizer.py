#!/usr/bin/env python3
# usage: check_tokenizer.py MODEL_DIR   (needs `tokenizers`; builds tok_dump first)
import subprocess, sys, os, tempfile
from tokenizers import Tokenizer
d = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
subprocess.check_call(["cc", "-O2", "-std=gnu11", "-o", here + "/tok_dump", here + "/tok_dump.c", "-lm"])
cases = [
 "Hello, world!", "The quick brown fox jumps over the lazy dog.", "I'm sure you've said it'll work; they'd say we're OK. DON'T SHOUT'S",
 "今日はとても良い天気ですね。", "日本語とEnglishが混ざった文章です。123456789", "  leading and trailing   spaces  ",
 "line1\\nline2\\n\\n\\nline3\\t\\ttabbed", "def f(x):\\n    return x**2 + 3  # comment\\n", "https://example.com/a/b?c=d&e=f#frag",
 "camelCaseWord PascalCase ALLCAPS snake_case_word", "Ünïcödé àccénts and ñ ß ø", "emoji 😀🎉 and 𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "1234567890123 3.14159 -42 +7",
 "<|message_user|><|content_text|>Hi there<|end_message|>", "a<|end_message|>b<|unused|>c", "$$$ !!! ??? ... --- /// \\n/ \\r\\n", "x" * 50,
 "ab" * 300, "   ", "\\n\\n", " \\n ", "word  \\n  word", "Hello\\u00a0World\\u3000です", "ａｂｃ　ＡＢＣ", "\u0915\u094d\u0937\u093f \u0645\u0631\u062d\u0628\u0627 \u05e9\u05dc\u05d5\u05dd",
]
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
    for c in cases: f.write(c.encode().decode("unicode_escape") if False else c.replace("\\u00a0","\u00a0").replace("\\u3000","\u3000").replace("\\r","\r") + "\n")
    path = f.name
out = subprocess.check_output([here + "/tok_dump", d + "/tokenizer.json", path]).decode().split("\n")
tk = Tokenizer.from_file(d + "/tokenizer.json")
bad = 0
for i, c in enumerate(cases):
    text = c.replace("\\n", "\n").replace("\\t", "\t").replace("\\u00a0","\u00a0").replace("\\u3000","\u3000").replace("\\r","\r")
    want = tk.encode(text, add_special_tokens=False).ids
    got = [int(x) for x in out[i].split()]
    if want != got:
        bad += 1
        print("MISMATCH", repr(c)); print(" want", want[:40]); print(" got ", got[:40])
print("cases", len(cases), "mismatches", bad)
sys.exit(1 if bad else 0)
