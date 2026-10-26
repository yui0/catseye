#!/usr/bin/env python3
"""Kanji -> hiragana reading for the Japanese TTS front end.

  ja_reading.py FILE      (UTF-8 text file) -> hiragana words separated by spaces on stdout

Uses pykakasi (pip install pykakasi). Particles は / へ / を are respelled わ / え / お
so the kana-to-IPA rules in lang_ja.h pronounce them correctly.
"""
import sys

def main():
    import pykakasi
    text = open(sys.argv[1], encoding="utf-8").read()
    kks = pykakasi.kakasi()
    out = []
    prev_kanji = False
    for item in kks.convert(text):
        orig, hira = item["orig"], item["hira"]
        if orig in ("は", "へ", "を") and out:
            hira = {"は": "わ", "へ": "え", "を": "お"}[orig]
        out.append(hira)
    print(" ".join(out))

if __name__ == "__main__":
    main()
