# pronunciation — English / Portuguese / Japanese TTS and pronunciation check

Header-only C, built on the same conventions as the rest of catseye.

| File | Purpose | Status |
|---|---|---|
| `phon.h` | UTF-8 / IPA helpers, text normalizer (digits -> words) | done |
| `latphon.h` | language registry `cp_langs[]` + G2P `cp_g2p(lang, text, out, cap)` -> IPA; `cp_g2p_hook` for the neural LatPhon model | done, LatPhon pending |
| `lang_en.h` `lang_pt.h` `lang_ja.h` | one header per language (rules + lexicon) | done |
| `klatt.h` | Klatt-style formant synthesizer (no model files); **reference only**, the voice is robotic and hard to listen to | done |
| `tts_piper.h` | Piper (VITS) voices on `../transcript/cats_onnx.h` | done (en_US lessac, pt_BR faber) |
| `tts.c` | CLI | `-e klatt` (default, reference only) / `piper` / `kokoro` / `qwen` |
| `align.h`, `pcheck.h`, `pcheck.c` | pronunciation check: CTC forced alignment, per-phoneme score (GOP), heard-vs-expected comparison | done; needs a recognizer's posteriors |
| `tools/phoneme_logits.py` | WAV -> CTC phoneme posteriors (wav2vec2-xlsr-53-espeak-cv-ft, needs `transformers`, ~1.2 GB download) | done |
| `gguf.h`, `tts_kokoro.h` | GGUF reader and Kokoro-82M (Q8_0) engine: PL-BERT, prosody predictor, decoder, iSTFTNet | done (en / pt / ja voices) |
| `tts_qwen.h` | Qwen3-TTS 12 Hz 0.6B (Q8_0): byte-level BPE, 28-layer talker, 5-layer code predictor, codec decoder | done (en / pt / ja) |
| LatPhon | see `/root/.claude/plans/fluttering-wondering-moon.md` | pending |

```bash
make && make test
# neural engines (use these for listening); models are described in the sections below
./tts -e piper  -L en -t "Hello world, this is a test." -o en.wav   # needs models/en.onnx(.json) / pt.onnx(.json)
./tts -e kokoro -L pt -t "Olá mundo, bom dia." -o pt.wav
./tts -e qwen   -L ja -t "こんにちは、ありがとうございます。" -o ja.wav
# the default engine is the Klatt formant synthesizer: REFERENCE ONLY (robotic, not meant for listening).
# It needs no model files, so it is the engine for tests and for the pronunciation-check pipeline.
./tts -L en -t "Hello world, this is a test." -o en_klatt.wav
./tts -L pt -t "cidade" --ipa      # sidˈadʒi
```

## Pronunciation check

```bash
make pcheck
./pcheck -L en -t "Hello world" -z models/vocab.txt -w learner.wav          # runs tools/phoneme_logits.py (writes vocab.txt on first use)
python3 tools/phoneme_logits.py --wav x.wav --out x.cpost --vocab models/vocab.txt
./pcheck -L pt -t "Olá mundo" -z models/vocab.txt --post x.cpost --json
```

Each expected phoneme is reported `ok`, `weak` (matches but scores < 60), `substituted` (with what was heard),
`deleted`, and extra phonemes as `inserted`; words and the whole utterance get a 0-100 score.

Known limits: the recognizer is multilingual and leans toward English; Mandarin tone tokens are masked.
The Klatt voice is recognized well in English but Portuguese Klatt speech is not recognized reliably, so
Portuguese checking has only been tested with synthetic posteriors (`test/test_pcheck.c`), not real speech.

## Piper voices

Download once into `models/` (en_US-lessac-medium as `en.onnx`, pt_BR-faber-medium as `pt.onnx`, plus the `.onnx.json` files)
from <https://huggingface.co/rhasspy/piper-voices>. Running them needed these operators added to
`../transcript/cats_onnx.h`: ConvTranspose (1-D), GatherND, ScatterND, NonZero, RandomNormalLike (fixed seed, so output is
reproducible), Ceil; INT64 open-end sentinels in initializers; `CATS_ONNX_DEBUG=1` dumps the inputs of a failing node.
Synthesis of a short sentence takes about 1.5x its duration on a multi-core CPU (Conv / ConvTranspose use OpenMP).

## Adding a language

1. `lang_xx.h`: write `static void cp_xx_word(const char *word, char *ipa, size_t cap)` (one normalized word in, IPA out;
   put `ˈ` before the stressed vowel). If words are not separated by spaces or digits/punctuation need special handling, also
   write `static int cp_xx_normalize(const char *in, char *out, size_t cap)` (see `lang_ja.h`); otherwise reuse
   `cp_normalize_default`, which lower-cases, spells digits and keeps `, . ? !` as pause tokens.
2. `phon.h`: add `CP_XX` before `CP_NLANG`. `latphon.h`: include the header and add one row to `cp_langs[]`
   (ISO code, name, word fn, normalize fn, Piper voice stem in `models/` or `NULL`).
3. `klatt.h`: add any phonemes the Klatt voice does not have yet to `ck_tab[]` (formants, duration, frication).
4. `test/test_g2p.c`: add `chk(CP_XX, "...", "...")` lines.

`tts` and `pcheck` pick the language up from `-L xx` automatically.

## Japanese

`./tts -L ja -t "こんにちは、ありがとうございます。"` reads hiragana/katakana by rule (no pitch accent). Kanji need
`pip install pykakasi`: `tools/ja_reading.py` converts them to hiragana first (particles は/へ/を are respelled only in kanji text,
because pykakasi does not segment pure hiragana). Piper has no Japanese voice, so only the Klatt engine speaks Japanese for now;
Kokoro and Qwen3-TTS will add natural Japanese. Klatt Japanese is intelligible-sounding speech from rules but a multilingual
recognizer scores it low (the same holds for Klatt Portuguese), so use it for the pipeline, not as a reference voice.

> **Klatt output (`./tts` without `-e`) is for reference only.** It is rule-based formant synthesis: intelligible at best in English,
> unpleasant to listen to in every language. Use `-e piper`, `-e kokoro` or `-e qwen` when the sound matters.

## English pronunciation lexicon (recommended for piper / kokoro)

The built-in English rules (NRL letter-to-sound) are a fallback and often put the stress in the wrong place (`hɛˈloʊ`).
The Piper and Kokoro voices were trained on espeak-ng phonemes, so feed them espeak-like phonemes from a dictionary:

```bash
python3 tools/make_lexicon_en.py            # downloads CMUdict, writes models/lexicon_en.txt (125k words, 2.8 MB)
./tts -e piper -L en -t "Hello world, this is a test." --ipa     # həlˈoʊ wˈɜːld, ðɪs ɪz ɐ tˈɛst.
```

`tts` and `pcheck` load `lexicon_en.txt` from `-m DIR` (or `models/`) when it exists, otherwise they fall back to the rules.
CMUdict ([BSD-style license](https://github.com/cmusphinx/cmudict/blob/master/LICENSE), (c) Carnegie Mellon University) supplies the
phones and the stress; the ARPAbet -> IPA step is a small decision tree per phone fitted to espeak-ng output over the whole dictionary
(`tools/en_g2p_tree.json`, held-out token accuracy 92%), and the weak forms of function words come from espeak-ng in context
(`tools/en_function_words.json`). espeak-ng is not needed to build or use the lexicon. Unknown inflected words (`walked`, `dogs`,
`quickly`) are built from their base word; `the` / `to` become `ðɪ` / `tʊ` before a vowel; punctuation is attached to the previous word
(`wˈɜːld,`) the way espeak writes it. `tools/espeak_ipa.py` (needs `pip install espeakng-loader`) prints the espeak-ng phonemes
for comparison, and `tts --from-ipa` synthesizes an IPA string directly.

Effect on `pcheck` scores (piper medium, same expected phonemes; piper's sampling noise makes single runs vary by a few points):
"Hello world" 63 -> 77, "I want to eat the apple, then walk to school." 78 -> 86, other test sentences unchanged within noise.

Another English voice, `en_US-lessac-high` from
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/high), runs with `-v en_high` after saving
it as `models/en_high.onnx` + `en_high.onnx.json`. By the `pcheck` score it is not clearly better than the medium voice (91 vs 85 and
83 vs 86 on two sentences) and it is about 6x slower in the current ONNX runtime, so medium stays the default.
`--noise X --noise-w Y` change Piper's sampling noise (0.667 / 0.8 by default).

## Kokoro-82M (Q8_0)

```bash
cd models
curl -LO https://huggingface.co/cstr/kokoro-82m-GGUF/resolve/main/kokoro-82m-q8_0.gguf && mv kokoro-82m-q8_0.gguf kokoro-q8.gguf
for v in af_heart pf_dora jf_alpha; do   # English, Brazilian Portuguese, Japanese
  curl -LO https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/$v.pt && python3 ../tools/convert_kokoro_voice.py $v.pt $v.bin
done
cd .. && ./tts -e kokoro -L pt -t "Olá mundo, bom dia." -o pt.wav      # -v NAME picks another voice, --f32 dequantizes the weights
```

`catseye.h` gained Q8_0-weight kernels (`cats_dot_q8_0w_q8_0`, `cats_q8_0w_dequantize_row`) and `catseye_llm.h` loads
GGML type 8 tensors, so Q8_0 GGUFs work for the LLM runtime too. Matrices stay Q8_0 in memory; convolutions run in float.
`tools/kokoro_ref.py` dumps PyTorch intermediates (needs `pip install --no-deps kokoro loguru` plus torch, transformers, scipy);
`test/test_kokoro` compares every stage (BERT, durations, F0/N, text encoder, spectrum of the final audio) against them.
Observed: stages agree within 0.1-5% rms (the GGUF is quantized, the reference fp32); final audio log-spectrum differs by
0.07-0.08 decades. A 1.5 s utterance takes about 2.5 s on 24 threads.
Japanese: Kokoro's own Japanese front end (misaki + pyopenjtalk, pitch-accent marks) is not reproduced; the mora IPA from
`lang_ja.h` is fed instead, which pronounces correctly but without Kokoro's accent symbols.

## Qwen3-TTS 0.6B (Q8_0)

```bash
cd models
for f in qwen-talker-0.6b-base-Q8_0 qwen-tokenizer-12hz-Q8_0; do curl -LO https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main/$f.gguf; done
cd .. && ./tts -e qwen -L ja -t "こんにちは、ありがとうございます。" -o ja.wav      # --seed N, --greedy
```

The model reads raw text (no G2P step) and picks the language from `-L` (`en` / `pt` / `ja`, or any model language name).
The base checkpoint has no named speakers; the voice is whatever the sampler lands on, so use `--seed` to repeat a voice.
Checked against qwentts.cpp (`--dump`): the 14 prompt token ids and all 16 prompt embeddings match exactly, prefill logits
differ by 1% rms, and the first generated frame (16 codes) is identical. Later greedy frames drift (Q8 activation rounding,
f32 instead of f16 KV cache). The codec decoder alone matches the reference waveform to 0.7% rms (`test/test_qwen_codec`).
Speed (Xeon E5-2697 v2, AVX only, 12 cores, with other jobs running on the machine): 1.84 s of audio in 2.9 s, about 1.6x slower than real time (was 3x; qwentts.cpp: 1.4x). The codec convolutions use a packed, register-tiled kernel (`qw_pc_run`), Q8 matrices are streamed once against all quantized rows, and the thread count defaults to one per physical core (`OMP_NUM_THREADS` overrides): OpenMP spin-waiting on hyperthreads was 30x slower when the machine was shared. `QW_PROF=1` prints the codec stage times. The talker is now close to memory bandwidth (about 1.7 GB of weights are read per 12.5 Hz frame).
Voice cloning (speaker encoder, `--ref-wav`) is not implemented.
