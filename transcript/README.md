# Cat's eye — Transcription

[Project overview](../README.md) · [日本語](README.ja.md)

### Small C inference for speech recognition, ONNX models and LLMs. No Python at runtime.

[![License: MPL 2.0](https://img.shields.io/badge/license-MPL--2.0-5b5bd6.svg)](#license)
[![Language](https://img.shields.io/badge/language-C11-20232a.svg)](#repository-map)
[![Acceleration](https://img.shields.io/badge/accel-AVX2%20%7C%20AVX%20%7C%20OpenMP%20%7C%20GPU-20232a.svg)](#speed)

[日本語](README.ja.md)

**Cat's eye turns audio files into text with plain C, offline.** Its main feature is a native
implementation of [EmbeddingGemma 2 Transcribe](https://eg2.haku.sh/): the audio tower, the reader,
word search and decoding all run in one small executable, with no PyTorch, no ONNX Runtime and no
server. Python is only needed once, to convert the model files.

The same code base also holds a small ONNX runtime with an RNN-T speech recognizer, and a
LLaMA-style text generator that reads GGUF files.

> [!NOTE]
> **Status.** English transcription is tested end to end on a real model (a 225 s clip comes out
> as correct English text). Japanese needs a Reader you train yourself with the notebook in
> [`colab/`](colab/); that path is implemented but its accuracy has not been measured. Word error
> rates have not been measured either; the original authors report about 5% for English.

## Quick start

```bash
make                       # builds ./cats  (OpenMP; GPU support if EGL/GLESv2 headers exist)
```

**1. Get the models** (once; this step needs Python with `huggingface_hub`, `torch` and `numpy`):

```bash
python3 tools/prepare_eg2_native.py --download --preset official-v2 --out models
```

This downloads the published weights from Hugging Face and writes two files:
`models/encoder.eg2m` (shared audio encoder) and `models/en.eg2m` (English reader and vocabulary).
Already have the converted ZIP? `make bundle-english MODEL_ZIP=embeddinggemma2-native-official-v2.zip`.

**2. Transcribe**

```bash
./cats -a eg2-native-asr -w speech.wav            # English, models in ./models
./cats -a eg2-native-asr -L ja -w speech_ja.wav   # Japanese (needs models/ja.eg2m)
./cats -a eg2-native-asr -m /path/to/models -w speech.wav
./cats -a eg2-native-asr -L auto -w mixed.wav  # bilingual English/Japanese: requires models/mix.eg2m
```

Any WAV works: 8/16/24/32-bit PCM or float, mono or multi-channel, any sample rate, any length.
The text is printed to standard output.

**3. Optional: halve the model size**

```bash
make fp16-models           # writes models_fp16/ (encoder 1.22 GB -> 0.62 GB)
./cats -a eg2-native-asr -m models_fp16 -w speech.wav
```

The large matrices are stored as float16 and widened to float32 on load. On the English model the
weights round-trip with a relative error of 2e-7 and the transcript is unchanged.

## Options

| Option | Meaning |
|---|---|
| `-a eg2-native-asr` | Full pipeline: WAV → text. |
| `-m DIR` | Model directory (default `models`). |
| `-w FILE` | Input WAV. |
| `-L en\|ja\|auto` | Reader language; `auto` uses a trained bilingual `mix.eg2m` (default `en`). |
| `-j N` | Threads. Default: one per physical core (or `OMP_NUM_THREADS`). |
| `-G auto\|off\|N` | GPU GEMM. `auto` (default) uses a hardware GPU when there is one; `N` picks an EGL device. |
| `-o FILE` | Also write the per-frame scores (for debugging). |

Environment: `CATS_SIMD=scalar\|avx\|avx2\|avx512\|auto`, `OMP_NUM_THREADS`, `CATS_GPU=off\|auto\|N`.
Run `./cats -a cpu-info` to see what the CPU offers. `./cats help` lists every mode.

## Speed

Measured on 225 s of 48 kHz speech, Xeon E5-2697 v2 (12 cores, AVX but no AVX2/FMA), Radeon HD 7870 XT:

| Build | Time | Real-time factor |
|---|---|---|
| First version (one thread, one token at a time) | 38 min 47 s | 10× slower than real time |
| Now, CPU only | about 33 s | 0.15 |
| Now, with GPU GEMM | about 30 s | 0.13 |

Peak memory is about 1.6 GB. The Mac app that inspired this project reports 15 minutes of audio in
about 3 seconds on Apple Silicon; this program does not reach that without a modern GPU path.

What made the difference:

- **Batched GEMM.** Every Conformer stage handles all 750 tokens of a 30 s window at once, in cache-blocked
  tiles (AVX2/FMA, plain AVX or scalar, chosen at run time). A layer's 100 MB of weights is read once
  instead of once per token.
- **Threads.** OpenMP over tokens, output blocks and the vocabulary search. Hyper-threads are not used by default;
  they share the floating-point units and only add waiting.
- **Half-size models** (above) halve the bytes read.
- **GPU (optional).** `cats_gpu.h` runs the same GEMM as an OpenGL ES 3.1 compute shader on a headless
  EGL device (no window, no GLFW). Weights stay in GPU memory while a layer is used. Software rasterizers are
  skipped, and any GPU error falls back to the CPU kernels, so the transcript never depends on the GPU.
  The shader reaches roughly 500 GFLOPS here against about 210 GFLOPS for the CPU kernel; the rest of the
  pipeline limits the end-to-end gain. Splitting rows between CPU and GPU was slower (`CATS_GPU_SHARE` tunes it).

Build switches: `make OPENMP=0` (single thread), `make GPU=0` (no GPU code), `make GPU=1` (require it).
Results match across all of these to float rounding: same words, scores within about 0.01.

## What is in the box

| File | Content |
|---|---|
| `cats.c` | The `cats` command line program. |
| `cats_speech.h` | WAV reading and resampling, log-mel features, voice-activity detection. |
| `cats_rnnt.h` | EmbeddingGemma 2 audio tower, reader, word search, CTC; RNN-T beam search. |
| `cats_onnx.h` | ONNX parser, CPU SIMD kernels and a small graph runtime. |
| `../cats_gpu.h` | Optional GPU GEMM (EGL + OpenGL ES 3.1 compute). |
| `cats_hayamimi.h` | Optional bridge to Sherpa-ONNX (`-a asr`, build with `make hayamimi`). |
| `../catseye.h`, `../catseye_llm.h` | The original Cat's eye neural-network core and LLM runtime. |

Each header can be included on its own.

## Other modes

```bash
# Text generation (GGUF, or a llama2.c checkpoint with -c model.bin -z tokenizer.bin)
./cats -m model.gguf -n 256 -p "Once upon a time"
./cats -a chat -m model.gguf

# ONNX inspection and small-model execution
./cats -a onnx-info  -m model.onnx
./cats -a onnx-check -m model.onnx          # exits non-zero if an operator is unsupported
./cats -a onnx-run   -m small.onnx -p 1,2,3

# RNN-T speech recognition with encoder / decoder / joiner ONNX files (experimental)
./cats -a asr-native -m encoder.onnx -D decoder.onnx -J joiner.onnx -z tokens.txt -w speech.wav

# 80-bin log-mel features
./cats -a fbank -w speech.wav -o features.cfb
```

The step-by-step EmbeddingGemma 2 modes (`eg2-features`, `eg2-subsample`, `eg2-layer`,
`eg2-conformer`, `eg2-project`, `eg2-reader`, `eg2-decode`, `eg2-quantize`, `eg2-asr`) run one stage at a
time and are meant for debugging and for checking a new model against reference output.

## Models

| Item | Details |
|---|---|
| `encoder.eg2m` | Shared: Conv2D subsampler, 12 Conformer layers, projection, mel filters. 1.22 GB (0.62 GB as float16). |
| `en.eg2m` | English reader (19.5M parameters), 200,000-word INT8 table, vocabulary. 234 MB. |
| `ja.eg2m` | Japanese reader trained in Colab. Only this file is language specific; keep one encoder. |

`.eg2m` files are plain containers (`EG2PKG1`, no compression) that cats reads in place, one layer at a
time. Handy commands:

```bash
python3 tools/eg2_bundle.py --verify models/encoder.eg2m      # list the contents
python3 tools/eg2_bundle.py --from-zip old.zip --out models   # repack an older ZIP
python3 tools/prepare_eg2_native.py --list-models              # other sources and presets
```

The older flags `-R reader.eg2r -Q table.eg2q -z vocab.txt` still work with unpacked files.

**Japanese.** Open `colab/EmbeddingGemma2_Japanese_Reader_Colab.ipynb` in Google Colab. It freezes the
encoder, rebuilds a Japanese word list and trains a reader, then writes `ja.eg2m`. To build the English
packages in Colab instead, use `colab/EmbeddingGemma2_Native_Colab.ipynb`.

## Repository map

```text
cats.c                     command line program
cats_*.h  catseye*.h       the library headers (see above)
Makefile                   build, tests, model helpers (make help)
run_tests.sh               all C and Python tests
tools/                     Python: model conversion, bundling, Colab generators, Japanese training
colab/                     generated notebooks (rebuild: python3 tools/make_eg2_colab.py)
test/                      regression tests
fixtures/                  small synthetic model files used by the tests
bench/                     micro-benchmarks (GEMM, SIMD, INT8)
```

## Tests

```bash
make test-smoke            # 4 focused tests, seconds
make test                  # 44 C tests and 7 Python tests, about 4 minutes
./cats -a cpu-info
cc -O3 -fopenmp -I. bench/bench_eg2.c -lm -o bench_eg2 && ./bench_eg2   # GEMM speed
```

The tests use synthetic models, so they check that every stage computes what the reference math says,
not recognition accuracy. The GPU test is skipped when no GPU is present.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `invalid weights, audio or inference failure` | Check that `models/encoder.eg2m` and `models/<lang>.eg2m` exist and pass `eg2_bundle.py --verify`; check the file really is a WAV. |
| `missing models/encoder.eg2m` | Pass `-m DIR`, or run from the folder that contains `models/`. |
| Very slow | Run `./cats -a cpu-info`. `scalar` means no SIMD is used; check `CATS_SIMD`. A busy machine also slows the thread barriers. |
| `cats-gpu: ... falling back to the CPU` | The driver rejected the shader. Results stay correct; use `-G off` to silence it. |
| Build fails on `GLES3/gl32.h` | Install the EGL/GLESv2 development files, or `make GPU=0`. |
| Garbled Japanese output | Use `-L ja` together with a `ja.eg2m` trained for your data. |

## License

[MPL-2.0](https://mozilla.org/MPL/2.0/). Model weights keep their own licenses (the EmbeddingGemma 2
weights are under Google's Gemma terms).

© 2016–2026 Yuichiro Nakada

## English + Japanese in a single recording

The new `-L auto` mode uses a **jointly trained single CTC Reader and vocabulary**
(`models/mix.eg2m`) with the original shared `encoder.eg2m`. It can output
Japanese characters and English words in the same utterance. English word
boundaries are indicated by U+2581 (`▁`) vocabulary prefixes.

Open `colab/EmbeddingGemma2_English_Japanese_Mixed_Colab.ipynb` to train
and package the joint Reader. The default Colab demonstration mixes separate
Japanese and English FLEURS sentences; real intra-sentence code switching
requires training/validation audio containing actual Japanese/English switches.
No bilingual trained weights are distributed in this source archive, and real
mixed-language accuracy remains untested. `-L auto` deliberately errors when
`mix.eg2m` is missing; placing independent `en.eg2m` and `ja.eg2m` files
beside one another is **not** equivalent to joint bilingual training.
