# Inkling-2-Mini (Cat's eye)

Text-only CPU inference of Thinking Machines' [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(26.4B parameters, 3.4B active per token, 5.4B effective with the three loop passes) in dependency-free C, plus a
converter that shrinks the 20 GB checkpoint to **one file**.

Only what is specific to the model is in `inkling/`. The reusable parts went into `../catseye.h`:
FP8 / E8M0 / FP4 and NVFP4 / MXFP8 decoding, `cats_softcap`, `cats_l2_clip`, a byte-level BPE tokenizer
(`cats_bpe`, including the GPT-style pre-tokenizer regex and a `tokenizer.json` converter), a small JSON helper,
next to the existing safetensors reader, Q4_0 / Q8_0 GEMM and RMSNorm.

## What the model is

Decoder with 24 layers and sparse MoE FFNs (128 routed experts, 8 per token, sigmoid router, softcapped SwiGLU).
Layers 6-17 are run three times with shared weights; the second and third pass receive *messages* from the previous
pass (learned queries attend over the previous pass's block states, then a low-rank gate mixes them into the q, k, v
and MLP inputs). Residual connections are *block attention residuals* (learned queries over the block states), not plain
sums. Attention is hybrid (20 sliding-window layers, 4 global ones) with learned relative-position biases, a per-token
attention sink, L2-clipped q/k, short causal convolutions on k/v (dynamic taps in the global layers) and on the
layer outputs, and log-scaled queries in the global layers. Everything is implemented from the Hugging Face
`inkling2_loop` modeling code.

## Usage

```sh
make                                              # inkling and inkling_quantize
hf download thinkingmachines/Inkling-2-Mini --local-dir Inkling-2-Mini
./inkling_quantize -i Inkling-2-Mini -o inkling-q4.safetensors      # ~15.5 GB, one file (checkpoint: 20 GB)

./inkling -m inkling-q4.safetensors -p "Explain why the sky is blue in one sentence."
./inkling -m inkling-q4.safetensors -i                 # interactive chat, the conversation stays in the KV cache
./inkling -m inkling-q4.safetensors -f prompt.txt --effort low -n 256 -v
```

`./inkling` prints its options. Notable ones: `--effort` (thinking effort of the chat template: none, minimal, low,
medium, high, xhigh, max or a number up to 0.99), `--think` (also show the thinking), `--raw` (show control tokens),
`--passes N` (use fewer loop passes: faster, lower quality), `-c` (context length, default 4096; the global layers
cache every token, the sliding ones only 1024).

## Running on an 8 GB Mac (Apple M2)

The default Q4 file (15.5 GB) does not fit in 8 GB of RAM, so there is a smaller recipe plus streaming from the SSD:

```sh
# on any machine with ~32 GB RAM (the quantizer keeps the result in memory), then copy the file to the Mac
./inkling_quantize -i Inkling-2-Mini -o inkling-q3.safetensors -b q3 -a q4 -e q4        # ~11.9 GB

# on the Mac
brew install libomp                    # OpenMP for Apple clang; without it the build is single-threaded
make && make test                      # kernel_test must report "all dot kernels agree"; the tiny-model test must pass
./inkling -m inkling-q3.safetensors -c 2048 --batch 32 --topk 6 -v -p "..."      # recommended: Q3 + top-6
```

* `-b q3` is a 3.5-bit format (Q3_0: 32 weights share an fp16 scale); `-a q4 -e q4` shrinks attention and the vocabulary
  tables too (`-a q8 -e q8` is better but ~0.8 GB bigger). The file is memory-mapped: only ~1.5 GB (attention, the output
  table, caches) has to stay resident, the 24.4B expert weights are streamed from the SSD as the router picks them.
  When the experts do not fit in memory (the process's RAM or cgroup limit) an LRU **expert cache** is turned on
  automatically (`--cache MB` to set it, `--cache 0` for plain mmap): whole experts are read with large parallel `pread()`s,
  which is ~3x faster than demand paging (page faults read only 4-16 KB each and saturate a drive's random IOPS).
  Reading less is the main speed knob: `--topk 6` (fewer experts) is the recommended one.
  **Avoid `--passes 2`/`1`**: measured against the default, they change the output a lot (KL 0.61 / 1.07, top-1
  agreement 73% / 61%).
* Measured (24-core x86 Linux, USB SSD that does ~250 MB/s on 2 MB random reads, process limited to 5 GB, cold file,
  Q3 with the automatic 2.1 GB cache): 0.57 tokens/s with 8 experts, **0.85 tokens/s with `--topk 6`** (26% less read).
  With everything in the page cache the same file decodes at ~3.5 tokens/s. Quality against the Q4 file (610 tokens of
  English / Japanese / code): Q3 perplexity 3.105 vs 3.050, KL 0.060, top-1 agreement 91%; Q3 + `--topk 6` 3.185, KL 0.080,
  90%. An internal Apple SSD should be several times faster than that USB drive but has not been measured. Prompts are
  processed `--batch` tokens at a time (every expert is read once per chunk), so they are much faster per token than
  generation. `--prefetch N` (read the N likeliest experts of the next layer in the background; the router of layer l+1 applied
  to layer l's input names 77% of the experts in its top 8) made things slower on that drive, as the guesses add reads to an
  already saturated disk; it is off by default.
* The code has NEON kernels (SDOT) for Q3_0 / Q4_0 / Q8_0 and the float dot, and only the performance cores are used. They
  were compile-checked for aarch64 but written without access to Apple hardware: `make test` is how to confirm them
  (`tests/kernel_test` compares every SIMD kernel with its scalar reference).

## The single file and the quantization

`inkling_quantize` writes a safetensors container (so `cats_st` maps it directly) holding GGML Q4_0 / Q8_0 blocks and,
as U8 tensors, the tokenizer and `config.json`. The 128 experts of each layer are stored as one 2-D matrix, expert *e*
being the row range `[e * rows, (e + 1) * rows)`.

| Part | Default (`-b` / `-a` / `-e`) | Notes |
|---|---|---|
| routed experts (24.4B weights) | Q4_0 (`-b q4`); `-b q3` = Q3_0, 3.5 bit | decoded from NVFP4 (layers 0-19) / MXFP8 (layers 20-23), re-quantized with a per-block scale search (`--fast` = plain Q4_0) |
| attention matrices | Q8_0 (`-a q8`) | `q4` or `f16` possible |
| embedding / output table | Q8_0 (`-e q8`) | unpadded vocabulary (200,058 rows) |
| norms, routers, relative-position tables, convolutions, loop controllers | F32 | small |

The quantizer keeps the whole result in memory (~15.5 GB) before writing; the input is memory-mapped.
Q4_0 has the same 4.5 bits per weight as NVFP4, so it is a re-quantization, not a size win over the experts; Q3_0 (3.5 bits)
is the smaller option. The vision tower and the multi-token-prediction head are not converted.

## Verification

* `make test` needs only python3 with torch, numpy and safetensors: it builds a tiny random checkpoint with the same tensor
  layout (NVFP4 and MXFP8 experts, sliding and global layers, loop passes), checks the converter's decoding and the gate/up
  de-interleave against independent PyTorch formulas, and compares the C logits with an independent PyTorch implementation
  of the whole model (exact to ~4e-4 on 26 tokens with a window of 8, so the caches, ring buffers and convolution states
  are covered). Quantized (Q8 / Q4) files must run.
* `make test-tokenizer MODEL_DIR=Inkling-2-Mini` compares the tokenizer with Hugging Face `tokenizers` on 25 cases
  (CJK, emoji, whitespace, contractions, special tokens): identical.
* Generation quality of the real 26B model has to be judged by running it.

## Chat format

The chat template of the checkpoint has no generation prompt: the model starts its reply itself with
`<|message_model|>`, optionally `<|content_thinking|>...<|end_message|>`, then `<|content_text|>...<|end_message|>`,
and ends with `<|content_model_end_sampling|>`. `inkling` builds `<|message_system|>` (optional system text, then the
thinking-effort line), `<|message_user|>` and feeds them without a BOS token.

## License

The weights are published by Thinking Machines under their own license; check the model card.
