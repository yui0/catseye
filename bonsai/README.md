# bonsai

[Ternary-Bonsai-2.5-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF) (PrismML, Apache-2.0) on the CPU with catseye.
The `PTQ1_0` GGUF (ternary weights, 1.75 bit/weight, 6 GB) is used as it is.

```sh
make model        # downloads models/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf (6 GB)
make mmproj       # optional: the vision tower (0.6 GB), for image input
make
./bonsai models/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf -p "Explain quantum computing in simple terms."
./bonsai models/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf          # interactive chat, empty line quits
./bonsai models/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf --mmproj models/Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf \
         -p "@photo.jpg What is in this picture?"            # a message starting with @path attaches an image
make test
```

The model thinks first (`<think> ... </think>`) and then answers. `--no-think` skips the reasoning
(the card suggests `--temp 0.7 --top-p 0.8` for that mode). Defaults are the thinking-mode sampling settings
(temp 1.0, top-k 20, top-p 0.95, min-p 0). Keep `-n` large enough for the reasoning trace.
By default one thread per physical core is used (`-j N` to change; hyper-threads make it slower).

Needs ~14 GB of RAM (the weights are repacked at load, see below; the file stays mapped but is only paged in once),
plus the KV cache (`-c`, 128 KB per token), plus ~2 GB for the vision tower. Measured on a Xeon E5-2697 v2
(12 cores, AVX only, DDR3 about 46 GB/s): **about 4 tokens/s generation**, 4.5 tokens/s prompt processing.

## Layout

| | |
|---|---|
| `../catseye.h` | the generic parts: `cats_ptq1_0_*` (PTQ1_0 decode, repacked 2-bit-plane kernels `cats_ptq1_0_r2_pack` / `cats_matmul_r2_q8_0` for scalar, AVX and AVX2), `cats_fwht` / `cats_hadamard_rows`, `cats_bpe_from_gguf` + Qwen pre-tokenizer for `cats_bpe`, `cats_sample_logits` |
| `bonsai.h` | the `qwen35` graph: Gated DeltaNet layers (48), gated GQA attention layers (16), SwiGLU FFN, M-RoPE, the MTP head |
| `vision.h` | the Qwen3-VL vision tower (`bv_encode`) and feeding image embeddings (`bn_feed_embeddings`) |
| `bonsai.c` | CLI / chat loop (ChatML template), speculative decoding loop |
| `test/` | `test_ptq1_0` (kernels, Hadamard), `test_bpe MODEL.gguf` (tokenizer round trips) |

`../pronunciation/gguf.h` (mmap GGUF reader) gained the PTQ1_0 tensor size.

## Speed: what the kernel does

* The file's `PTQ1_0` packs 5 trits per byte in base 3, which needs multiplications to decode. At load the weights are repacked
  into 2-bit planes (`trit + 1`, 32 bytes per 128 weights, plane k = Q8_0 block k): decoding is a shift and a mask.
* The dot product is integer only: `maddubs(u, y)` adds the products of the 2-bit values with the int8 activations, and
  `sum(y)` (computed once per matmul, not per weight row) is subtracted. float is used only to apply one scale per 32 values.
* Measured, single thread, same data: this kernel 0.08 ns/weight; a bit-serial version with 64-bit `AND` + `popcnt` on
  8 activation bit planes 0.19 ns/weight, so plain bit operations are slower than `maddubs` for int8 activations.
* Generation is now limited by instruction throughput, not by memory (7 GB per token at 46 GB/s would allow ~6.5 tokens/s).
  Because of that, batching tokens barely shares any work: 2 tokens cost 1.7x one token.
* `BONSAI_NOREPACK=1` keeps the original kernels (slower, a bit less memory).

## Format notes

* **PTQ1_0** (ggml type 143): 128 weights = 28 bytes = `qs[24]` (5 trits/byte, base 3) + `qh[2]` (4 trits/byte) + fp16 scale.
  Layout from the PrismML llama.cpp fork (`ptq1_0-base`, upstream PR 29672); the trits of a byte are interleaved, see `cats_ptq1_0_decode_scalar`.
* **Hadamard folding** (`prism.hadamard.*`): weights were stored as `W' = W D H` (D = +-1 signs per input width, H = normalized
  Walsh-Hadamard over blocks of 1024), so every matmul input is first rotated `x -> H(D x)`. `token_embd` is the exception:
  rows are rotated on disk and are restored with `H` then `D` after the lookup.
* **Gated DeltaNet head order**: the delta rule runs in tiled order (value head `h` reads key head `h % 16`), but `ssm_out` was folded
  with grouped input channels (`prism.hadamard.gdn_v_grouped`), so its input is permuted `[hd, nk, rep] -> [hd, rep, nk]` first.
* RMSNorm weights already contain the `+1` of Qwen3-Next, and `ssm_a` is already `-exp(A_log)`.
* **M-RoPE** (`rope.dimension_sections = [11, 11, 10, 0]`, interleaved t, h, w): text tokens have the same position in all three.
  Image tokens get `t = start`, `h = start + row`, `w = start + column`, and the text after an image continues at
  `start + max(nx, ny)`, so the RoPE position and the KV-cache slot are separate (`m->rpos`, `m->pos`). Image tokens attend causally.

## Image input (`--mmproj`)

`vision.h` follows the mmproj `qwen3vl_merger` (27 ViT blocks, 1152 wide, 16 heads, patch 16, LayerNorm, fused qkv, 2D RoPE on
the first two of four position sections, GELU(tanh) MLP, 2x2 merge to 5120). The image is resized with `smart_resize`
(multiples of 32 px, at most `--image-tokens` tokens, default 256), normalized with mean = std = 0.5, and its embeddings
replace token embeddings between `<|vision_start|>` and `<|vision_end|>`. The vision weights are expanded to float32 at load.
On the test machine a 640x488 picture (234 tokens) takes ~10 s in the vision tower and ~50 s for the LLM to read the tokens.

## Speculative decoding with the MTP head (`--draft N`)

Implemented and exact: the head (`blk.64.*`, Q4_0, not Hadamard folded) drafts N tokens, the main model verifies them in
one batch, and a draft is accepted only if it equals the token sampled from the main model, so the output is the same as without
drafting (checked: identical text for N = 1, 2 with `--greedy`). The recurrent state of rejected rows is rolled back by pointer swap.
Use the `-mtp.gguf` as MODEL, or `--mtp FILE` to take the head from it.

**On the AVX-only test CPU it is slower, so it is off by default**: acceptance is good (78% with N = 1) but a verify batch of two
tokens costs 1.7x one token, giving 3.8 tokens/s against 4.05 without drafting (N = 2: 2.7). It only pays off where tokens are
cheap to batch (memory-bound hardware); try it on your machine. Not combinable with `--mmproj`.

## Status

Verified: generation matches expectations in English and Japanese (17 x 23 = 391, multi-turn memory, "日本の首都は東京です。"),
images are read correctly (a newspaper front page: name, date, headlines), and the graph was cross-read against the PrismML
llama.cpp fork. Not verified: the logits and the image embeddings were not compared numerically against llama.cpp (in one test the
model read "ASTRONAUTS LAND ON PLAIN" as "... ON MOON"; resolution or the model, not determined), and the AVX2/FMA kernels
(`cats_ptq1_0_decode_avx2`, `cats_ptq1_0_rowdot_avx2`, `cats_r2_body_avx2`) compile but could not be run on the development
machine (AVX only); `CATS_SIMD=avx` forces the AVX path on an AVX2 CPU, `test_ptq1_0` checks the AVX2 kernels when it runs on one.
