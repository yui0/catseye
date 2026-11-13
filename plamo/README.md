# PLaMo 3 Translate (Cat's eye)

A translation CLI in dependency-free C for Preferred Networks' [PLaMo 3 Translate](https://huggingface.co/collections/pfnet/plamo-3-translate). It uses the smallest model, **0.8B** (53 languages), and can shrink it further with quantization.

Only the PLaMo-specific code lives in `plamo/` (config, weight names, the sandwich-norm decoder layer, the PFML3 prompt). Everything reusable is in `../catseye.h`: safetensors reader (F32/F16/BF16/Q4_0/Q8_0), quantized GEMM, YaRN RoPE, KV-cached GQA attention, offset RMSNorm and the Unigram tokenizer.

## Usage

```sh
make                                     # plamo and plamo_quantize
hf download pfnet/plamo-3-translate-0.8b --local-dir hf-0.8b
./plamo_quantize -i hf-0.8b -o models/q4          # 482 MB (original 1.78 GB)

./plamo -m models/q4 -t en -p "今日はとても良い天気ですね。"
echo "Hello, world." | ./plamo -m models/q4 -t ja --polite
./plamo -m models/q4 -t ja -g "catseye=キャッツアイ" -p "I use catseye every day."
cat doc.txt | ./plamo -m models/q4 -t en -l       # one line at a time
```

Run `./plamo` for all options: `-s` source language, `-c` reference context, `-g src=tgt` glossary, `--polite`/`--plain` Japanese register, `-v` timing. Language codes are lowercase: `en` `ja` `zh-hans` `zh-hant` `ko` `fr` `de` `ja-easy` ...

## Making it smaller

`plamo_quantize -b body -e embedding -h lm_head` (types `f16`, `q8`, `q4`). Sizes for 0.8B:

| Setup | Size | Notes |
|---|---|---|
| original BF16 | 1.78 GB | the official weights load as they are |
| `-b q8` | 906 MB | translations nearly identical to BF16 |
| `-b q4 -e q8 -h q8` | 639 MB | 4-bit body, 8-bit vocabulary tables |
| `-b q4` (default) | 482 MB | occasionally picks different wording |

The next step down would be restricting the vocabulary (the 107,520 x 1536 embedding and output matrices are nearly 40% of the model) to the languages you use; that is not implemented. `--fast` is plain GGML Q4_0 (faster to produce, slightly worse).

## Verification

`tests/check_reference.py` compares tokenizer ids and greedy translations with the official Hugging Face implementation (needs torch, transformers, numba):

```sh
cd tests && python3 check_reference.py ../hf-0.8b ../models/q4
```

With the BF16 weights the tokenizer matched on every case and 5 of 6 translations were identical (the other differs only through the reference's bf16 rounding). The shared building blocks are tested by `make test-core` (`../test/test_catseye_core.c`).

## License

The weights use the [PLaMo Community License](https://plamo.preferredai.jp/): free for personal, learning and non-commercial (including academic) use; commercial use up to 1 billion yen annual revenue needs registration; above that needs an agreement with PFN. Check the model card for the current terms.
