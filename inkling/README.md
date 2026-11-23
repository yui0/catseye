# Inkling 2 Mini on Cat's eye

[日本語](README.ja.md)

Run Thinking Machines' Inkling 2 Mini locally on a CPU. Inkling supports text generation from the terminal, interactive chat, and an OpenAI-compatible API for clients such as OpenCode. The converter packages the quantized text weights, tokenizer, and model configuration in a single safetensors file. See the [Inkling 2 Mini model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) for model details and the weights license.

## Build

From the root of the Cat's eye checkout, enter `inkling/` and build. You need `make` and a C11 compiler. The build produces `inkling` and `inkling_quantize`.

```sh
cd inkling
make
```

On macOS, install OpenMP with `brew install libomp` before building for multithreaded inference. You can build for a single thread with `make OPENMP=0`.

## Prepare the model

If you already have a file produced by `inkling_quantize`, pass its path to `-m` and skip this step. To convert the original checkpoint, download the required files with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

```sh
hf download thinkingmachines/Inkling-2-Mini \
  model-00001-of-00001.safetensors config.json tokenizer.json \
  --local-dir Inkling-2-Mini

./inkling_quantize -i Inkling-2-Mini -o inkling-q4.safetensors
```

The default output is about 15.5 GB. Allow disk space for both the checkpoint and the converted file. Use a machine with around 32 GB of RAM for conversion, since the converter holds the output in memory before writing it.

## Run from the terminal

Generate a single reply, start an interactive conversation, or read a prompt from a UTF-8 text file.

```sh
./inkling -m inkling-q4.safetensors -p "Explain why the sky is blue in one sentence."
./inkling -m inkling-q4.safetensors -i
./inkling -m inkling-q4.safetensors -f prompt.txt --effort low -n 256 -v
```

Interactive mode keeps the conversation in the KV cache until you exit. Use `Ctrl+D` to finish. Use `-s` to supply a system message and `--think` to display the model's thinking. `-n` sets the maximum number of generated tokens, with a default of 512. `-c` sets the context length, with a default of 4096. Use `--effort low` for shorter reasoning and `-v` to see timings.

## Use with OpenCode or an API client

Start the server from `inkling/`. This command matches the context and output limits in the bundled OpenCode 2 configuration.

```sh
OMP_WAIT_POLICY=PASSIVE ./inkling -m inkling-q4.safetensors \
  --serve 8080 -c 16384 -n 2048 --effort low -v
```

The server listens on `127.0.0.1` and handles one inference request at a time. A longer context uses more RAM. To check the server and send a short prompt, run the following from another terminal.

```sh
curl -fsS http://127.0.0.1:8080/v1/models

curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling","messages":[{"role":"user","content":"Say hello."}],"max_tokens":128}'
```

The chat API supports text input, JSON responses, SSE streaming with `"stream":true`, thinking text, and function calls with tool results in the conversation history. Function selection supports `auto` and `none`.

For OpenCode 2, use the bundled [opencode.json](opencode.json). From the project you want OpenCode to work on, set `OPENCODE_CONFIG` to the absolute path to that file in your Cat's eye checkout.

```sh
OPENCODE_CONFIG=/path/to/catseye/inkling/opencode.json \
  opencode run --standalone -m catseye/inkling "Describe this project's structure."
```

If you change the server port or context length, update `settings.baseURL` or `limit.context` in the provider configuration. The API model ID is `inkling`. [OpenCode custom provider documentation](https://opencode.ai/docs/models/#compatible)

## Run with less memory

Use Q3 experts and Q4 attention and vocabulary tables to reduce the model file to about 11.9 GB. You can convert it on a machine with more RAM, then copy the resulting file to the inference machine.

```sh
./inkling_quantize -i Inkling-2-Mini -o inkling-q3.safetensors -b q3 -a q4 -e q4
./inkling -m inkling-q3.safetensors -c 2048 --batch 32 -p "Summarize the water cycle." -v
```

Inkling memory-maps the model and reads routed experts as needed. When the experts exceed the memory budget, Inkling enables an expert cache automatically. A fast SSD helps when the weights cannot stay in memory. Start with the model's default expert count and loop passes for quality. If storage reads dominate, `--topk 6` can reduce the amount of expert data read, with a potential quality trade-off.

## Tests

`make test` checks the SIMD kernels and compares inference on a small generated checkpoint with a PyTorch reference. It needs Python, PyTorch, NumPy, and `safetensors`. `make test-tokenizer MODEL_DIR=Inkling-2-Mini` checks the tokenizer against Hugging Face `tokenizers` using the downloaded tokenizer file.
