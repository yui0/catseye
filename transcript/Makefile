# Cat's eye / Hayamimi / EmbeddingGemma 2 — native C build
# Requires a GNU-compatible C11 compiler; Python is used only for model packaging.
.DEFAULT_GOAL := all

CC ?= cc
PYTHON ?= python3
CPPFLAGS ?=
CFLAGS ?= -O3 -std=gnu11
# Multi-threaded EmbeddingGemma 2 kernels (OpenMP). `make OPENMP=0` builds serial.
OPENMP ?= 1
OMP_FLAGS := $(if $(filter 1,$(OPENMP)),-fopenmp,)
# GPU GEMM through EGL + OpenGL ES 3.1 compute (`make GPU=0` to leave it out).
# Auto: enabled when the EGL/GLESv2 development files are installed.
GPU ?= auto
ifeq ($(GPU),auto)
GPU := $(shell pkg-config --exists egl glesv2 2>/dev/null && echo '#include <GLES3/gl32.h>' | $(CC) -E - >/dev/null 2>&1 && echo 1 || echo 0)
endif
GPU_FLAGS := $(if $(filter 1,$(GPU)),-DCATS_GPU $(shell pkg-config --cflags egl glesv2 2>/dev/null),)
GPU_LIBS  := $(if $(filter 1,$(GPU)),$(shell pkg-config --libs egl glesv2 2>/dev/null),)
LDFLAGS ?=
LDLIBS ?= -lm

TARGET = cats
MODEL_DIR ?= models
ASR_LANG ?= en
WAV ?=
MODEL_ZIP ?=
TEST_OPT ?= 2

HEADERS := $(wildcard *.h) ../catseye_llm.h ../cats_gpu.h
NATIVE_FLAGS := -DCATS_SPEECH_NATIVE

.PHONY: all help debug hayamimi test test-smoke test-adapter check clean asr \
        bundle-english bundle-japanese bundle-mixed verify-models fp16-models

all: $(TARGET)

$(TARGET): cats.c $(HEADERS)
	$(CC) $(CPPFLAGS) $(NATIVE_FLAGS) $(OMP_FLAGS) $(GPU_FLAGS) $(CFLAGS) -o $@ cats.c $(LDFLAGS) $(LDLIBS) $(GPU_LIBS)

# Debug and Hayamimi are separate executables; they never overwrite the release build.
debug: cats-debug
cats-debug: cats.c $(HEADERS)
	$(CC) $(CPPFLAGS) $(NATIVE_FLAGS) $(OMP_FLAGS) $(GPU_FLAGS) -O0 -g3 -std=gnu11 -o $@ cats.c $(LDFLAGS) $(LDLIBS) $(GPU_LIBS)

hayamimi: cats-hayamimi
cats-hayamimi: cats.c $(HEADERS)
	$(CC) $(CPPFLAGS) $(NATIVE_FLAGS) $(OMP_FLAGS) $(GPU_FLAGS) -DCATS_HAYAMIMI $(CFLAGS) -o $@ cats.c $(LDFLAGS) $(LDLIBS) $(GPU_LIBS)

check: $(TARGET)
	./$(TARGET) -a cpu-info

# Full existing regression suite, including older ONNX and speech tests.
test:
	CC="$(CC)" CATS_TEST_OPT="$(TEST_OPT)" sh ./run_tests.sh

# Fast focused check of the long-WAV, bundle and GEMM paths.
test-smoke:
	@set -eu; dir=$$(mktemp -d); trap 'rm -rf "$$dir"' EXIT HUP INT TERM; \
	for src in test/test_eg2_stage31.c test/test_eg2_stage35.c test/test_eg2_stage39.c test/test_eg2_mixed.c test/test_eg2_hybrid.c; do \
	  name=$$(basename $${src%.c}); \
	  $(CC) $(CPPFLAGS) $(OMP_FLAGS) -O$(TEST_OPT) -std=gnu11 -I. -o "$$dir/$$name" "$$src" $(LDFLAGS) $(LDLIBS); \
	  "$$dir/$$name"; \
	done; \
	printf 'PASS: 5 focused C tests\n'

# Small adapter unit and synthetic end-to-end tests (require numpy/torch).
test-adapter:
	$(PYTHON) test/test_eg2_adapter.py
	$(PYTHON) test/test_eg2_adapter_training.py

# These commands use already-generated models; they DO NOT train or download weights.
bundle-english:
	@test -n "$(MODEL_ZIP)" || { echo 'Usage: make bundle-english MODEL_ZIP=embeddinggemma2-native-official-v2.zip'; exit 2; }
	$(PYTHON) tools/eg2_bundle.py --from-zip "$(MODEL_ZIP)" --out "$(MODEL_DIR)"

bundle-japanese:
	@test -n "$(MODEL_ZIP)" || { echo 'Usage: make bundle-japanese MODEL_ZIP=embeddinggemma2-reader-ja.zip'; exit 2; }
	$(PYTHON) tools/eg2_bundle.py --from-zip "$(MODEL_ZIP)" --source ja --out "$(MODEL_DIR)"

# Package a jointly-trained EN/JA Reader (not two monolingual readers joined).
bundle-mixed:
	@test -d "$(MIX_DIR)" || { echo 'Usage: make bundle-mixed MIX_DIR=/path/to/trained-reader'; exit 2; }
	$(PYTHON) tools/eg2_bundle.py --mix-dir "$(MIX_DIR)" --out "$(MODEL_DIR)"

verify-models:
	@test -f "$(MODEL_DIR)/encoder.eg2m" || { echo 'Missing encoder.eg2m in $(MODEL_DIR)'; exit 2; }
	$(PYTHON) tools/eg2_bundle.py --verify "$(MODEL_DIR)/encoder.eg2m"
	@for lang in en ja mix; do \
	  if test -f "$(MODEL_DIR)/$$lang.eg2m"; then \
	    $(PYTHON) tools/eg2_bundle.py --verify "$(MODEL_DIR)/$$lang.eg2m" || exit 1; \
	  fi; \
	done

# Half-precision encoder (about half the size, same transcript); the language
# packages are copied unchanged next to it.
fp16-models:
	@test -f "$(MODEL_DIR)/encoder.eg2m" || { echo 'Missing encoder.eg2m in $(MODEL_DIR)'; exit 2; }
	$(PYTHON) tools/eg2_bundle.py --to-fp16 "$(MODEL_DIR)/encoder.eg2m" --out "$(MODEL_DIR)_fp16"
	@for lang in en ja mix; do \
	  if test -f "$(MODEL_DIR)/$$lang.eg2m"; then cp -n "$(MODEL_DIR)/$$lang.eg2m" "$(MODEL_DIR)_fp16/"; fi; \
	done

asr: $(TARGET)
	@test -n "$(WAV)" || { echo 'Usage: make asr WAV=audio.wav [ASR_LANG=auto] [MODEL_DIR=models]'; exit 2; }
	./$(TARGET) -a eg2-native-asr -m "$(MODEL_DIR)" -L "$(ASR_LANG)" -w "$(WAV)"

clean:
	rm -f "$(TARGET)" cats-debug cats-hayamimi

help:
	@printf '%s\n' \
	  'make                         Build ./cats (OpenMP, and GPU GEMM when EGL/GLESv2 are installed)' \
	  'make OPENMP=0 GPU=0          Plain single-thread CPU build' \
	  'make clean && make CC=clang  Rebuild with Clang' \
	  'make debug                   Build ./cats-debug (-O0 -g3)' \
	  'make hayamimi                Build ./cats-hayamimi (optional ASR path)' \
	  'make check                   Show supported CPU SIMD' \
	  'make test-smoke              Run 4 focused native C tests' \
	  'make test                    Run all C and Python tests (duration depends on hardware)' \
	  'make test-adapter            Test C/Python adapter parity and synthetic CTC training' \
	  'make asr WAV=audio.wav [ASR_LANG=auto] [MODEL_DIR=models]' \
	  'make bundle-english MODEL_ZIP=official.zip' \
	  'make bundle-japanese MODEL_ZIP=japanese.zip' \
	  'make bundle-mixed MIX_DIR=/path/to/trained-mix' \
	  'make verify-models           Validate existing *.eg2m files' \
	  'make fp16-models             Write a half-size encoder to models_fp16/' \
		  'make clean                   Remove compiled binaries (not models)'
