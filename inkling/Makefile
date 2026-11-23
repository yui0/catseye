# Inkling-2-Mini on Cat's eye (C11 + OpenMP, no other dependencies)
CC ?= cc
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
OPENMP ?= 1
LDLIBS ?= -lm -pthread
ifeq ($(UNAME_M),arm64)
CFLAGS ?= -O3 -std=gnu11 -mcpu=native
else ifeq ($(UNAME_M),aarch64)
CFLAGS ?= -O3 -std=gnu11 -mcpu=native
else
CFLAGS ?= -O3 -std=gnu11 -march=native
endif
ifeq ($(UNAME_S),Darwin)
# Apple clang has no OpenMP of its own: use Homebrew's libomp (brew install libomp). Without it the build is single-threaded.
LIBOMP := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(and $(filter 1,$(OPENMP)),$(LIBOMP)),)
OMP_FLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
LDLIBS += -L$(LIBOMP)/lib -lomp
else
$(warning libomp not found: building single-threaded. Run `brew install libomp` for all performance cores.)
OMP_FLAGS :=
endif
else
OMP_FLAGS := $(if $(filter 1,$(OPENMP)),-fopenmp,)
endif
HEADERS := ../catseye.h ../catseye_data.h

all: inkling inkling_quantize

inkling: inkling.c openai_adapter.h ../catseye_openai.h $(HEADERS)
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ inkling.c $(LDLIBS)

inkling_quantize: quantize.c $(HEADERS)
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ quantize.c $(LDLIBS)

# tiny random model vs PyTorch (no download); test-tokenizer needs the real tokenizer.json: make test-tokenizer MODEL_DIR=...
test: all
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o tests/kernel_test tests/kernel_test.c $(LDLIBS) && tests/kernel_test
	tests/run_tiny.sh

test-tokenizer: tests/tok_dump.c $(HEADERS)
	python3 tests/check_tokenizer.py $(MODEL_DIR)

clean:
	rm -f inkling inkling_quantize tests/tok_dump tests/kernel_test

.PHONY: all clean test test-tokenizer
