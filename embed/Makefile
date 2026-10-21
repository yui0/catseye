# eg2e: EmbeddingGemma 2 image / text vectors and similarity search (C, OpenMP, no Python at run time)
CC ?= cc
CFLAGS ?= -O3 -std=gnu11 -fopenmp
WARN = -Wall -Wno-comment -Wno-unused-function
LDLIBS = -lm

# GPU GEMM through EGL + OpenGL ES 3.1 compute (make GPU=1); off by default because weights are fp16 on the GPU
GPU ?= 0
GPU_FLAGS := $(if $(filter 1,$(GPU)),-DCATS_GPU $(shell pkg-config --cflags egl glesv2 2>/dev/null),)
GPU_LIBS  := $(if $(filter 1,$(GPU)),$(shell pkg-config --libs egl glesv2 2>/dev/null),)

all: eg2e

eg2e: eg2e.c eg2_embed.h eg2_tokenizer.h ../catseye.h ../cats_gpu.h
	$(CC) $(CFLAGS) $(GPU_FLAGS) $(WARN) -o $@ eg2e.c $(LDLIBS) $(GPU_LIBS)

test_ref: test_ref.c eg2_embed.h ../catseye.h
	$(CC) $(CFLAGS) $(WARN) -o $@ test_ref.c $(LDLIBS)

# Layer-by-layer comparison with the Python reference:
#   make test MODEL=/path/to/embeddinggemma-2 REF=/path/to/ref NAMES="text_0 image_0"
test: test_ref
	./test_ref $(MODEL) $(REF) $(NAMES)

clean:
	rm -f eg2e test_ref

.PHONY: all test clean
