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

# Browser build (Emscripten, WASM SIMD):  source /root/image/emsdk/emsdk_env.sh; make web
#   eg2e_web.js     single thread; works anywhere
#   eg2e_web_mt.js  pthreads (SharedArrayBuffer: the page needs COOP/COEP headers, see web/serve.py)
WEB_FLAGS = -O3 -msimd128 -Wno-comment -Wno-unused-function eg2e_web.c \
	-s MODULARIZE=1 -s EXPORT_NAME=Eg2eModule -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=3GB -s INITIAL_MEMORY=64MB \
	-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","FS","HEAPU8","HEAPF32","UTF8ToString","stringToUTF8"]' \
	-s EXPORTED_FUNCTIONS='["_malloc","_free"]' -s ENVIRONMENT=web,worker,node -s FILESYSTEM=1
WEB_THREADS ?= 11

web: eg2e_web.js eg2e_web_mt.js
eg2e_web.js: eg2e_web.c eg2_embed.h eg2_tokenizer.h ../catseye.h
	emcc $(WEB_FLAGS) -o $@
eg2e_web_mt.js: eg2e_web.c eg2_embed.h eg2_tokenizer.h ../catseye.h
	emcc $(WEB_FLAGS) -pthread -s PTHREAD_POOL_SIZE=$(WEB_THREADS) -o $@

clean:
	rm -f eg2e test_ref eg2e_web.js eg2e_web.wasm eg2e_web_mt.js eg2e_web_mt.wasm eg2e_web_mt.worker.js

.PHONY: all test clean web
