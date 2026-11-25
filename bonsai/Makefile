CC ?= cc
CFLAGS ?= -O3 -march=native -std=gnu11 -fopenmp -Wall -Wno-unused-function
LDLIBS = -lm -fopenmp
MODEL ?= models/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf
URL = https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF/resolve/main/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf

all: bonsai

bonsai: bonsai.c bonsai.h vision.h ../catseye.h ../pronunciation/gguf.h ../stb_image.h ../stb_image_resize.h
	$(CC) $(CFLAGS) -o $@ bonsai.c $(LDLIBS)

test/test_ptq1_0: test/test_ptq1_0.c ../catseye.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

test: test/test_ptq1_0
	./test/test_ptq1_0

# 6 GB (model) and 0.6 GB (vision tower, optional)
model:
	mkdir -p models && curl -L -C - -o $(MODEL) $(URL)
mmproj:
	mkdir -p models && curl -L -C - -o models/Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf $(URL:PTQ1_0.gguf=mmproj-Q8_0.gguf)

run: bonsai
	./bonsai $(MODEL) -p "Explain quantum computing in simple terms."

clean:
	rm -f bonsai test/test_ptq1_0

.PHONY: all test model mmproj run clean
