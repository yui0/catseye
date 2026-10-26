CC ?= cc
CFLAGS ?= -O3 -march=native -std=gnu11 -fopenmp -Wall -Wextra -Wno-unused-function -Wno-missing-field-initializers
LDLIBS = -lm -fopenmp

all: tts pcheck

tts: tts.c lang_en.h lang_pt.h lang_ja.h ja_reader.h latphon.h phon.h klatt.h tts_piper.h tts_kokoro.h gguf.h ../transcript/cats_onnx.h
	$(CC) $(CFLAGS) -o $@ tts.c $(LDLIBS)

test: test/test_g2p test/test_klatt test/test_pcheck test/test_piper test/test_kokoro test/test_piper
	./test/test_g2p
	./test/test_klatt
	./test/test_pcheck
	./test/test_piper
	@if [ -f models/kokoro-q8.gguf ] && [ -f models/af_heart.bin ] && [ -f /tmp/kokoro_ref/meta.json ]; then ./test/test_kokoro models/kokoro-q8.gguf models/af_heart.bin /tmp/kokoro_ref q8 && ./test/test_kokoro models/kokoro-q8.gguf models/af_heart.bin /tmp/kokoro_ref; else echo 'kokoro skipped (needs models/kokoro-q8.gguf, models/af_heart.bin and tools/kokoro_ref.py output in /tmp/kokoro_ref)'; fi

pcheck: pcheck.c lang_en.h lang_pt.h lang_ja.h ja_reader.h pcheck.h align.h latphon.h phon.h
	$(CC) $(CFLAGS) -o $@ pcheck.c $(LDLIBS)

test/%: test/%.c tts_kokoro.h gguf.h lang_en.h lang_pt.h lang_ja.h latphon.h phon.h klatt.h align.h pcheck.h tts_piper.h ../transcript/cats_onnx.h
	$(CC) $(CFLAGS) -I. -o $@ $< $(LDLIBS)

clean:
	rm -f tts pcheck test/test_pcheck test/test_g2p test/test_klatt test/test_piper test/test_kokoro

.PHONY: all test clean
