# Project entry points. Application-specific options live in each directory.
.DEFAULT_GOAL := help
.PHONY: help all transcript check test test-core clean legacy
help:
	@printf '%s\n' 'make transcript    Build the transcription CLI in transcript/' 'make check         Build the CLI and show CPU capabilities' 'make test          Run the library tests (gradient check, kernels) and the transcription tests' 'make legacy        Build the original training examples' 'make clean         Remove transcription executables' 'make -C transcript help  Show transcription and model commands'
all: transcript
transcript:
	$(MAKE) -C transcript
check clean:
	$(MAKE) -C transcript $@
	@if [ "$@" = clean ]; then rm -rf test/bin; fi
# catseye.h itself: gradient check of every layer type + inference kernels / file formats (sanitizers if available)
CC ?= cc
SAN := $(shell echo 'int main(){return 0;}' | $(CC) -x c - -fsanitize=address,undefined -o /dev/null 2>/dev/null && echo -fsanitize=address,undefined)
test-core:
	@mkdir -p test/bin
	$(CC) -O1 -g -w -D_GNU_SOURCE $(SAN) -fopenmp -o test/bin/grad test/test_catseye_grad.c -lm
	$(CC) -O1 -g -w -D_GNU_SOURCE $(SAN) -fopenmp -o test/bin/core test/test_catseye_core.c -lm
	ASAN_OPTIONS=detect_leaks=0 test/bin/grad
	$(CC) -O2 -g -w -D_GNU_SOURCE $(SAN) -fopenmp -o test/bin/train test/test_catseye_train.c -lm
	ASAN_OPTIONS=detect_leaks=0 test/bin/core
	ASAN_OPTIONS=detect_leaks=0 test/bin/train
test: test-core
	$(MAKE) -C transcript test
legacy:
	$(MAKE) -f Makefile.legacy
