# Project entry points. Application-specific options live in each directory.
.DEFAULT_GOAL := help
.PHONY: help all transcript check test clean legacy
help:
	@printf '%s\n' 'make transcript    Build the transcription CLI in transcript/' 'make check         Build the CLI and show CPU capabilities' 'make test          Run transcription regression tests' 'make legacy        Build the original training examples' 'make clean         Remove transcription executables' 'make -C transcript help  Show transcription and model commands'
all: transcript
transcript:
	$(MAKE) -C transcript
check test clean:
	$(MAKE) -C transcript $@
legacy:
	$(MAKE) -f Makefile.legacy
