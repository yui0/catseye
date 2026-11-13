# PLaMo 3 Translate on Cat's eye (C11 + OpenMP, no other dependencies)
CC ?= cc
CFLAGS ?= -O3 -std=gnu11 -march=native
OPENMP ?= 1
OMP_FLAGS := $(if $(filter 1,$(OPENMP)),-fopenmp,)
LDLIBS ?= -lm
HEADERS := ../catseye.h

all: plamo plamo_quantize

plamo: plamo.c $(HEADERS)
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ plamo.c $(LDLIBS)

plamo_quantize: quantize.c $(HEADERS)
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ quantize.c $(LDLIBS)

clean:
	rm -f plamo plamo_quantize

.PHONY: all clean
