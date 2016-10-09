# ©2016 YUICHIRO NAKADA

PROGRAM = $(patsubst %.c,%,$(wildcard *.c))
#OBJS = $(patsubst %.c,%.o,$(wildcard *.c))

CC = clang
CFLAGS = -Ofast -march=native -funroll-loops -mf16c
LDFLAGS = -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CFLAGS += `pkg-config --libs --cflags OpenCL`
endif
ifeq ($(UNAME_S),Darwin)
	CFLAGS += -framework opencl
endif

.PHONY: all
all: depend $(PROGRAM)

%.o : %.c $(HEAD)
	$(CC) $(LDFLAGS) $(CFLAGS) -c $(@F:.o=.c) -o $@

.PHONY: clean
clean:
	$(RM) $(PROGRAM) $(OBJS) _depend.inc

.PHONY: depend
depend: $(OBJS:.o=.c)
	-@ $(RM) _depend.inc
	-@ for i in $^; do cpp -MM $$i | sed "s/\ [_a-zA-Z0-9][_a-zA-Z0-9]*\.c//g" >> _depend.inc; done

-include _depend.inc
