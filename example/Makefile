# ©2016 YUICHIRO NAKADA

PROGRAM = $(patsubst %.c,%,$(wildcard *.c))
OBJS = $(patsubst %.c,%.o,$(wildcard *.c))

# 定義済みマクロの再定義
CC = clang
CFLAGS = -Os -Ofast
LDFLAGS = -lm

# サフィックスルール適用対象の拡張子の定義
.SUFFIXES: .c .o

# プライマリターゲット
.PHONY: all
all: depend $(PROGRAM)

# プログラムの生成ルール
#$(PROGRAM): $(OBJS)
#$(PROGRAM): $(patsubst %.c,%,$(PROGRAM))
#	$(CC) $(LDFLAG) -o $(PROGRAM) $^
#$(PROGRAM): $@
#	$(CC) $(LDFLAGS) -o $(PROGRAM) $@
%.o : %.c $(HEAD)
	$(CC) $(LDFLAGS) $(CFLAGS) -c $(@F:.o=.c) -o $@

# サフィックスルール
#.c.o:
#	$(CC) $(CFLAGS) -c $<

# ファイル削除用ターゲット
.PHONY: clean
clean:
	$(RM) $(PROGRAM) $(OBJS) _depend.inc

# ヘッダファイルの依存関係
.PHONY: depend
depend: $(OBJS:.o=.c)
	-@ $(RM) _depend.inc
	-@ for i in $^; do cpp -MM $$i | sed "s/\ [_a-zA-Z0-9][_a-zA-Z0-9]*\.c//g" >> _depend.inc; done

-include _depend.inc
