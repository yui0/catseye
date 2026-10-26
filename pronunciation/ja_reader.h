/* ja_reader.h - wires cp_ja_reader to tools/ja_reading.py (pykakasi) for kanji text. */
#ifndef CATS_JA_READER_H
#define CATS_JA_READER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "latphon.h"

static const char *cp_tools_dir = "tools";	/* directory containing ja_reading.py */

static int cp_ja_reader_py(const char *text, char *hira, size_t cap)
{
	char tmp[] = "/tmp/jareadXXXXXX";
	int fd = mkstemp(tmp);
	if (fd < 0) return -1;
	size_t n = strlen(text);
	if (write(fd, text, n) != (ssize_t)n) { close(fd); remove(tmp); return -1; }
	close(fd);
	char cmd[1024];
	snprintf(cmd, sizeof cmd, "python3 '%s/ja_reading.py' '%s' 2>/dev/null", cp_tools_dir, tmp);
	FILE *p = popen(cmd, "r");
	if (!p) { remove(tmp); return -1; }
	size_t got = fread(hira, 1, cap - 1, p);
	hira[got] = 0;
	int rc = pclose(p);
	remove(tmp);
	if (rc || !got) { fprintf(stderr, "warning: kanji reading unavailable (pip install pykakasi); kanji skipped\n"); return -1; }
	return 0;
}

static inline void cp_ja_reader_enable(const char *argv0)
{
	static char dir[512];
	snprintf(dir, sizeof dir, "%s", argv0);
	char *sl = strrchr(dir, '/');
	if (sl) { *sl = 0; strncat(dir, "/tools", sizeof dir - strlen(dir) - 1); } else snprintf(dir, sizeof dir, "tools");
	cp_tools_dir = dir;
	cp_ja_reader = cp_ja_reader_py;
}

#endif
