/*
 * gguf.h - minimal GGUF (v2/v3) reader: metadata lookup and tensor table over an mmap of the file.
 * Tensors keep their on-disk type; dims are in ggml order (dim[0] is the fastest-varying one).
 */
#ifndef CATS_GGUF_H
#define CATS_GGUF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

enum { CG_F32 = 0, CG_F16 = 1, CG_Q4_0 = 2, CG_Q8_0 = 8 };
enum { CG_U8, CG_I8, CG_U16, CG_I16, CG_U32, CG_I32, CG_F32V, CG_BOOL, CG_STR, CG_ARR, CG_U64, CG_I64, CG_F64 };

typedef struct { char name[128]; int nd; uint64_t dim[4]; int type; uint64_t n; const void *data; size_t bytes; } cg_tensor;
typedef struct { char key[96]; int type; const uint8_t *val; } cg_kv;	/* val points at the raw value (arrays: elem type u32, count u64, elements) */

typedef struct {
	uint8_t *map; size_t size;
	cg_kv *kv; int nkv;
	cg_tensor *t; int nt;
} cg_file;

static inline uint64_t cg_rd64(const uint8_t **p) { uint64_t v; memcpy(&v, *p, 8); *p += 8; return v; }
static inline uint32_t cg_rd32(const uint8_t **p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }

static inline size_t cg_type_bytes(int type, uint64_t n)
{
	switch (type) {
	case CG_F32: return n * 4;
	case CG_F16: return n * 2;
	case CG_Q4_0: return n / 32 * 18;
	case CG_Q8_0: return n / 32 * 34;
	}
	return 0;
}

/* skip one value of the given type, returning the new position */
static inline const uint8_t *cg_skip(const uint8_t *p, int type)
{
	switch (type) {
	case CG_U8: case CG_I8: case CG_BOOL: return p + 1;
	case CG_U16: case CG_I16: return p + 2;
	case CG_U32: case CG_I32: case CG_F32V: return p + 4;
	case CG_U64: case CG_I64: case CG_F64: return p + 8;
	case CG_STR: { uint64_t l = cg_rd64(&p); return p + l; }
	case CG_ARR: {
		int et = (int)cg_rd32(&p); uint64_t n = cg_rd64(&p);
		for (uint64_t i = 0; i < n; i++) p = cg_skip(p, et);
		return p;
	}
	}
	return p;
}

static inline void cg_close(cg_file *g)
{
	if (g->map) munmap(g->map, g->size);
	free(g->kv); free(g->t);
	memset(g, 0, sizeof *g);
}

static inline int cg_open(cg_file *g, const char *path)
{
	memset(g, 0, sizeof *g);
	int fd = open(path, O_RDONLY);
	if (fd < 0) return -1;
	struct stat st;
	if (fstat(fd, &st)) { close(fd); return -1; }
	g->size = (size_t)st.st_size;
	g->map = (uint8_t *)mmap(NULL, g->size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (g->map == MAP_FAILED) { g->map = NULL; return -1; }
	const uint8_t *p = g->map;
	if (memcmp(p, "GGUF", 4)) { cg_close(g); return -1; }
	p += 4;
	uint32_t ver = cg_rd32(&p);
	if (ver < 2 || ver > 3) { cg_close(g); return -1; }
	uint64_t nt = cg_rd64(&p), nkv = cg_rd64(&p);
	g->kv = (cg_kv *)calloc(nkv ? nkv : 1, sizeof(cg_kv)); g->nkv = (int)nkv;
	g->t = (cg_tensor *)calloc(nt ? nt : 1, sizeof(cg_tensor)); g->nt = (int)nt;
	for (uint64_t i = 0; i < nkv; i++) {
		uint64_t l = cg_rd64(&p);
		snprintf(g->kv[i].key, sizeof g->kv[i].key, "%.*s", (int)(l < 95 ? l : 95), (const char *)p);
		p += l;
		g->kv[i].type = (int)cg_rd32(&p);
		g->kv[i].val = p;
		p = cg_skip(p, g->kv[i].type);
	}
	uint64_t align = 32;
	for (uint64_t i = 0; i < nt; i++) {
		cg_tensor *t = &g->t[i];
		uint64_t l = cg_rd64(&p);
		snprintf(t->name, sizeof t->name, "%.*s", (int)(l < 127 ? l : 127), (const char *)p);
		p += l;
		t->nd = (int)cg_rd32(&p);
		t->n = 1;
		for (int d = 0; d < t->nd; d++) { t->dim[d] = cg_rd64(&p); t->n *= t->dim[d]; }
		t->type = (int)cg_rd32(&p);
		t->data = (const void *)(uintptr_t)cg_rd64(&p);	/* offset for now */
		t->bytes = cg_type_bytes(t->type, t->n);
	}
	for (int i = 0; i < g->nkv; i++) if (!strcmp(g->kv[i].key, "general.alignment")) { const uint8_t *v = g->kv[i].val; align = cg_rd32(&v); }
	size_t base = ((size_t)(p - g->map) + align - 1) / align * align;
	for (uint64_t i = 0; i < nt; i++) {
		g->t[i].data = g->map + base + (uintptr_t)g->t[i].data;
		if ((const uint8_t *)g->t[i].data + g->t[i].bytes > g->map + g->size) { cg_close(g); return -1; }
	}
	return 0;
}

static inline const cg_tensor *cg_find(const cg_file *g, const char *name)
{
	for (int i = 0; i < g->nt; i++) if (!strcmp(g->t[i].name, name)) return &g->t[i];
	return NULL;
}

static inline const cg_kv *cg_kvfind(const cg_file *g, const char *key)
{
	for (int i = 0; i < g->nkv; i++) if (!strcmp(g->kv[i].key, key)) return &g->kv[i];
	return NULL;
}

static inline uint32_t cg_u32(const cg_file *g, const char *key, uint32_t def)
{
	const cg_kv *k = cg_kvfind(g, key);
	if (!k || (k->type != CG_U32 && k->type != CG_I32)) return def;
	const uint8_t *p = k->val;
	return cg_rd32(&p);
}

/* string array: returns count, fills out[i] with malloc'ed copies (caller frees) */
static inline int cg_str_array(const cg_file *g, const char *key, char ***out)
{
	const cg_kv *k = cg_kvfind(g, key);
	if (!k || k->type != CG_ARR) return -1;
	const uint8_t *p = k->val;
	int et = (int)cg_rd32(&p); uint64_t n = cg_rd64(&p);
	if (et != CG_STR) return -1;
	char **o = (char **)calloc(n ? n : 1, sizeof(char *));
	for (uint64_t i = 0; i < n; i++) {
		uint64_t l = cg_rd64(&p);
		o[i] = (char *)malloc(l + 1); memcpy(o[i], p, l); o[i][l] = 0; p += l;
	}
	*out = o;
	return (int)n;
}

#endif
