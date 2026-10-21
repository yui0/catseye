//---------------------------------------------------------
//	Cat's eye - EmbeddingGemma 2 tokenizer (byte-fallback BPE)
//
//		Reads the .eg2t file written by tools/eg2_embed_tokenizer.py. Matches the Hugging Face
//		tokenizer: spaces become U+2581, the whole text is one BPE chunk, unknown characters fall
//		back to <0xXX> byte tokens, and the result is wrapped as [<bos>, ..., <eos>].
//		Literal special-token strings in the text (e.g. "<bos>") are not recognised.
//---------------------------------------------------------

#ifndef EG2_TOKENIZER_H
#define EG2_TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
	uint32_t n_vocab, n_merges;
	char *pool;			// all token strings back to back
	uint32_t *off;			// n_vocab + 1 offsets into pool
	uint32_t *shash;		// open-addressing table: token id + 1 (0 = empty)
	uint32_t smask;
	uint64_t *pkey;			// (left << 32 | right) + 1, 0 = empty
	uint32_t *prank, *pres;
	uint32_t pmask;
	int byte_id[256];		// id of <0xXX>, -1 if absent
} eg2e_tok;

static uint32_t eg2t_hash(const char *s, size_t n)
{
	uint32_t h = 2166136261u;
	for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
	return h;
}
static uint32_t eg2t_pair_hash(uint32_t a, uint32_t b)
{
	uint64_t x = ((uint64_t)a << 32 | b) * 0x9E3779B97F4A7C15ull;
	return (uint32_t)(x >> 32);
}
static int eg2t_find(const eg2e_tok *t, const char *s, size_t n)
{
	for (uint32_t i = eg2t_hash(s, n) & t->smask;; i = (i + 1) & t->smask) {
		uint32_t e = t->shash[i];
		if (!e) return -1;
		uint32_t id = e - 1, len = t->off[id + 1] - t->off[id];
		if (len == n && !memcmp(t->pool + t->off[id], s, n)) return (int)id;
	}
}
static void eg2t_free(eg2e_tok *t)
{
	free(t->pool); free(t->off); free(t->shash); free(t->pkey); free(t->prank); free(t->pres);
	memset(t, 0, sizeof(*t));
}
static int eg2t_load(eg2e_tok *t, const char *path)
{
	memset(t, 0, sizeof(*t));
	FILE *f = fopen(path, "rb");
	if (!f) return -1;
	unsigned char hd[12];
	if (fread(hd, 1, 12, f) != 12 || memcmp(hd, "EG2T", 4)) { fclose(f); return -1; }
	memcpy(&t->n_vocab, hd + 4, 4);
	memcpy(&t->n_merges, hd + 8, 4);
	fseek(f, 0, SEEK_END);
	long total = ftell(f);
	fseek(f, 12, SEEK_SET);
	t->off = (uint32_t*)malloc(((size_t)t->n_vocab + 1) * 4);
	t->pool = (char*)malloc((size_t)total);
	if (!t->off || !t->pool) goto bad;
	size_t used = 0;
	for (uint32_t i = 0; i < t->n_vocab; i++) {
		uint16_t len;
		if (fread(&len, 2, 1, f) != 1 || fread(t->pool + used, 1, len, f) != len) goto bad;
		t->off[i] = (uint32_t)used;
		used += len;
	}
	t->off[t->n_vocab] = (uint32_t)used;
	for (t->smask = 1; t->smask < t->n_vocab * 2; t->smask <<= 1) ;
	t->smask--;
	t->shash = (uint32_t*)calloc((size_t)t->smask + 1, 4);
	for (t->pmask = 1; t->pmask < t->n_merges * 2; t->pmask <<= 1) ;
	t->pmask--;
	t->pkey = (uint64_t*)calloc((size_t)t->pmask + 1, 8);
	t->prank = (uint32_t*)malloc(((size_t)t->pmask + 1) * 4);
	t->pres = (uint32_t*)malloc(((size_t)t->pmask + 1) * 4);
	if (!t->shash || !t->pkey || !t->prank || !t->pres) goto bad;
	for (uint32_t id = 0; id < t->n_vocab; id++) {
		uint32_t i = eg2t_hash(t->pool + t->off[id], t->off[id + 1] - t->off[id]) & t->smask;
		while (t->shash[i]) i = (i + 1) & t->smask;
		t->shash[i] = id + 1;
	}
	for (uint32_t r = 0; r < t->n_merges; r++) {
		uint32_t m[3];
		if (fread(m, 4, 3, f) != 3) goto bad;
		uint32_t i = eg2t_pair_hash(m[0], m[1]) & t->pmask;
		while (t->pkey[i]) i = (i + 1) & t->pmask;
		t->pkey[i] = ((uint64_t)m[0] << 32 | m[1]) + 1;
		t->prank[i] = r;
		t->pres[i] = m[2];
	}
	for (int b = 0; b < 256; b++) {
		char name[8];
		snprintf(name, sizeof name, "<0x%02X>", b);
		t->byte_id[b] = eg2t_find(t, name, 6);
	}
	fclose(f);
	return 0;
bad:
	fclose(f);
	eg2t_free(t);
	return -1;
}
static int eg2t_merge_lookup(const eg2e_tok *t, uint32_t a, uint32_t b, uint32_t *rank, uint32_t *res)
{
	uint64_t key = ((uint64_t)a << 32 | b) + 1;
	for (uint32_t i = eg2t_pair_hash(a, b) & t->pmask; t->pkey[i]; i = (i + 1) & t->pmask)
		if (t->pkey[i] == key) { *rank = t->prank[i]; *res = t->pres[i]; return 1; }
	return 0;
}

// UTF-8 text -> [<bos>, ids..., <eos>]. Returns the id count (capped at `cap`), or -1 on error.
static int eg2t_encode(const eg2e_tok *t, const char *text, int *ids, int cap)
{
	size_t n = strlen(text);
	int *sym = (int*)malloc((n + 2) * 3 * sizeof(int));	// a space expands to 3 bytes, a byte to one token
	if (!sym) return -1;
	size_t ns = 0;
	for (size_t i = 0; i < n;) {
		unsigned char c = (unsigned char)text[i];
		size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
		if (i + len > n) len = 1;
		const char *piece = text + i;
		static const char sp[] = "\xE2\x96\x81";			// U+2581
		if (c == ' ') { piece = sp; len = 3; i += 1; }
		else i += len;
		int id = eg2t_find(t, piece, len);
		if (id >= 0) sym[ns++] = id;
		else for (size_t b = 0; b < len; b++) {		// byte fallback
			int bid = t->byte_id[(unsigned char)piece[b]];
			sym[ns++] = bid >= 0 ? bid : 3;
		}
	}
	while (ns > 1) {					// merge the best-ranked adjacent pair until none is left
		uint32_t best = 0xffffffffu, res = 0;
		size_t at = 0;
		for (size_t i = 0; i + 1 < ns; i++) {
			uint32_t r, rr;
			if (eg2t_merge_lookup(t, (uint32_t)sym[i], (uint32_t)sym[i + 1], &r, &rr) && r < best) { best = r; res = rr; at = i; }
		}
		if (best == 0xffffffffu) break;
		sym[at] = (int)res;
		memmove(sym + at + 1, sym + at + 2, (ns - at - 2) * sizeof(int));
		ns--;
	}
	int out = 0;
	if (out < cap) ids[out++] = 2;				// <bos>
	for (size_t i = 0; i < ns && out < cap - 1; i++) ids[out++] = sym[i];
	if (out < cap) ids[out++] = 1;				// <eos>
	free(sym);
	return out;
}

#endif /* EG2_TOKENIZER_H */
