/*
 * align.h - phoneme-level alignment and scoring for the pronunciation checker.
 *
 *   cpa_logsoftmax      raw CTC logits -> log posteriors
 *   cpa_force_align     CTC forced alignment of a reference phoneme sequence
 *   cpa_greedy          greedy CTC decode (what the recognizer "heard")
 *   cpa_edit_align      Levenshtein alignment of reference vs heard phonemes
 *
 * Posteriors are T x V row-major floats; blank is id 0 (wav2vec2 <pad>).
 */
#ifndef CATS_ALIGN_H
#define CATS_ALIGN_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

typedef struct { char **sym; int n; } cpa_vocab;

static inline void cpa_vocab_free(cpa_vocab *v)
{
	for (int i = 0; i < v->n; i++) free(v->sym[i]);
	free(v->sym); v->sym = NULL; v->n = 0;
}

/* one symbol per line; line number = id; id 0 is the blank */
static inline int cpa_vocab_load(cpa_vocab *v, const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) return -1;
	char line[256]; int cap = 0; v->n = 0; v->sym = NULL;
	while (fgets(line, sizeof line, f)) {
		size_t l = strlen(line);
		while (l && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
		if (v->n == cap) { cap = cap ? cap * 2 : 128; v->sym = (char **)realloc(v->sym, cap * sizeof(char *)); }
		v->sym[v->n++] = strdup(line);
	}
	fclose(f);
	return v->n ? 0 : -1;
}

/* silence symbols a recognizer should never use for en/pt: special tokens, Mandarin tone digits, '.', '?' */
static inline void cpa_mask(const cpa_vocab *v, float *x, int T)
{
	for (int i = 1; i < v->n; i++) {
		const char *s = v->sym[i]; int bad = (s[0] == '<' && s[1]);
		for (; *s; s++) if ((*s >= '0' && *s <= '9') || *s == '.' || *s == '?') bad = 1;
		if (bad) for (int t = 0; t < T; t++) x[(size_t)t * v->n + i] = -1e4f;
	}
}

static inline void cpa_logsoftmax(float *x, int T, int V)
{
	for (int t = 0; t < T; t++) {
		float *r = x + (size_t)t * V, m = r[0], s = 0;
		for (int v = 1; v < V; v++) if (r[v] > m) m = r[v];
		for (int v = 0; v < V; v++) s += expf(r[v] - m);
		float ls = m + logf(s);
		for (int v = 0; v < V; v++) r[v] -= ls;
	}
}

/*
 * Comparison units. Recognizers emit compound symbols (ɑːɹ, ɛɹ, əl, long vowels);
 * both sides are reduced to single-segment units so they can be compared:
 * length marks are dropped and compounds are split, except affricates and
 * English diphthongs which stay whole. Writes vocab ids to out, returns count.
 */
static inline int cpa_find(const cpa_vocab *v, const char *s, int l)
{
	for (int i = 1; i < v->n; i++) if ((int)strlen(v->sym[i]) == l && !strncmp(v->sym[i], s, l)) return i;
	return -1;
}

static inline int cpa_units(const cpa_vocab *v, int id, int *out, int cap)
{
	static const char *keep[] = {"a\xc9\xaa", "e\xc9\xaa", "o\xca\x8a", "a\xca\x8a", "\xc9\x94\xc9\xaa", "t\xca\x83", "d\xca\x92", "ts", "t\xc9\x95", NULL};
	char base[64]; int bl = 0;
	for (const char *p = v->sym[id]; *p && bl < 60;) {
		/* length marks and nasalization are not compared: recognizers spell nasal vowels inconsistently */
		if (!strncmp(p, "\xcb\x90", 2) || !strncmp(p, "\xcc\x83", 2)) { p += 2; continue; }
		static const char *pre[][2] = {{"\xc3\xa3", "a"}, {"\xe1\xba\xbd", "e"}, {"\xc4\xa9", "i"}, {"\xc3\xb5", "o"}, {"\xc5\xa9", "u"}};
		int hit = 0;
		for (int k = 0; k < 5 && !hit; k++) { int l = (int)strlen(pre[k][0]); if (!strncmp(p, pre[k][0], l)) { base[bl++] = pre[k][1][0]; p += l; hit = 1; } }
		if (hit) continue;
		base[bl++] = *p++;
	}
	base[bl] = 0;
	for (int k = 0; keep[k]; k++) if (!strcmp(base, keep[k])) { int b = cpa_find(v, base, bl); out[0] = b > 0 ? b : id; return 1; }
	int n = 0;
	for (const char *p = base; *p && n < cap;) {
		const char *q = p; q++;
		while ((*q & 0xC0) == 0x80) q++;
		/* combining marks and modifier letters belong to the previous segment */
		while (!strncmp(q, "\xcc\x83", 2) || !strncmp(q, "\xca\xb2", 2) || !strncmp(q, "\xcc\xaf", 2)) q += 2;
		int u = cpa_find(v, p, (int)(q - p));
		if (u < 0) { out[0] = id; return 1; }
		out[n++] = u; p = q;
	}
	if (!n) { out[0] = id; return 1; }
	return n;
}

/* longest-match tokenization of an IPA word into vocab ids; stress marks are skipped.
 * Precomposed nasal vowels fall back to base vowel + U+0303. Returns count. */
static inline int cpa_tokenize(const cpa_vocab *v, const char *ipa, int *ids, int cap)
{
	static const char *nas[][2] = {{"\xc3\xa3", "a\xcc\x83"}, {"\xe1\xba\xbd", "e\xcc\x83"}, {"\xc4\xa9", "i\xcc\x83"}, {"\xc3\xb5", "o\xcc\x83"}, {"\xc5\xa9", "u\xcc\x83"}};
	int n = 0;
	for (const char *p = ipa; *p && n < cap;) {
		if (!strncmp(p, "\xcb\x88", 2) || !strncmp(p, "\xcb\x8c", 2)) { p += 2; continue; }
		if (*p == ' ' || *p == ',' || *p == '.' || *p == '?' || *p == '!') { p++; continue; }
		int best = -1, bl = 0;
		if (*p == 'g') { int g = cpa_find(v, "\xc9\xa1", 2); if (g > 0) { ids[n++] = g; p++; continue; } }
		for (int i = 1; i < v->n; i++) {
			int l = (int)strlen(v->sym[i]);
			if (l > bl && !strncmp(p, v->sym[i], l)) { best = i; bl = l; }
		}
		if (best >= 0) { ids[n++] = best; p += bl; continue; }
		int done = 0;
		for (int k = 0; k < 5 && !done; k++) {
			int l = (int)strlen(nas[k][0]);
			if (strncmp(p, nas[k][0], l)) continue;
			for (int i = 1; i < v->n; i++) {
				if (!strcmp(v->sym[i], nas[k][1])) { ids[n++] = i; done = 1; break; }
			}
			if (!done) for (int i = 1; i < v->n; i++) if (!strncmp(v->sym[i], nas[k][1], 1) && strlen(v->sym[i]) == 1) { ids[n++] = i; done = 1; break; }
			if (done) { p += l; }
		}
		if (done) continue;
		/* unknown symbol: skip one code point */
		p++; while ((*p & 0xC0) == 0x80) p++;
	}
	return n;
}

typedef struct { int start, end; float gop; float score; } cpa_seg;

/*
 * CTC forced alignment (Viterbi). lp: T x V log posteriors, tok: L ids (non-blank).
 * seg[i] receives the frames assigned to token i (end exclusive), the goodness of
 * pronunciation gop = mean over those frames of (lp[tok] - max_v lp[v]) <= 0, and
 * score = 100 * exp(gop). Returns 0, or -1 if the audio is too short for the tokens.
 */
static inline int cpa_force_align(const float *lp, int T, int V, const int *tok, int L, cpa_seg *seg)
{
	if (!L) return 0;
	int S = 2 * L + 1;
	if (T < L) return -1;
	float *dp = (float *)malloc(sizeof(float) * (size_t)T * S);
	unsigned char *bp = (unsigned char *)malloc((size_t)T * S);
	if (!dp || !bp) { free(dp); free(bp); return -1; }
	for (size_t i = 0; i < (size_t)T * S; i++) dp[i] = -FLT_MAX;
	#define EMIT(t, s) (lp[(size_t)(t) * V + (((s) & 1) ? tok[(s) / 2] : 0)])
	dp[0] = EMIT(0, 0); bp[0] = 0;
	if (S > 1) { dp[1] = EMIT(0, 1); bp[1] = 0; }
	for (int t = 1; t < T; t++) {
		for (int s = 0; s < S; s++) {
			float best = dp[(size_t)(t - 1) * S + s]; unsigned char from = 0;
			if (s >= 1 && dp[(size_t)(t - 1) * S + s - 1] > best) { best = dp[(size_t)(t - 1) * S + s - 1]; from = 1; }
			if (s >= 2 && (s & 1) && tok[s / 2] != tok[s / 2 - 1] && dp[(size_t)(t - 1) * S + s - 2] > best) { best = dp[(size_t)(t - 1) * S + s - 2]; from = 2; }
			if (best <= -FLT_MAX / 2) continue;
			dp[(size_t)t * S + s] = best + EMIT(t, s); bp[(size_t)t * S + s] = from;
		}
	}
	int s = S - 1;
	if (S > 1 && dp[(size_t)(T - 1) * S + S - 2] > dp[(size_t)(T - 1) * S + S - 1]) s = S - 2;
	if (dp[(size_t)(T - 1) * S + s] <= -FLT_MAX / 2) { free(dp); free(bp); return -1; }
	for (int i = 0; i < L; i++) { seg[i].start = T; seg[i].end = 0; seg[i].gop = 0; seg[i].score = 0; }
	int *state = (int *)malloc(sizeof(int) * T);
	for (int t = T - 1; t >= 0; t--) { state[t] = s; s -= bp[(size_t)t * S + s]; }
	for (int t = 0; t < T; t++) {
		int st = state[t];
		if (!(st & 1)) continue;
		int i = st / 2; const float *r = lp + (size_t)t * V; float m = r[0];
		for (int v = 1; v < V; v++) if (r[v] > m) m = r[v];
		if (t < seg[i].start) seg[i].start = t;
		if (t + 1 > seg[i].end) seg[i].end = t + 1;
		seg[i].gop += r[tok[i]] - m;
	}
	for (int i = 0; i < L; i++) {
		int n = seg[i].end - seg[i].start;
		if (n > 0) seg[i].gop /= n;
		seg[i].score = 100.0f * expf(seg[i].gop);
	}
	free(state); free(dp); free(bp);
	#undef EMIT
	return 0;
}

/* greedy CTC decode: collapse repeats, drop blanks. Returns token count. */
static inline int cpa_greedy(const float *lp, int T, int V, int *ids, int cap)
{
	int n = 0, prev = -1;
	for (int t = 0; t < T; t++) {
		const float *r = lp + (size_t)t * V; int b = 0;
		for (int v = 1; v < V; v++) if (r[v] > r[b]) b = v;
		if (b != prev && b != 0 && n < cap) ids[n++] = b;
		prev = b;
	}
	return n;
}

enum { CPA_OK, CPA_SUB, CPA_DEL, CPA_INS };
typedef struct { int op, ref, hyp; } cpa_edit;	/* ref/hyp are indices into the sequences, -1 if absent */

/* Levenshtein alignment; returns number of edit entries written to out (cap >= nr + nh) */
static inline int cpa_edit_align(const int *ref, int nr, const int *hyp, int nh, cpa_edit *out)
{
	int *d = (int *)malloc(sizeof(int) * (nr + 1) * (nh + 1));
	#define D(i, j) d[(i) * (nh + 1) + (j)]
	for (int i = 0; i <= nr; i++) D(i, 0) = i;
	for (int j = 0; j <= nh; j++) D(0, j) = j;
	for (int i = 1; i <= nr; i++) for (int j = 1; j <= nh; j++) {
		int c = D(i - 1, j - 1) + (ref[i - 1] != hyp[j - 1]);
		if (D(i - 1, j) + 1 < c) c = D(i - 1, j) + 1;
		if (D(i, j - 1) + 1 < c) c = D(i, j - 1) + 1;
		D(i, j) = c;
	}
	int n = 0, i = nr, j = nh;
	while (i > 0 || j > 0) {
		cpa_edit e;
		if (i > 0 && j > 0 && D(i, j) == D(i - 1, j - 1) + (ref[i - 1] != hyp[j - 1])) {
			e.op = ref[i - 1] == hyp[j - 1] ? CPA_OK : CPA_SUB; e.ref = i - 1; e.hyp = j - 1; i--; j--;
		} else if (i > 0 && D(i, j) == D(i - 1, j) + 1) { e.op = CPA_DEL; e.ref = i - 1; e.hyp = -1; i--; }
		else { e.op = CPA_INS; e.ref = -1; e.hyp = j - 1; j--; }
		out[n++] = e;
	}
	#undef D
	free(d);
	for (int a = 0, b = n - 1; a < b; a++, b--) { cpa_edit t = out[a]; out[a] = out[b]; out[b] = t; }
	return n;
}

#endif
