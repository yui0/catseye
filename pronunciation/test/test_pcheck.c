#include <stdio.h>
#include "pcheck.h"

static const char *syms[] = {"<pad>","h","ɛ","l","oʊ","w","ɚ","d","ð","t","θ","ɪ","s","ʌ","n","ə","k","ʃ","ˈ"};
#define NS ((int)(sizeof syms / sizeof syms[0]))

static cpa_vocab vocab(void)
{
	cpa_vocab v; v.n = NS; v.sym = (char **)malloc(sizeof(char *) * NS);
	for (int i = 0; i < NS; i++) v.sym[i] = strdup(syms[i]);
	return v;
}

/* fake recognizer output: each phoneme gets 5 confident frames followed by a blank frame */
static float *logits(const cpa_vocab *v, const char *ipa, int *T)
{
	int ids[64], n = cpa_tokenize(v, ipa, ids, 64);
	*T = n * 6;
	float *x = (float *)calloc((size_t)*T * v->n, sizeof(float));
	for (int i = 0; i < n; i++) for (int f = 0; f < 6; f++) {
		float *r = x + (size_t)(i * 6 + f) * v->n;
		for (int k = 0; k < v->n; k++) r[k] = -4.0f;
		r[f == 5 ? 0 : ids[i]] = 6.0f;
	}
	return x;
}

static int run(const char *text, const char *heard, cpc_report *r, const cpa_vocab *v)
{
	int T; float *x = logits(v, heard, &T);
	int rc = cpc_check(CP_EN, text, v, x, T, r);
	free(x);
	return rc;
}

int main(void)
{
	cpa_vocab v = vocab(); cpc_report r; int fails = 0;
	/* "hello" -> hɛˈloʊ */
	run("hello", "hɛloʊ", &r, &v);
	if (r.overall < 90) { printf("FAIL perfect overall %.0f\n", r.overall); fails++; }
	for (int i = 0; i < r.ntok; i++) if (r.tok[i].status != CPA_OK) { puts("FAIL perfect has errors"); fails++; }
	cpc_free(&r);
	/* l heard as n */
	run("hello", "hɛnoʊ", &r, &v);
	if (r.tok[2].status != CPA_SUB || r.tok[2].heard != 14 || r.overall > 90) { printf("FAIL substitution (status %d heard %d)\n", r.tok[2].status, r.tok[2].heard); fails++; }
	cpc_free(&r);
	/* l dropped */
	run("hello", "hɛoʊ", &r, &v);
	if (r.tok[2].status != CPA_DEL || r.words[0].score > 85) { puts("FAIL deletion"); fails++; }
	cpc_free(&r);
	/* extra s */
	run("hello", "hɛsloʊ", &r, &v);
	if (r.nins != 1 || r.ins_id[0] != 12) { puts("FAIL insertion"); fails++; }
	cpc_free(&r);
	/* two words, only the second is wrong: "the" ð ə, "thin"? use "this" ð ɪ s vs t ɪ s */
	run("the this", "ðət" "ɪs", &r, &v);
	if (r.nwords != 2 || r.words[0].score < 90 || r.words[1].score > 80) { printf("FAIL per-word (%.0f %.0f)\n", r.words[0].score, r.words[1].score); fails++; }
	cpc_free(&r);
	if (fails) return 1;
	puts("pcheck ok");
	return 0;
}
