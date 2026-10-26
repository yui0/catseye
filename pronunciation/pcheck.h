/*
 * pcheck.h - phoneme-level pronunciation report.
 *
 * Given the expected text and CTC phoneme posteriors for the learner's speech,
 * cpc_check() aligns the expected phonemes (from cp_g2p) to the audio, scores
 * each one (goodness of pronunciation) and compares them with what the
 * recognizer heard, labelling every phoneme ok / weak / substituted / deleted.
 */
#ifndef CATS_PCHECK_H
#define CATS_PCHECK_H

#include <stdio.h>
#include "latphon.h"
#include "align.h"

#define CPC_WEAK 60.0f	/* below this a matching phoneme is reported as "weak" */

typedef struct {
	char word[128], ipa[256];
	int first, count;	/* tokens in cpc_report.tok */
	float score;
} cpc_word;

typedef struct {
	int id, word;		/* reference phoneme id (vocab) and owning word */
	int status;		/* CPA_OK / CPA_SUB / CPA_DEL */
	int heard;		/* vocab id of what was heard for SUB, else -1 */
	int weak;
	float score;
	int start, end;		/* frames */
} cpc_tok;

typedef struct {
	cpc_word *words; int nwords;
	cpc_tok *tok; int ntok;
	int *ins_after; int *ins_id; int nins;	/* inserted phonemes: heard id, placed after tok index */
	float overall;
	int aligned;		/* 0 if forced alignment failed (audio too short) */
} cpc_report;

static inline void cpc_free(cpc_report *r)
{
	free(r->words); free(r->tok); free(r->ins_after); free(r->ins_id);
	memset(r, 0, sizeof *r);
}

/* lp: T x V raw logits (modified in place into log posteriors) */
static inline int cpc_check(int lang, const char *text, const cpa_vocab *v, float *lp, int T, cpc_report *rep)
{
	memset(rep, 0, sizeof *rep);
	size_t nl = strlen(text) * 4 + 64;
	char *norm = (char *)malloc(nl);
	cp_normalize(lang, text, norm, nl);
	int wcap = 64, *ids = NULL, ntok = 0, icap = 0;
	rep->words = (cpc_word *)calloc(wcap, sizeof(cpc_word));
	for (char *p = norm; *p;) {
		while (*p == ' ') p++;
		if (!*p) break;
		char *e = p; while (*e && *e != ' ') e++;
		size_t wl = (size_t)(e - p);
		if (wl == 1 && strchr(",.?!", *p)) { p = e; continue; }
		if (rep->nwords == wcap) { wcap *= 2; rep->words = (cpc_word *)realloc(rep->words, wcap * sizeof(cpc_word)); }
		cpc_word *w = &rep->words[rep->nwords++];
		memset(w, 0, sizeof *w);
		if (wl > 120) wl = 120;
		memcpy(w->word, p, wl);
		cp_g2p(lang, w->word, w->ipa, sizeof w->ipa);
		int tmp[128], n = cpa_tokenize(v, w->ipa, tmp, 128);
		if (ntok + n > icap) { icap = (ntok + n) * 2 + 64; ids = (int *)realloc(ids, icap * sizeof(int)); }
		w->first = ntok; w->count = n;
		memcpy(ids + ntok, tmp, n * sizeof(int)); ntok += n;
		p = e;
	}
	free(norm);
	if (!ntok) { free(ids); return -1; }
	int V = v->n;
	cpa_mask(v, lp, T);
	cpa_logsoftmax(lp, T, V);
	cpa_seg *seg = (cpa_seg *)calloc(ntok, sizeof(cpa_seg));
	rep->aligned = cpa_force_align(lp, T, V, ids, ntok, seg) == 0;
	int *hyp = (int *)malloc(sizeof(int) * (T + 1));
	int nh = cpa_greedy(lp, T, V, hyp, T);
	/* compare in single-segment units; every unit remembers the reference token it came from */
	int *ru = (int *)malloc(sizeof(int) * ntok * 4), *ro = (int *)malloc(sizeof(int) * ntok * 4), nru = 0;
	for (int i = 0; i < ntok; i++) { int u[8], n = cpa_units(v, ids[i], u, 8); for (int k = 0; k < n; k++) { ru[nru] = u[k]; ro[nru++] = i; } }
	int *hu = (int *)malloc(sizeof(int) * (nh + 1) * 4), nhu = 0;
	for (int i = 0; i < nh; i++) { int u[8], n = cpa_units(v, hyp[i], u, 8); for (int k = 0; k < n; k++) hu[nhu++] = u[k]; }
	cpa_edit *ed = (cpa_edit *)malloc(sizeof(cpa_edit) * (nru + nhu + 1));
	int ne = cpa_edit_align(ru, nru, hu, nhu, ed);
	rep->tok = (cpc_tok *)calloc(ntok, sizeof(cpc_tok)); rep->ntok = ntok;
	rep->ins_after = (int *)malloc(sizeof(int) * (nhu + 1)); rep->ins_id = (int *)malloc(sizeof(int) * (nhu + 1));
	for (int w = 0; w < rep->nwords; w++) for (int k = 0; k < rep->words[w].count; k++) rep->tok[rep->words[w].first + k].word = w;
	int last_ref = -1;
	for (int i = 0; i < ntok; i++) { rep->tok[i].id = ids[i]; rep->tok[i].heard = -1; rep->tok[i].score = rep->aligned ? seg[i].score : 0; rep->tok[i].start = seg[i].start; rep->tok[i].end = seg[i].end; }
	for (int k = 0; k < ne; k++) {
		if (ed[k].op == CPA_INS) { rep->ins_after[rep->nins] = last_ref; rep->ins_id[rep->nins++] = hu[ed[k].hyp]; continue; }
		int ti = ro[ed[k].ref]; cpc_tok *t = &rep->tok[ti]; last_ref = ti;
		if (ed[k].op == CPA_OK) continue;
		if (ed[k].op == CPA_SUB) { t->status = CPA_SUB; t->heard = hu[ed[k].hyp]; }
		else if (t->status != CPA_SUB) t->status = CPA_DEL;
	}
	for (int i = 0; i < ntok; i++) if (rep->tok[i].status == CPA_OK && rep->tok[i].score < CPC_WEAK) rep->tok[i].weak = 1;
	double all = 0;
	for (int w = 0; w < rep->nwords; w++) {
		cpc_word *wd = &rep->words[w]; double s = 0;
		for (int k = 0; k < wd->count; k++) {
			cpc_tok *t = &rep->tok[wd->first + k];
			float sc = t->status == CPA_DEL ? 0 : t->status == CPA_SUB ? (t->score < 40 ? t->score : 40) : t->score;
			t->score = sc; s += sc; all += sc;
		}
		wd->score = wd->count ? (float)(s / wd->count) : 0;
	}
	rep->overall = (float)(all / ntok);
	free(ids); free(seg); free(hyp); free(ed); free(ru); free(ro); free(hu);
	return 0;
}

static inline const char *cpc_status(const cpc_tok *t)
{
	return t->status == CPA_DEL ? "deleted" : t->status == CPA_SUB ? "substituted" : t->weak ? "weak" : "ok";
}

static inline void cpc_print(const cpc_report *r, const cpa_vocab *v, FILE *f)
{
	fprintf(f, "overall: %.0f / 100%s\n", r->overall, r->aligned ? "" : "  (audio too short to align)");
	int ins = 0;
	for (int w = 0; w < r->nwords; w++) {
		const cpc_word *wd = &r->words[w];
		fprintf(f, "\n%-14s /%s/  %.0f\n", wd->word, wd->ipa, wd->score);
		for (int k = 0; k < wd->count; k++) {
			int ti = wd->first + k; const cpc_tok *t = &r->tok[ti];
			if (t->status == CPA_OK && !t->weak) continue;
			if (t->status == CPA_SUB) fprintf(f, "  %s -> %s  substituted  (%.0f)\n", v->sym[t->id], v->sym[t->heard], t->score);
			else fprintf(f, "  %s  %s  (%.0f)\n", v->sym[t->id], cpc_status(t), t->score);
		}
		for (; ins < r->nins && r->ins_after[ins] < wd->first + wd->count; ins++)
			if (r->ins_after[ins] >= wd->first - 0) fprintf(f, "  +%s  inserted\n", v->sym[r->ins_id[ins]]);
	}
}

static inline void cpc_json(const cpc_report *r, const cpa_vocab *v, FILE *f)
{
	fprintf(f, "{\"overall\":%.1f,\"words\":[", r->overall);
	for (int w = 0; w < r->nwords; w++) {
		const cpc_word *wd = &r->words[w];
		fprintf(f, "%s{\"word\":\"%s\",\"ipa\":\"%s\",\"score\":%.1f,\"phonemes\":[", w ? "," : "", wd->word, wd->ipa, wd->score);
		for (int k = 0; k < wd->count; k++) {
			const cpc_tok *t = &r->tok[wd->first + k];
			fprintf(f, "%s{\"p\":\"%s\",\"status\":\"%s\",\"score\":%.1f", k ? "," : "", v->sym[t->id], cpc_status(t), t->score);
			if (t->heard >= 0) fprintf(f, ",\"heard\":\"%s\"", v->sym[t->heard]);
			fputc('}', f);
		}
		fputs("]}", f);
	}
	fputs("]}\n", f);
}

#endif
