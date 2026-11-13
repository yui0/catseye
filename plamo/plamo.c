//---------------------------------------------------------
//	Cat's eye
//
//		©2026 Yuichiro Nakada
//---------------------------------------------------------

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <ctype.h>
#include <time.h>
#include <unistd.h>
#define CATS_INFER_ONLY
#include "../catseye.h"

#define PLM_TOK_BOS	1
#define PLM_TOK_EOS	2
#define PLM_TOK_TAG	16	// <|plamo:tag|> also ends a translation

typedef struct {
	int hidden, inter, layers, nq, nkv, hd, vocab, window, pattern;
	float eps, theta_full, theta_local, yarn_factor, yarn_orig, beta_fast, beta_slow;
} plm_cfg;

typedef struct {
	const cats_st_tensor *qkv, *o, *gu, *down;
	float *qn, *kn, *pre_mix, *post_mix, *pre_mlp, *post_mlp;
} plm_layer;

typedef struct {
	plm_cfg c;
	cats_st st;
	cats_unigram tok;
	plm_layer *L;
	const cats_st_tensor *embed, *head;
	float *final_norm;
	float *inv_full, *inv_local;
	float conc_full;
	float *kc, *vc;			// [layer][ctx][hd * nkv]
	int ctx;
	float *logits;
} plm_model;

//---------------------------------------------------------
// config.json: only flat numeric keys are needed
static int json_num(const char *j, const char *key, double *v)
{
	char k[96];
	snprintf(k, sizeof(k), "\"%s\"", key);
	const char *p = strstr(j, k);
	if (!p) return -1;
	p = strchr(p + strlen(k), ':');
	if (!p) return -1;
	char *e;
	double x = strtod(p + 1, &e);
	if (e == p + 1) return -1;
	*v = x;
	return 0;
}
static char *read_file(const char *path, size_t *size)
{
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *b = (char*)malloc((size_t)n + 1);
	if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
	b[n] = 0;
	fclose(f);
	if (size) *size = (size_t)n;
	return b;
}
static int plm_config(plm_cfg *c, const char *path)
{
	char *j = read_file(path, NULL);
	if (!j) return -1;
	double v;
#define NUM(field, key, type, def) do { c->field = json_num(j, key, &v) ? (type)(def) : (type)v; } while (0)
	NUM(hidden, "hidden_size", int, 1536);
	NUM(inter, "intermediate_size", int, 6144);
	NUM(layers, "num_hidden_layers", int, 16);
	NUM(nq, "num_attention_heads", int, 16);
	NUM(nkv, "num_key_value_heads", int, 1);
	NUM(hd, "head_dim", int, 128);
	NUM(vocab, "vocab_size", int, 107520);
	NUM(window, "window_size", int, 2048);
	NUM(pattern, "sliding_window_pattern", int, 8);
	NUM(eps, "rms_norm_eps", float, 1e-6);
	NUM(theta_full, "rope_theta", float, 1000000);
	NUM(theta_local, "rope_local_theta", float, 10000);
	NUM(yarn_factor, "rope_scaling_factor", float, 1);
	NUM(yarn_orig, "original_max_position_embeddings", float, 4096);
	NUM(beta_fast, "beta_fast", float, 32);
	NUM(beta_slow, "beta_slow", float, 1);
#undef NUM
	free(j);
	return 0;
}

//---------------------------------------------------------
static const cats_st_tensor *need(const cats_st *st, const char *fmt, int l)
{
	char name[192];
	snprintf(name, sizeof(name), fmt, l);
	const cats_st_tensor *t = cats_st_find(st, name);
	if (!t) fprintf(stderr, "missing tensor: %s\n", name);
	return t;
}
static float *need_f32(const cats_st *st, const char *fmt, int l)
{
	const cats_st_tensor *t = need(st, fmt, l);
	if (!t) return NULL;
	float *w = (float*)malloc((size_t)cats_st_numel(t) * sizeof(float));
	if (!w || cats_st_read_f32(st, t, w)) { free(w); return NULL; }
	return w;
}

static void plm_free(plm_model *m)
{
	if (m->L) for (int l = 0; l < m->c.layers; l++) {
		free(m->L[l].qn); free(m->L[l].kn); free(m->L[l].pre_mix); free(m->L[l].post_mix);
		free(m->L[l].pre_mlp); free(m->L[l].post_mlp);
	}
	free(m->L); free(m->final_norm); free(m->inv_full); free(m->inv_local); free(m->kc); free(m->vc); free(m->logits);
	cats_unigram_free(&m->tok);
	cats_st_close(&m->st);
	memset(m, 0, sizeof(*m));
}

static int plm_load(plm_model *m, const char *dir, int ctx)
{
	memset(m, 0, sizeof(*m));
	char path[1024];
	snprintf(path, sizeof(path), "%s/config.json", dir);
	if (plm_config(&m->c, path)) { fprintf(stderr, "cannot read %s\n", path); return -1; }
	snprintf(path, sizeof(path), "%s/model.safetensors", dir);
	if (cats_st_open(&m->st, path)) { fprintf(stderr, "cannot open %s\n", path); return -1; }
	snprintf(path, sizeof(path), "%s/tokenizer.jsonl", dir);
	if (cats_unigram_load(&m->tok, path)) { fprintf(stderr, "cannot read %s\n", path); return -1; }
	m->tok.break_spaces = 2;	// tokenizer_config.json: break_around_consecutive_spaces_threshold
	m->tok.break_repeat = 4;	//                         break_around_repeated_chars_threshold

	const plm_cfg *c = &m->c;
	m->L = (plm_layer*)calloc((size_t)c->layers, sizeof(plm_layer));
	for (int l = 0; l < c->layers; l++) {
		plm_layer *y = &m->L[l];
		char fmt[160];
#define T(field, suffix) do { snprintf(fmt, sizeof(fmt), "model.layers.layers.%%d." suffix); y->field = need(&m->st, fmt, l); if (!y->field) return -1; } while (0)
#define F(field, suffix) do { snprintf(fmt, sizeof(fmt), "model.layers.layers.%%d." suffix); y->field = need_f32(&m->st, fmt, l); if (!y->field) return -1; } while (0)
		T(qkv, "mixer.qkv_proj.weight"); T(o, "mixer.o_proj.weight");
		T(gu, "mlp.gate_up_proj.weight"); T(down, "mlp.down_proj.weight");
		F(qn, "mixer.q_norm.weight"); F(kn, "mixer.k_norm.weight");
		F(pre_mix, "pre_mixer_norm.weight"); F(post_mix, "post_mixer_norm.weight");
		F(pre_mlp, "pre_mlp_norm.weight"); F(post_mlp, "post_mlp_norm.weight");
#undef T
#undef F
	}
	m->embed = cats_st_find(&m->st, "model.embed_tokens.weight");
	m->head = cats_st_find(&m->st, "lm_head.weight");
	m->final_norm = need_f32(&m->st, "model.norm.weight%.0d", 0);
	if (!m->embed || !m->head || !m->final_norm) { fprintf(stderr, "missing embedding / head / final norm\n"); return -1; }
	if (m->embed->shape[1] != c->hidden || m->head->shape[1] != c->hidden) { fprintf(stderr, "weights do not match config\n"); return -1; }

	m->inv_full = (float*)malloc((size_t)c->hd / 2 * sizeof(float));
	m->inv_local = (float*)malloc((size_t)c->hd / 2 * sizeof(float));
	m->conc_full = cats_rope_inv_freq_yarn(m->inv_full, (unsigned)c->hd, c->theta_full, c->yarn_factor, c->yarn_orig, c->beta_fast, c->beta_slow);
	cats_rope_inv_freq_yarn(m->inv_local, (unsigned)c->hd, c->theta_local, 1.0f, 0, 0, 0);

	m->ctx = ctx;
	size_t kv = (size_t)c->layers * ctx * c->nkv * c->hd;
	m->kc = (float*)malloc(kv * sizeof(float));
	m->vc = (float*)malloc(kv * sizeof(float));
	m->logits = (float*)malloc((size_t)m->head->shape[0] * sizeof(float));
	return m->kc && m->vc && m->logits ? 0 : -1;
}

//---------------------------------------------------------
// Runs n tokens at positions pos0.. through the decoder and leaves the logits of the last one in m->logits.
static int plm_forward(plm_model *m, const int *ids, int n, int pos0)
{
	const plm_cfg *c = &m->c;
	const int H = c->hidden, qd = c->nq * c->hd, kd = c->nkv * c->hd, I = c->inter, hd = c->hd;
	const float scale = 1.0f / sqrtf((float)hd);
	if (pos0 + n > m->ctx) return -1;
	float *x = (float*)malloc((size_t)n * H * 4), *h = (float*)malloc((size_t)n * H * 4);
	float *qkv = (float*)malloc((size_t)n * (qd + 2 * kd) * 4), *att = (float*)malloc((size_t)n * qd * 4);
	float *gu = (float*)malloc((size_t)n * 2 * I * 4), *sc = (float*)malloc((size_t)c->nq * (pos0 + n) * 4);
	float *cs = (float*)malloc((size_t)hd * 4), *sn = (float*)malloc((size_t)hd * 4);
	int rc = -1;
	if (!x || !h || !qkv || !att || !gu || !sc || !cs || !sn) goto done;

	for (int i = 0; i < n; i++) if (cats_st_read_row_f32(&m->st, m->embed, ids[i], x + (size_t)i * H)) goto done;

	for (int l = 0; l < c->layers; l++) {
		const plm_layer *y = &m->L[l];
		const int full = !((l + 1) % c->pattern);
		const float *inv = full ? m->inv_full : m->inv_local;
		const float conc = full ? m->conc_full : 1.0f;
		float *kc = m->kc + (size_t)l * m->ctx * kd, *vc = m->vc + (size_t)l * m->ctx * kd;

		for (int i = 0; i < n; i++) cats_rmsnorm_offset(h + (size_t)i * H, x + (size_t)i * H, y->pre_mix, 1.0f, H, c->eps);
		if (cats_st_gemm_nt(qkv, h, &m->st, y->qkv, (size_t)n, NULL)) goto done;
		for (int i = 0; i < n; i++) {
			float *q = qkv + (size_t)i * (qd + 2 * kd), *k = q + qd, *v = k + kd;
			const int pos = pos0 + i;
			cats_rope_pos_table(cs, sn, (size_t)pos, inv, conc, (unsigned)hd);
			for (int hh = 0; hh < c->nq; hh++) {
				cats_rmsnorm_offset(q + hh * hd, q + hh * hd, y->qn, 1.0f, (unsigned)hd, c->eps);
				cats_rope_block(q + hh * hd, cs, sn, (unsigned)hd);
			}
			for (int hh = 0; hh < c->nkv; hh++) {
				cats_rmsnorm_offset(k + hh * hd, k + hh * hd, y->kn, 1.0f, (unsigned)hd, c->eps);
				cats_rope_block(k + hh * hd, cs, sn, (unsigned)hd);
			}
			memcpy(kc + (size_t)pos * kd, k, (size_t)kd * 4);
			memcpy(vc + (size_t)pos * kd, v, (size_t)kd * 4);
			cats_attention_cached(att + (size_t)i * qd, q, kc, vc, (size_t)pos, (unsigned)c->nq, (unsigned)c->nkv,
					      (unsigned)hd, scale, full ? 0 : (size_t)c->window, sc);
		}
		if (cats_st_gemm_nt(h, att, &m->st, y->o, (size_t)n, NULL)) goto done;	// h: attention output
		for (int i = 0; i < n; i++) {
			float *o = h + (size_t)i * H, *xi = x + (size_t)i * H;
			cats_rmsnorm_offset(o, o, y->post_mix, 0.2f, H, c->eps);
			for (int j = 0; j < H; j++) xi[j] += o[j];
			cats_rmsnorm_offset(o, xi, y->pre_mlp, 1.0f, H, c->eps);
		}
		if (cats_st_gemm_nt(gu, h, &m->st, y->gu, (size_t)n, NULL)) goto done;
		for (int i = 0; i < n; i++) {
			float *g = gu + (size_t)i * 2 * I;
			for (int j = 0; j < I; j++) g[j] = cats_silu(g[j]) * g[I + j];
		}
		// the activated rows are contiguous in gu with stride 2*I; repack so the GEMM sees [n][I]
		for (int i = 0; i < n; i++) memmove(gu + (size_t)i * I, gu + (size_t)i * 2 * I, (size_t)I * 4);
		if (cats_st_gemm_nt(h, gu, &m->st, y->down, (size_t)n, NULL)) goto done;
		for (int i = 0; i < n; i++) {
			float *o = h + (size_t)i * H, *xi = x + (size_t)i * H;
			cats_rmsnorm_offset(o, o, y->post_mlp, 0.08944272f, H, c->eps);	// 1 / 5^1.5
			for (int j = 0; j < H; j++) xi[j] += o[j];
		}
	}
	cats_rmsnorm_offset(h, x + (size_t)(n - 1) * H, m->final_norm, 1.0f, H, c->eps);
	rc = cats_st_gemm_nt(m->logits, h, &m->st, m->head, 1, NULL);
done:
	free(x); free(h); free(qkv); free(att); free(gu); free(sc); free(cs); free(sn);
	return rc;
}

//---------------------------------------------------------
// PFML3 translation prompt (chat_template.jinja of the model)
typedef struct {
	const char *src, *dst, *form, *context;
	const char *glossary[64][2];
	int nglossary;
} plm_req;

static char *plm_prompt(const plm_req *r, const char *text)
{
	size_t cap = strlen(text) + (r->context ? strlen(r->context) : 0) + 1024;
	for (int i = 0; i < r->nglossary; i++) cap += strlen(r->glossary[i][0]) + strlen(r->glossary[i][1]) + 64;
	char *p = (char*)malloc(cap), *w = p;
	w += sprintf(w, "<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation");
	if (r->context && *r->context) w += sprintf(w, "<|plamo:tag|>context<|plamo:msg|>%s", r->context);
	if (r->nglossary) {
		w += sprintf(w, "<|plamo:tag|>glossary");
		for (int i = 0; i < r->nglossary; i++) w += sprintf(w, "<|plamo:index|>%s<|plamo:item|>%s", r->glossary[i][0], r->glossary[i][1]);
	}
	w += sprintf(w, "<|plamo:tag|>input");
	if (r->src) w += sprintf(w, "<|plamo:key|>lang<|plamo:val|>%s", r->src);
	w += sprintf(w, "<|plamo:msg|>%s<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>%s", text, r->dst);
	if (r->form) w += sprintf(w, "<|plamo:key|>form<|plamo:val|>%s", r->form);
	strcpy(w, "<|plamo:msg|>");
	return p;
}

static int valid_lang(const char *s)
{
	size_t n = strlen(s), dash = 0;
	for (size_t i = 0; i < n; i++) {
		if (s[i] == '-') { dash = i; continue; }
		if (s[i] < 'a' || s[i] > 'z') return 0;
	}
	if (!dash) return n == 2 || n == 3;
	return (dash == 2 || dash == 3) && n - dash - 1 == 4 && !strchr(s + dash + 1, '-');
}
static int blank(const char *s)
{
	for (; *s; s++) if (!isspace((unsigned char)*s)) return 0;
	return 1;
}

static double now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec * 1e-9;
}

static int argmax(const float *x, int n)
{
	int b = 0;
	for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
	return b;
}

// Translates one text, streaming to stdout. Returns the number of generated tokens or -1.
static int dump_ids;
static int translate(plm_model *m, const plm_req *r, const char *text, int max_new, int verbose)
{
	if (strstr(text, "<|plamo:")) { fprintf(stderr, "input must not contain reserved <|plamo: tokens\n"); return -1; }
	char *prompt = plm_prompt(r, text);
	int cap = (int)strlen(prompt) + 16;
	int *ids = (int*)malloc((size_t)cap * sizeof(int));
	int n = cats_unigram_encode(&m->tok, prompt, 1, ids, cap);
	free(prompt);
	if (n <= 0 || n + 1 > m->ctx) { fprintf(stderr, "prompt too long (%d tokens, context %d)\n", n, m->ctx); free(ids); return -1; }

	if (dump_ids) {	// prompt token ids, for comparing the tokenizer with the reference implementation
		for (int i = 0; i < n; i++) printf("%d%c", ids[i], i + 1 < n ? ' ' : '\n');
		free(ids);
		return 0;
	}
	double t0 = now();
	const int chunk = 128;
	for (int i = 0; i < n; i += chunk) {
		if (plm_forward(m, ids + i, n - i < chunk ? n - i : chunk, i)) { free(ids); return -1; }
	}
	double t1 = now();

	char buf[4096];
	size_t held = 0;
	int gen = 0, pos = n;
	char tmp[1024];
	while (gen < max_new && pos < m->ctx) {
		int id = argmax(m->logits, (int)m->head->shape[0]);
		if (id == PLM_TOK_BOS || id == PLM_TOK_EOS || id == PLM_TOK_TAG) break;
		size_t w = cats_unigram_piece(&m->tok, id, tmp, sizeof(tmp));
		if (held + w >= sizeof(buf)) { fwrite(buf, 1, held, stdout); held = 0; }
		memcpy(buf + held, tmp, w);
		held += w;
		size_t ok = held;	// hold back an incomplete UTF-8 tail
		for (size_t back = 1; back <= 3 && back <= held; back++) {
			unsigned char ch = (unsigned char)buf[held - back];
			if ((ch & 0xC0) == 0x80) continue;
			if (cats_utf8_len(ch) > (int)back) ok = held - back;
			break;
		}
		fwrite(buf, 1, ok, stdout);
		memmove(buf, buf + ok, held - ok);
		held -= ok;
		fflush(stdout);
		gen++;
		if (plm_forward(m, &id, 1, pos++)) break;
	}
	fwrite(buf, 1, held, stdout);
	putchar('\n');
	double t2 = now();
	if (verbose) fprintf(stderr, "[prompt %d tokens %.2fs (%.1f tok/s), generated %d tokens %.2fs (%.1f tok/s)]\n",
			     n, t1 - t0, n / (t1 - t0), gen, t2 - t1, gen / (t2 - t1 > 1e-9 ? t2 - t1 : 1e-9));
	free(ids);
	return gen;
}

static void usage(const char *a)
{
	fprintf(stderr,
		"PLaMo 3 Translate on Cat's eye\n"
		"Usage: %s -m MODEL_DIR -t TARGET_LANG [options] [-p TEXT | -f FILE]   (default: read stdin)\n"
		"  -m DIR      model directory (model.safetensors, config.json, tokenizer.jsonl)\n"
		"  -t LANG     target language code: en, ja, zh-hans, zh-hant, ko, fr, de, ja-easy, ...\n"
		"  -s LANG     source language (default: auto-detect)\n"
		"  -p TEXT     text to translate; -f FILE reads a file; otherwise stdin\n"
		"  -l          translate every input line separately (default: the whole input is one text)\n"
		"  --polite | --plain   Japanese register (target ja only)\n"
		"  -c TEXT     reference context     -g SRC=TGT   glossary entry (repeatable)\n"
		"  -n N        max new tokens (default 1024)    -C N   context size (default 4096)\n"
		"  -v          timing    --dump-ids   print the prompt token ids and exit\n", a);
}

int main(int argc, char **argv)
{
	const char *dir = NULL, *text = NULL, *file = NULL;
	plm_req r = {0};
	int lines = 0, max_new = 1024, ctx = 4096, verbose = 0;
	for (int i = 1; i < argc; i++) {
		const char *a = argv[i];
#define ARG (i + 1 < argc ? argv[++i] : (usage(argv[0]), exit(1), (char*)0))
		if (!strcmp(a, "-m")) dir = ARG;
		else if (!strcmp(a, "-t")) r.dst = ARG;
		else if (!strcmp(a, "-s")) r.src = ARG;
		else if (!strcmp(a, "-p")) text = ARG;
		else if (!strcmp(a, "-f")) file = ARG;
		else if (!strcmp(a, "-c")) r.context = ARG;
		else if (!strcmp(a, "-n")) max_new = atoi(ARG);
		else if (!strcmp(a, "-C")) ctx = atoi(ARG);
		else if (!strcmp(a, "-g")) {
			char *g = ARG, *eq = strchr(g, '=');
			if (!eq || r.nglossary >= 64) { usage(argv[0]); return 1; }
			*eq = 0;
			r.glossary[r.nglossary][0] = g;
			r.glossary[r.nglossary++][1] = eq + 1;
		}
		else if (!strcmp(a, "--polite")) r.form = "polite";
		else if (!strcmp(a, "--plain")) r.form = "plain";
		else if (!strcmp(a, "-l")) lines = 1;
		else if (!strcmp(a, "-v")) verbose = 1;
		else if (!strcmp(a, "--dump-ids")) dump_ids = 1;
		else { usage(argv[0]); return 1; }
	}
	if (!dir || !r.dst) { usage(argv[0]); return 1; }
	if (!valid_lang(r.dst) || (r.src && !valid_lang(r.src))) { fprintf(stderr, "language codes are lowercase, e.g. en, ja, zh-hans\n"); return 1; }
	if (r.form && strcmp(r.dst, "ja")) { fprintf(stderr, "--polite / --plain need -t ja\n"); return 1; }
	if (r.context && strstr(r.context, "<|plamo:")) { fprintf(stderr, "context must not contain <|plamo: tokens\n"); return 1; }

	char *input = NULL;
	if (text) input = strdup(text);
	else if (file) input = read_file(file, NULL);
	else {
		size_t cap = 1 << 16, len = 0, got;
		input = (char*)malloc(cap);
		while ((got = fread(input + len, 1, cap - len - 1, stdin)) > 0) {
			len += got;
			if (len + 1 >= cap) input = (char*)realloc(input, cap *= 2);
		}
		input[len] = 0;
	}
	if (!input) { fprintf(stderr, "no input\n"); return 1; }
	size_t il = strlen(input);
	while (il && (input[il - 1] == '\n' || input[il - 1] == '\r')) input[--il] = 0;	// trailing newline of echo / files

	plm_model m;
	double t0 = now();
	if (plm_load(&m, dir, ctx)) { plm_free(&m); return 1; }
	if (verbose) fprintf(stderr, "[loaded in %.2fs, SIMD %s]\n", now() - t0, cats_cpu_simd_name());

	int rc = 0;
	if (lines) {
		for (char *p = input, *e; p; p = e ? e + 1 : NULL) {
			e = strchr(p, '\n');
			if (e) *e = 0;
			if (!blank(p) && translate(&m, &r, p, max_new, verbose) < 0) rc = 1;
			else if (blank(p)) putchar('\n');
		}
	} else if (blank(input)) {
		fprintf(stderr, "empty input\n");
		rc = 1;
	} else if (translate(&m, &r, input, max_new, verbose) < 0) rc = 1;
	plm_free(&m);
	free(input);
	return rc;
}
