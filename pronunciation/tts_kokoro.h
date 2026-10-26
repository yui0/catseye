/*
 * tts_kokoro.h - Kokoro-82M (StyleTTS2 + iSTFTNet) from a GGUF file, Q8_0 weights.
 *
 * Pipeline (batch 1): PL-BERT (ALBERT) -> duration / prosody predictor (BiLSTMs,
 * AdaLN, AdaIN res-blocks) -> text encoder -> AdaIN decoder -> iSTFTNet generator.
 * Q8_0 matrices stay quantized in memory and are multiplied with Q8_0-quantized
 * activations (cats_dot_q8_0w_q8_0 in ../catseye.h); convolutions run in float.
 * Weights: cstr/kokoro-82m-GGUF (kokoro-82m-q8_0.gguf); voices: tools/convert_kokoro_voice.py.
 */
#ifndef CATS_TTS_KOKORO_H
#define CATS_TTS_KOKORO_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include "gguf.h"
#include "phon.h"
#ifndef CATS_CATSEYE_INCLUDED
#define CATS_CATSEYE_INCLUDED
#define CATS_INFER_ONLY
#include "../catseye.h"
#undef CATS_INFER_ONLY
#endif

#define KK_RATE 24000
#define KK_MAXTOK 510

/* ------------------------------------------------------------ tensor kinds */

typedef struct { const uint8_t *q; float *f; int in, out; float *b; } kk_lin;
typedef struct { float *w, *b; int cin, cout, k, stride, pad, dil; } kk_conv;
typedef struct { kk_lin fc; int c; } kk_adain;
typedef struct {
	kk_conv c1, c2, c1x1; int has1x1;
	kk_adain n1, n2;
	float *poolw, *poolb; int upsample, din, dout;
} kk_blk;
typedef struct { kk_lin ih[2], hh[2]; float *bih[2], *bhh[2]; int in, H; } kk_lstm;
typedef struct {
	int c, k, dil[3];
	kk_conv c1[3], c2[3]; kk_adain n1[3], n2[3]; float *a1[3], *a2[3];
} kk_res1;

typedef struct {
	cg_file g; int q8;
	char **tok; int ntok;
	const uint8_t *tok_emb, *pos_emb, *tt_emb, *te_emb; int bert_emb;
	float *emb_ln_w, *emb_ln_b, *attn_ln_w, *attn_ln_b, *ffn_ln_w, *ffn_ln_b;
	kk_lin emb_proj, aq, ak, av, ao, up, down, bert_proj;
	struct { kk_conv c; float *g, *b; } cnn[3];
	kk_lstm te_lstm;
	struct { kk_lstm l; kk_adain ln; } dur[3];
	kk_lstm dur_lstm, shared; kk_lin dur_proj;
	kk_blk F0[3], N[3]; kk_conv F0_proj, N_proj;
	kk_conv f0_conv, n_conv, asr_res;
	kk_blk enc, dec[4];
	float *msrc_w, msrc_b;
	kk_conv ups[2]; int ups_k[2], ups_s[2];
	kk_conv noise_conv[2]; kk_res1 noise_res[2], res[6];
	kk_conv post;
	int deterministic;	/* no random phase / noise: matches the PyTorch reference run */
	uint64_t rng;
} kk_model;

/* ----------------------------------------------------------------- loading */

static int kk_err;
static const cg_tensor *kk_need(kk_model *m, const char *name)
{
	const cg_tensor *t = cg_find(&m->g, name);
	if (!t) { fprintf(stderr, "kokoro: missing tensor %s\n", name); kk_err = 1; }
	return t;
}

/* any float-like tensor -> malloc'ed float array (Q8_0 is dequantized row by row) */
static float *kk_f32(kk_model *m, const char *name, size_t *n_out)
{
	const cg_tensor *t = kk_need(m, name);
	if (!t) return NULL;
	float *o = (float *)malloc(sizeof(float) * (t->n ? t->n : 1));
	if (t->type == CG_F32) memcpy(o, t->data, t->n * 4);
	else if (t->type == CG_F16) for (uint64_t i = 0; i < t->n; i++) o[i] = cats_half_to_float(cats_load_u16((const uint8_t *)t->data + 2 * i));
	else if (t->type == CG_Q8_0) {
		size_t row = t->dim[0], rb = cats_q8_0w_row_bytes(row);
		for (uint64_t r = 0; r < t->n / row; r++) cats_q8_0w_dequantize_row((const uint8_t *)t->data + r * rb, row, o + r * row);
	} else { fprintf(stderr, "kokoro: unsupported type %d for %s\n", t->type, name); kk_err = 1; }
	if (n_out) *n_out = t->n;
	return o;
}

static int kk_load_lin(kk_model *m, kk_lin *L, const char *wname, const char *bname)
{
	const cg_tensor *t = kk_need(m, wname);
	if (!t) return -1;
	memset(L, 0, sizeof *L);
	L->in = (int)t->dim[0]; L->out = t->nd > 1 ? (int)t->dim[1] : 1;
	if (t->type == CG_Q8_0 && m->q8) L->q = (const uint8_t *)t->data; else L->f = kk_f32(m, wname, NULL);
	if (bname) L->b = kk_f32(m, bname, NULL);
	return 0;
}

static int kk_load_linp(kk_model *m, kk_lin *L, const char *prefix)
{
	char w[260], b[260];
	snprintf(w, sizeof w, "%s.weight", prefix); snprintf(b, sizeof b, "%s.bias", prefix);
	return kk_load_lin(m, L, w, cg_find(&m->g, b) ? b : NULL);
}

static int kk_load_conv(kk_model *m, kk_conv *c, const char *prefix, int stride, int pad, int dil)
{
	char w[260], b[260];
	snprintf(w, sizeof w, "%s.weight", prefix); snprintf(b, sizeof b, "%s.bias", prefix);
	const cg_tensor *t = kk_need(m, w);
	if (!t) return -1;
	memset(c, 0, sizeof *c);
	c->k = (int)t->dim[0]; c->cin = t->nd > 1 ? (int)t->dim[1] : 1; c->cout = t->nd > 2 ? (int)t->dim[2] : 1;
	c->stride = stride; c->pad = pad; c->dil = dil;
	c->w = kk_f32(m, w, NULL);
	c->b = cg_find(&m->g, b) ? kk_f32(m, b, NULL) : NULL;
	return 0;
}

static void kk_load_adain(kk_model *m, kk_adain *a, const char *prefix, int c)
{
	kk_load_linp(m, &a->fc, prefix); a->c = c;
}

static void kk_load_lstm(kk_model *m, kk_lstm *l, const char *prefix, int in, int H)
{
	static const char *sfx[2] = {"", "_reverse"};
	l->in = in; l->H = H;
	for (int d = 0; d < 2; d++) {
		char n[200];
		snprintf(n, sizeof n, "%s.weight_ih_l0%s", prefix, sfx[d]); kk_load_lin(m, &l->ih[d], n, NULL);
		snprintf(n, sizeof n, "%s.weight_hh_l0%s", prefix, sfx[d]); kk_load_lin(m, &l->hh[d], n, NULL);
		snprintf(n, sizeof n, "%s.bias_ih_l0%s", prefix, sfx[d]); l->bih[d] = kk_f32(m, n, NULL);
		snprintf(n, sizeof n, "%s.bias_hh_l0%s", prefix, sfx[d]); l->bhh[d] = kk_f32(m, n, NULL);
	}
}

static void kk_load_blk(kk_model *m, kk_blk *b, const char *p, int din, int dout, int upsample)
{
	char n[200];
	memset(b, 0, sizeof *b);
	b->din = din; b->dout = dout; b->upsample = upsample;
	snprintf(n, sizeof n, "%s.conv1", p); kk_load_conv(m, &b->c1, n, 1, 1, 1);
	snprintf(n, sizeof n, "%s.conv2", p); kk_load_conv(m, &b->c2, n, 1, 1, 1);
	snprintf(n, sizeof n, "%s.adain1", p); kk_load_adain(m, &b->n1, n, din);
	snprintf(n, sizeof n, "%s.adain2", p); kk_load_adain(m, &b->n2, n, dout);
	if (din != dout) { snprintf(n, sizeof n, "%s.conv1x1", p); kk_load_conv(m, &b->c1x1, n, 1, 0, 1); b->has1x1 = 1; }
	if (upsample) {
		snprintf(n, sizeof n, "%s.pool.weight", p); b->poolw = kk_f32(m, n, NULL);
		snprintf(n, sizeof n, "%s.pool.bias", p); b->poolb = kk_f32(m, n, NULL);
	}
}

static void kk_load_res1(kk_model *m, kk_res1 *r, const char *p, int c, int k)
{
	static const int dil[3] = {1, 3, 5};
	r->c = c; r->k = k;
	for (int i = 0; i < 3; i++) {
		char n[200];
		r->dil[i] = dil[i];
		snprintf(n, sizeof n, "%s.convs1.%d", p, i); kk_load_conv(m, &r->c1[i], n, 1, (k * dil[i] - dil[i]) / 2, dil[i]);
		snprintf(n, sizeof n, "%s.convs2.%d", p, i); kk_load_conv(m, &r->c2[i], n, 1, (k - 1) / 2, 1);
		snprintf(n, sizeof n, "%s.adain1.%d", p, i); kk_load_adain(m, &r->n1[i], n, c);
		snprintf(n, sizeof n, "%s.adain2.%d", p, i); kk_load_adain(m, &r->n2[i], n, c);
		snprintf(n, sizeof n, "%s.alpha1.%d", p, i); r->a1[i] = kk_f32(m, n, NULL);
		snprintf(n, sizeof n, "%s.alpha2.%d", p, i); r->a2[i] = kk_f32(m, n, NULL);
	}
}

static inline void kk_close(kk_model *m)
{
	if (m->tok) { for (int i = 0; i < m->ntok; i++) free(m->tok[i]); free(m->tok); }
	cg_close(&m->g);	/* weight arrays are left to process exit: the model lives for the whole run */
	memset(m, 0, sizeof *m);
}

static inline int kk_open(kk_model *m, const char *path, int q8)
{
	memset(m, 0, sizeof *m);
	kk_err = 0; m->q8 = q8; m->rng = 0x2545F4914F6CDD1Dull;
	if (cg_open(&m->g, path)) return -1;
	m->ntok = cg_str_array(&m->g, "tokenizer.ggml.tokens", &m->tok);
	if (m->ntok <= 0) return -1;
	char n[200];
	const cg_tensor *t;
	if ((t = kk_need(m, "bert.embd.tok.weight"))) m->tok_emb = (const uint8_t *)t->data;
	m->pos_emb = (const uint8_t *)kk_need(m, "bert.embd.pos.weight")->data;
	m->tt_emb = (const uint8_t *)kk_need(m, "bert.embd.tt.weight")->data;
	m->emb_ln_w = kk_f32(m, "bert.embd.ln.weight", NULL); m->emb_ln_b = kk_f32(m, "bert.embd.ln.bias", NULL);
	kk_load_linp(m, &m->emb_proj, "bert.embd_proj");
	kk_load_linp(m, &m->aq, "bert.attn_q"); kk_load_linp(m, &m->ak, "bert.attn_k"); kk_load_linp(m, &m->av, "bert.attn_v"); kk_load_linp(m, &m->ao, "bert.attn_o");
	m->attn_ln_w = kk_f32(m, "bert.attn_ln.weight", NULL); m->attn_ln_b = kk_f32(m, "bert.attn_ln.bias", NULL);
	kk_load_linp(m, &m->up, "bert.ffn_up"); kk_load_linp(m, &m->down, "bert.ffn_down");
	m->ffn_ln_w = kk_f32(m, "bert.ffn_ln.weight", NULL); m->ffn_ln_b = kk_f32(m, "bert.ffn_ln.bias", NULL);
	kk_load_linp(m, &m->bert_proj, "bert_proj");

	m->te_emb = (const uint8_t *)kk_need(m, "text_enc.embd.weight")->data;
	for (int i = 0; i < 3; i++) {
		snprintf(n, sizeof n, "text_enc.cnn.%d.conv", i); kk_load_conv(m, &m->cnn[i].c, n, 1, 2, 1);
		snprintf(n, sizeof n, "text_enc.cnn.%d.ln.gamma", i); m->cnn[i].g = kk_f32(m, n, NULL);
		snprintf(n, sizeof n, "text_enc.cnn.%d.ln.beta", i); m->cnn[i].b = kk_f32(m, n, NULL);
	}
	kk_load_lstm(m, &m->te_lstm, "text_enc.lstm", 512, 256);

	for (int i = 0; i < 3; i++) {
		snprintf(n, sizeof n, "pred.dur_enc.%d.lstm", i); kk_load_lstm(m, &m->dur[i].l, n, 640, 256);
		snprintf(n, sizeof n, "pred.dur_enc.%d.adaln", i); kk_load_adain(m, &m->dur[i].ln, n, 512);
	}
	kk_load_lstm(m, &m->dur_lstm, "pred.lstm", 640, 256);
	kk_load_lstm(m, &m->shared, "pred.shared", 640, 256);
	kk_load_linp(m, &m->dur_proj, "pred.dur_proj");
	static const int pin[3] = {512, 512, 256}, pout[3] = {512, 256, 256};
	for (int i = 0; i < 3; i++) {
		snprintf(n, sizeof n, "pred.F0.%d", i); kk_load_blk(m, &m->F0[i], n, pin[i], pout[i], i == 1);
		snprintf(n, sizeof n, "pred.N.%d", i); kk_load_blk(m, &m->N[i], n, pin[i], pout[i], i == 1);
	}
	kk_load_conv(m, &m->F0_proj, "pred.F0_proj", 1, 0, 1); kk_load_conv(m, &m->N_proj, "pred.N_proj", 1, 0, 1);

	kk_load_conv(m, &m->f0_conv, "dec.F0_conv", 2, 1, 1); kk_load_conv(m, &m->n_conv, "dec.N_conv", 2, 1, 1);
	kk_load_conv(m, &m->asr_res, "dec.asr_res", 1, 0, 1);
	kk_load_blk(m, &m->enc, "dec.encode", 514, 1024, 0);
	for (int i = 0; i < 4; i++) { snprintf(n, sizeof n, "dec.decode.%d", i); kk_load_blk(m, &m->dec[i], n, 1090, i == 3 ? 512 : 1024, i == 3); }

	m->msrc_w = kk_f32(m, "dec.gen.m_source.weight", NULL);
	{ float *b = kk_f32(m, "dec.gen.m_source.bias", NULL); m->msrc_b = b ? b[0] : 0; free(b); }
	static const int uk[2] = {20, 12}, us[2] = {10, 6}, nk[2] = {12, 1}, ns[2] = {6, 1}, np[2] = {3, 0}, nc[2] = {256, 128}, nrk[2] = {7, 11};
	for (int i = 0; i < 2; i++) {
		snprintf(n, sizeof n, "dec.gen.ups.%d", i); kk_load_conv(m, &m->ups[i], n, us[i], (uk[i] - us[i]) / 2, 1);
		{ int t = m->ups[i].cin; m->ups[i].cin = m->ups[i].cout; m->ups[i].cout = t; }	/* ConvTranspose dims are (K, Cout, Cin) */
		m->ups_k[i] = uk[i]; m->ups_s[i] = us[i];
		snprintf(n, sizeof n, "dec.gen.noise_convs.%d", i); kk_load_conv(m, &m->noise_conv[i], n, ns[i], np[i], 1);
		(void)nk;
		snprintf(n, sizeof n, "dec.gen.noise_res.%d", i); kk_load_res1(m, &m->noise_res[i], n, nc[i], nrk[i]);
	}
	static const int rk[3] = {3, 7, 11};
	for (int i = 0; i < 6; i++) { snprintf(n, sizeof n, "dec.gen.resblocks.%d", i); kk_load_res1(m, &m->res[i], n, i < 3 ? 256 : 128, rk[i % 3]); }
	kk_load_conv(m, &m->post, "dec.gen.conv_post", 1, 3, 1);
	return kk_err ? -1 : 0;
}

/* -------------------------------------------------------------- primitives */

static inline float kk_gelu_new(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
static inline float kk_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* y[out] = W x + b for one vector */
static void kk_lin_run(const kk_lin *L, const float *x, float *y)
{
	if (L->q) {
		int8_t xq[(2048 / 32) * CATS_Q8_0_BYTES];
		cats_q8_0_quantize(xq, x, L->in);
		size_t rb = cats_q8_0w_row_bytes(L->in);
		for (int o = 0; o < L->out; o++) y[o] = cats_dot_q8_0w_q8_0(L->q + (size_t)o * rb, xq, L->in) + (L->b ? L->b[o] : 0.f);
	} else {
		for (int o = 0; o < L->out; o++) {
			const float *w = L->f + (size_t)o * L->in;
			float s = 0;
			for (int i = 0; i < L->in; i++) s += w[i] * x[i];
			y[o] = s + (L->b ? L->b[o] : 0.f);
		}
	}
}

/* Y[T][out] = X[T][in] W^T + b, parallel over rows */
static void kk_lin_seq(const kk_lin *L, const float *X, int T, float *Y)
{
	#pragma omp parallel for schedule(static)
	for (int t = 0; t < T; t++) kk_lin_run(L, X + (size_t)t * L->in, Y + (size_t)t * L->out);
}

static void kk_layernorm(float *x, int n, const float *g, const float *b, float eps)
{
	double mu = 0, var = 0;
	for (int i = 0; i < n; i++) mu += x[i];
	mu /= n;
	for (int i = 0; i < n; i++) { double d = x[i] - mu; var += d * d; }
	var /= n;
	float inv = (float)(1.0 / sqrt(var + eps));
	for (int i = 0; i < n; i++) x[i] = (float)((x[i] - mu) * inv) * (g ? g[i] : 1.f) + (b ? b[i] : 0.f);
}

/* x[C][T] -> y[C][Tout]; zero padding, any stride/dilation */
static int kk_conv_run(const kk_conv *c, const float *x, int T, float *y)
{
	int To = (T + 2 * c->pad - c->dil * (c->k - 1) - 1) / c->stride + 1;
	#pragma omp parallel for schedule(static)
	for (int oc = 0; oc < c->cout; oc++) {
		float *yo = y + (size_t)oc * To;
		float bv = c->b ? c->b[oc] : 0.f;
		for (int t = 0; t < To; t++) yo[t] = bv;
		for (int ic = 0; ic < c->cin; ic++) {
			const float *xi = x + (size_t)ic * T, *w = c->w + ((size_t)oc * c->cin + ic) * c->k;
			for (int k = 0; k < c->k; k++) {
				float wv = w[k];
				int sh = k * c->dil - c->pad;
				if (c->stride == 1) {
					int t0 = sh < 0 ? -sh : 0, t1 = To < T - sh ? To : T - sh;
					for (int t = t0; t < t1; t++) yo[t] += wv * xi[t + sh];
				} else {
					for (int t = 0; t < To; t++) { int s = t * c->stride + sh; if (s >= 0 && s < T) yo[t] += wv * xi[s]; }
				}
			}
		}
	}
	return To;
}

/* ConvTranspose1d: w is [Cin][Cout/groups][K]; returns Tout */
static int kk_convT_run(const float *w, const float *b, int cin, int cout, int K, int stride, int pad, int outpad, int groups, const float *x, int T, float *y)
{
	int To = (T - 1) * stride - 2 * pad + K + outpad, cog = cout / groups, cig = cin / groups;
	#pragma omp parallel for schedule(static)
	for (int oc = 0; oc < cout; oc++) {
		float *yo = y + (size_t)oc * To;
		int g = oc / cog, ol = oc % cog;
		for (int t = 0; t < To; t++) yo[t] = b ? b[oc] : 0.f;
		for (int il = 0; il < cig; il++) {
			int ic = g * cig + il;
			const float *xi = x + (size_t)ic * T, *wk = w + ((size_t)ic * cog + ol) * K;
			for (int k = 0; k < K; k++) {
				float wv = wk[k];
				for (int t = 0; t < T; t++) { int o = t * stride + k - pad; if (o >= 0 && o < To) yo[o] += xi[t] * wv; }
			}
		}
	}
	return To;
}

static void kk_leaky(float *x, size_t n, float a) { for (size_t i = 0; i < n; i++) if (x[i] < 0) x[i] *= a; }

/* AdaIN1d in place: x[C][T] = (1+gamma) * instance_norm(x) + beta, [gamma beta] = fc(style) */
static void kk_adain_run(const kk_adain *a, float *x, int T, const float *style)
{
	int C = a->c;
	float *h = (float *)malloc(sizeof(float) * 2 * C);
	kk_lin_run(&a->fc, style, h);
	#pragma omp parallel for schedule(static)
	for (int c = 0; c < C; c++) {
		float *r = x + (size_t)c * T;
		double mu = 0, var = 0;
		for (int t = 0; t < T; t++) mu += r[t];
		mu /= T;
		for (int t = 0; t < T; t++) { double d = r[t] - mu; var += d * d; }
		var /= T;
		float inv = (float)(1.0 / sqrt(var + 1e-5)), g = 1.f + h[c], bt = h[C + c];
		for (int t = 0; t < T; t++) r[t] = (float)((r[t] - mu) * inv) * g + bt;
	}
	free(h);
}

/* AdainResBlk1d: x[din][T] -> y[dout][T or 2T]; returns Tout */
static int kk_blk_run(const kk_blk *b, const float *x, int T, const float *style, float *y)
{
	int To = b->upsample ? 2 * T : T;
	float *r = (float *)malloc(sizeof(float) * (size_t)b->din * T);
	memcpy(r, x, sizeof(float) * (size_t)b->din * T);
	kk_adain_run(&b->n1, r, T, style);
	kk_leaky(r, (size_t)b->din * T, 0.2f);
	float *p = r;
	if (b->upsample) {
		p = (float *)malloc(sizeof(float) * (size_t)b->din * To);
		kk_convT_run(b->poolw, b->poolb, b->din, b->din, 3, 2, 1, 1, b->din, r, T, p);
		free(r);
	}
	float *c1 = (float *)malloc(sizeof(float) * (size_t)b->dout * To);
	kk_conv_run(&b->c1, p, To, c1);
	free(p);
	kk_adain_run(&b->n2, c1, To, style);
	kk_leaky(c1, (size_t)b->dout * To, 0.2f);
	float *c2 = (float *)malloc(sizeof(float) * (size_t)b->dout * To);
	kk_conv_run(&b->c2, c1, To, c2);
	free(c1);
	/* shortcut */
	float *s;
	if (b->upsample) {
		float *u = (float *)malloc(sizeof(float) * (size_t)b->din * To);
		for (int c = 0; c < b->din; c++) for (int t = 0; t < T; t++) u[(size_t)c * To + 2 * t] = u[(size_t)c * To + 2 * t + 1] = x[(size_t)c * T + t];
		if (b->has1x1) { s = (float *)malloc(sizeof(float) * (size_t)b->dout * To); kk_conv_run(&b->c1x1, u, To, s); free(u); } else s = u;
	} else if (b->has1x1) { s = (float *)malloc(sizeof(float) * (size_t)b->dout * To); kk_conv_run(&b->c1x1, x, T, s); }
	else s = (float *)x;
	const float isq = 0.70710678118654752f;
	for (size_t i = 0; i < (size_t)b->dout * To; i++) y[i] = (c2[i] + s[i]) * isq;
	free(c2);
	if (s != x) free(s);
	return To;
}

/* AdaINResBlock1 in place on x[C][T] */
static void kk_res1_run(const kk_res1 *r, float *x, int T, const float *style)
{
	size_t n = (size_t)r->c * T;
	float *xt = (float *)malloc(sizeof(float) * n), *yt = (float *)malloc(sizeof(float) * n);
	for (int i = 0; i < 3; i++) {
		memcpy(xt, x, sizeof(float) * n);
		kk_adain_run(&r->n1[i], xt, T, style);
		for (int c = 0; c < r->c; c++) { float a = r->a1[i][c]; float *p = xt + (size_t)c * T; for (int t = 0; t < T; t++) { float s = sinf(a * p[t]); p[t] += s * s / a; } }
		kk_conv_run(&r->c1[i], xt, T, yt);
		kk_adain_run(&r->n2[i], yt, T, style);
		for (int c = 0; c < r->c; c++) { float a = r->a2[i][c]; float *p = yt + (size_t)c * T; for (int t = 0; t < T; t++) { float s = sinf(a * p[t]); p[t] += s * s / a; } }
		kk_conv_run(&r->c2[i], yt, T, xt);
		for (size_t j = 0; j < n; j++) x[j] += xt[j];
	}
	free(xt); free(yt);
}

/* BiLSTM over x[T][in] -> y[T][2H] (torch gate order i f g o) */
static void kk_lstm_run(const kk_lstm *l, const float *x, int T, float *y)
{
	int H = l->H;
	float *gx = (float *)malloc(sizeof(float) * (size_t)T * 4 * H), *gh = (float *)malloc(sizeof(float) * 4 * H);
	float *h = (float *)calloc(H, sizeof(float)), *c = (float *)calloc(H, sizeof(float));
	for (int d = 0; d < 2; d++) {
		kk_lin tmp = l->ih[d]; tmp.b = l->bih[d];
		kk_lin_seq(&tmp, x, T, gx);
		memset(h, 0, sizeof(float) * H); memset(c, 0, sizeof(float) * H);
		for (int s = 0; s < T; s++) {
			int t = d ? T - 1 - s : s;
			kk_lin hh = l->hh[d]; hh.b = l->bhh[d];
			kk_lin_run(&hh, h, gh);
			const float *g = gx + (size_t)t * 4 * H;
			for (int j = 0; j < H; j++) {
				float ig = kk_sigmoid(g[j] + gh[j]), fg = kk_sigmoid(g[H + j] + gh[H + j]);
				float gg = tanhf(g[2 * H + j] + gh[2 * H + j]), og = kk_sigmoid(g[3 * H + j] + gh[3 * H + j]);
				c[j] = fg * c[j] + ig * gg;
				h[j] = og * tanhf(c[j]);
				y[(size_t)t * 2 * H + d * H + j] = h[j];
			}
		}
	}
	free(gx); free(gh); free(h); free(c);
}

/* ------------------------------------------------------------ vocabulary */

static inline int kk_tok_id(const kk_model *m, const char *s, int len)
{
	for (int i = 1; i < m->ntok; i++) if ((int)strlen(m->tok[i]) == len && !strncmp(m->tok[i], s, len)) return i;
	return -1;
}

/* our IPA -> Kokoro/misaki symbols -> ids (without BOS/EOS). Returns count. */
static inline int kk_ipa_ids(const kk_model *m, const char *ipa, int *ids, int cap)
{
	static const char *map[][2] = {
		{"e\xc9\xaa", "A"}, {"a\xc9\xaa", "I"}, {"o\xca\x8a", "O"}, {"a\xca\x8a", "W"}, {"\xc9\x94\xc9\xaa", "Y"},
		{"t\xca\x83", "\xca\xa7"}, {"d\xca\x92", "\xca\xa4"}, {"t\xc9\x95", "\xca\xa8"}, {"d\xca\x91", "\xca\xa5"}, {"ts", "\xca\xa6"},
		{"g", "\xc9\xa1"},
		{"\xc3\xa3", "a\xcc\x83"}, {"\xe1\xba\xbd", "e\xcc\x83"}, {"\xc4\xa9", "i\xcc\x83"}, {"\xc3\xb5", "o\xcc\x83"}, {"\xc5\xa9", "u\xcc\x83"},
	};
	int n = 0;
	for (const char *p = ipa; *p && n < cap;) {
		int best = -1, bl = 0;
		for (int i = 0; i < (int)(sizeof map / sizeof *map); i++) { int l = (int)strlen(map[i][0]); if (l > bl && !strncmp(p, map[i][0], l)) { best = i; bl = l; } }
		const char *s = p; int sl;
		if (best >= 0) { s = map[best][1]; sl = (int)strlen(s); p += bl; }
		else { uint32_t c; sl = cp_utf8_decode((const unsigned char *)p, &c); p += sl; }
		/* a replacement may be several symbols (a + combining tilde): emit each code point */
		for (const char *q = s; q < s + sl && n < cap;) {
			uint32_t c; int l = cp_utf8_decode((const unsigned char *)q, &c);
			int id = kk_tok_id(m, q, l);
			if (id >= 0) ids[n++] = id;
			q += l;
		}
	}
	return n;
}

/* --------------------------------------------------------------- inference */

typedef struct {	/* optional intermediates for validation (all malloc'ed by kk_run, caller frees) */
	float *d_en, *d, *dur, *en, *F0, *N, *t_en, *asr, *audio;
	int T, F, nsamp;
} kk_dump;

static inline float kk_randn(kk_model *m)
{
	double u[2];
	for (int k = 0; k < 2; k++) {
		m->rng ^= m->rng << 13; m->rng ^= m->rng >> 7; m->rng ^= m->rng << 17;
		u[k] = ((double)(m->rng >> 11) + 1.0) / 9007199254740994.0;
	}
	return (float)(sqrt(-2.0 * log(u[0])) * cos(6.283185307179586 * u[1]));
}
static inline float kk_rand(kk_model *m)
{
	m->rng ^= m->rng << 13; m->rng ^= m->rng >> 7; m->rng ^= m->rng << 17;
	return (float)((m->rng >> 11) * (1.0 / 9007199254740992.0));
}

/* ids includes BOS/EOS (0 ... 0); style = 256 floats [decoder 128 | predictor 128] */
static int kk_run(kk_model *m, const int *ids, int T, const float *style, float speed, float **pcm, size_t *count, kk_dump *dump)
{
	const float *sd = style, *sp = style + 128;
	/* ---- PL-BERT ---- */
	float *x = (float *)malloc(sizeof(float) * (size_t)T * 128);
	{
		float row[128], r2[128];
		for (int t = 0; t < T; t++) {
			cats_q8_0w_dequantize_row(m->tok_emb + (size_t)ids[t] * cats_q8_0w_row_bytes(128), 128, row);
			cats_q8_0w_dequantize_row(m->pos_emb + (size_t)t * cats_q8_0w_row_bytes(128), 128, r2);
			for (int i = 0; i < 128; i++) row[i] += r2[i];
			cats_q8_0w_dequantize_row(m->tt_emb, 128, r2);
			for (int i = 0; i < 128; i++) row[i] += r2[i];
			kk_layernorm(row, 128, m->emb_ln_w, m->emb_ln_b, 1e-12f);
			memcpy(x + (size_t)t * 128, row, sizeof row);
		}
	}
	float *h = (float *)malloc(sizeof(float) * (size_t)T * 768);
	kk_lin_seq(&m->emb_proj, x, T, h);
	free(x);
	{
		float *q = (float *)malloc(sizeof(float) * (size_t)T * 768), *k = (float *)malloc(sizeof(float) * (size_t)T * 768);
		float *v = (float *)malloc(sizeof(float) * (size_t)T * 768), *ctx = (float *)malloc(sizeof(float) * (size_t)T * 768);
		float *o = (float *)malloc(sizeof(float) * (size_t)T * 768), *u = (float *)malloc(sizeof(float) * (size_t)T * 2048);
		for (int layer = 0; layer < 12; layer++) {
			kk_lin_seq(&m->aq, h, T, q); kk_lin_seq(&m->ak, h, T, k); kk_lin_seq(&m->av, h, T, v);
			#pragma omp parallel for schedule(static)
			for (int hd = 0; hd < 12; hd++) {
				float *sc = (float *)malloc(sizeof(float) * T);
				for (int i = 0; i < T; i++) {
					float mx = -1e30f;
					for (int j = 0; j < T; j++) {
						float s = 0;
						for (int d = 0; d < 64; d++) s += q[(size_t)i * 768 + hd * 64 + d] * k[(size_t)j * 768 + hd * 64 + d];
						sc[j] = s * 0.125f; if (sc[j] > mx) mx = sc[j];
					}
					float sum = 0;
					for (int j = 0; j < T; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
					for (int d = 0; d < 64; d++) {
						float a = 0;
						for (int j = 0; j < T; j++) a += sc[j] * v[(size_t)j * 768 + hd * 64 + d];
						ctx[(size_t)i * 768 + hd * 64 + d] = a / sum;
					}
				}
				free(sc);
			}
			kk_lin_seq(&m->ao, ctx, T, o);
			for (int t = 0; t < T; t++) { for (int i = 0; i < 768; i++) o[(size_t)t * 768 + i] += h[(size_t)t * 768 + i]; kk_layernorm(o + (size_t)t * 768, 768, m->attn_ln_w, m->attn_ln_b, 1e-12f); }
			kk_lin_seq(&m->up, o, T, u);
			for (size_t i = 0; i < (size_t)T * 2048; i++) u[i] = kk_gelu_new(u[i]);
			kk_lin_seq(&m->down, u, T, h);
			for (int t = 0; t < T; t++) { for (int i = 0; i < 768; i++) h[(size_t)t * 768 + i] += o[(size_t)t * 768 + i]; kk_layernorm(h + (size_t)t * 768, 768, m->ffn_ln_w, m->ffn_ln_b, 1e-12f); }
		}
		free(q); free(k); free(v); free(ctx); free(o); free(u);
	}
	float *den = (float *)malloc(sizeof(float) * (size_t)T * 512);	/* [T][512] */
	kk_lin_seq(&m->bert_proj, h, T, den);
	free(h);
	if (dump) { dump->T = T; dump->d_en = (float *)malloc(sizeof(float) * 512 * T); for (int t = 0; t < T; t++) for (int c = 0; c < 512; c++) dump->d_en[(size_t)c * T + t] = den[(size_t)t * 512 + c]; }

	/* ---- duration encoder: [T][640] ---- */
	float *d = (float *)malloc(sizeof(float) * (size_t)T * 640), *l = (float *)malloc(sizeof(float) * (size_t)T * 512);
	for (int t = 0; t < T; t++) { memcpy(d + (size_t)t * 640, den + (size_t)t * 512, 512 * sizeof(float)); memcpy(d + (size_t)t * 640 + 512, sp, 128 * sizeof(float)); }
	free(den);
	for (int i = 0; i < 3; i++) {
		kk_lstm_run(&m->dur[i].l, d, T, l);
		float *hh = (float *)malloc(sizeof(float) * 1024);
		kk_lin_run(&m->dur[i].ln.fc, sp, hh);
		for (int t = 0; t < T; t++) {
			float *r = l + (size_t)t * 512;
			kk_layernorm(r, 512, NULL, NULL, 1e-5f);
			for (int c = 0; c < 512; c++) r[c] = (1.f + hh[c]) * r[c] + hh[512 + c];
			memcpy(d + (size_t)t * 640, r, 512 * sizeof(float)); memcpy(d + (size_t)t * 640 + 512, sp, 128 * sizeof(float));
		}
		free(hh);
	}
	if (dump) { dump->d = (float *)malloc(sizeof(float) * 640 * T); memcpy(dump->d, d, sizeof(float) * 640 * T); }
	/* ---- durations ---- */
	kk_lstm_run(&m->dur_lstm, d, T, l);
	int *dur = (int *)malloc(sizeof(int) * T); int F = 0;
	float *dl = (float *)malloc(sizeof(float) * 50);
	if (dump) dump->dur = (float *)malloc(sizeof(float) * T);
	for (int t = 0; t < T; t++) {
		kk_lin_run(&m->dur_proj, l + (size_t)t * 512, dl);
		float s = 0; for (int i = 0; i < 50; i++) s += kk_sigmoid(dl[i]);
		s /= speed;
		if (dump) dump->dur[t] = s;
		int di = (int)nearbyintf(s); if (di < 1) di = 1;
		dur[t] = di; F += di;
	}
	free(dl);
	/* ---- expand to frames ---- */
	int *tokof = (int *)malloc(sizeof(int) * F);
	for (int t = 0, f = 0; t < T; t++) for (int j = 0; j < dur[t]; j++) tokof[f++] = t;
	float *enT = (float *)malloc(sizeof(float) * (size_t)F * 640);	/* [F][640] */
	for (int f = 0; f < F; f++) memcpy(enT + (size_t)f * 640, d + (size_t)tokof[f] * 640, 640 * sizeof(float));
	if (dump) { dump->F = F; dump->en = (float *)malloc(sizeof(float) * 640 * F); for (int f = 0; f < F; f++) for (int c = 0; c < 640; c++) dump->en[(size_t)c * F + f] = enT[(size_t)f * 640 + c]; }
	free(d); free(l);
	/* ---- F0 / N ---- */
	float *sh = (float *)malloc(sizeof(float) * (size_t)F * 512), *xin = (float *)malloc(sizeof(float) * (size_t)F * 512);
	kk_lstm_run(&m->shared, enT, F, sh);
	free(enT);
	for (int f = 0; f < F; f++) for (int c = 0; c < 512; c++) xin[(size_t)c * F + f] = sh[(size_t)f * 512 + c];
	free(sh);
	float *curve[2];
	for (int br = 0; br < 2; br++) {
		const kk_blk *B = br ? m->N : m->F0;
		float *a = (float *)malloc(sizeof(float) * (size_t)512 * F), *b = (float *)malloc(sizeof(float) * (size_t)512 * 2 * F);
		memcpy(a, xin, sizeof(float) * (size_t)512 * F);
		int Tc = F;
		for (int i = 0; i < 3; i++) { Tc = kk_blk_run(&B[i], a, Tc, sp, b); float *tmp = a; a = b; b = tmp; }
		curve[br] = (float *)malloc(sizeof(float) * Tc);
		kk_conv_run(br ? &m->N_proj : &m->F0_proj, a, Tc, curve[br]);
		free(a); free(b);
	}
	free(xin);
	float *F0 = curve[0], *N = curve[1];
	if (dump) { dump->F0 = (float *)malloc(sizeof(float) * 2 * F); memcpy(dump->F0, F0, sizeof(float) * 2 * F); dump->N = (float *)malloc(sizeof(float) * 2 * F); memcpy(dump->N, N, sizeof(float) * 2 * F); }

	/* ---- text encoder ---- */
	float *te = (float *)malloc(sizeof(float) * (size_t)512 * T), *tb = (float *)malloc(sizeof(float) * (size_t)512 * T);
	for (int t = 0; t < T; t++) {
		float row[512];
		cats_q8_0w_dequantize_row(m->te_emb + (size_t)ids[t] * cats_q8_0w_row_bytes(512), 512, row);
		for (int c = 0; c < 512; c++) te[(size_t)c * T + t] = row[c];
	}
	for (int i = 0; i < 3; i++) {
		kk_conv_run(&m->cnn[i].c, te, T, tb);
		for (int t = 0; t < T; t++) {
			float col[512];
			for (int c = 0; c < 512; c++) col[c] = tb[(size_t)c * T + t];
			kk_layernorm(col, 512, m->cnn[i].g, m->cnn[i].b, 1e-5f);
			for (int c = 0; c < 512; c++) { float v = col[c]; te[(size_t)c * T + t] = v < 0 ? v * 0.2f : v; }
		}
	}
	{
		float *seq = (float *)malloc(sizeof(float) * (size_t)T * 512), *o = (float *)malloc(sizeof(float) * (size_t)T * 512);
		for (int t = 0; t < T; t++) for (int c = 0; c < 512; c++) seq[(size_t)t * 512 + c] = te[(size_t)c * T + t];
		kk_lstm_run(&m->te_lstm, seq, T, o);
		for (int t = 0; t < T; t++) for (int c = 0; c < 512; c++) te[(size_t)c * T + t] = o[(size_t)t * 512 + c];
		free(seq); free(o);
	}
	if (dump) { dump->t_en = (float *)malloc(sizeof(float) * 512 * T); memcpy(dump->t_en, te, sizeof(float) * 512 * T); }
	float *asr = (float *)malloc(sizeof(float) * (size_t)512 * F);
	for (int c = 0; c < 512; c++) for (int f = 0; f < F; f++) asr[(size_t)c * F + f] = te[(size_t)c * T + tokof[f]];
	if (dump) { dump->asr = (float *)malloc(sizeof(float) * 512 * F); memcpy(dump->asr, asr, sizeof(float) * 512 * F); }
	free(te); free(tb); free(tokof); free(dur);

	/* ---- decoder ---- */
	float *f0c = (float *)malloc(sizeof(float) * F), *nc = (float *)malloc(sizeof(float) * F);
	kk_conv_run(&m->f0_conv, F0, 2 * F, f0c);
	kk_conv_run(&m->n_conv, N, 2 * F, nc);
	float *cat = (float *)malloc(sizeof(float) * (size_t)514 * F), *xa = (float *)malloc(sizeof(float) * (size_t)1024 * 2 * F);
	memcpy(cat, asr, sizeof(float) * (size_t)512 * F); memcpy(cat + (size_t)512 * F, f0c, sizeof(float) * F); memcpy(cat + (size_t)513 * F, nc, sizeof(float) * F);
	kk_blk_run(&m->enc, cat, F, sd, xa);
	free(cat);
	float *ar = (float *)malloc(sizeof(float) * (size_t)64 * F);
	kk_conv_run(&m->asr_res, asr, F, ar);
	float *xb = (float *)malloc(sizeof(float) * (size_t)1090 * F), *xc = (float *)malloc(sizeof(float) * (size_t)1024 * 2 * F);
	int Tf = F;
	for (int i = 0; i < 4; i++) {
		memcpy(xb, xa, sizeof(float) * (size_t)1024 * F);
		memcpy(xb + (size_t)1024 * F, ar, sizeof(float) * (size_t)64 * F);
		memcpy(xb + (size_t)1088 * F, f0c, sizeof(float) * F); memcpy(xb + (size_t)1089 * F, nc, sizeof(float) * F);
		Tf = kk_blk_run(&m->dec[i], xb, F, sd, xc);
		float *tmp = xa; xa = xc; xc = tmp;
	}
	free(xb); free(xc); free(ar); free(f0c); free(nc); free(asr);
	/* xa: [512][2F] */

	/* ---- generator ---- */
	int L = 2 * F * 300;
	float *har = NULL; int HT = L / 5 + 1;
	{
		/* sine source with 9 harmonics */
		float *f0s = (float *)malloc(sizeof(float) * L);
		for (int n = 0; n < L; n++) f0s[n] = F0[n / 300];
		int D = 2 * F;
		float *src = (float *)calloc(L, sizeof(float));
		float *rad = (float *)malloc(sizeof(float) * L), *ph = (float *)malloc(sizeof(float) * D), *sines = (float *)malloc(sizeof(float) * (size_t)9 * L);
		for (int hm = 0; hm < 9; hm++) {
			float ini = (hm && !m->deterministic) ? kk_rand(m) : 0.f;
			for (int n = 0; n < L; n++) { float r = f0s[n] * (hm + 1) / 24000.f; r -= floorf(r); rad[n] = r; }
			rad[0] += ini;
			double acc = 0;
			for (int i = 0; i < D; i++) {
				double sidx = (i + 0.5) * 300 - 0.5; int i0 = (int)floor(sidx); double lam = sidx - i0; int i1 = i0 + 1 < L ? i0 + 1 : L - 1;
				if (i0 < 0) { i0 = 0; lam = 0; }
				acc += rad[i0] * (1 - lam) + rad[i1] * lam;
				ph[i] = (float)(acc * 2.0 * M_PI * 300.0);
			}
			for (int n = 0; n < L; n++) {
				double sidx = (n + 0.5) / 300.0 - 0.5; int i0 = (int)floor(sidx); double lam = sidx - i0;
				if (i0 < 0) { i0 = 0; lam = 0; }
				int i1 = i0 + 1 < D ? i0 + 1 : D - 1;
				if (i0 >= D) i0 = D - 1;
				sines[(size_t)hm * L + n] = sinf((float)(ph[i0] * (1 - lam) + ph[i1] * lam));
			}
		}
		for (int n = 0; n < L; n++) {
			float uv = f0s[n] > 10.f ? 1.f : 0.f, namp = uv * 0.003f + (1.f - uv) * 0.1f / 3.f, s = m->msrc_b;
			for (int hm = 0; hm < 9; hm++) {
				float v = sines[(size_t)hm * L + n] * 0.1f * uv + (m->deterministic ? 0.f : namp * kk_randn(m));
				s += m->msrc_w[hm] * v;
			}
			src[n] = tanhf(s);
		}
		free(f0s); free(rad); free(ph); free(sines);
		/* STFT n_fft 20, hop 5, hann, center/reflect -> [22][HT] (mag 11, phase 11) */
		har = (float *)malloc(sizeof(float) * (size_t)22 * HT);
		float win[20], cs[20][11], sn[20][11];
		for (int i = 0; i < 20; i++) { win[i] = 0.5f - 0.5f * cosf(2.f * (float)M_PI * i / 20.f); for (int k = 0; k < 11; k++) { cs[i][k] = cosf(2.f * (float)M_PI * i * k / 20.f); sn[i][k] = sinf(2.f * (float)M_PI * i * k / 20.f); } }
		#pragma omp parallel for schedule(static)
		for (int t = 0; t < HT; t++) {
			float fr[20];
			for (int i = 0; i < 20; i++) {
				int p = t * 5 + i - 10;
				if (p < 0) p = -p; if (p >= L) p = 2 * (L - 1) - p;
				fr[i] = src[p] * win[i];
			}
			for (int k = 0; k < 11; k++) {
				float re = 0, im = 0;
				for (int i = 0; i < 20; i++) { re += fr[i] * cs[i][k]; im -= fr[i] * sn[i][k]; }
				har[(size_t)k * HT + t] = hypotf(re, im); har[(size_t)(11 + k) * HT + t] = atan2f(im, re);
			}
		}
		free(src);
	}
	int Tc = Tf;
	float *g = xa;
	size_t cap = (size_t)128 * (L / 5 + 2);
	float *gb = (float *)malloc(sizeof(float) * cap), *xs = (float *)malloc(sizeof(float) * cap);
	float *acc = (float *)malloc(sizeof(float) * cap), *tmp = (float *)malloc(sizeof(float) * cap);
	for (int i = 0; i < 2; i++) {
		int cc = m->ups[i].cout;
		kk_leaky(g, (size_t)m->ups[i].cin * Tc, 0.1f);
		int Tn = kk_conv_run(&m->noise_conv[i], har, HT, xs);
		kk_res1_run(&m->noise_res[i], xs, Tn, sd);
		int Tu = kk_convT_run(m->ups[i].w, m->ups[i].b, m->ups[i].cin, cc, m->ups_k[i], m->ups_s[i], (m->ups_k[i] - m->ups_s[i]) / 2, 0, 1, g, Tc, gb);
		if (i == 1) {	/* ReflectionPad1d((1, 0)) */
			for (int c = cc - 1; c >= 0; c--) {
				float *r = gb + (size_t)c * (Tu + 1);
				memmove(r + 1, gb + (size_t)c * Tu, sizeof(float) * Tu);
				r[0] = r[2];
			}
			Tu++;
		}
		if (Tn != Tu) { fprintf(stderr, "kokoro: generator length mismatch %d vs %d\n", Tn, Tu); return -1; }
		for (size_t j = 0; j < (size_t)cc * Tu; j++) gb[j] += xs[j];
		memset(acc, 0, sizeof(float) * (size_t)cc * Tu);
		for (int j = 0; j < 3; j++) {
			memcpy(tmp, gb, sizeof(float) * (size_t)cc * Tu);
			kk_res1_run(&m->res[i * 3 + j], tmp, Tu, sd);
			for (size_t q = 0; q < (size_t)cc * Tu; q++) acc[q] += tmp[q];
		}
		for (size_t q = 0; q < (size_t)cc * Tu; q++) acc[q] /= 3.f;
		if (i == 0) { free(xa); g = (float *)malloc(sizeof(float) * cap); }
		memcpy(g, acc, sizeof(float) * (size_t)cc * Tu);
		Tc = Tu;
	}
	kk_leaky(g, (size_t)128 * Tc, 0.01f);
	float *pst = (float *)malloc(sizeof(float) * (size_t)22 * Tc);
	kk_conv_run(&m->post, g, Tc, pst);
	free(g); free(gb); free(xs); free(acc); free(tmp); free(har);
	free(F0); free(N);
	/* ---- inverse STFT ---- */
	{
		int total = (Tc - 1) * 5 + 20;
		float *out = (float *)calloc(total, sizeof(float)), *env = (float *)calloc(total, sizeof(float));
		float win[20], cs[20][10], sn[20][10];
		for (int i = 0; i < 20; i++) { win[i] = 0.5f - 0.5f * cosf(2.f * (float)M_PI * i / 20.f); for (int k = 1; k < 10; k++) { cs[i][k] = cosf(2.f * (float)M_PI * i * k / 20.f); sn[i][k] = sinf(2.f * (float)M_PI * i * k / 20.f); } }
		for (int t = 0; t < Tc; t++) {
			float re[11], im[11];
			for (int k = 0; k < 11; k++) { float mag = expf(pst[(size_t)k * Tc + t]), ph = sinf(pst[(size_t)(11 + k) * Tc + t]); re[k] = mag * cosf(ph); im[k] = mag * sinf(ph); }
			for (int n = 0; n < 20; n++) {
				float v = re[0] + ((n & 1) ? -re[10] : re[10]);
				for (int k = 1; k < 10; k++) v += 2.f * (re[k] * cs[n][k] - im[k] * sn[n][k]);
				out[t * 5 + n] += v / 20.f * win[n]; env[t * 5 + n] += win[n] * win[n];
			}
		}
		int ns = (Tc - 1) * 5;
		*pcm = (float *)malloc(sizeof(float) * ns); *count = (size_t)ns;
		for (int i = 0; i < ns; i++) (*pcm)[i] = out[10 + i] / (env[10 + i] > 1e-11f ? env[10 + i] : 1.f);
		free(out); free(env);
	}
	free(pst);
	if (dump) { dump->nsamp = (int)*count; dump->audio = (float *)malloc(sizeof(float) * *count); memcpy(dump->audio, *pcm, sizeof(float) * *count); }
	return 0;
}

/* ---- text-level API ---- */

typedef struct { kk_model m; float *pack; int npack; } kk_voice;

static inline int kk_voice_open(kk_voice *v, const char *gguf, const char *voice_bin, int q8)
{
	memset(v, 0, sizeof *v);
	if (kk_open(&v->m, gguf, q8)) return -1;
	FILE *f = fopen(voice_bin, "rb");
	if (!f) return -1;
	fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
	v->npack = (int)(sz / (256 * 4));
	v->pack = (float *)malloc(sz);
	if (fread(v->pack, 1, sz, f) != (size_t)sz) { fclose(f); return -1; }
	fclose(f);
	return v->npack > 0 ? 0 : -1;
}

/* IPA (from cp_g2p) -> PCM. Splits at sentence ends so each chunk fits the 510-token window. */
static inline int kk_synth(kk_voice *v, const char *ipa, float speed, float **pcm, size_t *count)
{
	int all[4096], na = kk_ipa_ids(&v->m, ipa, all, 4096);
	float *out = NULL; size_t on = 0;
	int comma = kk_tok_id(&v->m, ",", 1), stop[3] = {kk_tok_id(&v->m, ".", 1), kk_tok_id(&v->m, "!", 1), kk_tok_id(&v->m, "?", 1)};
	for (int i = 0; i < na;) {
		int j = i, last = -1;
		while (j < na && j - i < 400) { if (all[j] == stop[0] || all[j] == stop[1] || all[j] == stop[2]) last = j; j++; }
		int end = j < na ? (last >= i ? last + 1 : j) : na;
		if (j < na && last < i) { for (int k = j; k > i; k--) if (all[k - 1] == comma) { end = k; break; } }
		int ids[KK_MAXTOK + 2], n = 0;
		ids[n++] = 0;
		for (int k = i; k < end; k++) ids[n++] = all[k];
		ids[n++] = 0;
		int row = n - 3; if (row < 0) row = 0; if (row >= v->npack) row = v->npack - 1;
		float *p; size_t c;
		if (kk_run(&v->m, ids, n, v->pack + (size_t)row * 256, speed > 0.1f ? speed : 1.f, &p, &c, NULL)) { free(out); return -1; }
		out = (float *)realloc(out, (on + c) * sizeof(float));
		memcpy(out + on, p, c * sizeof(float)); on += c; free(p);
		i = end;
	}
	if (!on) return -1;
	*pcm = out; *count = on;
	return 0;
}

#endif
