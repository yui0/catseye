/*
 * tts_qwen.h - Qwen3-TTS 12 Hz from GGUF (Serveurperso/Qwen3-TTS-GGUF, Q8_0).
 *
 * Codec decoder: 16 RVQ codes per 12.5 Hz frame -> 24 kHz audio
 *   RVQ lookup -> causal conv k3 -> 8-layer sliding-window transformer -> 2x(convT + ConvNeXt)
 *   -> DAC decoder (snake, convT 8/5/4/3, 3 dilated residual units per stage).
 * Talker: byte-level BPE -> prompt embeddings -> 28-layer Qwen3 LM (codec frame 0) -> 5-layer
 *   code predictor (frames 1..15) -> codec decoder.
 * Helpers (kk_lin, kk_conv_run, ...) come from tts_kokoro.h.
 */
#ifndef CATS_TTS_QWEN_H
#define CATS_TTS_QWEN_H

#include "tts_kokoro.h"

#include <time.h>
#include <omp.h>
#define QW_RATE 24000
#define QW_HOP 1920
#define QW_NCB 16

/* hyperthreads add little to these kernels and OpenMP spin-waits badly when the machine is shared:
 * default to one thread per physical core unless OMP_NUM_THREADS says otherwise */
static void qw_threads_init(void)
{
	if (getenv("OMP_NUM_THREADS")) return;
	int sib = 1; char buf[128] = "";
	FILE *f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
	if (f) { if (fgets(buf, sizeof buf, f)) { sib = 1; for (char *p = buf; *p; p++) if (*p == ',' || *p == '-') sib++; } fclose(f); }
	int n = omp_get_num_procs() / sib;
	omp_set_num_threads(n > 0 ? n : 1);
}

static double qw_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int qw_prof(void) { static int v = -1; if (v < 0) v = getenv("QW_PROF") != NULL; return v; }
#define QW_LAP(name) do { if (qw_prof()) { double n_ = qw_now(); fprintf(stderr, "  [prof] %-14s %7.3fs\n", name, n_ - qw_t0); qw_t0 = n_; } } while (0)

/* ------------------------------------------------ packed, register-tiled convolution */

static void qw_lin(const kk_lin *L, const float *X, int T, float *Y);
typedef float qw_v8 __attribute__((vector_size(32), aligned(4), may_alias));
typedef struct { float *pw, *b; int cin, cout, K, dil; } qw_pc;	/* pw: [ceil(cout/4)][cin][K][4], zero padded */

static void qw_pc_pack(qw_pc *p, const float *w, int cout, int cin, int K, int dil, const float *bias)
{
	int nb = (cout + 3) / 4;
	p->pw = (float *)calloc((size_t)nb * cin * K * 4, 4);
	for (int oc = 0; oc < cout; oc++) for (int ic = 0; ic < cin; ic++) for (int k = 0; k < K; k++)
		p->pw[(((size_t)(oc / 4) * cin + ic) * K + k) * 4 + oc % 4] = w[((size_t)oc * cin + ic) * K + k];
	p->b = bias ? (float *)memcpy(malloc(sizeof(float) * cout), bias, sizeof(float) * cout) : NULL;
	p->cin = cin; p->cout = cout; p->K = K; p->dil = dil;
}

/* ConvTranspose weights [cin][cout][K] seen as a 1x1 conv with cout*K rows */
static void qw_pc_pack_t(qw_pc *p, const float *w, int cout, int cin, int K)
{
	int R = cout * K, nb = (R + 3) / 4;
	p->pw = (float *)calloc((size_t)nb * cin * 4, 4);
	for (int ic = 0; ic < cin; ic++) for (int r = 0; r < R; r++) p->pw[((size_t)(r / 4) * cin + ic) * 4 + r % 4] = w[(size_t)ic * R + r];
	p->b = NULL; p->cin = cin; p->cout = R; p->K = 1; p->dil = 1;
}

/* y[cout][T] = causal conv of x[cin][T] (left zero padding (K-1)*dil); 4 channels x 16 steps per register tile */
static void qw_pc_run(const qw_pc *p, const float *x, int T, float *y)
{
	const int pad = (p->K - 1) * p->dil, Tr = (T + 15) & ~15, Wp = pad + Tr, nb = (p->cout + 3) / 4, ntb = Tr / 16;
	float *xp = (float *)calloc((size_t)p->cin * Wp + 16, 4);
	#pragma omp parallel for schedule(static)
	for (int ic = 0; ic < p->cin; ic++) memcpy(xp + (size_t)ic * Wp + pad, x + (size_t)ic * T, (size_t)T * 4);
	#pragma omp parallel for schedule(dynamic, 4)
	for (int tile = 0; tile < nb * ntb; tile++) {
		int ob = tile / ntb, tb = tile % ntb;
		qw_v8 a00, a01, a10, a11, a20, a21, a30, a31;
		float b0 = 0, b1 = 0, b2 = 0, b3 = 0;
		if (p->b) { int o = ob * 4; b0 = p->b[o]; b1 = o + 1 < p->cout ? p->b[o + 1] : 0; b2 = o + 2 < p->cout ? p->b[o + 2] : 0; b3 = o + 3 < p->cout ? p->b[o + 3] : 0; }
		a00 = a01 = (qw_v8){ b0, b0, b0, b0, b0, b0, b0, b0 }; a10 = a11 = (qw_v8){ b1, b1, b1, b1, b1, b1, b1, b1 };
		a20 = a21 = (qw_v8){ b2, b2, b2, b2, b2, b2, b2, b2 }; a30 = a31 = (qw_v8){ b3, b3, b3, b3, b3, b3, b3, b3 };
		const float *w = p->pw + (size_t)ob * p->cin * p->K * 4;
		for (int ic = 0; ic < p->cin; ic++) {
			const float *xr = xp + (size_t)ic * Wp + (size_t)tb * 16;
			for (int k = 0; k < p->K; k++, w += 4) {
				qw_v8 x0 = *(const qw_v8 *)(xr + k * p->dil), x1 = *(const qw_v8 *)(xr + k * p->dil + 8);
				float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
				a00 += w0 * x0; a01 += w0 * x1; a10 += w1 * x0; a11 += w1 * x1;
				a20 += w2 * x0; a21 += w2 * x1; a30 += w3 * x0; a31 += w3 * x1;
			}
		}
		float tmp[4][16];
		*(qw_v8 *)tmp[0] = a00; *(qw_v8 *)(tmp[0] + 8) = a01; *(qw_v8 *)tmp[1] = a10; *(qw_v8 *)(tmp[1] + 8) = a11;
		*(qw_v8 *)tmp[2] = a20; *(qw_v8 *)(tmp[2] + 8) = a21; *(qw_v8 *)tmp[3] = a30; *(qw_v8 *)(tmp[3] + 8) = a31;
		int t0 = tb * 16, nt = T - t0 < 16 ? T - t0 : 16;
		for (int i = 0; i < 4; i++) { int oc = ob * 4 + i; if (oc < p->cout) memcpy(y + (size_t)oc * T + t0, tmp[i], (size_t)nt * 4); }
	}
	free(xp);
}

/* ConvTranspose1d with kernel == stride or 2*stride, causal (output trimmed to T*stride) */
static void qw_pct_run(const qw_pc *p, const float *bias, int cout, int K, int stride, const float *x, int T, float *y)
{
	float *col = (float *)malloc((size_t)p->cout * T * 4);
	qw_pc_run(p, x, T, col);
	int To = T * stride;
	#pragma omp parallel for schedule(static)
	for (int oc = 0; oc < cout; oc++) {
		float *yo = y + (size_t)oc * To, bv = bias ? bias[oc] : 0.f;
		for (int u = 0; u < To; u++) {
			int a = u / stride, m = u % stride;
			float v = col[((size_t)oc * K + m) * T + a];
			if (K == 2 * stride && a >= 1) v += col[((size_t)oc * K + stride + m) * T + a - 1];
			yo[u] = v + bv;
		}
	}
	free(col);
}

typedef struct { float *a, *ib; int c; } qw_snake;
typedef struct { qw_snake s1, s2; qw_pc c1, c2; int dil; } qw_res;
typedef struct { qw_snake snake; qw_pc tw; float *tb; int in, out, stride; qw_res ru[3]; } qw_dblk;
typedef struct { kk_lin q, k, v, o, gate, up, down; float *an, *fn, *as, *fs; } qw_tlayer;
typedef struct { float *dww, *dwb, *nw, *nb, *gamma; kk_lin pw1, pw2; qw_pc tw; float *tb; } qw_cnx;

typedef struct {
	kk_model m;	/* only .g and .q8 are used */
	float *cb[QW_NCB], *vq_first_proj, *vq_rest_proj;
	qw_pc pre_conv;
	kk_lin in_proj, out_proj; float *in_b_dummy;
	qw_tlayer tl[8]; float *tnorm;
	int heads, hd, window; float eps, theta;
	qw_cnx up[2];
	qw_pc dac_pre, dac_post; qw_dblk blk[4]; qw_snake snake_post;
} qw_codec;

static void qw_load_snake(kk_model *m, qw_snake *s, const char *prefix)
{
	char n[260]; size_t c;
	snprintf(n, sizeof n, "%s.alpha", prefix); float *a = kk_f32(m, n, &c);
	snprintf(n, sizeof n, "%s.beta", prefix); float *b = kk_f32(m, n, NULL);
	s->c = (int)c; s->a = a; s->ib = b;
	for (size_t i = 0; i < c; i++) { a[i] = expf(a[i]); b[i] = 1.f / (expf(b[i]) + 1e-9f); }
}

static void qw_load_pc(kk_model *m, qw_pc *p, const char *prefix, int dil)
{
	kk_conv c; kk_load_conv(m, &c, prefix, 1, 0, dil);
	qw_pc_pack(p, c.w, c.cout, c.cin, c.k, dil, c.b);
	free(c.w); free(c.b);
}

static float qw_kvf(const cg_file *g, const char *key, float def)
{
	const cg_kv *kv = cg_kvfind(g, key);
	float v; if (!kv) return def; memcpy(&v, kv->val, 4); return v;
}

static inline void qw_codec_close(qw_codec *c) { cg_close(&c->m.g); }

static inline int qw_codec_open(qw_codec *c, const char *path, int q8)
{
	memset(c, 0, sizeof *c); qw_threads_init();
	kk_err = 0; c->m.q8 = q8;
	if (cg_open(&c->m.g, path)) return -1;
	kk_model *m = &c->m;
	char n[260];
	for (int k = 0; k < QW_NCB; k++) {
		snprintf(n, sizeof n, k == 0 ? "tok_dec.vq_first.0.codebook" : "tok_dec.vq_rest.%d.codebook", k - 1);
		c->cb[k] = kk_f32(m, n, NULL);
	}
	c->vq_first_proj = kk_f32(m, "tok_dec.vq_first.output_proj.weight", NULL);
	c->vq_rest_proj = kk_f32(m, "tok_dec.vq_rest.output_proj.weight", NULL);
	qw_load_pc(m, &c->pre_conv, "tok_dec.pre_conv", 1);
	kk_load_linp(m, &c->in_proj, "tok_dec.pre_tfm.input_proj");
	kk_load_linp(m, &c->out_proj, "tok_dec.pre_tfm.output_proj");
	c->tnorm = kk_f32(m, "tok_dec.pre_tfm.norm.weight", NULL);
	c->heads = (int)cg_u32(&m->g, "qwen3-tts-tokenizer.decoder.num_attention_heads", 16);
	c->hd = (int)cg_u32(&m->g, "qwen3-tts-tokenizer.decoder.head_dim", 64);
	c->window = (int)cg_u32(&m->g, "qwen3-tts-tokenizer.decoder.sliding_window", 72);
	c->eps = qw_kvf(&m->g, "qwen3-tts-tokenizer.decoder.rms_norm_eps", 1e-5f);
	c->theta = qw_kvf(&m->g, "qwen3-tts-tokenizer.decoder.rope_theta", 10000.f);
	for (int l = 0; l < 8; l++) {
		qw_tlayer *L = &c->tl[l];
		#define P(f, s) snprintf(n, sizeof n, "tok_dec.pre_tfm.blk.%d." s, l)
		P(0, "attn_q"); kk_load_linp(m, &L->q, n);
		P(0, "attn_k"); kk_load_linp(m, &L->k, n);
		P(0, "attn_v"); kk_load_linp(m, &L->v, n);
		P(0, "attn_output"); kk_load_linp(m, &L->o, n);
		P(0, "ffn_gate"); kk_load_linp(m, &L->gate, n);
		P(0, "ffn_up"); kk_load_linp(m, &L->up, n);
		P(0, "ffn_down"); kk_load_linp(m, &L->down, n);
		P(0, "attn_norm.weight"); L->an = kk_f32(m, n, NULL);
		P(0, "ffn_norm.weight"); L->fn = kk_f32(m, n, NULL);
		P(0, "attn_scale"); L->as = kk_f32(m, n, NULL);
		P(0, "ffn_scale"); L->fs = kk_f32(m, n, NULL);
		#undef P
	}
	for (int i = 0; i < 2; i++) {
		qw_cnx *u = &c->up[i];
		snprintf(n, sizeof n, "tok_dec.upsample.%d.conv.weight", i); { const cg_tensor *tt = kk_need(m, n); float *w = kk_f32(m, n, NULL); qw_pc_pack_t(&u->tw, w, (int)tt->dim[1], (int)tt->dim[2], (int)tt->dim[0]); free(w); }
		snprintf(n, sizeof n, "tok_dec.upsample.%d.conv.bias", i); u->tb = kk_f32(m, n, NULL);
		snprintf(n, sizeof n, "tok_dec.upsample.%d.dwconv.weight", i); u->dww = kk_f32(m, n, NULL);
		snprintf(n, sizeof n, "tok_dec.upsample.%d.dwconv.bias", i); u->dwb = kk_f32(m, n, NULL);
		snprintf(n, sizeof n, "tok_dec.upsample.%d.norm.weight", i); u->nw = kk_f32(m, n, NULL);
		snprintf(n, sizeof n, "tok_dec.upsample.%d.norm.bias", i); u->nb = kk_f32(m, n, NULL);
		snprintf(n, sizeof n, "tok_dec.upsample.%d.gamma", i); u->gamma = kk_f32(m, n, NULL);
		snprintf(n, sizeof n, "tok_dec.upsample.%d.pwconv1", i); kk_load_linp(m, &u->pw1, n);
		snprintf(n, sizeof n, "tok_dec.upsample.%d.pwconv2", i); kk_load_linp(m, &u->pw2, n);
	}
	qw_load_pc(m, &c->dac_pre, "tok_dec.dec.0.conv", 1);
	static const int strides[4] = { 8, 5, 4, 3 }, chs[5] = { 1536, 768, 384, 192, 96 }, dils[3] = { 1, 3, 9 };
	for (int b = 0; b < 4; b++) {
		qw_dblk *B = &c->blk[b];
		B->in = chs[b]; B->out = chs[b + 1]; B->stride = strides[b];
		snprintf(n, sizeof n, "tok_dec.dec.%d.snake", b + 1); qw_load_snake(m, &B->snake, n);
		snprintf(n, sizeof n, "tok_dec.dec.%d.conv_t.weight", b + 1); { const cg_tensor *tt = kk_need(m, n); float *w = kk_f32(m, n, NULL); qw_pc_pack_t(&B->tw, w, (int)tt->dim[1], (int)tt->dim[2], (int)tt->dim[0]); free(w); }
		snprintf(n, sizeof n, "tok_dec.dec.%d.conv_t.bias", b + 1); B->tb = kk_f32(m, n, NULL);
		for (int r = 0; r < 3; r++) {
			qw_res *R = &B->ru[r]; R->dil = dils[r];
			snprintf(n, sizeof n, "tok_dec.dec.%d.res.%d.act1", b + 1, r); qw_load_snake(m, &R->s1, n);
			snprintf(n, sizeof n, "tok_dec.dec.%d.res.%d.act2", b + 1, r); qw_load_snake(m, &R->s2, n);
			snprintf(n, sizeof n, "tok_dec.dec.%d.res.%d.conv1", b + 1, r); qw_load_pc(m, &R->c1, n, R->dil);
			snprintf(n, sizeof n, "tok_dec.dec.%d.res.%d.conv2", b + 1, r); qw_load_pc(m, &R->c2, n, 1);
		}
	}
	qw_load_snake(m, &c->snake_post, "tok_dec.dec.5.snake");
	qw_load_pc(m, &c->dac_post, "tok_dec.dec.6.conv", 1);
	return kk_err ? -1 : 0;
}

/* channel-major x[C][T] -> y[C][T]: causal conv (left pad (k-1)*dil) */
static void qw_cconv(const kk_conv *c, const float *x, int T, float *y)
{
	int pad = (c->k - 1) * c->dil;
	float *xp = (float *)calloc((size_t)c->cin * (T + pad), 4);
	for (int ic = 0; ic < c->cin; ic++) memcpy(xp + (size_t)ic * (T + pad) + pad, x + (size_t)ic * T, (size_t)T * 4);
	kk_conv cc = *c; cc.pad = 0;
	kk_conv_run(&cc, xp, T + pad, y);
	free(xp);
}

static void qw_snake_run(const qw_snake *s, float *x, int T)
{
	#pragma omp parallel for schedule(static)
	for (int c = 0; c < s->c; c++) {
		float a = s->a[c], ib = s->ib[c];
		for (int t = 0; t < T; t++) { float v = x[(size_t)c * T + t], sn = sinf(a * v); x[(size_t)c * T + t] = v + ib * sn * sn; }
	}
}

static void qw_rmsnorm(float *x, int n, const float *g, float eps)
{
	double ss = 0;
	for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
	float r = (float)(1.0 / sqrt(ss / n + eps));
	for (int i = 0; i < n; i++) x[i] = x[i] * r * g[i];
}

static inline float qw_gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678f)); }

/* NEOX rotary embedding on one head vector of size hd */
static void qw_rope(float *v, int hd, int pos, float theta)
{
	int h = hd / 2;
	for (int i = 0; i < h; i++) {
		float f = (float)pos * powf(theta, -2.f * i / hd), cs = cosf(f), sn = sinf(f);
		float a = v[i], b = v[i + h];
		v[i] = a * cs - b * sn; v[i + h] = b * cs + a * sn;
	}
}

/* sliding-window causal transformer over h[T][1024] in place (1024 -> 512 -> 1024) */
static void qw_tfm(const qw_codec *c, float *h, int T)
{
	const int D = 512, H = c->heads, hd = c->hd, A = H * hd, F = c->tl[0].gate.out;
	float *x = (float *)malloc((size_t)T * D * 4), *t = (float *)malloc((size_t)T * D * 4);
	float *q = (float *)malloc((size_t)T * A * 4), *k = (float *)malloc((size_t)T * A * 4), *v = (float *)malloc((size_t)T * A * 4), *at = (float *)malloc((size_t)T * A * 4);
	float *g = (float *)malloc((size_t)T * F * 4), *u = (float *)malloc((size_t)T * F * 4);
	qw_lin(&c->in_proj, h, T, x);
	for (int l = 0; l < 8; l++) {
		const qw_tlayer *L = &c->tl[l];
		memcpy(t, x, (size_t)T * D * 4);
		for (int i = 0; i < T; i++) qw_rmsnorm(t + (size_t)i * D, D, L->an, c->eps);
		qw_lin(&L->q, t, T, q); qw_lin(&L->k, t, T, k); qw_lin(&L->v, t, T, v);
		for (int i = 0; i < T; i++) for (int hh = 0; hh < H; hh++) { qw_rope(q + (size_t)i * A + hh * hd, hd, i, c->theta); qw_rope(k + (size_t)i * A + hh * hd, hd, i, c->theta); }
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < T; i++) for (int hh = 0; hh < H; hh++) {
			int j0 = i - c->window + 1; if (j0 < 0) j0 = 0;
			float sc[256], mx = -1e30f, sum = 0, *o = at + (size_t)i * A + hh * hd;
			const float *qi = q + (size_t)i * A + hh * hd;
			for (int j = j0; j <= i; j++) {
				const float *kj = k + (size_t)j * A + hh * hd; float s = 0;
				for (int d = 0; d < hd; d++) s += qi[d] * kj[d];
				s /= sqrtf((float)hd); sc[j - j0] = s; if (s > mx) mx = s;
			}
			for (int j = j0; j <= i; j++) { sc[j - j0] = expf(sc[j - j0] - mx); sum += sc[j - j0]; }
			for (int d = 0; d < hd; d++) o[d] = 0;
			for (int j = j0; j <= i; j++) { float w = sc[j - j0] / sum; const float *vj = v + (size_t)j * A + hh * hd; for (int d = 0; d < hd; d++) o[d] += w * vj[d]; }
		}
		qw_lin(&L->o, at, T, t);
		for (int i = 0; i < T; i++) for (int d = 0; d < D; d++) x[(size_t)i * D + d] += t[(size_t)i * D + d] * L->as[d];
		memcpy(t, x, (size_t)T * D * 4);
		for (int i = 0; i < T; i++) qw_rmsnorm(t + (size_t)i * D, D, L->fn, c->eps);
		qw_lin(&L->gate, t, T, g); qw_lin(&L->up, t, T, u);
		for (size_t i = 0; i < (size_t)T * F; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
		qw_lin(&L->down, g, T, t);
		for (int i = 0; i < T; i++) for (int d = 0; d < D; d++) x[(size_t)i * D + d] += t[(size_t)i * D + d] * L->fs[d];
	}
	for (int i = 0; i < T; i++) qw_rmsnorm(x + (size_t)i * D, D, c->tnorm, c->eps);
	qw_lin(&c->out_proj, x, T, h);
	free(x); free(t); free(q); free(k); free(v); free(at); free(g); free(u);
}

/* codes[k * T + t] (k < 16) -> malloc'ed float PCM of T * 1920 samples */
static int qw_codec_decode(const qw_codec *c, const int32_t *codes, int T, float **pcm)
{
	double qw_t0 = qw_now();
	/* RVQ: sum codebooks per group, 256 -> 512 projection (1x1 conv) */
	float *e1 = (float *)calloc((size_t)T * 256, 4), *e2 = (float *)calloc((size_t)T * 256, 4);
	for (int t = 0; t < T; t++) for (int k = 0; k < QW_NCB; k++) {
		const float *row = c->cb[k] + (size_t)codes[(size_t)k * T + t] * 256; float *d = (k == 0 ? e1 : e2) + (size_t)t * 256;
		for (int i = 0; i < 256; i++) d[i] += row[i];
	}
	float *h = (float *)malloc((size_t)512 * T * 4);	/* channel-major [512][T] */
	for (int o = 0; o < 512; o++) for (int t = 0; t < T; t++) {
		const float *w1 = c->vq_first_proj + (size_t)o * 256, *w2 = c->vq_rest_proj + (size_t)o * 256, *a = e1 + (size_t)t * 256, *b = e2 + (size_t)t * 256;
		float s = 0; for (int i = 0; i < 256; i++) s += w1[i] * a[i] + w2[i] * b[i];
		h[(size_t)o * T + t] = s;
	}
	free(e1); free(e2); QW_LAP("rvq");
	float *cm = (float *)malloc((size_t)1024 * T * 4);
	qw_pc_run(&c->pre_conv, h, T, cm); free(h);
	float *tm = (float *)malloc((size_t)1024 * T * 4);	/* time-major for the transformer */
	for (int ch = 0; ch < 1024; ch++) for (int t = 0; t < T; t++) tm[(size_t)t * 1024 + ch] = cm[(size_t)ch * T + t];
	free(cm);
	QW_LAP("pre_conv"); qw_tfm(c, tm, T); QW_LAP("transformer");
	int Tc = T;
	float *x = (float *)malloc((size_t)1024 * Tc * 4);
	for (int ch = 0; ch < 1024; ch++) for (int t = 0; t < Tc; t++) x[(size_t)ch * Tc + t] = tm[(size_t)t * 1024 + ch];
	free(tm);
	for (int b = 0; b < 2; b++) {
		const qw_cnx *u = &c->up[b];
		float *y = (float *)malloc((size_t)1024 * Tc * 2 * 4);
		qw_pct_run(&u->tw, u->tb, 1024, 2, 2, x, Tc, y);
		free(x); x = y; Tc *= 2;
		/* ConvNeXt: depthwise k7 causal, LayerNorm, MLP, layer scale */
		float *dw = (float *)malloc((size_t)1024 * Tc * 4);
		#pragma omp parallel for schedule(static)
		for (int ch = 0; ch < 1024; ch++) for (int t = 0; t < Tc; t++) {
			float s = u->dwb[ch];
			for (int k = 0; k < 7; k++) { int ti = t - 6 + k; if (ti >= 0) s += u->dww[ch * 7 + k] * x[(size_t)ch * Tc + ti]; }
			dw[(size_t)ch * Tc + t] = s;
		}
		float *r = (float *)malloc((size_t)Tc * 1024 * 4), *m1 = (float *)malloc((size_t)Tc * 4096 * 4), *m2 = (float *)malloc((size_t)Tc * 1024 * 4);
		for (int t = 0; t < Tc; t++) { for (int ch = 0; ch < 1024; ch++) r[(size_t)t * 1024 + ch] = dw[(size_t)ch * Tc + t]; kk_layernorm(r + (size_t)t * 1024, 1024, u->nw, u->nb, 1e-6f); }
		qw_lin(&u->pw1, r, Tc, m1);
		for (size_t i = 0; i < (size_t)Tc * 4096; i++) m1[i] = qw_gelu(m1[i]);
		qw_lin(&u->pw2, m1, Tc, m2);
		for (int ch = 0; ch < 1024; ch++) for (int t = 0; t < Tc; t++) x[(size_t)ch * Tc + t] += m2[(size_t)t * 1024 + ch] * u->gamma[ch];
		free(dw); free(r); free(m1); free(m2);
	}
	QW_LAP("upsample");
	/* DAC decoder */
	float *y = (float *)malloc((size_t)1536 * Tc * 4);
	qw_pc_run(&c->dac_pre, x, Tc, y); free(x); x = y;
	for (int b = 0; b < 4; b++) {
		const qw_dblk *B = &c->blk[b];
		qw_snake_run(&B->snake, x, Tc);
		int K = 2 * B->stride, To = Tc * B->stride;
		float *ny = (float *)malloc((size_t)B->out * To * 4);
		qw_pct_run(&B->tw, B->tb, B->out, K, B->stride, x, Tc, ny);
		free(x); x = ny; Tc = To; QW_LAP("dac convT");
		for (int r = 0; r < 3; r++) {
			const qw_res *R = &B->ru[r];
			float *s = (float *)malloc((size_t)B->out * Tc * 4), *s2 = (float *)malloc((size_t)B->out * Tc * 4);
			memcpy(s, x, (size_t)B->out * Tc * 4);
			qw_snake_run(&R->s1, s, Tc); qw_pc_run(&R->c1, s, Tc, s2);
			qw_snake_run(&R->s2, s2, Tc); qw_pc_run(&R->c2, s2, Tc, s);
			for (size_t i = 0; i < (size_t)B->out * Tc; i++) x[i] += s[i];
			free(s); free(s2);
		}
		QW_LAP("dac res");
	}
	qw_snake_run(&c->snake_post, x, Tc);
	float *o = (float *)malloc((size_t)Tc * 4);
	qw_pc_run(&c->dac_post, x, Tc, o); free(x);
	for (int i = 0; i < Tc; i++) o[i] = o[i] > 1.f ? 1.f : o[i] < -1.f ? -1.f : o[i];
	*pcm = o;
	return Tc;
}


/* ============================================================== byte-level BPE */

typedef struct { char **tok; int ntok; int *tsorted; char **mrg; int nmrg; int *msorted; char *b2s[256]; } qw_bpe;

static char **qw_sort_keys;
static int qw_cmp_idx(const void *a, const void *b) { return strcmp(qw_sort_keys[*(const int *)a], qw_sort_keys[*(const int *)b]); }

static int *qw_sorted_index(char **keys, int n)
{
	int *ix = (int *)malloc(sizeof(int) * n);
	for (int i = 0; i < n; i++) ix[i] = i;
	qw_sort_keys = keys; qsort(ix, n, sizeof(int), qw_cmp_idx);
	return ix;
}

static int qw_lookup(char **keys, const int *sorted, int n, const char *s)
{
	int lo = 0, hi = n - 1;
	while (lo <= hi) {
		int mid = (lo + hi) / 2, c = strcmp(keys[sorted[mid]], s);
		if (!c) return sorted[mid];
		if (c < 0) lo = mid + 1; else hi = mid - 1;
	}
	return -1;
}

static void qw_put_utf8(char *o, int cp)
{
	if (cp < 0x80) { o[0] = (char)cp; o[1] = 0; }
	else { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3F)); o[2] = 0; }
}

static int qw_bpe_open(qw_bpe *b, const cg_file *g)
{
	memset(b, 0, sizeof *b);
	b->ntok = cg_str_array(g, "tokenizer.ggml.tokens", &b->tok);
	b->nmrg = cg_str_array(g, "tokenizer.ggml.merges", &b->mrg);
	if (b->ntok <= 0 || b->nmrg <= 0) return -1;
	b->tsorted = qw_sorted_index(b->tok, b->ntok);
	b->msorted = qw_sorted_index(b->mrg, b->nmrg);
	int n = 0, used[256] = { 0 };
	for (int c = 0; c < 256; c++) {
		int vis = (c >= '!' && c <= '~') || (c >= 0xA1 && c <= 0xAC) || (c >= 0xAE && c <= 0xFF);
		char t[4]; qw_put_utf8(t, vis ? c : 256 + n++);
		b->b2s[c] = strdup(t); used[c] = vis;
	}
	return 0;
}

static int qw_cp(const unsigned char *s, int *adv)
{
	if (s[0] < 0x80) { *adv = 1; return s[0]; }
	if ((s[0] & 0xE0) == 0xC0) { *adv = 2; return ((s[0] & 0x1F) << 6) | (s[1] & 0x3F); }
	if ((s[0] & 0xF0) == 0xE0) { *adv = 3; return ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); }
	if ((s[0] & 0xF8) == 0xF0) { *adv = 4; return ((s[0] & 7) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); }
	*adv = 1; return s[0];
}
static int qw_is_nl(int c) { return c == '\n' || c == '\r'; }
static int qw_is_ws(int c) { return c == ' ' || (c >= 9 && c <= 13) || c == 0xA0 || c == 0x3000 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x85; }
static int qw_is_dig(int c) { return c >= '0' && c <= '9'; }
static int qw_is_let(int c)
{
	if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return 1;
	if (c < 0xAA) return 0;
	if (c == 0xAA || c == 0xB5 || c == 0xBA) return 1;
	if (c >= 0xC0 && c <= 0x24F) return c != 0xD7 && c != 0xF7;
	if (c >= 0x250 && c <= 0x2FF) return 1;
	if (c >= 0x370 && c <= 0x1FFF) return !(c >= 0x1680 && c < 0x1681);
	if (c >= 0x2C00 && c <= 0x2DFF) return 1;
	if (c >= 0x3040 && c <= 0x30FF && c != 0x30FB && c != 0x3099 && c != 0x309A) return 1;	/* kana, but not the middle dot */
	if (c >= 0x3400 && c <= 0x9FFF) return 1;
	if (c >= 0xAC00 && c <= 0xD7AF) return 1;
	if (c >= 0xF900 && c <= 0xFAFF) return 1;
	if (c >= 0xFF21 && c <= 0xFF3A) return 1;
	if (c >= 0xFF41 && c <= 0xFF5A) return 1;
	if (c >= 0xFF66 && c <= 0xFF9F) return 1;
	if (c >= 0x20000) return 1;
	return 0;
}
static int qw_is_oth(int c) { return !qw_is_ws(c) && !qw_is_let(c) && !qw_is_dig(c); }

/* the Qwen2 pre-tokenizer regex, hand-matched; returns the byte length of the next chunk */
static int qw_pretok(const unsigned char *s, int len)
{
	int adv, c = qw_cp(s, &adv);
	if (c == '\'' || c == 0x2019) {
		static const char *suf[] = { "s", "t", "re", "ve", "m", "ll", "d" };
		for (int k = 0; k < 7; k++) {
			int sl = (int)strlen(suf[k]), ok = adv + sl <= len;
			for (int j = 0; ok && j < sl; j++) { int ch = s[adv + j]; if (ch >= 'A' && ch <= 'Z') ch += 32; ok = ch == suf[k][j]; }
			if (ok) return adv + sl;
		}
	}
	int a2, c2;
	if (c != '\r' && c != '\n' && !qw_is_dig(c)) {	/* [^\r\n\p{L}\p{N}]?\p{L}+ */
		int i = adv;
		if (qw_is_let(c)) i = 0;
		else if (adv >= len || !qw_is_let(qw_cp(s + adv, &a2))) goto not2;
		while (i < len) { c2 = qw_cp(s + i, &a2); if (!qw_is_let(c2)) break; i += a2; }
		return i;
	}
not2:
	if (qw_is_dig(c)) return adv;
	{	/*  ?[^\s\p{L}\p{N}]+[\r\n]* */
		int i = 0;
		if (c == ' ' && adv < len && qw_is_oth(qw_cp(s + adv, &a2))) i = adv;
		else if (!qw_is_oth(c)) goto not4;
		int st = i;
		while (i < len) { c2 = qw_cp(s + i, &a2); if (!qw_is_oth(c2)) break; i += a2; }
		if (i > st) { while (i < len && qw_is_nl(s[i])) i++; return i; }
	}
not4:
	if (qw_is_ws(c)) {
		int e = 0, lastnl = -1, n;
		while (e < len) { c2 = qw_cp(s + e, &a2); if (!qw_is_ws(c2)) break; e += a2; if (qw_is_nl(c2)) lastnl = e; }
		if (lastnl > 0) return lastnl;	/* \s*[\r\n]+ */
		if (e >= len) return e;	/* \s+(?!\S) at the end of the text */
		/* all but the last whitespace character */
		n = 0; { int i = 0, prev = 0; while (i < e) { prev = i; qw_cp(s + i, &a2); i += a2; } n = prev; }
		return n > 0 ? n : e;
	}
	return adv;
}

static int qw_bpe_chunk(const qw_bpe *b, const unsigned char *s, int len, int *out)
{
	enum { MAXS = 512 };
	static char sym[MAXS][64]; int ns = 0;
	for (int i = 0; i < len && ns < MAXS; i++) { strcpy(sym[ns++], b->b2s[s[i]]); }
	for (;;) {
		int best = -1, bi = -1;
		for (int i = 0; i + 1 < ns; i++) {
			char key[140]; snprintf(key, sizeof key, "%s %s", sym[i], sym[i + 1]);
			int r = qw_lookup(b->mrg, b->msorted, b->nmrg, key);
			if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; }
		}
		if (bi < 0) break;
		strcat(sym[bi], sym[bi + 1]);
		for (int i = bi + 1; i + 1 < ns; i++) strcpy(sym[i], sym[i + 1]);
		ns--;
	}
	int n = 0;
	for (int i = 0; i < ns; i++) { int id = qw_lookup(b->tok, b->tsorted, b->ntok, sym[i]); if (id >= 0) out[n++] = id; }
	return n;
}

/* text -> token ids; "<|...|>" strings that exist in the vocabulary map to their ids */
static int qw_bpe_encode(const qw_bpe *b, const char *text, int *ids, int cap)
{
	int n = 0, len = (int)strlen(text), i = 0;
	while (i < len && n < cap - 64) {
		if (text[i] == '<' && text[i + 1] == '|') {
			const char *e = strstr(text + i, "|>");
			if (e && e - (text + i) < 40) {
				char t[48]; int tl = (int)(e - (text + i)) + 2; memcpy(t, text + i, tl); t[tl] = 0;
				int id = qw_lookup(b->tok, b->tsorted, b->ntok, t);
				if (id >= 0) { ids[n++] = id; i += tl; continue; }
			}
		}
		int j = i; /* plain text up to the next special marker */
		while (j < len && !(text[j] == '<' && text[j + 1] == '|')) j++;
		if (j == i) j = i + 1;
		int p = i;
		while (p < j) { int cl = qw_pretok((const unsigned char *)text + p, j - p); n += qw_bpe_chunk(b, (const unsigned char *)text + p, cl, ids + n); p += cl; }
		i = j;
	}
	return n;
}

/* ============================================================ talker + code predictor */

typedef struct { kk_lin q, k, v, o, gate, up, down; float *an, *fn, *qn, *kn; } qw_lmlayer;

typedef struct {
	kk_model m; qw_bpe bpe;
	qw_lmlayer tl[28], cl[5];
	int ntl, ncl, heads, kvh, hd, F, D;
	float eps, theta, ceps, ctheta;
	float *tnorm, *cnorm;
	kk_lin codec_head, fc1, fc2, lm_head[15];
	const cg_tensor *codec_embd, *text_embd, *cp_embd[15];
	int pad_id, bos_id, eos_id, think_id, nothink_id, think_bos_id, think_eos_id;
	int tts_pad, tts_bos, tts_eos;
	char **lang_names; int *lang_ids; int nlang;
	qw_codec codec;
	uint64_t rng;
} qw_tts;

static void qw_load_layer(kk_model *m, qw_lmlayer *L, const char *p)
{
	char n[260];
	#define LN(f, s) snprintf(n, sizeof n, "%s." s, p); kk_load_linp(m, &L->f, n)
	LN(q, "attn_q"); LN(k, "attn_k"); LN(v, "attn_v"); LN(o, "attn_output"); LN(gate, "ffn_gate"); LN(up, "ffn_up"); LN(down, "ffn_down");
	#undef LN
	snprintf(n, sizeof n, "%s.attn_norm.weight", p); L->an = kk_f32(m, n, NULL);
	snprintf(n, sizeof n, "%s.ffn_norm.weight", p); L->fn = kk_f32(m, n, NULL);
	snprintf(n, sizeof n, "%s.attn_q_norm.weight", p); L->qn = kk_f32(m, n, NULL);
	snprintf(n, sizeof n, "%s.attn_k_norm.weight", p); L->kn = kk_f32(m, n, NULL);
}

static int *qw_kv_i32_array(const cg_file *g, const char *key, int *n)
{
	const cg_kv *kv = cg_kvfind(g, key);
	if (!kv || kv->type != CG_ARR) return NULL;
	const uint8_t *p = kv->val; int et = (int)cg_rd32(&p); uint64_t c = cg_rd64(&p);
	int *o = (int *)malloc(sizeof(int) * (c ? c : 1));
	for (uint64_t i = 0; i < c; i++) { if (et == CG_I64 || et == CG_U64) { int64_t v; memcpy(&v, p + 8 * i, 8); o[i] = (int)v; } else { int32_t v; memcpy(&v, p + 4 * i, 4); o[i] = v; } }
	*n = (int)c; return o;
}

static inline void qw_close(qw_tts *t) { qw_codec_close(&t->codec); cg_close(&t->m.g); }

static inline int qw_open(qw_tts *t, const char *talker, const char *codec, int q8)
{
	memset(t, 0, sizeof *t); qw_threads_init();
	kk_err = 0; t->m.q8 = q8; t->rng = 0x9E3779B97F4A7C15ull;
	if (cg_open(&t->m.g, talker)) return -1;
	kk_model *m = &t->m; const cg_file *g = &m->g;
	if (qw_bpe_open(&t->bpe, g)) { fprintf(stderr, "qwen: no tokenizer in %s\n", talker); return -1; }
	t->ntl = (int)cg_u32(g, "qwen3-tts.talker.block_count", 28); t->ncl = (int)cg_u32(g, "qwen3-tts.code_pred.block_count", 5);
	t->heads = (int)cg_u32(g, "qwen3-tts.talker.attention.head_count", 16); t->kvh = (int)cg_u32(g, "qwen3-tts.talker.attention.head_count_kv", 8);
	t->hd = (int)cg_u32(g, "qwen3-tts.talker.attention.key_length", 128); t->D = (int)cg_u32(g, "qwen3-tts.talker.embedding_length", 1024);
	t->F = (int)cg_u32(g, "qwen3-tts.talker.feed_forward_length", 3072);
	t->eps = qw_kvf(g, "qwen3-tts.talker.attention.layer_norm_rms_epsilon", 1e-6f); t->theta = qw_kvf(g, "qwen3-tts.talker.rope.freq_base", 1e6f);
	t->ceps = qw_kvf(g, "qwen3-tts.code_pred.attention.layer_norm_rms_epsilon", 1e-6f); t->ctheta = qw_kvf(g, "qwen3-tts.code_pred.rope.freq_base", 1e6f);
	char n[260];
	for (int l = 0; l < t->ntl; l++) { snprintf(n, sizeof n, "talker.blk.%d", l); qw_load_layer(m, &t->tl[l], n); }
	for (int l = 0; l < t->ncl; l++) { snprintf(n, sizeof n, "code_pred.blk.%d", l); qw_load_layer(m, &t->cl[l], n); }
	t->tnorm = kk_f32(m, "talker.output_norm.weight", NULL); t->cnorm = kk_f32(m, "code_pred.output_norm.weight", NULL);
	kk_load_linp(m, &t->codec_head, "talker.codec_head"); kk_load_linp(m, &t->fc1, "talker.text_proj.fc1"); kk_load_linp(m, &t->fc2, "talker.text_proj.fc2");
	t->codec_embd = kk_need(m, "talker.codec_embd.weight"); t->text_embd = kk_need(m, "talker.text_embd.weight");
	for (int i = 0; i < 15; i++) {
		snprintf(n, sizeof n, "code_pred.lm_head.%d", i); kk_load_linp(m, &t->lm_head[i], n);
		snprintf(n, sizeof n, "code_pred.codec_embd.%d.weight", i); t->cp_embd[i] = kk_need(m, n);
	}
	t->pad_id = (int)cg_u32(g, "qwen3-tts.codec.pad_id", 0); t->bos_id = (int)cg_u32(g, "qwen3-tts.codec.bos_id", 0); t->eos_id = (int)cg_u32(g, "qwen3-tts.codec.eos_id", 0);
	t->think_id = (int)cg_u32(g, "qwen3-tts.codec.think_id", 0); t->nothink_id = (int)cg_u32(g, "qwen3-tts.codec.nothink_id", 0);
	t->think_bos_id = (int)cg_u32(g, "qwen3-tts.codec.think_bos_id", 0); t->think_eos_id = (int)cg_u32(g, "qwen3-tts.codec.think_eos_id", 0);
	t->tts_pad = (int)cg_u32(g, "qwen3-tts.text.tts_pad_id", 0); t->tts_bos = (int)cg_u32(g, "qwen3-tts.text.tts_bos_id", 0); t->tts_eos = (int)cg_u32(g, "qwen3-tts.text.tts_eos_id", 0);
	t->nlang = cg_str_array(g, "qwen3-tts.codec.language_names", &t->lang_names);
	int ni = 0; t->lang_ids = qw_kv_i32_array(g, "qwen3-tts.codec.language_ids", &ni);
	if (ni < t->nlang) t->nlang = ni;
	if (kk_err) return -1;
	return qw_codec_open(&t->codec, codec, q8);
}

/* Y[T][out] = X[T][in] W^T (+ bias). Rows are quantized once; each weight row is then streamed against all of them,
 * parallel over output rows (also good for T = 1 decode steps). */
static void qw_lin(const kk_lin *L, const float *X, int T, float *Y)
{
	if (!L->q) {
		#pragma omp parallel for schedule(static)
		for (int o = 0; o < L->out; o++) {
			const float *w = L->f + (size_t)o * L->in;
			for (int t = 0; t < T; t++) { const float *x = X + (size_t)t * L->in; float s = 0; for (int i = 0; i < L->in; i++) s += w[i] * x[i]; Y[(size_t)t * L->out + o] = s + (L->b ? L->b[o] : 0.f); }
		}
		return;
	}
	size_t qb = (size_t)(L->in / 32) * CATS_Q8_0_BYTES, rb = cats_q8_0w_row_bytes(L->in);
	int8_t *xq = (int8_t *)malloc(qb * T);
	for (int t = 0; t < T; t++) cats_q8_0_quantize(xq + qb * t, X + (size_t)t * L->in, L->in);
	#pragma omp parallel for schedule(static)
	for (int o = 0; o < L->out; o++) {
		const uint8_t *w = L->q + (size_t)o * rb; float bv = L->b ? L->b[o] : 0.f;
		for (int t = 0; t < T; t++) Y[(size_t)t * L->out + o] = cats_dot_q8_0w_q8_0(w, xq + qb * t, L->in) + bv;
	}
	free(xq);
}

/* rows x[T][D] through nl Qwen3 layers with a KV cache (kc/vc[l] = [maxpos][kvh*hd]); rope positions n_past.. */
static void qw_lm_run(const qw_lmlayer *Ls, int nl, float **kc, float **vc, int H, int KV, int hd, int F, int D, float eps, float theta, float *x, int T, int n_past)
{
	const int A = H * KV / KV * hd, B = KV * hd;
	float *t = (float *)malloc((size_t)T * D * 4), *q = (float *)malloc((size_t)T * A * 4), *k = (float *)malloc((size_t)T * B * 4), *v = (float *)malloc((size_t)T * B * 4);
	float *at = (float *)malloc((size_t)T * A * 4), *g = (float *)malloc((size_t)T * F * 4), *u = (float *)malloc((size_t)T * F * 4);
	for (int l = 0; l < nl; l++) {
		const qw_lmlayer *L = &Ls[l];
		memcpy(t, x, (size_t)T * D * 4);
		for (int i = 0; i < T; i++) qw_rmsnorm(t + (size_t)i * D, D, L->an, eps);
		qw_lin(&L->q, t, T, q); qw_lin(&L->k, t, T, k); qw_lin(&L->v, t, T, v);
		for (int i = 0; i < T; i++) {
			for (int h = 0; h < H; h++) { float *p = q + (size_t)i * A + h * hd; qw_rmsnorm(p, hd, L->qn, eps); qw_rope(p, hd, n_past + i, theta); }
			for (int h = 0; h < KV; h++) { float *p = k + (size_t)i * B + h * hd; qw_rmsnorm(p, hd, L->kn, eps); qw_rope(p, hd, n_past + i, theta); }
			memcpy(kc[l] + (size_t)(n_past + i) * B, k + (size_t)i * B, (size_t)B * 4);
			memcpy(vc[l] + (size_t)(n_past + i) * B, v + (size_t)i * B, (size_t)B * 4);
		}
		#pragma omp parallel for schedule(static) collapse(2)
		for (int i = 0; i < T; i++) for (int h = 0; h < H; h++) {
			int p = n_past + i, kh = h / (H / KV);
			float *sc = (float *)malloc(sizeof(float) * (p + 1)), mx = -1e30f, sum = 0, *o = at + (size_t)i * A + h * hd;
			const float *qi = q + (size_t)i * A + h * hd;
			for (int j = 0; j <= p; j++) { const float *kj = kc[l] + (size_t)j * B + kh * hd; float s = 0; for (int d = 0; d < hd; d++) s += qi[d] * kj[d]; s /= sqrtf((float)hd); sc[j] = s; if (s > mx) mx = s; }
			for (int j = 0; j <= p; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
			for (int d = 0; d < hd; d++) o[d] = 0;
			for (int j = 0; j <= p; j++) { float w = sc[j] / sum; const float *vj = vc[l] + (size_t)j * B + kh * hd; for (int d = 0; d < hd; d++) o[d] += w * vj[d]; }
			free(sc);
		}
		qw_lin(&L->o, at, T, t);
		for (size_t i = 0; i < (size_t)T * D; i++) x[i] += t[i];
		memcpy(t, x, (size_t)T * D * 4);
		for (int i = 0; i < T; i++) qw_rmsnorm(t + (size_t)i * D, D, L->fn, eps);
		qw_lin(&L->gate, t, T, g); qw_lin(&L->up, t, T, u);
		for (size_t i = 0; i < (size_t)T * F; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
		qw_lin(&L->down, g, T, t);
		for (size_t i = 0; i < (size_t)T * D; i++) x[i] += t[i];
	}
	free(t); free(q); free(k); free(v); free(at); free(g); free(u);
}

static void qw_embed_row(const cg_tensor *e, int id, float *out)
{
	size_t row = e->dim[0];
	if (e->type == CG_Q8_0) cats_q8_0w_dequantize_row((const uint8_t *)e->data + (size_t)id * cats_q8_0w_row_bytes(row), row, out);
	else if (e->type == CG_F16) for (size_t i = 0; i < row; i++) out[i] = cats_half_to_float(cats_load_u16((const uint8_t *)e->data + 2 * ((size_t)id * row + i)));
	else memcpy(out, (const float *)e->data + (size_t)id * row, row * 4);
}

/* text-token ids -> projected embeddings [n][D] (embedding -> fc1 -> silu -> fc2) */
static void qw_text_proj(const qw_tts *t, const int *ids, int n, float *out)
{
	int E = (int)t->text_embd->dim[0];
	float *e = (float *)malloc((size_t)n * E * 4), *h = (float *)malloc((size_t)n * t->fc1.out * 4);
	for (int i = 0; i < n; i++) qw_embed_row(t->text_embd, ids[i], e + (size_t)i * E);
	qw_lin(&t->fc1, e, n, h);
	for (size_t i = 0; i < (size_t)n * t->fc1.out; i++) h[i] = h[i] / (1.f + expf(-h[i]));
	qw_lin(&t->fc2, h, n, out);
	free(e); free(h);
}

/* prompt for plain TTS (no speaker / clone): returns rows T_ctx, out is [T_ctx][D] malloc'ed; -1 on an unknown language */
static int qw_build_prompt(const qw_tts *t, const char *text, const char *lang, float **out, int *ids_dbg, int *nids_dbg)
{
	const int D = t->D;
	char *full = (char *)malloc(strlen(text) + 80);
	snprintf(full, strlen(text) + 80, "<|im_start|>assistant\n%s<|im_end|>\n<|im_start|>assistant\n", text);
	int *ids = (int *)malloc(sizeof(int) * (strlen(full) + 16)), N = qw_bpe_encode(&t->bpe, full, ids, (int)strlen(full) + 16);
	free(full);
	if (ids_dbg) { memcpy(ids_dbg, ids, sizeof(int) * N); *nids_dbg = N; }
	int Nt = N - 3 - 5;
	if (Nt <= 0) { free(ids); return -1; }
	int lid = -1;
	if (lang && strcmp(lang, "auto")) { for (int i = 0; i < t->nlang; i++) if (!strcasecmp(t->lang_names[i], lang)) lid = t->lang_ids[i]; if (lid < 0) { free(ids); return -2; } }
	int pre[8], np = 0;
	if (lid < 0) { pre[np++] = t->nothink_id; pre[np++] = t->think_bos_id; pre[np++] = t->think_eos_id; }
	else { pre[np++] = t->think_id; pre[np++] = t->think_bos_id; pre[np++] = lid; pre[np++] = t->think_eos_id; }
	pre[np++] = t->pad_id;
	int T = 3 + np + Nt + 2;
	float *x = (float *)calloc((size_t)T * D, 4), *tmp = (float *)malloc((size_t)D * 4);
	float *sp = (float *)malloc((size_t)3 * D * 4); int sid[3] = { t->tts_bos, t->tts_eos, t->tts_pad };
	qw_text_proj(t, sid, 3, sp);
	const float *bos = sp, *eos = sp + D, *pad = sp + 2 * D;
	qw_text_proj(t, ids, 3, x);
	for (int i = 0; i < np; i++) {
		float *r = x + (size_t)(3 + i) * D; memcpy(r, i == np - 1 ? bos : pad, (size_t)D * 4);
		qw_embed_row(t->codec_embd, pre[i], tmp); for (int d = 0; d < D; d++) r[d] += tmp[d];
	}
	float *cpad = (float *)malloc((size_t)D * 4), *cbos = (float *)malloc((size_t)D * 4);
	qw_embed_row(t->codec_embd, t->pad_id, cpad); qw_embed_row(t->codec_embd, t->bos_id, cbos);
	qw_text_proj(t, ids + 3, Nt, x + (size_t)(3 + np) * D);
	for (int i = 0; i < Nt; i++) { float *r = x + (size_t)(3 + np + i) * D; for (int d = 0; d < D; d++) r[d] += cpad[d]; }
	float *r = x + (size_t)(3 + np + Nt) * D; for (int d = 0; d < D; d++) r[d] = eos[d] + cpad[d];
	r += D; for (int d = 0; d < D; d++) r[d] = pad[d] + cbos[d];
	/* the text pad embedding rides along every later step */
	t = t;
	free(ids); free(tmp); free(sp); free(cpad); free(cbos);
	*out = x;
	return T;
}

static void qw_final_norm(float *h, int D, const float *g, float eps) { qw_rmsnorm(h, D, g, eps); }

static uint64_t qw_rnd(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }
static float qw_unif(uint64_t *s) { return (float)((qw_rnd(s) >> 40) * (1.0 / 16777216.0)); }

/* greedy when temp <= 0; otherwise repetition penalty -> temperature -> top-k -> multinomial */
static int qw_sample(float *lg, int V, float temp, int topk, float rep, const int *hist, int nh, uint64_t *rng)
{
	if (temp <= 0) { int b = 0; for (int i = 1; i < V; i++) if (lg[i] > lg[b]) b = i; return b; }
	for (int h = 0; h < nh; h++) { int tk = hist[h]; int seen = 0; for (int j = 0; j < h; j++) if (hist[j] == tk) seen = 1; if (!seen && tk >= 0 && tk < V) lg[tk] = lg[tk] < 0 ? lg[tk] * rep : lg[tk] / rep; }
	for (int i = 0; i < V; i++) lg[i] /= temp;
	if (topk > 0 && topk < V) {
		float *c = (float *)malloc(sizeof(float) * V); memcpy(c, lg, sizeof(float) * V);
		for (int k = 0; k < topk; k++) { int b = k; for (int i = k + 1; i < V; i++) if (c[i] > c[b]) b = i; float x = c[k]; c[k] = c[b]; c[b] = x; }
		float th = c[topk - 1]; free(c);
		for (int i = 0; i < V; i++) if (lg[i] < th) lg[i] = -INFINITY;
	}
	float mx = -INFINITY, sum = 0; for (int i = 0; i < V; i++) if (lg[i] > mx) mx = lg[i];
	for (int i = 0; i < V; i++) { lg[i] = expf(lg[i] - mx); sum += lg[i]; }
	float r = qw_unif(rng) * sum, acc = 0;
	for (int i = 0; i < V; i++) { acc += lg[i]; if (acc >= r) return i; }
	return V - 1;
}

typedef struct { float temp, sub_temp, rep; int topk, sub_topk, max_frames; } qw_params;
static inline qw_params qw_default_params(void) { qw_params p = { 0.9f, 0.9f, 1.05f, 50, 50, 2048 }; return p; }

typedef struct { int32_t *codes; int T; float *prefill_hidden; float *prefill_logits; } qw_gen;

/* generate frames; codes laid out [k * T + t] as the codec decoder expects. */
static int qw_generate(qw_tts *t, const char *text, const char *lang, const qw_params *P, qw_gen *gen)
{
	const int D = t->D, KVD = t->kvh * t->hd, V = 3072;
	float *x; int T0 = qw_build_prompt(t, text, lang, &x, NULL, NULL);
	if (T0 < 0) return T0;
	int maxpos = T0 + P->max_frames + 4;
	float *kc[28], *vc[28], *ckc[5], *cvc[5];
	for (int l = 0; l < t->ntl; l++) { kc[l] = (float *)malloc((size_t)maxpos * KVD * 4); vc[l] = (float *)malloc((size_t)maxpos * KVD * 4); }
	for (int l = 0; l < t->ncl; l++) { ckc[l] = (float *)malloc((size_t)18 * KVD * 4); cvc[l] = (float *)malloc((size_t)18 * KVD * 4); }
	float *pad; { int id = t->tts_pad; pad = (float *)malloc((size_t)D * 4); qw_text_proj(t, &id, 1, pad); }
	qw_lm_run(t->tl, t->ntl, kc, vc, t->heads, t->kvh, t->hd, t->F, D, t->eps, t->theta, x, T0, 0);
	float *h = (float *)malloc((size_t)D * 4), *lg = (float *)malloc(sizeof(float) * V);
	memcpy(h, x + (size_t)(T0 - 1) * D, (size_t)D * 4); free(x);
	qw_final_norm(h, D, t->tnorm, t->eps);
	int cap = P->max_frames, nf = 0, pos = T0;
	int32_t *frames = (int32_t *)malloc(sizeof(int32_t) * 16 * cap); int *hist = (int *)malloc(sizeof(int) * cap);
	if (gen) { gen->prefill_hidden = (float *)malloc((size_t)D * 4); memcpy(gen->prefill_hidden, h, (size_t)D * 4); }
	float *emb = (float *)malloc((size_t)D * 4), *tmp = (float *)malloc((size_t)D * 4), *cx = (float *)malloc((size_t)2 * D * 4), *cl = (float *)malloc(sizeof(float) * 2048);
	while (nf < cap) {
		qw_lin(&t->codec_head, h, 1, lg);
		if (gen && nf == 0) { gen->prefill_logits = (float *)malloc(sizeof(float) * V); memcpy(gen->prefill_logits, lg, sizeof(float) * V); }
		for (int i = V - 1024; i < V; i++) if (i != t->eos_id) lg[i] = -INFINITY;
		int c0 = qw_sample(lg, V, P->temp, P->topk, P->rep, hist, nf, &t->rng);
		if (c0 == t->eos_id) break;
		int32_t *fr = frames + (size_t)nf * 16; fr[0] = c0; hist[nf] = c0;
		/* code predictor: [talker hidden, embed(c0)] then one code per pass */
		memcpy(cx, h, (size_t)D * 4); qw_embed_row(t->codec_embd, c0, cx + D);
		for (int g = 0; g < 15; g++) {
			int Tn = g == 0 ? 2 : 1, past = g == 0 ? 0 : g + 1;
			if (g > 0) qw_embed_row(t->cp_embd[g - 1], fr[g], cx);
			qw_lm_run(t->cl, t->ncl, ckc, cvc, t->heads, t->kvh, t->hd, t->F, D, t->ceps, t->ctheta, cx, Tn, past);
			memcpy(tmp, cx + (size_t)(Tn - 1) * D, (size_t)D * 4); qw_final_norm(tmp, D, t->cnorm, t->ceps);
			qw_lin(&t->lm_head[g], tmp, 1, cl);
			fr[g + 1] = qw_sample(cl, 2048, P->sub_temp, P->sub_topk, 1.f, NULL, 0, &t->rng);
		}
		/* next talker input: sum of the 16 code embeddings + text pad */
		qw_embed_row(t->codec_embd, c0, emb);
		for (int g = 0; g < 15; g++) { qw_embed_row(t->cp_embd[g], fr[g + 1], tmp); for (int d = 0; d < D; d++) emb[d] += tmp[d]; }
		for (int d = 0; d < D; d++) emb[d] += pad[d];
		nf++;
		qw_lm_run(t->tl, t->ntl, kc, vc, t->heads, t->kvh, t->hd, t->F, D, t->eps, t->theta, emb, 1, pos++);
		memcpy(h, emb, (size_t)D * 4); qw_final_norm(h, D, t->tnorm, t->eps);
	}
	int32_t *codes = (int32_t *)malloc(sizeof(int32_t) * 16 * (nf ? nf : 1));
	for (int f = 0; f < nf; f++) for (int k = 0; k < 16; k++) codes[(size_t)k * nf + f] = frames[(size_t)f * 16 + k];
	if (gen) { gen->codes = codes; gen->T = nf; } else free(codes);
	for (int l = 0; l < t->ntl; l++) { free(kc[l]); free(vc[l]); }
	for (int l = 0; l < t->ncl; l++) { free(ckc[l]); free(cvc[l]); }
	free(frames); free(hist); free(pad); free(h); free(lg); free(emb); free(tmp); free(cx); free(cl);
	return nf;
}

#endif
