/*
 * bonsai.h - Ternary-Bonsai 2.5 27B (PrismML, GGUF arch "qwen35", weights in PTQ1_0) on the CPU.
 *
 * The model is a Qwen3.5-style hybrid: every 4th layer is full attention (GQA, gated output, partial RoPE),
 * the others are Gated DeltaNet (causal conv + delta-rule recurrent state). All big weights are PTQ1_0 ternary
 * (see catseye.h: cats_ptq1_0_*) and were folded with a blockwise Hadamard rotation W' = W D H, so every matmul input
 * is first rotated with y = H(D x) (cats_hadamard_rows); token_embd is stored rotated and is un-rotated on lookup.
 * Everything generic (PTQ1_0 kernels, Hadamard, BPE, sampler) lives in ../catseye.h; this file is the model graph.
 */
#ifndef BONSAI_H
#define BONSAI_H

#define CATS_INFER_ONLY
#include "../catseye.h"
#include "../pronunciation/gguf.h"
#include <sys/time.h>

#define BN_MAXT 32		/* tokens per forward() call */

typedef struct {
	int full;		/* full attention (1) or Gated DeltaNet (0) */
	int idx;		/* index among layers of the same kind (state / KV cache slot) */
	const cg_tensor *qkv, *gate, *ssm_out;			/* Gated DeltaNet */
	const cg_tensor *wq, *wk, *wv, *wo;			/* full attention */
	const cg_tensor *ffn_gate, *ffn_up, *ffn_down;
	const float *attn_norm, *post_norm, *qn, *kn;
	const float *ssm_a, *dt_bias, *ssm_norm, *conv, *alpha, *beta;
} bn_layer;

typedef struct {
	cg_file g;
	cats_bpe bpe; uint8_t *bpe_blob; char **toks, **mrgs; int8_t *ttype;
	int nl, D, F, nh, nkv, hd, rope_dim, vocab;
	float rope_base, eps;
	int conv_k, sd /* head dim of the delta rule */, nk /* k heads */, nv /* v heads */, inner, conv_ch, interval;
	int hblock, nsign, swidth[4]; int8_t *sign[4];
	int v_grouped, emb_alt;
	bn_layer *L; int nfull, ngdn;
	const cg_tensor *embd, *output; const float *onorm;
	cats_ptq1_0_r2 *rp;	/* repacked weights, indexed like g.t[] (rp[i].w == NULL: use the original PTQ1_0 data) */
	int ctx, pos;		/* pos: KV cache slot of the next token */
	int rpos;		/* RoPE position of the next text token (differs from pos after an image: an image advances it by max(nx, ny)) */
	int rsec[4]; float *rfreq;	/* mrope sections [t, h, w, e] and the frequencies base^(-2j/rope_dim) */
	const float *emb_in;	/* NULL, or nt x D input embeddings (image tokens) replacing the token embeddings of the next forward() */
	const int *p3;		/* NULL, or nt x 3 RoPE positions (t, h, w) of the next forward(); NULL = text positions rpos + i */
	float *kc, *vc;		/* [nfull][nkv][ctx][hd] */
	float *S;		/* [ngdn][nv][sd][sd]  (value dim major) */
	float *cs;		/* [ngdn][conv_ch][conv_k - 1] */
	/* scratch (BN_MAXT tokens) */
	float *x, *h, *att, *qkv, *z, *qf, *kf, *vf, *g1, *g2, *al, *be, *sc, *logits;
	int8_t *xq;
	int eos, eos2;
	/* speculative decoding with the MTP head (see bn_mtp_open) */
	int nextn;		/* nextn_predict_layers of the GGUF: the last block(s) are the MTP head and not part of the main pass */
	int spec;		/* > 0: rows after the first of this batch write the recurrent state to separate buffers (rollback by pointer swap) */
	float *Sspec[3], *cspec[3];
	cg_file mg; int mg_open;
	struct { const cg_tensor *eh, *wq, *wk, *wv, *wo, *gate, *up, *down; const float *enorm, *hnorm, *attn_norm, *post_norm, *qn, *kn, *head_norm; float *kc, *vc; int ok; } mtp;
} bn_model;

static double bn_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static const cg_tensor *bn_need(const cg_file *g, const char *fmt, int i)
{
	char n[160];
	snprintf(n, sizeof n, fmt, i);
	const cg_tensor *t = cg_find(g, n);
	if (!t) fprintf(stderr, "bonsai: missing tensor %s\n", n);
	return t;
}
static const float *bn_f32(const cg_file *g, const char *fmt, int i, size_t n)
{
	const cg_tensor *t = bn_need(g, fmt, i);
	if (!t) return NULL;
	if (t->type != CG_F32 || t->n != n) { fprintf(stderr, "bonsai: %s: expected %zu f32 values\n", t->name, n); return NULL; }
	return (const float*)t->data;
}
static float bn_kv_f32(const cg_file *g, const char *key, float def)
{
	const cg_kv *k = cg_kvfind(g, key);
	if (!k || k->type != CG_F32V) return def;
	float v; memcpy(&v, k->val, 4);
	return v;
}
/* numeric array -> int8 signs (+1 / -1) */
static int8_t *bn_kv_signs(const cg_file *g, const char *key, int *n)
{
	const cg_kv *k = cg_kvfind(g, key);
	if (!k || k->type != CG_ARR) return NULL;
	const uint8_t *p = k->val;
	int et = (int)cg_rd32(&p); uint64_t cnt = cg_rd64(&p);
	int8_t *o = (int8_t*)malloc(cnt ? cnt : 1);
	for (uint64_t i = 0; i < cnt; i++) {
		double v = 0;
		switch (et) {
		case CG_I8: v = *(const int8_t*)p; p += 1; break;
		case CG_U8: case CG_BOOL: v = *p ? 1 : -1; p += 1; break;
		case CG_I16: { int16_t x; memcpy(&x, p, 2); v = x; p += 2; break; }
		case CG_I32: { int32_t x; memcpy(&x, p, 4); v = x; p += 4; break; }
		case CG_F32V: { float x; memcpy(&x, p, 4); v = x; p += 4; break; }
		case CG_I64: { int64_t x; memcpy(&x, p, 8); v = (double)x; p += 8; break; }
		default: free(o); return NULL;
		}
		o[i] = v < 0 ? -1 : 1;
	}
	*n = (int)cnt;
	return o;
}
static int *bn_kv_u32_array(const cg_file *g, const char *key, int *n)
{
	const cg_kv *k = cg_kvfind(g, key);
	if (!k || k->type != CG_ARR) return NULL;
	const uint8_t *p = k->val;
	int et = (int)cg_rd32(&p); uint64_t cnt = cg_rd64(&p);
	if (et != CG_U32 && et != CG_I32 && et != CG_U64 && et != CG_I64) return NULL;
	int *o = (int*)malloc(cnt * sizeof(int));
	for (uint64_t i = 0; i < cnt; i++) {
		if (et == CG_U32 || et == CG_I32) { uint32_t x; memcpy(&x, p, 4); o[i] = (int)x; p += 4; } else { uint64_t x; memcpy(&x, p, 8); o[i] = (int)x; p += 8; }
	}
	*n = (int)cnt;
	return o;
}

static void bn_close(bn_model *m)
{
	cats_bpe_free(&m->bpe);
	free(m->bpe_blob);
	for (int i = 0; i < 4; i++) free(m->sign[i]);
	for (int i = 0; i < 3; i++) { free(m->Sspec[i]); free(m->cspec[i]); }
	free(m->mtp.kc); free(m->mtp.vc); free(m->rfreq);
	if (m->mg_open) cg_close(&m->mg);
	if (m->rp) for (int i = 0; i < m->g.nt; i++) cats_ptq1_0_r2_free(&m->rp[i]);
	free(m->rp); free(m->L); free(m->kc); free(m->vc); free(m->S); free(m->cs);
	free(m->x); free(m->h); free(m->att); free(m->qkv); free(m->z); free(m->qf); free(m->kf); free(m->vf);
	free(m->g1); free(m->g2); free(m->al); free(m->be); free(m->sc); free(m->logits); free(m->xq);
	cg_close(&m->g);
	memset(m, 0, sizeof *m);
}

/* ctx: tokens of context to reserve (KV cache); returns 0 on success */
static int bn_open(bn_model *m, const char *path, int ctx)
{
	memset(m, 0, sizeof *m);
	if (cg_open(&m->g, path)) { fprintf(stderr, "bonsai: cannot open %s (truncated download?)\n", path); return -1; }
	const cg_file *g = &m->g;
	const cg_kv *arch = cg_kvfind(g, "general.architecture");
	if (!arch || arch->type != CG_STR || memcmp(arch->val + 8, "qwen35", 6)) { fprintf(stderr, "bonsai: architecture is not qwen35\n"); return -1; }
	m->nextn = (int)cg_u32(g, "qwen35.nextn_predict_layers", 0);
	m->nl = (int)cg_u32(g, "qwen35.block_count", 0) - m->nextn;
	m->D = (int)cg_u32(g, "qwen35.embedding_length", 0);
	m->F = (int)cg_u32(g, "qwen35.feed_forward_length", 0);
	m->nh = (int)cg_u32(g, "qwen35.attention.head_count", 0);
	m->nkv = (int)cg_u32(g, "qwen35.attention.head_count_kv", 0);
	m->hd = (int)cg_u32(g, "qwen35.attention.key_length", 0);
	m->rope_dim = (int)cg_u32(g, "qwen35.rope.dimension_count", m->hd);
	m->rope_base = bn_kv_f32(g, "qwen35.rope.freq_base", 10000.0f);
	{
		int ns = 0, *sec = bn_kv_u32_array(g, "qwen35.rope.dimension_sections", &ns);
		for (int i = 0; i < 4; i++) m->rsec[i] = sec && i < ns ? sec[i] : (i == 0 ? m->rope_dim / 2 : 0);
		free(sec);
		m->rfreq = (float*)malloc(sizeof(float) * (m->rope_dim / 2));
		for (int j = 0; j < m->rope_dim / 2; j++) m->rfreq[j] = powf(m->rope_base, -2.0f * j / m->rope_dim);
	}
	m->eps = bn_kv_f32(g, "qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
	m->conv_k = (int)cg_u32(g, "qwen35.ssm.conv_kernel", 4);
	m->sd = (int)cg_u32(g, "qwen35.ssm.state_size", 128);
	m->nk = (int)cg_u32(g, "qwen35.ssm.group_count", 16);
	m->nv = (int)cg_u32(g, "qwen35.ssm.time_step_rank", 48);
	m->inner = (int)cg_u32(g, "qwen35.ssm.inner_size", 6144);
	m->interval = (int)cg_u32(g, "qwen35.full_attention_interval", 4);
	m->conv_ch = 2 * m->nk * m->sd + m->nv * m->sd;
	if (!m->nl || !m->D || !m->nh || !m->nkv || !m->hd || m->inner != m->nv * m->sd || m->D > CATS_PTQ1_0_MAX_IN || m->F > CATS_PTQ1_0_MAX_IN) { fprintf(stderr, "bonsai: unsupported hyper-parameters\n"); return -1; }
	/* Hadamard folding */
	m->hblock = (int)cg_u32(g, "prism.hadamard.block_size", 0);
	int nw = 0, ns = 0, ne;
	int *widths = bn_kv_u32_array(g, "prism.hadamard.sign_widths", &nw);
	int8_t *signs = bn_kv_signs(g, "prism.hadamard.sign_values", &ns);
	(void)ne;
	if (m->hblock && (!widths || !signs || nw > 4)) { fprintf(stderr, "bonsai: bad prism.hadamard.* metadata\n"); return -1; }
	if (m->hblock) {
		int off = 0;
		for (int i = 0; i < nw; i++) {
			if (off + widths[i] > ns || widths[i] % m->hblock) { fprintf(stderr, "bonsai: sign table does not fit\n"); return -1; }
			m->swidth[i] = widths[i];
			m->sign[i] = (int8_t*)malloc(widths[i]);
			memcpy(m->sign[i], signs + off, widths[i]);
			off += widths[i];
		}
		m->nsign = nw;
	}
	free(widths); free(signs);
	const cg_kv *vg = cg_kvfind(g, "prism.hadamard.gdn_v_grouped");
	m->v_grouped = vg && vg->val[0] ? 1 : 0;
	m->emb_alt = getenv("BONSAI_EMB_ALT") != NULL;
	/* tensors */
	m->embd = cg_find(g, "token_embd.weight");
	m->output = cg_find(g, "output.weight");
	m->onorm = bn_f32(g, "output_norm.weight", 0, m->D);
	if (!m->embd || !m->output || !m->onorm || m->embd->type != CG_PTQ1_0 || m->output->type != CG_PTQ1_0) { fprintf(stderr, "bonsai: missing or unsupported token_embd/output\n"); return -1; }
	m->vocab = (int)m->output->dim[1];
	m->L = (bn_layer*)calloc(m->nl, sizeof(bn_layer));
	for (int i = 0; i < m->nl; i++) {
		bn_layer *L = &m->L[i];
		L->full = (i + 1) % m->interval == 0;
		L->idx = L->full ? m->nfull++ : m->ngdn++;
		L->attn_norm = bn_f32(g, "blk.%d.attn_norm.weight", i, m->D);
		L->post_norm = bn_f32(g, "blk.%d.post_attention_norm.weight", i, m->D);
		L->ffn_gate = bn_need(g, "blk.%d.ffn_gate.weight", i);
		L->ffn_up = bn_need(g, "blk.%d.ffn_up.weight", i);
		L->ffn_down = bn_need(g, "blk.%d.ffn_down.weight", i);
		if (L->full) {
			L->wq = bn_need(g, "blk.%d.attn_q.weight", i); L->wk = bn_need(g, "blk.%d.attn_k.weight", i);
			L->wv = bn_need(g, "blk.%d.attn_v.weight", i); L->wo = bn_need(g, "blk.%d.attn_output.weight", i);
			L->qn = bn_f32(g, "blk.%d.attn_q_norm.weight", i, m->hd); L->kn = bn_f32(g, "blk.%d.attn_k_norm.weight", i, m->hd);
			if (!L->wq || !L->wk || !L->wv || !L->wo || !L->qn || !L->kn) return -1;
		} else {
			L->qkv = bn_need(g, "blk.%d.attn_qkv.weight", i); L->gate = bn_need(g, "blk.%d.attn_gate.weight", i);
			L->ssm_out = bn_need(g, "blk.%d.ssm_out.weight", i);
			L->ssm_a = bn_f32(g, "blk.%d.ssm_a", i, m->nv); L->dt_bias = bn_f32(g, "blk.%d.ssm_dt.bias", i, m->nv);
			L->ssm_norm = bn_f32(g, "blk.%d.ssm_norm.weight", i, m->sd);
			L->conv = bn_f32(g, "blk.%d.ssm_conv1d.weight", i, (size_t)m->conv_k * m->conv_ch);
			L->alpha = bn_f32(g, "blk.%d.ssm_alpha.weight", i, (size_t)m->D * m->nv); L->beta = bn_f32(g, "blk.%d.ssm_beta.weight", i, (size_t)m->D * m->nv);
			if (!L->qkv || !L->gate || !L->ssm_out || !L->ssm_a || !L->dt_bias || !L->ssm_norm || !L->conv || !L->alpha || !L->beta) return -1;
		}
		if (!L->attn_norm || !L->post_norm || !L->ffn_gate || !L->ffn_up || !L->ffn_down) return -1;
	}
	/* tokenizer */
	int nt = cg_str_array(g, "tokenizer.ggml.tokens", &m->toks), nm = cg_str_array(g, "tokenizer.ggml.merges", &m->mrgs);
	const cg_kv *ty = cg_kvfind(g, "tokenizer.ggml.token_type");
	if (nt <= 0 || nm <= 0 || !ty) { fprintf(stderr, "bonsai: no BPE vocabulary in the GGUF\n"); return -1; }
	m->ttype = (int8_t*)malloc(nt);
	{ const uint8_t *p = ty->val; cg_rd32(&p); cg_rd64(&p); for (int i = 0; i < nt; i++) { int32_t v; memcpy(&v, p, 4); p += 4; m->ttype[i] = (int8_t)v; } }
	if (cats_bpe_from_gguf(&m->bpe, &m->bpe_blob, m->toks, nt, m->ttype, m->mrgs, nm, 1)) { fprintf(stderr, "bonsai: tokenizer build failed\n"); return -1; }
	m->eos = (int)cg_u32(g, "tokenizer.ggml.eos_token_id", 248046);
	m->eos2 = (int)cg_u32(g, "tokenizer.ggml.bos_token_id", 248044);
	/* state */
	m->ctx = ctx;
	m->kc = (float*)calloc((size_t)m->nfull * m->nkv * ctx * m->hd, sizeof(float));
	m->vc = (float*)calloc((size_t)m->nfull * m->nkv * ctx * m->hd, sizeof(float));
	m->S = (float*)calloc((size_t)m->ngdn * m->nv * m->sd * m->sd, sizeof(float));
	m->cs = (float*)calloc((size_t)m->ngdn * m->conv_ch * (m->conv_k - 1), sizeof(float));
	const size_t T = BN_MAXT, maxin = m->F > m->D ? m->F : m->D;
	m->x = (float*)malloc(T * m->D * 4); m->h = (float*)malloc(T * maxin * 4); m->att = (float*)malloc(T * m->D * 4);
	m->qkv = (float*)malloc(T * m->conv_ch * 4); m->z = (float*)malloc(T * m->inner * 4);
	m->qf = (float*)malloc(T * m->nh * m->hd * 2 * 4); m->kf = (float*)malloc(T * m->nkv * m->hd * 4); m->vf = (float*)malloc(T * m->nkv * m->hd * 4);
	m->g1 = (float*)malloc(T * m->F * 4); m->g2 = (float*)malloc(T * m->F * 4);
	m->al = (float*)malloc(T * m->nv * 4); m->be = (float*)malloc(T * m->nv * 4);
	m->sc = (float*)malloc((size_t)(omp_get_max_threads() + 1) * ctx * 4);
	m->logits = (float*)malloc((size_t)m->vocab * 4);
	m->xq = (int8_t*)malloc(T * cats_q8_0_row_bytes(maxin > (size_t)m->inner ? maxin : m->inner));
	if (!m->kc || !m->vc || !m->S || !m->cs || !m->x || !m->h || !m->xq || !m->sc || !m->logits) { fprintf(stderr, "bonsai: out of memory\n"); return -1; }
	if (!getenv("BONSAI_NOREPACK")) {
		/* 2-bit planes for the integer kernels (+~1 GB over the file); token_embd is only read row by row and stays as is */
		const double t0 = bn_now();
		m->rp = (cats_ptq1_0_r2*)calloc(m->g.nt, sizeof(cats_ptq1_0_r2));
		for (int i = 0; i < m->g.nt; i++) {
			const cg_tensor *t = &m->g.t[i];
			if (t->type != CG_PTQ1_0 || t == m->embd || t->nd != 2) continue;
			if (cats_ptq1_0_r2_pack(&m->rp[i], (const uint8_t*)t->data, (size_t)t->dim[1], (size_t)t->dim[0])) { fprintf(stderr, "bonsai: out of memory repacking %s\n", t->name); return -1; }
		}
		fprintf(stderr, "bonsai: repacked weights in %.1fs\n", bn_now() - t0);
	}
	return 0;
}

static void bn_reset(bn_model *m)
{
	m->pos = m->rpos = 0;
	memset(m->S, 0, (size_t)m->ngdn * m->nv * m->sd * m->sd * sizeof(float));
	memset(m->cs, 0, (size_t)m->ngdn * m->conv_ch * (m->conv_k - 1) * sizeof(float));
}

/* rotate x[nt][in] (in place) with the folded Hadamard, quantize to Q8_0 into m->xq */
static void bn_prep(bn_model *m, float *x, int nt, int in)
{
	if (m->hblock) {
		const int8_t *sg = NULL;
		for (int i = 0; i < m->nsign; i++) if (m->swidth[i] == in) sg = m->sign[i];
		if (!sg) { fprintf(stderr, "bonsai: no sign table for width %d\n", in); exit(1); }
		#pragma omp parallel for schedule(static) if(nt > 1)
		for (int t = 0; t < nt; t++) cats_hadamard_rows(x + (size_t)t * in, 1, in, m->hblock, sg, 0);
	}
	const size_t arow = cats_q8_0_row_bytes(in);
	#pragma omp parallel for schedule(static) if(nt > 1)
	for (int t = 0; t < nt; t++) cats_q8_0_quantize(m->xq + t * arow, x + (size_t)t * in, in);
}
static void bn_mm(const bn_model *m, float *out, const cg_tensor *w, int nt)
{
	const cats_ptq1_0_r2 *r = m->rp ? &m->rp[w - m->g.t] : NULL;
	if (r && r->w) { cats_matmul_r2_q8_0(out, r, m->xq, nt); return; }
	cats_matmul_ptq1_0_q8_0(out, (const uint8_t*)w->data, m->xq, nt, (size_t)w->dim[0], (size_t)w->dim[1]);
}
static void bn_embed(const bn_model *m, int id, float *out)
{
	cats_ptq1_0_dequantize_row((const uint8_t*)m->embd->data + (size_t)id * cats_ptq1_0_row_bytes(m->D), m->D, out);
	if (!m->hblock) return;
	const int8_t *sg = NULL;
	for (int i = 0; i < m->nsign; i++) if (m->swidth[i] == m->D) sg = m->sign[i];
	if (m->emb_alt) {	/* alternative order for experiments: signs after H are undone first */
		for (int i = 0; i < m->D; i++) out[i] *= sg[i];
		for (int b = 0; b < m->D; b += m->hblock) cats_fwht(out + b, m->hblock);
	} else cats_hadamard_rows(out, 1, m->D, m->hblock, sg, 1);
}
static inline float bn_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static void bn_l2norm(float *dst, const float *src, int n, float scale)
{
	double s = 0;
	for (int i = 0; i < n; i++) s += (double)src[i] * src[i];
	const float f = scale / sqrtf((float)s + 1e-6f);
	for (int i = 0; i < n; i++) dst[i] = src[i] * f;
}

/* Gated DeltaNet layer on nt tokens: h (normalized, nt x D, destroyed) -> m->att */
static void bn_gdn(bn_model *m, const bn_layer *L, float *h, int nt)
{
	const int D = m->D, sd = m->sd, nk = m->nk, nv = m->nv, C = m->conv_ch, K = m->conv_k - 1;
	/* gates from the unrotated activations (f32 weights) */
	#pragma omp parallel for schedule(static)
	for (int j = 0; j < nv; j++) for (int t = 0; t < nt; t++) {
		const float *a = L->alpha + (size_t)j * D, *b = L->beta + (size_t)j * D, *hx = h + (size_t)t * D;
		float sa = 0, sb = 0;
		for (int i = 0; i < D; i++) { sa += a[i] * hx[i]; sb += b[i] * hx[i]; }
		m->al[t * nv + j] = sa; m->be[t * nv + j] = sb;
	}
	bn_prep(m, h, nt, D);
	bn_mm(m, m->qkv, L->qkv, nt);
	bn_mm(m, m->z, L->gate, nt);
	/* causal depthwise conv (kernel conv_k) + SiLU */
	float *cs = m->cs + (size_t)L->idx * C * K;
	for (int t = 0; t < nt; t++) {
		float *row = m->qkv + (size_t)t * C;
		#pragma omp parallel for schedule(static)
		for (int c = 0; c < C; c++) {
			const float *w = L->conv + (size_t)c * m->conv_k;
			float *s = cs + (size_t)c * K;
			float acc = w[K] * row[c];
			for (int k = 0; k < K; k++) acc += w[k] * s[k];
			for (int k = 0; k + 1 < K; k++) s[k] = s[k + 1];
			s[K - 1] = row[c];
			row[c] = cats_silu(acc);
		}
		if (m->spec && t < m->spec) memcpy(m->cspec[t] + (size_t)L->idx * C * K, cs, (size_t)C * K * sizeof(float));
	}
	/* delta rule, one value head per thread */
	float *S = m->S + (size_t)L->idx * nv * sd * sd;
	float *o = m->h;	/* reuse: nt x inner */
	const int ratio = nv / nk;
	#pragma omp parallel for schedule(static)
	for (int hv = 0; hv < nv; hv++) {
		const int hk = hv % nk;	/* the graph is tiled: value head hv reads key head hv % nk */
		float *St = S + (size_t)hv * sd * sd, qn[256], kn[256];
		const float g = L->ssm_a[hv];
		for (int t = 0; t < nt; t++) {
			/* verify batches: row 0 updates the state in place, row r >= 1 reads row r-1's state and writes Sspec[r-1] */
			float *src = St, *dst = St;
			if (m->spec && t >= 1) { dst = m->Sspec[t - 1] + (St - m->S); if (t >= 2) src = m->Sspec[t - 2] + (St - m->S); }
			const float *row = m->qkv + (size_t)t * C;
			bn_l2norm(qn, row + hk * sd, sd, 1.0f / sqrtf((float)sd));
			bn_l2norm(kn, row + nk * sd + hk * sd, sd, 1.0f);
			const float *v = row + 2 * nk * sd + hv * sd;
			const float decay = expf(g * cats_softplus(m->al[t * nv + hv] + L->dt_bias[hv])), beta = bn_sigmoid(m->be[t * nv + hv]);
			float *out = o + (size_t)t * m->inner + hv * sd;
			for (int j = 0; j < sd; j++) {
				float *sr = dst + (size_t)j * sd, kv = 0;
				const float *sa = src + (size_t)j * sd;
				for (int i = 0; i < sd; i++) { sr[i] = sa[i] * decay; kv += sr[i] * kn[i]; }
				const float d = (v[j] - kv) * beta;
				float acc = 0;
				for (int i = 0; i < sd; i++) { sr[i] += kn[i] * d; acc += sr[i] * qn[i]; }
				out[j] = acc;
			}
		}
	}
	/* gated RMSNorm per head: rmsnorm(o) * ssm_norm * silu(z) */
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < nt * nv; i++) {
		float *p = o + (size_t)i * sd;
		const float *zz = m->z + (size_t)i * sd;
		cats_rmsnorm(p, p, L->ssm_norm, sd);
		for (int j = 0; j < sd; j++) p[j] *= cats_silu(zz[j]);
	}
	if (m->v_grouped) {
		/* ssm_out was folded with its input channels in grouped head order [hd, rep, nk]; the delta rule produced tiled [hd, nk, rep] */
		float *tmp = m->g1;
		for (int t = 0; t < nt; t++) for (int r = 0; r < ratio; r++) for (int hk = 0; hk < nk; hk++)
			memcpy(tmp + (size_t)t * m->inner + (hk * ratio + r) * sd, o + (size_t)t * m->inner + (r * nk + hk) * sd, sd * sizeof(float));
		memcpy(o, tmp, (size_t)nt * m->inner * sizeof(float));
	}
	bn_prep(m, o, nt, m->inner);
	bn_mm(m, m->att, L->ssm_out, nt);
}

/* full attention layer on nt tokens at positions pos..pos+nt-1 */
/* q/k norm, RoPE, KV cache and attention for the projections in m->qf / kf / vf; the gated outputs (nt x nh*hd) go to m->g2 */
/* RoPE position of token t of this batch for the pair index j: interleaved M-RoPE (t, h, w, t, h, w, ... with the section limits) */
static inline int bn_rope_pos(const bn_model *m, int t, int rbase, int j)
{
	if (!m->p3) return rbase + t;
	const int *s = m->rsec;
	int c = 3;
	if (j % 3 == 1 && j < 3 * s[1]) c = 1; else if (j % 3 == 2 && j < 3 * s[2]) c = 2; else if (j % 3 == 0 && j < 3 * s[0]) c = 0;
	return c == 3 ? 0 : m->p3[3 * t + c];
}
static void bn_attn_core(bn_model *m, float *kc, float *vc, const float *qnw, const float *knw, int nt, int pos, int rbase)
{
	const int hd = m->hd, nh = m->nh, nkv = m->nkv, rd = m->rope_dim, ctx = m->ctx;
	/* q/k norm + RoPE (NEOX pairing on the first rope_dim dims), cache k/v */
	float *qn = m->h;	/* nt x nh x hd normalized queries (h is free again after bn_prep) */
	float *gate = m->g1;	/* nt x nh x hd */
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < nt * nh; i++) {
		const int t = i / nh, hh = i % nh;
		const float *src = m->qf + ((size_t)t * nh + hh) * hd * 2;
		float *q = qn + ((size_t)t * nh + hh) * hd;
		cats_rmsnorm(q, src, qnw, hd);
		memcpy(gate + ((size_t)t * nh + hh) * hd, src + hd, hd * sizeof(float));
		for (int j = 0; j < rd / 2; j++) {
			const float ang = bn_rope_pos(m, t, rbase, j) * m->rfreq[j], c = cosf(ang), s = sinf(ang);
			const float a = q[j], b = q[j + rd / 2];
			q[j] = a * c - b * s; q[j + rd / 2] = a * s + b * c;
		}
	}
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < nt * nkv; i++) {
		const int t = i / nkv, kh = i % nkv;
		float *k = kc + ((size_t)kh * ctx + pos + t) * hd, *v = vc + ((size_t)kh * ctx + pos + t) * hd;
		cats_rmsnorm(k, m->kf + ((size_t)t * nkv + kh) * hd, knw, hd);
		memcpy(v, m->vf + ((size_t)t * nkv + kh) * hd, hd * sizeof(float));
		for (int j = 0; j < rd / 2; j++) {
			const float ang = bn_rope_pos(m, t, rbase, j) * m->rfreq[j], c = cosf(ang), s = sinf(ang);
			const float a = k[j], b = k[j + rd / 2];
			k[j] = a * c - b * s; k[j + rd / 2] = a * s + b * c;
		}
	}
	float *ao = m->g2;	/* nt x nh x hd */
	const float scale = 1.0f / sqrtf((float)hd);
	const int grp = nh / nkv;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < nt * nh; i++) {
		const int t = i / nh, hh = i % nh, kh = hh / grp, n = pos + t + 1;
		float *sc = m->sc + (size_t)omp_get_thread_num() * ctx;
		const float *q = qn + ((size_t)t * nh + hh) * hd, *K = kc + (size_t)kh * ctx * hd, *V = vc + (size_t)kh * ctx * hd;
		float mx = -1e30f;
		for (int p = 0; p < n; p++) {
			float s = 0;
			const float *k = K + (size_t)p * hd;
			for (int j = 0; j < hd; j++) s += q[j] * k[j];
			sc[p] = s * scale;
			if (sc[p] > mx) mx = sc[p];
		}
		float sum = 0;
		for (int p = 0; p < n; p++) { sc[p] = expf(sc[p] - mx); sum += sc[p]; }
		float *out = ao + ((size_t)t * nh + hh) * hd;
		memset(out, 0, hd * sizeof(float));
		for (int p = 0; p < n; p++) {
			const float w = sc[p] / sum, *v = V + (size_t)p * hd;
			for (int j = 0; j < hd; j++) out[j] += w * v[j];
		}
		const float *gt = gate + ((size_t)t * nh + hh) * hd;
		for (int j = 0; j < hd; j++) out[j] *= bn_sigmoid(gt[j]);
	}
}
static void bn_attn(bn_model *m, const bn_layer *L, float *h, int nt)
{
	const size_t slot = (size_t)L->idx * m->nkv * m->ctx * m->hd;
	bn_prep(m, h, nt, m->D);
	bn_mm(m, m->qf, L->wq, nt);
	bn_mm(m, m->kf, L->wk, nt);
	bn_mm(m, m->vf, L->wv, nt);
	bn_attn_core(m, m->kc + slot, m->vc + slot, L->qn, L->kn, nt, m->pos, m->rpos);
	bn_prep(m, m->g2, nt, m->nh * m->hd);
	bn_mm(m, m->att, L->wo, nt);
}

/* Runs nt <= BN_MAXT tokens through the model, appending to the context.
 * logits: NULL, or room for the logits of the last token (all_rows == 0) or of every token (nt x vocab).
 * hn:     NULL, or nt x D floats receiving the final-norm hidden state of every token (the input of the MTP head). */
static int bn_forward_ex(bn_model *m, const int *tok, int nt, float *logits, int all_rows, float *hn)
{
	const int D = m->D;
	if (nt > BN_MAXT || m->pos + nt > m->ctx) { fprintf(stderr, "bonsai: context full (%d)\n", m->ctx); return -1; }
	if (m->emb_in) memcpy(m->x, m->emb_in, (size_t)nt * D * sizeof(float));
	else for (int t = 0; t < nt; t++) bn_embed(m, tok[t], m->x + (size_t)t * D);
	for (int li = 0; li < m->nl; li++) {
		const bn_layer *L = &m->L[li];
		for (int t = 0; t < nt; t++) cats_rmsnorm(m->h + (size_t)t * D, m->x + (size_t)t * D, L->attn_norm, D);
		if (L->full) bn_attn(m, L, m->h, nt); else bn_gdn(m, L, m->h, nt);
		for (int i = 0; i < nt * D; i++) m->x[i] += m->att[i];
		for (int t = 0; t < nt; t++) cats_rmsnorm(m->h + (size_t)t * D, m->x + (size_t)t * D, L->post_norm, D);
		bn_prep(m, m->h, nt, D);
		bn_mm(m, m->g1, L->ffn_gate, nt);
		bn_mm(m, m->g2, L->ffn_up, nt);
		for (size_t i = 0; i < (size_t)nt * m->F; i++) m->g1[i] = cats_silu(m->g1[i]) * m->g2[i];
		bn_prep(m, m->g1, nt, m->F);
		bn_mm(m, m->att, L->ffn_down, nt);
		for (int i = 0; i < nt * D; i++) m->x[i] += m->att[i];
	}
	m->pos += nt; m->rpos += nt;
	if (logits || hn) {
		const int r0 = logits && !all_rows ? nt - 1 : 0, nr = logits && !all_rows ? 1 : nt;
		for (int t = r0; t < r0 + nr; t++) cats_rmsnorm(m->h + (size_t)(t - r0) * D, m->x + (size_t)t * D, m->onorm, D);
		if (hn) {
			if (nr == nt) memcpy(hn, m->h, (size_t)nt * D * sizeof(float));
			else for (int t = 0; t < nt; t++) cats_rmsnorm(hn + (size_t)t * D, m->x + (size_t)t * D, m->onorm, D);
		}
		if (logits) { bn_prep(m, m->h, nr, D); bn_mm(m, logits, m->output, nr); }
	}
	return 0;
}
static int bn_forward(bn_model *m, const int *tok, int nt, float *logits) { return bn_forward_ex(m, tok, nt, logits, 0, NULL); }

/* ---- MTP draft head (blk.<nl>: eh_proj, one gated-attention block, shared output head; Q4_0 weights, not Hadamard folded) ---- */
static int bn_mtp_open(bn_model *m, const char *path)
{
	const cg_file *g = &m->g;
	if (path) {
		if (cg_open(&m->mg, path)) { fprintf(stderr, "bonsai: cannot open MTP file %s\n", path); return -1; }
		m->mg_open = 1; g = &m->mg;
	}
	const int il = m->nl;
	char n[160];
#define MT(field, fmt) do { snprintf(n, sizeof n, fmt, il); m->mtp.field = cg_find(g, n); if (!m->mtp.field) { fprintf(stderr, "bonsai: MTP tensor %s not found\n", n); return -1; } } while (0)
	MT(eh, "blk.%d.nextn.eh_proj.weight"); MT(wq, "blk.%d.attn_q.weight"); MT(wk, "blk.%d.attn_k.weight"); MT(wv, "blk.%d.attn_v.weight");
	MT(wo, "blk.%d.attn_output.weight"); MT(gate, "blk.%d.ffn_gate.weight"); MT(up, "blk.%d.ffn_up.weight"); MT(down, "blk.%d.ffn_down.weight");
#undef MT
#define MF(field, fmt, cnt) do { snprintf(n, sizeof n, fmt, il); const cg_tensor *t_ = cg_find(g, n); if (!t_ || t_->type != CG_F32 || t_->n != (size_t)(cnt)) { fprintf(stderr, "bonsai: MTP tensor %s missing\n", n); return -1; } m->mtp.field = (const float*)t_->data; } while (0)
	MF(enorm, "blk.%d.nextn.enorm.weight", m->D); MF(hnorm, "blk.%d.nextn.hnorm.weight", m->D); MF(attn_norm, "blk.%d.attn_norm.weight", m->D);
	MF(post_norm, "blk.%d.post_attention_norm.weight", m->D); MF(qn, "blk.%d.attn_q_norm.weight", m->hd); MF(kn, "blk.%d.attn_k_norm.weight", m->hd);
	MF(head_norm, "blk.%d.nextn.shared_head_norm.weight", m->D);
#undef MF
	const cg_tensor *w[] = { m->mtp.eh, m->mtp.wq, m->mtp.wk, m->mtp.wv, m->mtp.wo, m->mtp.gate, m->mtp.up, m->mtp.down };
	for (int i = 0; i < 8; i++) if (w[i]->type != CG_Q4_0) { fprintf(stderr, "bonsai: MTP weight %s is not Q4_0 (type %d)\n", w[i]->name, w[i]->type); return -1; }
	m->mtp.kc = (float*)calloc((size_t)m->nkv * m->ctx * m->hd, sizeof(float));
	m->mtp.vc = (float*)calloc((size_t)m->nkv * m->ctx * m->hd, sizeof(float));
	if (!m->mtp.kc || !m->mtp.vc) return -1;
	m->mtp.ok = 1;
	return 0;
}
/* Allocates the rollback buffers for verify batches of up to `rows` draft tokens (rows <= 3, 150 MB each). */
static int bn_spec_alloc(bn_model *m, int rows)
{
	for (int i = 0; i < rows; i++) {
		if (m->Sspec[i]) continue;
		m->Sspec[i] = (float*)calloc((size_t)m->ngdn * m->nv * m->sd * m->sd, sizeof(float));
		m->cspec[i] = (float*)calloc((size_t)m->ngdn * m->conv_ch * (m->conv_k - 1), sizeof(float));
		if (!m->Sspec[i] || !m->cspec[i]) return -1;
	}
	return 0;
}
/* After a verify batch run with m->spec = D: keep the state after row `acc` (0 <= acc <= D) */
static void bn_spec_commit(bn_model *m, int acc)
{
	/* (the caller resets pos and rpos) */
	const int D = m->spec;
	if (acc >= 1) { float *t = m->S; m->S = m->Sspec[acc - 1]; m->Sspec[acc - 1] = t; }
	if (acc < D) memcpy(m->cs, m->cspec[acc], (size_t)m->ngdn * m->conv_ch * (m->conv_k - 1) * sizeof(float));
	m->spec = 0;
}
/* One MTP step on nt <= 3 tokens: toks[i] sits at position pos0 + i, hprev[i] is the main model's final-norm hidden state
 * of the position before it. Writes the MTP KV cache; logits (vocab) and hout (D) for the last token if not NULL. */
static void bn_mtp_step(bn_model *m, const int *toks, const float *hprev, int nt, int pos0, float *logits, float *hout)
{
	const int D = m->D, hd = m->hd, nh = m->nh;
	float *cat = (float*)malloc((size_t)nt * 2 * D * 4), *cur = (float*)malloc((size_t)nt * D * 4), *res = (float*)malloc((size_t)nt * D * 4);
	float *hh = (float*)malloc((size_t)nt * m->F * 4), *u = (float*)malloc((size_t)nt * m->F * 4), *ao = (float*)malloc((size_t)nt * nh * hd * 4);
	float *e = (float*)malloc((size_t)D * 4);
	for (int t = 0; t < nt; t++) {
		bn_embed(m, toks[t], e);
		cats_rmsnorm(cat + (size_t)t * 2 * D, e, m->mtp.enorm, D);
		cats_rmsnorm(cat + (size_t)t * 2 * D + D, hprev + (size_t)t * D, m->mtp.hnorm, D);
	}
	cats_gemm_nt_q4_0(cur, cat, (const uint8_t*)m->mtp.eh->data, nt, 2 * D, D, NULL);
	memcpy(res, cur, (size_t)nt * D * 4);
	for (int t = 0; t < nt; t++) cats_rmsnorm(hh + (size_t)t * D, cur + (size_t)t * D, m->mtp.attn_norm, D);
	cats_gemm_nt_q4_0(m->qf, hh, (const uint8_t*)m->mtp.wq->data, nt, D, nh * hd * 2, NULL);
	cats_gemm_nt_q4_0(m->kf, hh, (const uint8_t*)m->mtp.wk->data, nt, D, m->nkv * hd, NULL);
	cats_gemm_nt_q4_0(m->vf, hh, (const uint8_t*)m->mtp.wv->data, nt, D, m->nkv * hd, NULL);
	bn_attn_core(m, m->mtp.kc, m->mtp.vc, m->mtp.qn, m->mtp.kn, nt, pos0, pos0);
	memcpy(ao, m->g2, (size_t)nt * nh * hd * 4);
	cats_gemm_nt_q4_0(cur, ao, (const uint8_t*)m->mtp.wo->data, nt, nh * hd, D, NULL);
	for (int i = 0; i < nt * D; i++) res[i] += cur[i];
	for (int t = 0; t < nt; t++) cats_rmsnorm(hh + (size_t)t * D, res + (size_t)t * D, m->mtp.post_norm, D);
	{
		float *g = (float*)malloc((size_t)nt * m->F * 4);
		cats_gemm_nt_q4_0(g, hh, (const uint8_t*)m->mtp.gate->data, nt, D, m->F, NULL);
		cats_gemm_nt_q4_0(u, hh, (const uint8_t*)m->mtp.up->data, nt, D, m->F, NULL);
		for (size_t i = 0; i < (size_t)nt * m->F; i++) g[i] = cats_silu(g[i]) * u[i];
		cats_gemm_nt_q4_0(cur, g, (const uint8_t*)m->mtp.down->data, nt, m->F, D, NULL);
		free(g);
	}
	for (int i = 0; i < nt * D; i++) res[i] += cur[i];
	if (hout || logits) {
		float *hl = (float*)malloc((size_t)D * 4);
		cats_rmsnorm(hl, res + (size_t)(nt - 1) * D, m->mtp.head_norm, D);
		if (hout) memcpy(hout, hl, (size_t)D * 4);
		if (logits) { bn_prep(m, hl, 1, D); bn_mm(m, logits, m->output, 1); }
		free(hl);
	}
	free(cat); free(cur); free(res); free(hh); free(u); free(ao); free(e);
}

#endif
