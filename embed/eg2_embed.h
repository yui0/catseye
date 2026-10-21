//---------------------------------------------------------
//	Cat's eye - EmbeddingGemma 2 embeddings (text / image)
//
//		Native C port of google/embeddinggemma-2 for the vision tower and the shared text
//		backbone. Reads the original model.safetensors (bf16) directly; no Python at runtime.
//		The generic kernels (GEMM, RoPE, attention, safetensors) live in ../catseye.h.
//---------------------------------------------------------

#ifndef EG2_EMBED_H
#define EG2_EMBED_H

#define CATS_INFER_ONLY
#define CATS_NO_ALLOC_MACROS
#include "../catseye.h"
#undef CATS_INFER_ONLY
#undef CATS_NO_ALLOC_MACROS

#define EG2E_TEXT_DIM		512
#define EG2E_OUT_DIM		768
#define EG2E_TEXT_LAYERS	24
#define EG2E_PLE_DIM		512
#define EG2E_VIS_DIM		768
#define EG2E_VIS_LAYERS		16
#define EG2E_VIS_HEADS		12
#define EG2E_VIS_HEAD_DIM	64
#define EG2E_VIS_FF		3072
#define EG2E_PATCH		16
#define EG2E_POOL		3
#define EG2E_POS_TABLE		10240
#define EG2E_EPS		1e-6f
#define EG2E_TOK_BOS		2
#define EG2E_TOK_EOS		1
#define EG2E_TOK_BOI		255999
#define EG2E_TOK_EOI		258882
#define EG2E_TOK_IMAGE		258880
#define EG2E_SLIDING_WINDOW	512	// |i-j| <= 512 on the local layers

// A weight matrix [out][in]: float32 (widened from bf16 at load) or Q4_0 bytes used straight from the mapping.
typedef struct { float *f; const uint8_t *q; } eg2e_mat;

typedef struct {
	float *in_ln, *post_attn_ln, *pre_ff_ln, *post_ff_ln;
	float *q_norm, *k_norm;
	eg2e_mat q, k, v, o, gate, up, down;
	eg2e_mat ple_gate, ple_proj;			// text only
	float *ple_norm;				// text only
	float scalar;					// text only (layer_scalar)
	unsigned nq, nkv, hd;
	int full;					// full (global) attention layer
} eg2e_layer;

typedef struct {
	cats_st st;
	int has_vision;
	eg2e_layer text[EG2E_TEXT_LAYERS];
	float *text_norm;				// final RMSNorm
	eg2e_mat text_proj;				// Linear 512 -> 768
	eg2e_mat ple_proj;				// per-layer-input projection (12288 x 512)
	float *ple_norm;				// ... and its norm
	eg2e_layer vis[EG2E_VIS_LAYERS];
	eg2e_mat patch_w, embed_vision;			// patch Linear, Linear 768 -> 512
	float *pos_table;				// [2][10240][768]
	const cats_st_tensor *embed_tokens;		// bf16 table, widened row by row
} eg2e_model;

// ---- stage timers (eg2e -v prints them) ---------------------------------------------------------

enum { EG2E_P_VNORM, EG2E_P_VPROJ, EG2E_P_VROPE, EG2E_P_VATT, EG2E_P_VMLP, EG2E_P_VOTHER,
       EG2E_P_TNORM, EG2E_P_TPROJ, EG2E_P_TATT, EG2E_P_TMLP, EG2E_P_TPLE, EG2E_P_TOTHER, EG2E_P_COUNT };
static const char *const eg2e_prof_name[EG2E_P_COUNT] = { "vision norm", "vision qkvo", "vision rope/norm", "vision attention",
	"vision mlp", "vision other", "text norm", "text qkvo", "text attention", "text mlp", "text ple", "text other" };
static double eg2e_prof[EG2E_P_COUNT];
static int eg2e_prof_on;
static double eg2e_clock(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + 1e-9 * t.tv_nsec;
}
// EG2E_TICK(&t) at the start of a stage, EG2E_TOCK(&t, stage) at its end
#define EG2E_TICK(t)		do { if (eg2e_prof_on) *(t) = eg2e_clock(); } while (0)
#define EG2E_TOCK(t, stage)	do { if (eg2e_prof_on) { double n_ = eg2e_clock(); eg2e_prof[stage] += n_ - *(t); *(t) = n_; } } while (0)

// ---- weights -------------------------------------------------------------------------------

static float *eg2e_get(eg2e_model *m, const char *fmt, int idx, int64_t want)
{
	char name[256];
	snprintf(name, sizeof(name), fmt, idx);
	const cats_st_tensor *t = cats_st_find(&m->st, name);
	if (!t || (want > 0 && cats_st_numel(t) != want)) {
		fprintf(stderr, "eg2e: missing or mis-sized tensor %s\n", name);
		return NULL;
	}
	float *p = (float*)malloc((size_t)cats_st_numel(t) * sizeof(float));
	if (p && cats_st_read_f32(&m->st, t, p)) { free(p); p = NULL; }
	return p;
}
// Loads a weight matrix; Q4_0 tensors stay quantized (no copy), everything else becomes float32.
static int eg2e_getmat(eg2e_model *m, const char *fmt, int idx, int64_t want, eg2e_mat *out)
{
	char name[256];
	snprintf(name, sizeof(name), fmt, idx);
	const cats_st_tensor *t = cats_st_find(&m->st, name);
	if (!t || (want > 0 && cats_st_numel(t) != want)) {
		fprintf(stderr, "eg2e: missing or mis-sized tensor %s\n", name);
		return -1;
	}
	out->f = NULL; out->q = NULL;
	if (t->dtype == CATS_ST_Q4_0) {
		if (t->ndim != 2 || t->shape[1] % CATS_Q4_0_BLOCK || cats_st_bytes(t->dtype, cats_st_numel(t)) != t->end - t->begin) return -1;
		out->q = cats_st_raw(&m->st, t);
		return 0;
	}
	out->f = eg2e_get(m, fmt, idx, want);
	return out->f ? 0 : -1;
}
static void eg2e_mm(float *c, const float *x, const eg2e_mat *w, size_t n, size_t in, size_t out)
{
	if (w->q) cats_gemm_nt_q4_0(c, x, w->q, n, in, out, NULL);
	else cats_gemm_nt(c, x, w->f, n, in, out, NULL);
}
#define EG2E_NEED(p) do { if (!(p)) return -1; } while (0)

static int eg2e_load_text_layer(eg2e_model *m, int i)
{
	eg2e_layer *l = &m->text[i];
	l->full = (i % 6 == 5);			// every sixth layer is global (config.layer_types)
	l->hd = l->full ? 512 : 256;
	l->nq = 4;
	l->nkv = l->full ? 1 : 2;
	char f[200];
#define P(field, key, n) snprintf(f, sizeof f, "language_model.layers.%%d." key); \
	l->field = eg2e_get(m, f, i, n); EG2E_NEED(l->field)
#define PM(field, key, n) snprintf(f, sizeof f, "language_model.layers.%%d." key); \
	EG2E_NEED(!eg2e_getmat(m, f, i, n, &l->field))
	P(in_ln, "input_layernorm.weight", 512);
	P(post_attn_ln, "post_attention_layernorm.weight", 512);
	P(pre_ff_ln, "pre_feedforward_layernorm.weight", 512);
	P(post_ff_ln, "post_feedforward_layernorm.weight", 512);
	P(q_norm, "self_attn.q_norm.weight", l->hd);
	P(k_norm, "self_attn.k_norm.weight", l->hd);
	PM(q, "self_attn.q_proj.weight", (int64_t)l->nq * l->hd * 512);
	PM(k, "self_attn.k_proj.weight", (int64_t)l->nkv * l->hd * 512);
	PM(v, "self_attn.v_proj.weight", (int64_t)l->nkv * l->hd * 512);
	PM(o, "self_attn.o_proj.weight", (int64_t)l->nq * l->hd * 512);
	PM(gate, "mlp.gate_proj.weight", 2048 * 512);
	PM(up, "mlp.up_proj.weight", 2048 * 512);
	PM(down, "mlp.down_proj.weight", 2048 * 512);
	PM(ple_gate, "ple_block.per_layer_input_gate.weight", 512 * 512);
	PM(ple_proj, "ple_block.per_layer_projection.weight", 512 * 512);
	P(ple_norm, "ple_block.post_per_layer_input_norm.weight", 512);
	float *sc;
	snprintf(f, sizeof f, "language_model.layers.%%d.layer_scalar");
	sc = eg2e_get(m, f, i, 1);
	EG2E_NEED(sc);
	l->scalar = sc[0];
	free(sc);
#undef P
#undef PM
	return 0;
}

static int eg2e_load_vision_layer(eg2e_model *m, int i)
{
	eg2e_layer *l = &m->vis[i];
	l->hd = EG2E_VIS_HEAD_DIM;
	l->nq = l->nkv = EG2E_VIS_HEADS;
	char f[200];
#define P(field, key, n) snprintf(f, sizeof f, "vision_tower.encoder.layers.%%d." key); \
	l->field = eg2e_get(m, f, i, n); EG2E_NEED(l->field)
#define PM(field, key, n) snprintf(f, sizeof f, "vision_tower.encoder.layers.%%d." key); \
	EG2E_NEED(!eg2e_getmat(m, f, i, n, &l->field))
	P(in_ln, "input_layernorm.weight", 768);
	P(post_attn_ln, "post_attention_layernorm.weight", 768);
	P(pre_ff_ln, "pre_feedforward_layernorm.weight", 768);
	P(post_ff_ln, "post_feedforward_layernorm.weight", 768);
	P(q_norm, "self_attn.q_norm.weight", 64);
	P(k_norm, "self_attn.k_norm.weight", 64);
	PM(q, "self_attn.q_proj.linear.weight", 768 * 768);
	PM(k, "self_attn.k_proj.linear.weight", 768 * 768);
	PM(v, "self_attn.v_proj.linear.weight", 768 * 768);
	PM(o, "self_attn.o_proj.linear.weight", 768 * 768);
	PM(gate, "mlp.gate_proj.linear.weight", 768 * 3072);
	PM(up, "mlp.up_proj.linear.weight", 768 * 3072);
	PM(down, "mlp.down_proj.linear.weight", 768 * 3072);
#undef P
#undef PM
	return 0;
}

static void eg2e_free(eg2e_model *m)
{
	for (int i = 0; i < EG2E_TEXT_LAYERS; i++) {
		eg2e_layer *l = &m->text[i];
		float **p[] = { &l->in_ln, &l->post_attn_ln, &l->pre_ff_ln, &l->post_ff_ln, &l->q_norm, &l->k_norm, &l->q.f,
				&l->k.f, &l->v.f, &l->o.f, &l->gate.f, &l->up.f, &l->down.f, &l->ple_gate.f, &l->ple_proj.f, &l->ple_norm };
		for (unsigned j = 0; j < sizeof(p) / sizeof(*p); j++) { free(*p[j]); *p[j] = NULL; }
	}
	for (int i = 0; i < EG2E_VIS_LAYERS; i++) {
		eg2e_layer *l = &m->vis[i];
		float **p[] = { &l->in_ln, &l->post_attn_ln, &l->pre_ff_ln, &l->post_ff_ln, &l->q_norm, &l->k_norm, &l->q.f,
				&l->k.f, &l->v.f, &l->o.f, &l->gate.f, &l->up.f, &l->down.f };
		for (unsigned j = 0; j < sizeof(p) / sizeof(*p); j++) { free(*p[j]); *p[j] = NULL; }
	}
	free(m->text_norm); free(m->text_proj.f); free(m->ple_proj.f); free(m->ple_norm);
	free(m->patch_w.f); free(m->pos_table); free(m->embed_vision.f);
	cats_st_close(&m->st);
	memset(m, 0, sizeof(*m));
}

// dir holds model.safetensors from google/embeddinggemma-2. The vision tower is optional.
static int eg2e_load(eg2e_model *m, const char *dir, int with_vision)
{
	memset(m, 0, sizeof(*m));
	char path[1024];
	snprintf(path, sizeof(path), "%s/model.safetensors", dir);
	if (cats_st_open(&m->st, path)) { fprintf(stderr, "eg2e: cannot open %s\n", path); return -1; }
	int rc = -1;
	for (int i = 0; i < EG2E_TEXT_LAYERS; i++) if (eg2e_load_text_layer(m, i)) goto fail;
	if (!(m->text_norm = eg2e_get(m, "language_model.norm.weight", 0, 512))) goto fail;
	if (eg2e_getmat(m, "language_model.embedding_projection.weight", 0, 768 * 512, &m->text_proj)) goto fail;
	if (eg2e_getmat(m, "language_model.ple.per_layer_model_projection.weight", 0,
			(int64_t)EG2E_TEXT_LAYERS * EG2E_PLE_DIM * 512, &m->ple_proj)) goto fail;
	if (!(m->ple_norm = eg2e_get(m, "language_model.ple.per_layer_projection_norm.weight", 0, EG2E_PLE_DIM))) goto fail;
	m->embed_tokens = cats_st_find(&m->st, "language_model.embed_tokens.weight");
	if (!m->embed_tokens || m->embed_tokens->ndim != 2 || m->embed_tokens->shape[1] != 512) goto fail;
	if (with_vision) {
		for (int i = 0; i < EG2E_VIS_LAYERS; i++) if (eg2e_load_vision_layer(m, i)) goto fail;
		if (eg2e_getmat(m, "vision_tower.patch_embedder.input_proj.weight", 0, 768 * 768, &m->patch_w)) goto fail;
		if (!(m->pos_table = eg2e_get(m, "vision_tower.patch_embedder.position_embedding_table", 0,
					      2LL * EG2E_POS_TABLE * 768))) goto fail;
		if (eg2e_getmat(m, "embed_vision.embedding_projection.weight", 0, 512 * 768, &m->embed_vision)) goto fail;
		m->has_vision = 1;
	}
	rc = 0;
fail:
	if (rc) { fprintf(stderr, "eg2e: incomplete model in %s\n", dir); eg2e_free(m); }
	return rc;
}

// ---- shared pieces -------------------------------------------------------------------------

static void eg2e_rms_rows(float *dst, const float *src, const float *w, size_t rows, unsigned d)
{
	#pragma omp parallel for schedule(static)
	for (long r = 0; r < (long)rows; r++) {
		if (w) cats_rmsnorm(dst + r * d, src + r * d, w, d);
		else cats_rmsnorm_noscale(dst + r * d, src + r * d, d, EG2E_EPS);
	}
}
// Per-head RMSNorm over x[t][heads*hd] (the q/k/v normalisations act on every head vector).
static void eg2e_head_norm(float *x, const float *w, size_t t, unsigned heads, unsigned hd)
{
	#pragma omp parallel for schedule(static)
	for (long i = 0; i < (long)(t * heads); i++) {
		float *p = x + (size_t)i * hd;
		if (w) cats_rmsnorm(p, p, w, hd);
		else cats_rmsnorm_noscale(p, p, hd, EG2E_EPS);
	}
}
static void eg2e_gated_mlp(float *out, const float *x, const eg2e_layer *l, size_t t, unsigned d, unsigned ff)
{
	float *g = (float*)malloc(t * ff * sizeof(float)), *u = (float*)malloc(t * ff * sizeof(float));
	eg2e_mm(g, x, &l->gate, t, d, ff);
	eg2e_mm(u, x, &l->up, t, d, ff);
	#pragma omp parallel for schedule(static)
	for (long i = 0; i < (long)(t * ff); i++) g[i] = cats_gelu_tanh(g[i]) * u[i];
	eg2e_mm(out, g, &l->down, t, ff, d);
	free(g); free(u);
}

// ---- text backbone: inputs_embeds[t][512] -> mean pooled, L2-normalised vector[768] --------------

// hidden_dump (optional) receives the output of every layer: [24][t][512], for layer-by-layer checks.
static int eg2e_backbone(const eg2e_model *m, const float *embeds, size_t t, float *out768, float *hidden_dump)
{
	const unsigned D = 512;
	float *x = (float*)malloc(t * D * sizeof(float)), *h = (float*)malloc(t * D * sizeof(float));
	float *per = (float*)malloc(t * EG2E_TEXT_LAYERS * EG2E_PLE_DIM * sizeof(float));
	float *pos = (float*)malloc(t * sizeof(float));
	if (!x || !h || !per || !pos) { free(x); free(h); free(per); free(pos); return -1; }
	memcpy(x, embeds, t * D * sizeof(float));
	for (size_t i = 0; i < t; i++) pos[i] = (float)i;

	// per-layer inputs: projection of the embeddings, scaled, then RMSNorm over each 512-d slice
	eg2e_mm(per, embeds, &m->ple_proj, t, D, EG2E_TEXT_LAYERS * EG2E_PLE_DIM);
	const float ple_scale = 1.0f / sqrtf((float)D);
	for (size_t i = 0; i < t * EG2E_TEXT_LAYERS * EG2E_PLE_DIM; i++) per[i] *= ple_scale;
	eg2e_rms_rows(per, per, m->ple_norm, t * EG2E_TEXT_LAYERS, EG2E_PLE_DIM);

	for (int li = 0; li < EG2E_TEXT_LAYERS; li++) {
		const eg2e_layer *l = &m->text[li];
		const unsigned hd = l->hd, qd = l->nq * hd, kd = l->nkv * hd;
		float *q = (float*)malloc(t * qd * sizeof(float)), *k = (float*)malloc(t * kd * sizeof(float));
		float *v = (float*)malloc(t * kd * sizeof(float)), *a = (float*)malloc(t * qd * sizeof(float));
		float *cs = (float*)malloc(t * hd * sizeof(float)), *sn = (float*)malloc(t * hd * sizeof(float));
		float *o = (float*)malloc(t * D * sizeof(float));
		double tm = 0;
		EG2E_TICK(&tm);

		eg2e_rms_rows(h, x, l->in_ln, t, D);
		EG2E_TOCK(&tm, EG2E_P_TNORM);
		eg2e_mm(q, h, &l->q, t, D, qd);
		eg2e_mm(k, h, &l->k, t, D, kd);
		eg2e_mm(v, h, &l->v, t, D, kd);
		EG2E_TOCK(&tm, EG2E_P_TPROJ);
		eg2e_head_norm(q, l->q_norm, t, l->nq, hd);
		eg2e_head_norm(k, l->k_norm, t, l->nkv, hd);
		eg2e_head_norm(v, NULL, t, l->nkv, hd);
		cats_rope_table(cs, sn, pos, t, hd, l->full ? 1000000.0f : 10000.0f);
		#pragma omp parallel for schedule(static)
		for (long i = 0; i < (long)t; i++) {
			for (unsigned hh = 0; hh < l->nq; hh++) cats_rope_block(q + i*qd + hh*hd, cs + i*hd, sn + i*hd, hd);
			for (unsigned hh = 0; hh < l->nkv; hh++) cats_rope_block(k + i*kd + hh*hd, cs + i*hd, sn + i*hd, hd);
		}
		EG2E_TOCK(&tm, EG2E_P_TOTHER);
		cats_attention_bidir(a, q, k, v, t, l->nq, l->nkv, hd, 1.0f, l->full ? 0 : EG2E_SLIDING_WINDOW + 1);
		EG2E_TOCK(&tm, EG2E_P_TATT);
		eg2e_mm(o, a, &l->o, t, qd, D);
		EG2E_TOCK(&tm, EG2E_P_TPROJ);
		eg2e_rms_rows(o, o, l->post_attn_ln, t, D);
		for (size_t i = 0; i < t * D; i++) x[i] += o[i];

		eg2e_rms_rows(h, x, l->pre_ff_ln, t, D);
		EG2E_TOCK(&tm, EG2E_P_TNORM);
		eg2e_gated_mlp(o, h, l, t, D, 2048);
		EG2E_TOCK(&tm, EG2E_P_TMLP);
		eg2e_rms_rows(o, o, l->post_ff_ln, t, D);
		for (size_t i = 0; i < t * D; i++) x[i] += o[i];
		EG2E_TOCK(&tm, EG2E_P_TNORM);

		// per-layer-input block: gelu(gate(x)) * ple[layer] -> Linear -> RMSNorm, added to the residual
		eg2e_mm(h, x, &l->ple_gate, t, D, EG2E_PLE_DIM);
		for (size_t i = 0; i < t; i++) {
			const float *pl = per + (i * EG2E_TEXT_LAYERS + li) * EG2E_PLE_DIM;
			for (unsigned j = 0; j < EG2E_PLE_DIM; j++) h[i * EG2E_PLE_DIM + j] = cats_gelu_tanh(h[i * EG2E_PLE_DIM + j]) * pl[j];
		}
		eg2e_mm(o, h, &l->ple_proj, t, EG2E_PLE_DIM, D);
		eg2e_rms_rows(o, o, l->ple_norm, t, D);
		for (size_t i = 0; i < t * D; i++) x[i] = (x[i] + o[i]) * l->scalar;
		EG2E_TOCK(&tm, EG2E_P_TPLE);
		if (hidden_dump) memcpy(hidden_dump + (size_t)li * t * D, x, t * D * sizeof(float));
		free(q); free(k); free(v); free(a); free(cs); free(sn); free(o);
	}

	// final norm, Linear 512 -> 768, mean pool (the projection is linear, so order is irrelevant), L2 normalise
	eg2e_rms_rows(h, x, m->text_norm, t, D);
	float *proj = (float*)malloc(t * EG2E_OUT_DIM * sizeof(float));
	eg2e_mm(proj, h, &m->text_proj, t, D, EG2E_OUT_DIM);
	double acc[EG2E_OUT_DIM] = {0};
	for (size_t i = 0; i < t; i++) for (unsigned j = 0; j < EG2E_OUT_DIM; j++) acc[j] += proj[i * EG2E_OUT_DIM + j];
	double nrm = 0;
	for (unsigned j = 0; j < EG2E_OUT_DIM; j++) { acc[j] /= (double)t; nrm += acc[j] * acc[j]; }
	nrm = sqrt(nrm);
	for (unsigned j = 0; j < EG2E_OUT_DIM; j++) out768[j] = (float)(nrm > 0 ? acc[j] / nrm : 0);
	free(proj); free(x); free(h); free(per); free(pos);
	return 0;
}

// token ids -> scaled embeddings (embed_tokens * sqrt(512)); `soft` optionally replaces image positions
static int eg2e_embed_ids(const eg2e_model *m, const int *ids, size_t t, const float *soft, float *out)
{
	const float scale = sqrtf((float)EG2E_TEXT_DIM);
	size_t s = 0;
	for (size_t i = 0; i < t; i++) {
		float *dst = out + i * EG2E_TEXT_DIM;
		if (soft && ids[i] == EG2E_TOK_IMAGE) {	// soft tokens are inserted as they are (no scaling)
			memcpy(dst, soft + s++ * EG2E_TEXT_DIM, EG2E_TEXT_DIM * sizeof(float));
			continue;
		}
		if (cats_st_read_row_f32(&m->st, m->embed_tokens, ids[i], dst)) return -1;
		for (unsigned j = 0; j < EG2E_TEXT_DIM; j++) dst[j] *= scale;
	}
	return 0;
}

static int eg2e_embed_text_ids(const eg2e_model *m, const int *ids, size_t t, float *out768)
{
	float *emb = (float*)malloc(t * EG2E_TEXT_DIM * sizeof(float));
	if (!emb || eg2e_embed_ids(m, ids, t, NULL, emb)) { free(emb); return -1; }
	int rc = eg2e_backbone(m, emb, t, out768, NULL);
	free(emb);
	return rc;
}

// ---- vision tower ----------------------------------------------------------------------------

// pix[p][768] (patch pixels in [row][col][channel] order, 0..1) for a gw x gh grid of 16x16 patches
// -> soft[(gw*gh)/9][512]. layer_dump (optional) receives every layer's output [16][p][768].
static int eg2e_vision(const eg2e_model *m, const float *pix, int gw, int gh, float *soft, float *layer_dump)
{
	if (!m->has_vision || gw % EG2E_POOL || gh % EG2E_POOL) return -1;
	const size_t P = (size_t)gw * gh;
	const unsigned D = EG2E_VIS_DIM, hd = EG2E_VIS_HEAD_DIM, HEADS = EG2E_VIS_HEADS;
	float *x = (float*)malloc(P * D * sizeof(float)), *h = (float*)malloc(P * D * sizeof(float));
	float *in = (float*)malloc(P * D * sizeof(float));
	float *cs = (float*)malloc(P * hd * sizeof(float)), *sn = (float*)malloc(P * hd * sizeof(float));
	if (!x || !h || !in || !cs || !sn) { free(x); free(h); free(in); free(cs); free(sn); return -1; }

	for (size_t i = 0; i < P * D; i++) in[i] = 2.0f * (pix[i] - 0.5f);
	eg2e_mm(x, in, &m->patch_w, P, D, D);
	for (size_t p = 0; p < P; p++) {			// + x-embedding + y-embedding
		const float *ex = m->pos_table + (size_t)(p % gw) * D;
		const float *ey = m->pos_table + ((size_t)EG2E_POS_TABLE + p / gw) * D;
		for (unsigned j = 0; j < D; j++) x[p * D + j] += ex[j] + ey[j];
	}
	// axial 2-D RoPE: head dim 64 = [x block (32) | y block (32)], inverse frequencies 100^(-2j/32)
	for (size_t p = 0; p < P; p++) {
		float px = (float)(p % gw), py = (float)(p / gw);
		for (unsigned j = 0; j < 16; j++) {
			double f = pow(100.0, -2.0 * j / 32.0);
			float cx = (float)cos(px * f), sx = (float)sin(px * f), cy = (float)cos(py * f), sy = (float)sin(py * f);
			cs[p*hd + j] = cs[p*hd + 16 + j] = cx;  sn[p*hd + j] = sn[p*hd + 16 + j] = sx;
			cs[p*hd + 32 + j] = cs[p*hd + 48 + j] = cy;  sn[p*hd + 32 + j] = sn[p*hd + 48 + j] = sy;
		}
	}
	for (int li = 0; li < EG2E_VIS_LAYERS; li++) {
		const eg2e_layer *l = &m->vis[li];
		float *q = (float*)malloc(P * D * sizeof(float)), *k = (float*)malloc(P * D * sizeof(float));
		float *v = (float*)malloc(P * D * sizeof(float)), *a = (float*)malloc(P * D * sizeof(float));
		float *o = (float*)malloc(P * D * sizeof(float));
		double tm = 0;
		EG2E_TICK(&tm);
		eg2e_rms_rows(h, x, l->in_ln, P, D);
		EG2E_TOCK(&tm, EG2E_P_VNORM);
		eg2e_mm(q, h, &l->q, P, D, D);
		eg2e_mm(k, h, &l->k, P, D, D);
		eg2e_mm(v, h, &l->v, P, D, D);
		EG2E_TOCK(&tm, EG2E_P_VPROJ);
		eg2e_head_norm(q, l->q_norm, P, HEADS, hd);
		eg2e_head_norm(k, l->k_norm, P, HEADS, hd);
		eg2e_head_norm(v, NULL, P, HEADS, hd);
		#pragma omp parallel for schedule(static)
		for (long i = 0; i < (long)P; i++) for (unsigned hh = 0; hh < HEADS; hh++) {
			float *qq = q + i*D + hh*hd, *kk = k + i*D + hh*hd;
			cats_rope_block(qq, cs + i*hd, sn + i*hd, 32);
			cats_rope_block(qq + 32, cs + i*hd + 32, sn + i*hd + 32, 32);
			cats_rope_block(kk, cs + i*hd, sn + i*hd, 32);
			cats_rope_block(kk + 32, cs + i*hd + 32, sn + i*hd + 32, 32);
		}
		EG2E_TOCK(&tm, EG2E_P_VROPE);
		cats_attention_bidir(a, q, k, v, P, HEADS, HEADS, hd, 1.0f, 0);
		EG2E_TOCK(&tm, EG2E_P_VATT);
		eg2e_mm(o, a, &l->o, P, D, D);
		EG2E_TOCK(&tm, EG2E_P_VPROJ);
		eg2e_rms_rows(o, o, l->post_attn_ln, P, D);
		for (size_t i = 0; i < P * D; i++) x[i] += o[i];
		eg2e_rms_rows(h, x, l->pre_ff_ln, P, D);
		EG2E_TOCK(&tm, EG2E_P_VNORM);
		eg2e_gated_mlp(o, h, l, P, D, EG2E_VIS_FF);
		EG2E_TOCK(&tm, EG2E_P_VMLP);
		eg2e_rms_rows(o, o, l->post_ff_ln, P, D);
		for (size_t i = 0; i < P * D; i++) x[i] += o[i];
		EG2E_TOCK(&tm, EG2E_P_VNORM);
		if (layer_dump) memcpy(layer_dump + (size_t)li * P * D, x, P * D * sizeof(float));
		free(q); free(k); free(v); free(a); free(o);
	}
	// 3x3 average pooling over the patch grid, x sqrt(768), then embed_vision: RMSNorm (no scale) + Linear 768 -> 512
	const int ow = gw / EG2E_POOL, oh = gh / EG2E_POOL;
	float *pooled = (float*)calloc((size_t)ow * oh * D, sizeof(float));
	const float root = sqrtf((float)D);
	for (size_t p = 0; p < P; p++) {
		float *dst = pooled + (((p / gw) / EG2E_POOL) * ow + (p % gw) / EG2E_POOL) * D;
		for (unsigned j = 0; j < D; j++) dst[j] += x[p * D + j] * (root / (EG2E_POOL * EG2E_POOL));
	}
	eg2e_rms_rows(pooled, pooled, NULL, (size_t)ow * oh, D);
	eg2e_mm(soft, pooled, &m->embed_vision, (size_t)ow * oh, D, 512);
	free(pooled); free(x); free(h); free(in); free(cs); free(sn);
	return 0;
}

// ---- image preprocessing ---------------------------------------------------------------------

// Largest size whose patch count is <= max_soft*9 with both sides a multiple of 48 (aspect preserved).
static int eg2e_target_size(int h, int w, int max_soft, int *th, int *tw)
{
	const int side = EG2E_PATCH * EG2E_POOL, max_patches = max_soft * EG2E_POOL * EG2E_POOL;
	double factor = sqrt((double)max_patches * EG2E_PATCH * EG2E_PATCH / ((double)h * w));
	int nh = (int)floor(factor * h / side) * side, nw = (int)floor(factor * w / side) * side;
	int maxside = (max_patches / (EG2E_POOL * EG2E_POOL)) * side;
	if (!nh && !nw) return -1;
	if (!nh) { nh = side; nw = (int)floor((double)w / h) * side; if (nw > maxside) nw = maxside; }
	else if (!nw) { nw = side; nh = (int)floor((double)h / w) * side; if (nh > maxside) nh = maxside; }
	if (!nh || !nw || (long)nh * nw > (long)max_patches * EG2E_PATCH * EG2E_PATCH) return -1;
	*th = nh; *tw = nw;
	return 0;
}
// Rearranges an already resized RGB image (th x tw) into patches[p][(row*16+col)*3+channel] in 0..1.
static void eg2e_patchify(const unsigned char *rgb, int th, int tw, float *pix)
{
	const int gw = tw / EG2E_PATCH, gh = th / EG2E_PATCH;
	for (int py = 0; py < gh; py++) for (int px = 0; px < gw; px++) {
		float *dst = pix + ((size_t)py * gw + px) * 768;
		for (int r = 0; r < EG2E_PATCH; r++) for (int c = 0; c < EG2E_PATCH; c++) for (int ch = 0; ch < 3; ch++)
			dst[(r * EG2E_PATCH + c) * 3 + ch] =
				rgb[((size_t)(py * EG2E_PATCH + r) * tw + px * EG2E_PATCH + c) * 3 + ch] * (1.0f / 255.0f);
	}
}

// RGB image (8-bit, interleaved) -> unit embedding[768]. The image is resized with `resize`
// (a caller-supplied bicubic resampler, so this header needs no image library).
typedef int (*eg2e_resize_fn)(const unsigned char *src, int sw, int sh, unsigned char *dst, int dw, int dh);
static int eg2e_embed_image(const eg2e_model *m, const unsigned char *rgb, int w, int h, int max_soft,
			    eg2e_resize_fn resize, float *out768)
{
	int th, tw;
	if (eg2e_target_size(h, w, max_soft, &th, &tw)) return -1;
	unsigned char *img = (unsigned char*)malloc((size_t)th * tw * 3);
	if (!img) return -1;
	if (th == h && tw == w) memcpy(img, rgb, (size_t)th * tw * 3);
	else if (resize(rgb, w, h, img, tw, th)) { free(img); return -1; }
	const int gw = tw / EG2E_PATCH, gh = th / EG2E_PATCH, n = gw * gh / (EG2E_POOL * EG2E_POOL);
	float *pix = (float*)malloc((size_t)gw * gh * 768 * sizeof(float)), *soft = (float*)malloc((size_t)n * 512 * sizeof(float));
	int rc = -1;
	if (pix && soft) {
		eg2e_patchify(img, th, tw, pix);
		if (!eg2e_vision(m, pix, gw, gh, soft, NULL)) {
			// [BOS, boi, soft tokens x n, eoi, EOS]
			size_t t = (size_t)n + 4;
			int *ids = (int*)malloc(t * sizeof(int));
			float *emb = (float*)malloc(t * EG2E_TEXT_DIM * sizeof(float));
			if (ids && emb) {
				ids[0] = EG2E_TOK_BOS; ids[1] = EG2E_TOK_BOI;
				for (int i = 0; i < n; i++) ids[2 + i] = EG2E_TOK_IMAGE;
				ids[n + 2] = EG2E_TOK_EOI; ids[n + 3] = EG2E_TOK_EOS;
				if (!eg2e_embed_ids(m, ids, t, soft, emb)) rc = eg2e_backbone(m, emb, t, out768, NULL);
			}
			free(ids); free(emb);
		}
	}
	free(img); free(pix); free(soft);
	return rc;
}

static float eg2e_cosine(const float *a, const float *b)
{
	double s = 0;
	for (int i = 0; i < EG2E_OUT_DIM; i++) s += (double)a[i] * b[i];
	return (float)s;
}

#endif /* EG2_EMBED_H */
