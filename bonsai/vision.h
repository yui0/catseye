/*
 * vision.h - image input for Ternary-Bonsai: the Qwen3-VL vision tower from the mmproj GGUF (clip.projector_type = qwen3vl_merger).
 *
 *   image -> smart resize (multiple of 32 pixels, bicubic) -> 16x16 patches (two temporal kernels summed, a still image is
 *   both frames) -> + learned position embedding (bilinear resize of the 48x48 grid) -> 27 ViT blocks (LayerNorm, fused qkv,
 *   2D RoPE, GELU MLP) -> post LayerNorm -> 2x2 merge (4 neighbouring patches -> one 4608-wide vector) -> MLP -> 5120 = LLM width.
 *
 * Patches are ordered block by block (each 2x2 group of neighbours is contiguous) so the merge is a plain reshape.
 * The weights (Q8_0 / F16 / F32 in the file) are expanded to float32 at load (~1.9 GB) and run through cats_gemm_nt.
 */
#ifndef BONSAI_VISION_H
#define BONSAI_VISION_H

#include "bonsai.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../stb_image_resize.h"

typedef struct { float *ln1w, *ln1b, *qkvw, *qkvb, *ow, *ob, *ln2w, *ln2b, *upw, *upb, *dnw, *dnb; } bv_layer;
typedef struct {
	int nl, E, F, H, hd, patch, merge, proj, grid;
	float eps;
	float *pw /* E x 768, both temporal kernels summed */, *pb, *pos /* grid*grid x E */;
	bv_layer *L;
	float *plnw, *plnb, *m0w, *m0b, *m2w, *m2b;
} bn_vis;

static float *bv_f32(const cg_file *g, const char *name, size_t n)
{
	const cg_tensor *t = cg_find(g, name);
	if (!t || t->n != n) { fprintf(stderr, "bonsai: mmproj tensor %s missing or has the wrong size\n", name); return NULL; }
	float *o = (float*)malloc(n * sizeof(float));
	if (!o) return NULL;
	if (t->type == CG_F32) memcpy(o, t->data, n * sizeof(float));
	else if (t->type == CG_F16) for (size_t i = 0; i < n; i++) o[i] = cats_half_to_float(cats_load_u16((const uint8_t*)t->data + 2 * i));
	else if (t->type == CG_Q8_0) cats_q8_0w_dequantize_row((const uint8_t*)t->data, n, o);
	else { fprintf(stderr, "bonsai: mmproj tensor %s has unsupported type %d\n", name, t->type); free(o); return NULL; }
	return o;
}

static void bv_close(bn_vis *v)
{
	if (v->L) for (int i = 0; i < v->nl; i++) {
		bv_layer *l = &v->L[i];
		free(l->ln1w); free(l->ln1b); free(l->qkvw); free(l->qkvb); free(l->ow); free(l->ob); free(l->ln2w); free(l->ln2b);
		free(l->upw); free(l->upb); free(l->dnw); free(l->dnb);
	}
	free(v->L); free(v->pw); free(v->pb); free(v->pos); free(v->plnw); free(v->plnb); free(v->m0w); free(v->m0b); free(v->m2w); free(v->m2b);
	memset(v, 0, sizeof *v);
}

static int bv_open(bn_vis *v, const char *path, int llm_width)
{
	memset(v, 0, sizeof *v);
	cg_file g;
	if (cg_open(&g, path)) { fprintf(stderr, "bonsai: cannot open %s\n", path); return -1; }
	const cg_kv *pt = cg_kvfind(&g, "clip.projector_type");
	if (!pt || pt->type != CG_STR || memcmp(pt->val + 8, "qwen3vl_merger", 14)) { fprintf(stderr, "bonsai: %s is not a qwen3vl_merger mmproj\n", path); cg_close(&g); return -1; }
	v->nl = (int)cg_u32(&g, "clip.vision.block_count", 0);
	v->E = (int)cg_u32(&g, "clip.vision.embedding_length", 0);
	v->F = (int)cg_u32(&g, "clip.vision.feed_forward_length", 0);
	v->H = (int)cg_u32(&g, "clip.vision.attention.head_count", 0);
	v->patch = (int)cg_u32(&g, "clip.vision.patch_size", 16);
	v->merge = (int)cg_u32(&g, "clip.vision.spatial_merge_size", 2);
	v->proj = (int)cg_u32(&g, "clip.vision.projection_dim", 0);
	v->eps = bn_kv_f32(&g, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
	v->hd = v->H ? v->E / v->H : 0;
	if (!v->nl || !v->E || !v->H || v->merge != 2 || v->proj != llm_width || v->hd % 4 || v->patch * v->patch * 3 != 768) { fprintf(stderr, "bonsai: unsupported mmproj geometry\n"); cg_close(&g); return -1; }
	const int E = v->E, F = v->F;
	const cg_tensor *pe = cg_find(&g, "v.position_embd.weight");
	if (!pe) { cg_close(&g); return -1; }
	v->grid = (int)sqrt((double)pe->dim[1]);
	float *w0 = bv_f32(&g, "v.patch_embd.weight", (size_t)E * 768), *w1 = bv_f32(&g, "v.patch_embd.weight.1", (size_t)E * 768);
	if (!w0 || !w1) { free(w0); free(w1); cg_close(&g); return -1; }
	for (size_t i = 0; i < (size_t)E * 768; i++) w0[i] += w1[i];
	free(w1);
	v->pw = w0;
	v->pb = bv_f32(&g, "v.patch_embd.bias", E);
	v->pos = bv_f32(&g, "v.position_embd.weight", (size_t)v->grid * v->grid * E);
	v->plnw = bv_f32(&g, "v.post_ln.weight", E); v->plnb = bv_f32(&g, "v.post_ln.bias", E);
	v->m0w = bv_f32(&g, "mm.0.weight", (size_t)E * 4 * E * 4); v->m0b = bv_f32(&g, "mm.0.bias", E * 4);
	v->m2w = bv_f32(&g, "mm.2.weight", (size_t)E * 4 * v->proj); v->m2b = bv_f32(&g, "mm.2.bias", v->proj);
	v->L = (bv_layer*)calloc(v->nl, sizeof(bv_layer));
	int ok = v->pb && v->pos && v->plnw && v->plnb && v->m0w && v->m0b && v->m2w && v->m2b && v->L;
	for (int i = 0; ok && i < v->nl; i++) {
		bv_layer *l = &v->L[i];
		char n[96];
#define BL(field, fmt, cnt) do { snprintf(n, sizeof n, fmt, i); l->field = bv_f32(&g, n, (size_t)(cnt)); ok = ok && l->field; } while (0)
		BL(ln1w, "v.blk.%d.ln1.weight", E); BL(ln1b, "v.blk.%d.ln1.bias", E); BL(ln2w, "v.blk.%d.ln2.weight", E); BL(ln2b, "v.blk.%d.ln2.bias", E);
		BL(qkvw, "v.blk.%d.attn_qkv.weight", (size_t)E * 3 * E); BL(qkvb, "v.blk.%d.attn_qkv.bias", 3 * E);
		BL(ow, "v.blk.%d.attn_out.weight", (size_t)E * E); BL(ob, "v.blk.%d.attn_out.bias", E);
		BL(upw, "v.blk.%d.ffn_up.weight", (size_t)E * F); BL(upb, "v.blk.%d.ffn_up.bias", F);
		BL(dnw, "v.blk.%d.ffn_down.weight", (size_t)E * F); BL(dnb, "v.blk.%d.ffn_down.bias", E);
#undef BL
	}
	cg_close(&g);	/* everything was copied to float32 */
	if (!ok) { bv_close(v); return -1; }
	return 0;
}

static void bv_layernorm(float *x, int n, int d, const float *w, const float *b, float eps)
{
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++) {
		float *r = x + (size_t)i * d;
		double s = 0, q = 0;
		for (int j = 0; j < d; j++) s += r[j];
		const float mean = (float)(s / d);
		for (int j = 0; j < d; j++) { const float t = r[j] - mean; q += (double)t * t; }
		const float inv = 1.0f / sqrtf((float)(q / d) + eps);
		for (int j = 0; j < d; j++) r[j] = (r[j] - mean) * inv * w[j] + b[j];
	}
}

/* Qwen's smart_resize: edges rounded (half to even) to a multiple of `factor`, scaled to fit [min_pixels, max_pixels] */
static void bv_smart_resize(int w, int h, int factor, long min_pixels, long max_pixels, int *ow, int *oh)
{
	int hb = (int)nearbyint((double)h / factor) * factor, wb = (int)nearbyint((double)w / factor) * factor;
	if ((long)hb * wb > max_pixels) {
		const double beta = sqrt((double)h * w / max_pixels);
		hb = (int)floor(h / beta / factor) * factor; wb = (int)floor(w / beta / factor) * factor;
	} else if ((long)hb * wb < min_pixels) {
		const double beta = sqrt((double)min_pixels / ((double)h * w));
		hb = (int)ceil(h * beta / factor) * factor; wb = (int)ceil(w * beta / factor) * factor;
	}
	*ow = wb > factor ? wb : factor; *oh = hb > factor ? hb : factor;
}

/* rgb: w x h x 3 bytes. Returns the image embeddings (nx * ny x proj floats, malloc'ed) or NULL. nx / ny: token grid.
 * max_tokens limits the number of LLM tokens (each is 32x32 pixels of the resized image). */
static float *bv_encode(const bn_vis *v, const unsigned char *rgb, int w, int h, int max_tokens, int *nx, int *ny)
{
	const int E = v->E, P = v->patch, H = v->H, hd = v->hd, F = v->F;
	int W, Hh;
	bv_smart_resize(w, h, P * v->merge, 8L * 1024, (long)max_tokens * P * P * v->merge * v->merge, &W, &Hh);
	unsigned char *rs = (unsigned char*)malloc((size_t)W * Hh * 3);
	if (!stbir_resize_uint8_generic(rgb, w, h, 0, rs, W, Hh, 0, 3, -1, 0, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM, STBIR_COLORSPACE_LINEAR, NULL)) { free(rs); return NULL; }
	const int gw = W / P, gh = Hh / P, n = gw * gh;
	*nx = gw / 2; *ny = gh / 2;
	fprintf(stderr, "bonsai: image %dx%d -> %dx%d, %d patches, %d tokens\n", w, h, W, Hh, n, n / 4);
	/* patches in block order; position (y, x) of each */
	float *x = (float*)malloc((size_t)n * 768 * sizeof(float)), *res = (float*)malloc((size_t)n * E * sizeof(float));
	int *py = (int*)malloc(n * sizeof(int)), *px = (int*)malloc(n * sizeof(int));
	{
		int ptr = 0;
		for (int by = 0; by < gh / 2; by++) for (int bx = 0; bx < gw / 2; bx++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++, ptr++) {
			const int y = by * 2 + dy, xx = bx * 2 + dx;
			py[ptr] = y; px[ptr] = xx;
			float *o = x + (size_t)ptr * 768;
			for (int c = 0; c < 3; c++) for (int ky = 0; ky < P; ky++) for (int kx = 0; kx < P; kx++)
				*o++ = (rs[((size_t)(y * P + ky) * W + xx * P + kx) * 3 + c] / 255.0f - 0.5f) / 0.5f;
		}
	}
	free(rs);
	cats_gemm_nt(res, x, v->pw, n, 768, E, v->pb);
	free(x);
	/* learned position embedding, bilinear with aligned corners from grid x grid to gh x gw */
	{
		const int G = v->grid;
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; i++) {
			const float sy = gh > 1 ? py[i] * (float)(G - 1) / (gh - 1) : 0.0f, sx = gw > 1 ? px[i] * (float)(G - 1) / (gw - 1) : 0.0f;
			const int y0 = (int)sy, x0 = (int)sx, y1 = y0 + 1 < G ? y0 + 1 : y0, x1 = x0 + 1 < G ? x0 + 1 : x0;
			const float fy = sy - y0, fx = sx - x0;
			const float *a = v->pos + ((size_t)y0 * G + x0) * E, *b = v->pos + ((size_t)y0 * G + x1) * E, *c = v->pos + ((size_t)y1 * G + x0) * E, *d = v->pos + ((size_t)y1 * G + x1) * E;
			float *o = res + (size_t)i * E;
			for (int e = 0; e < E; e++) o[e] += (1 - fy) * ((1 - fx) * a[e] + fx * b[e]) + fy * ((1 - fx) * c[e] + fx * d[e]);
		}
	}
	/* 2D RoPE tables: pair j < hd/4 rotates with the row, hd/4 <= j < hd/2 with the column, frequency base^(-2k/(hd/2)) */
	const int half = hd / 2, sec = hd / 4;
	float *rc = (float*)malloc((size_t)n * half * sizeof(float)), *rsn = (float*)malloc((size_t)n * half * sizeof(float));
	for (int i = 0; i < n; i++) for (int j = 0; j < half; j++) {
		const int k = j % sec;
		const float ang = (j < sec ? py[i] : px[i]) * powf(10000.0f, -2.0f * k / half);
		rc[i * half + j] = cosf(ang); rsn[i * half + j] = sinf(ang);
	}
	float *qkv = (float*)malloc((size_t)n * 3 * E * sizeof(float)), *tmp = (float*)malloc((size_t)n * E * sizeof(float)), *att = (float*)malloc((size_t)n * E * sizeof(float));
	float *hid = (float*)malloc((size_t)n * F * sizeof(float));
	const float scale = 1.0f / sqrtf((float)hd);
	for (int li = 0; li < v->nl; li++) {
		const bv_layer *L = &v->L[li];
		memcpy(tmp, res, (size_t)n * E * sizeof(float));
		bv_layernorm(tmp, n, E, L->ln1w, L->ln1b, v->eps);
		cats_gemm_nt(qkv, tmp, L->qkvw, n, E, 3 * E, L->qkvb);
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; i++) for (int hh = 0; hh < H; hh++) for (int part = 0; part < 2; part++) {
			float *q = qkv + (size_t)i * 3 * E + part * E + (size_t)hh * hd;
			for (int j = 0; j < half; j++) {
				const float c = rc[i * half + j], s = rsn[i * half + j], a = q[j], b = q[j + half];
				q[j] = a * c - b * s; q[j + half] = a * s + b * c;
			}
		}
		#pragma omp parallel
		{
			float *sc = (float*)malloc((size_t)n * sizeof(float));
			#pragma omp for schedule(static)
			for (int idx = 0; idx < n * H; idx++) {
				const int i = idx / H, hh = idx % H;
				const float *q = qkv + (size_t)i * 3 * E + (size_t)hh * hd;
				float mx = -1e30f;
				for (int j = 0; j < n; j++) {
					const float *k = qkv + (size_t)j * 3 * E + E + (size_t)hh * hd;
					float s = 0;
					for (int d = 0; d < hd; d++) s += q[d] * k[d];
					sc[j] = s * scale;
					if (sc[j] > mx) mx = sc[j];
				}
				float sum = 0;
				for (int j = 0; j < n; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
				float *o = att + (size_t)i * E + (size_t)hh * hd;
				memset(o, 0, hd * sizeof(float));
				for (int j = 0; j < n; j++) {
					const float wgt = sc[j] / sum, *vv = qkv + (size_t)j * 3 * E + 2 * E + (size_t)hh * hd;
					for (int d = 0; d < hd; d++) o[d] += wgt * vv[d];
				}
			}
			free(sc);
		}
		cats_gemm_nt(tmp, att, L->ow, n, E, E, L->ob);
		for (size_t i = 0; i < (size_t)n * E; i++) res[i] += tmp[i];
		memcpy(tmp, res, (size_t)n * E * sizeof(float));
		bv_layernorm(tmp, n, E, L->ln2w, L->ln2b, v->eps);
		cats_gemm_nt(hid, tmp, L->upw, n, E, F, L->upb);
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < (size_t)n * F; i++) hid[i] = cats_gelu_tanh(hid[i]);
		cats_gemm_nt(tmp, hid, L->dnw, n, F, E, L->dnb);
		for (size_t i = 0; i < (size_t)n * E; i++) res[i] += tmp[i];
	}
	bv_layernorm(res, n, E, v->plnw, v->plnb, v->eps);
	/* merge 2x2: 4 consecutive patches -> one 4*E vector -> MLP -> LLM width */
	const int nt = n / 4, M = 4 * E;
	float *m0 = (float*)malloc((size_t)nt * M * sizeof(float)), *out = (float*)malloc((size_t)nt * v->proj * sizeof(float));
	cats_gemm_nt(m0, res, v->m0w, nt, M, M, v->m0b);
	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < (size_t)nt * M; i++) m0[i] = cats_gelu_tanh(m0[i]);
	cats_gemm_nt(out, m0, v->m2w, nt, M, v->proj, v->m2b);
	free(m0); free(qkv); free(tmp); free(att); free(hid); free(res); free(rc); free(rsn); free(py); free(px);
	return out;
}

/* Feeds image embeddings (nx * ny tokens of the LLM width) to the main model: M-RoPE positions t = start, h = start + row,
 * w = start + column; afterwards the text position continues at start + max(nx, ny). */
static int bn_feed_embeddings(bn_model *m, const float *emb, int nx, int ny)
{
	const int D = m->D, N = nx * ny, pos0 = m->rpos;
	int p3[3 * BN_MAXT];
	for (int i0 = 0; i0 < N; i0 += BN_MAXT) {
		const int c = N - i0 < BN_MAXT ? N - i0 : BN_MAXT;
		for (int t = 0; t < c; t++) { const int i = i0 + t; p3[3 * t] = pos0; p3[3 * t + 1] = pos0 + i / nx; p3[3 * t + 2] = pos0 + i % nx; }
		m->emb_in = emb + (size_t)i0 * D; m->p3 = p3;
		const int rc = bn_forward_ex(m, NULL, c, NULL, 0, NULL);
		m->emb_in = NULL; m->p3 = NULL;
		if (rc) return -1;
	}
	m->rpos = pos0 + (nx > ny ? nx : ny);
	return 0;
}

#endif
