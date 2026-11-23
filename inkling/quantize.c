//---------------------------------------------------------
//	Cat's eye
//
//		©2026 Yuichiro Nakada
//---------------------------------------------------------

// Inkling-2-Mini: shrink the Hugging Face checkpoint (NVFP4 / MXFP8 experts, BF16 rest, ~20 GB) into ONE file:
// a safetensors container with GGML Q4_0 / Q8_0 blocks (the layout `cats_st` and `inkling` read directly), the
// tokenizer and config.json embedded as U8 tensors. Experts are decoded from NVFP4 / MXFP8 and re-quantized,
// with the gate / up rows de-interleaved (the checkpoint stores them as gate0, up0, gate1, up1, ...).

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#define CATS_INFER_ONLY
#include "../catseye.h"

typedef struct { cats_st_entry *e; int n, cap; } ent_list;
static void ent_add(ent_list *L, const char *name, const char *dtype, int ndim, const int64_t *shape, void *data, uint64_t bytes)
{
	if (L->n == L->cap) { L->cap = L->cap ? L->cap * 2 : 256; L->e = (cats_st_entry*)realloc(L->e, (size_t)L->cap * sizeof(*L->e)); }
	cats_st_entry *e = &L->e[L->n++];
	memset(e, 0, sizeof(*e));
	e->name = strdup(name);
	e->dtype = dtype;
	e->ndim = ndim;
	for (int i = 0; i < ndim; i++) e->shape[i] = shape[i];
	e->data = data;
	e->bytes = bytes;
}

static int parse_type(const char *s)
{
	if (!strcmp(s, "f16")) return CATS_ST_F16;
	if (!strcmp(s, "q8")) return CATS_ST_Q8_0;
	if (!strcmp(s, "q4")) return CATS_ST_Q4_0;
	if (!strcmp(s, "q3")) return CATS_ST_Q3_0;
	return -1;
}
static const char *type_name(int t) { return t == CATS_ST_F16 ? "F16" : t == CATS_ST_Q8_0 ? "Q8_0" : t == CATS_ST_Q3_0 ? "Q3_0" : "Q4_0"; }

static cats_st st;
static ent_list L;
static int refine = 1;
// Ternary experiment (--tern G): the selected expert matrices are reduced to {-s, 0, +s} per group of G weights along the
// input dimension (the optimal number of non-zeros per group, s = their mean magnitude) and then stored as Q8_0, which
// holds those three values exactly. The file is big; it exists to measure how much quality a real ternary format could keep.
static int tern_g = 0, tern_part = 3, tern_from = 0, tern_to = 1000;
static int cmp_desc(const void *a, const void *b) { const float x = *(const float*)a, y = *(const float*)b; return x < y ? 1 : x > y ? -1 : 0; }
static void ternarize_row(float *row, int n, int G)
{
	float a[256], w[256];
	for (int b = 0; b + G <= n; b += G) {
		for (int j = 0; j < G; j++) { w[j] = row[b + j]; a[j] = fabsf(w[j]); }
		qsort(a, (size_t)G, sizeof(float), cmp_desc);
		double cs = 0, best = -1; int kb = 1;
		for (int k = 1; k <= G; k++) { cs += a[k - 1]; if (cs * cs / k > best) { best = cs * cs / k; kb = k; } }
		double sum = 0; for (int k = 0; k < kb; k++) sum += a[k];
		const float s = (float)(sum / kb), thr = a[kb - 1];
		for (int j = 0; j < G; j++) row[b + j] = fabsf(w[j]) >= thr && thr > 0 ? (w[j] < 0 ? -s : s) : 0.0f;
	}
}
static int w2_type = -1, hi_layers = 0, hi_type = CATS_ST_Q4_0;
static double total_bytes;

static const cats_st_tensor *must(const char *fmt, int a)
{
	char nm[256];
	snprintf(nm, sizeof(nm), fmt, a);
	const cats_st_tensor *t = cats_st_find(&st, nm);
	if (!t) { fprintf(stderr, "missing tensor: %s\n", nm); exit(1); }
	return t;
}
static const cats_st_tensor *maybe(const char *fmt, int a)
{
	char nm[256];
	snprintf(nm, sizeof(nm), fmt, a);
	return cats_st_find(&st, nm);
}

// F32 copy with an explicit target shape (drops singleton dims, e.g. conv weights [C,1,K] -> [C,K]).
static void put_f32(const char *newname, const cats_st_tensor *t, int ndim, int64_t s0, int64_t s1, int64_t s2)
{
	const int64_t n = cats_st_numel(t);
	float *d = (float*)malloc((size_t)n * sizeof(float));
	if (cats_st_read_f32(&st, t, d)) { fprintf(stderr, "cannot read %s\n", t->name); exit(1); }
	const int64_t shape[3] = { s0, s1, s2 };
	ent_add(&L, newname, "F32", ndim, shape, d, (uint64_t)n * 4);
	total_bytes += (double)n * 4;
}
static void put_same(const char *newname, const cats_st_tensor *t)
{
	put_f32(newname, t, t->ndim, t->shape[0], t->ndim > 1 ? t->shape[1] : 0, t->ndim > 2 ? t->shape[2] : 0);
}

static void quantize_row(int type, const float *row, int64_t cols, uint8_t *o)
{
	if (type == CATS_ST_Q4_0) cats_q4_0_quantize_row(row, (size_t)cols, o, refine);
	else if (type == CATS_ST_Q3_0) cats_q3_0_quantize_row(row, (size_t)cols, o, refine);
	else if (type == CATS_ST_Q8_0) cats_q8_0w_quantize_row(row, (size_t)cols, o);
	else for (int64_t k = 0; k < cols; k++) { uint16_t h = cats_float_to_half(row[k]); o[2 * k] = (uint8_t)h; o[2 * k + 1] = (uint8_t)(h >> 8); }
}
// A 2-D matrix (any source dtype), the first `rows` rows only, quantized to `type`.
static void put_matrix(const char *newname, const cats_st_tensor *t, int type, int64_t rows)
{
	if (rows <= 0) rows = t->shape[0];
	const int64_t cols = t->shape[1];
	if (type != CATS_ST_F16 && cols % CATS_Q4_0_BLOCK) { fprintf(stderr, "%s: %lld columns are not a multiple of 32\n", t->name, (long long)cols); exit(1); }
	const size_t rb = (size_t)cats_st_bytes(type, cols);
	uint8_t *data = (uint8_t*)malloc(rb * (size_t)rows);
	#pragma omp parallel
	{
		float *row = (float*)malloc((size_t)cols * sizeof(float));
		#pragma omp for schedule(static)
		for (long r = 0; r < rows; r++) {
			cats_st_read_row_f32(&st, t, r, row);
			quantize_row(type, row, cols, data + (size_t)r * rb);
		}
		free(row);
	}
	const int64_t shape[2] = { rows, cols };
	ent_add(&L, newname, type_name(type), 2, shape, data, (uint64_t)rb * (uint64_t)rows);
	total_bytes += (double)rb * (double)rows;
	fprintf(stderr, "%-24s %-4s %9.1f MB\n", newname, type_name(type), (double)rb * (double)rows / 1048576.0);
}

// One expert tensor (w13: [E, 2I, H], w2: [E, H, I]) decoded from NVFP4 / MXFP8 and stored as a 2-D
// [E * rows, cols] matrix, so that expert e is the row range [e * rows, (e + 1) * rows).
static void put_experts(int l, int w, int type, int E, int rows, int cols, int I)
{
	char base[256], nm[288];
	snprintf(base, sizeof(base), "model.llm.layers.%d.mlp.experts.w%d_weight", l, w);
	const cats_st_tensor *W = cats_st_find(&st, base);
	snprintf(nm, sizeof(nm), "%s_scale", base);
	const cats_st_tensor *S = cats_st_find(&st, nm);
	snprintf(nm, sizeof(nm), "%s_amax", base);
	const cats_st_tensor *A = cats_st_find(&st, nm);
	snprintf(nm, sizeof(nm), "%s_outer_scale_denominator", base);
	const cats_st_tensor *D = cats_st_find(&st, nm);
	if (!W || !S) { fprintf(stderr, "missing expert tensor %s\n", base); exit(1); }
	const int nvfp4 = A != NULL;
	if (nvfp4 && !D) { fprintf(stderr, "missing %s\n", nm); exit(1); }
	const size_t wrow = nvfp4 ? (size_t)cols / 2 : (size_t)cols, srow = nvfp4 ? (size_t)cols / 16 : (size_t)cols / 32;
	if ((uint64_t)W->shape[0] * W->shape[1] * wrow != W->end - W->begin || (uint64_t)W->shape[0] * W->shape[1] * srow != S->end - S->begin) {
		fprintf(stderr, "%s: unexpected size (shape %lld x %lld x %lld)\n", base, (long long)W->shape[0], (long long)W->shape[1], (long long)W->shape[2]);
		exit(1);
	}
	const uint8_t *wd = cats_st_raw(&st, W), *sd = cats_st_raw(&st, S);
	float amax[256], den = 1;
	if (nvfp4) {
		memcpy(amax, cats_st_raw(&st, A), (size_t)E * 4);
		memcpy(&den, cats_st_raw(&st, D), 4);
	}
	const int tern = tern_g > 0 && l >= tern_from && l < tern_to && (tern_part & (w == 13 ? 1 : 2));
	if (tern) type = CATS_ST_Q8_0;
	const size_t rb = (size_t)cats_st_bytes(type, cols);
	const long total = (long)E * rows;
	uint8_t *data = (uint8_t*)malloc(rb * (size_t)total);
	#pragma omp parallel
	{
		float *row = (float*)malloc((size_t)cols * sizeof(float));
		#pragma omp for schedule(static)
		for (long idx = 0; idx < total; idx++) {
			const int e = (int)(idx / rows), r = (int)(idx % rows);
			if (nvfp4) cats_nvfp4_dequantize_row(wd + (size_t)idx * wrow, sd + (size_t)idx * srow, (size_t)cols, amax[e] / den, row);
			else cats_mxfp8_dequantize_row(wd + (size_t)idx * wrow, sd + (size_t)idx * srow, (size_t)cols, row);
			if (tern) ternarize_row(row, cols, tern_g);
			const long dst = w == 13 ? (long)e * rows + ((r & 1) ? I + r / 2 : r / 2) : idx;	// gate = even rows, up = odd rows
			quantize_row(type, row, cols, data + (size_t)dst * rb);
		}
		free(row);
	}
	char name[64];
	snprintf(name, sizeof(name), "blk.%d.%s", l, w == 13 ? "w13" : "w2");
	const int64_t shape[2] = { total, cols };
	ent_add(&L, name, type_name(type), 2, shape, data, (uint64_t)rb * (uint64_t)total);
	total_bytes += (double)rb * (double)total;
	fprintf(stderr, "%-24s %-4s %9.1f MB  (%s)\n", name, type_name(type), (double)rb * (double)total / 1048576.0, nvfp4 ? "NVFP4" : "MXFP8");
}

int main(int argc, char **argv)
{
	const char *in = NULL, *out = NULL;
	int body = CATS_ST_Q4_0, attn = CATS_ST_Q8_0, emb = CATS_ST_Q8_0;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-i") && i + 1 < argc) in = argv[++i];
		else if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
		else if (!strcmp(argv[i], "-b") && i + 1 < argc) body = parse_type(argv[++i]);
		else if (!strcmp(argv[i], "-a") && i + 1 < argc) attn = parse_type(argv[++i]);
		else if (!strcmp(argv[i], "-e") && i + 1 < argc) emb = parse_type(argv[++i]);
		else if (!strcmp(argv[i], "--fast")) refine = 0;
		else if (!strcmp(argv[i], "--w2") && i + 1 < argc) w2_type = parse_type(argv[++i]);
		else if (!strcmp(argv[i], "--hi") && i + 1 < argc) hi_layers = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--tern") && i + 1 < argc) tern_g = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--tern-part") && i + 1 < argc) tern_part = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--tern-from") && i + 1 < argc) tern_from = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--tern-to") && i + 1 < argc) tern_to = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--hi-type") && i + 1 < argc) hi_type = parse_type(argv[++i]);
		else in = NULL;
	}
	if (!in || !out || body < 0 || attn < 0 || emb < 0) {
		fprintf(stderr,
			"Usage: %s -i HF_DIR -o OUT.safetensors [-b TYPE] [-a TYPE] [-e TYPE] [--fast]\n"
			"  -b  routed experts (q4 | q3 | q8, default q4; q3 = 3.5 bit for 8 GB machines; f16 only for debugging)        -a  attention matrices (q8 | q4 | q3 | f16, default q8)\n"
			"  -e  embedding / output tables (q8 | q4 | f16, default q8)\n"
			"  --w2 TYPE      different type for the experts' down projections (e.g. -b q3 --w2 q4)\n"
			"  --tern G       experiment: ternarize experts in groups of G (stored as Q8_0); --tern-part 1|2|3 (w13 / w2 / both),\n"
			"                 --tern-from A --tern-to B (layers [A,B))\n"
			"  --hi N         keep the experts of the first and last N layers at --hi-type (default q4)\n"
			"  Experts are decoded from NVFP4 / MXFP8 and re-quantized. Q4_0 uses a per-block scale search; --fast is\n"
			"  plain GGML Q4_0 (quicker, a little worse). Defaults give a ~15.5 GB file from the 20 GB checkpoint.\n", argv[0]);
		return 1;
	}
	char path[1024];
	snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors", in);
	if (cats_st_open(&st, path)) { snprintf(path, sizeof(path), "%s/model.safetensors", in); if (cats_st_open(&st, path)) { fprintf(stderr, "cannot open the checkpoint in %s\n", in); return 1; } }
	snprintf(path, sizeof(path), "%s/config.json", in);
	size_t cfg_size;
	char *cfg = cats_read_file(path, &cfg_size);
	if (!cfg) { fprintf(stderr, "cannot read %s\n", path); return 1; }
	double v;
	int H = 2304, NL = 24, E = 128, I = 1152, V = 200058, LS = 6, LE = 18;
#define CFG(var, key) do { if (!cats_json_num(cfg, key, &v)) var = (int)v; } while (0)
	CFG(H, "hidden_size"); CFG(NL, "num_hidden_layers"); CFG(E, "n_routed_experts"); CFG(I, "intermediate_size");
	CFG(V, "unpadded_vocab_size"); CFG(LS, "core_start"); CFG(LE, "core_end");
#undef CFG
	fprintf(stderr, "hidden %d layers %d experts %d x %d vocab %d loop [%d,%d)\n", H, NL, E, I, V, LS, LE);
	if (E > 256) { fprintf(stderr, "too many experts\n"); return 1; }

	// embeddings, final norms
	put_matrix("embed", must("model.llm.embed.weight", 0), emb, V);
	put_matrix("unembed", must("model.llm.unembed.weight", 0), emb, V);
	put_same("embed_norm", must("model.llm.embed_norm.weight", 0));
	put_same("final_norm", must("model.llm.norm.weight", 0));
	put_same("out_query", must("model.llm.attn_res.output_query", 0));
	{	// cross-pass message queries: (loop layer, [q, k, v, mlp], H)
		const cats_st_tensor *q = must("model.llm.middle_cycle_cross_pass_queries", 0), *kv = must("model.llm.middle_cycle_cross_pass_kv_queries", 0);
		const int nl = LE - LS;
		if (q->shape[0] != 2 * nl || kv->shape[0] != 2 * nl || q->shape[1] != H) { fprintf(stderr, "unexpected cross-pass query shape\n"); return 1; }
		float *a = (float*)malloc((size_t)2 * nl * H * 4), *b = (float*)malloc((size_t)2 * nl * H * 4), *m = (float*)malloc((size_t)nl * 4 * H * 4);
		cats_st_read_f32(&st, q, a); cats_st_read_f32(&st, kv, b);
		for (int i = 0; i < nl; i++) {
			memcpy(m + ((size_t)i * 4 + 0) * H, a + (size_t)(2 * i) * H, (size_t)H * 4);		// query
			memcpy(m + ((size_t)i * 4 + 1) * H, b + (size_t)(2 * i) * H, (size_t)H * 4);		// key
			memcpy(m + ((size_t)i * 4 + 2) * H, b + (size_t)(2 * i + 1) * H, (size_t)H * 4);	// value
			memcpy(m + ((size_t)i * 4 + 3) * H, a + (size_t)(2 * i + 1) * H, (size_t)H * 4);	// mlp
		}
		free(a); free(b);
		const int64_t s[3] = { nl, 4, H };
		ent_add(&L, "msg_query", "F32", 3, s, m, (uint64_t)nl * 4 * H * 4);
	}

	for (int l = 0; l < NL; l++) {
		char nm[96];
#define NAME(suffix) (snprintf(nm, sizeof(nm), "blk.%d." suffix, l), nm)
#define SRC(fmt) must("model.llm.layers.%d." fmt, l)
		const int full = maybe("model.llm.layers.%d.attn.k_sconv.static_w", l) != NULL;
		put_matrix(NAME("wq"), SRC("attn.wq_du.weight"), attn, 0);
		put_matrix(NAME("wk"), SRC("attn.wk_dv.weight"), attn, 0);
		put_matrix(NAME("wv"), SRC("attn.wv_dv.weight"), attn, 0);
		put_matrix(NAME("wo"), SRC("attn.wo_ud.weight"), attn, 0);
		put_matrix(NAME("wgate"), SRC("attn.gate_attn.weight"), attn, 0);
		put_same(NAME("wr"), SRC("attn.wr_du.weight"));
		put_same(NAME("sink"), SRC("attn.attn_sink_proj.weight"));
		put_same(NAME("qscale"), SRC("attn.q_l2_clip.scale"));
		put_same(NAME("rscale"), SRC("attn.rel_l2_clip.scale"));
		put_same(NAME("relproj"), SRC("attn.rel_logits_proj.proj"));
		if (full) {
			put_same(NAME("k_static"), SRC("attn.k_sconv.static_w"));
			put_same(NAME("k_wproj"), SRC("attn.k_sconv.weight_proj"));
			put_same(NAME("v_static"), SRC("attn.v_sconv.static_w"));
			put_same(NAME("v_wproj"), SRC("attn.v_sconv.weight_proj"));
			put_same(NAME("q_in_norm"), SRC("q_input_norm.weight"));
			put_same(NAME("k_in_norm"), SRC("k_input_norm.weight"));
			put_same(NAME("v_in_norm"), SRC("v_input_norm.weight"));
		} else {
			put_f32(NAME("k_conv"), SRC("attn.k_sconv.weight"), 2, SRC("attn.k_sconv.weight")->shape[0], SRC("attn.k_sconv.weight")->shape[2], 0);
			put_f32(NAME("v_conv"), SRC("attn.v_sconv.weight"), 2, SRC("attn.v_sconv.weight")->shape[0], SRC("attn.v_sconv.weight")->shape[2], 0);
			put_same(NAME("attn_norm"), SRC("attn_norm.weight"));
		}
		put_f32(NAME("attn_sconv"), SRC("attn_sconv.weight"), 2, H, SRC("attn_sconv.weight")->shape[2], 0);
		put_f32(NAME("mlp_sconv"), SRC("mlp_sconv.weight"), 2, H, SRC("mlp_sconv.weight")->shape[2], 0);
		put_same(NAME("post_norm"), SRC("mlp_norm.weight"));
		put_same(NAME("router"), SRC("mlp.gate.weight"));
		put_same(NAME("router_bias"), SRC("mlp.gate.bias"));
		if (l > 0) put_same(NAME("attn_res_q"), must("model.llm.attn_res.attn_queries.%d", l - 1));
		put_same(NAME("mlp_res_q"), must("model.llm.attn_res.mlp_queries.%d", l));
		if (maybe("model.llm.layers.%d.controllers.attn.down_weight", l)) {
			static const char *src[4] = { "attn", "k", "v", "mlp" }, *dst[4] = { "q", "k", "v", "m" };
			static const char *part[3] = { "down", "forget", "gain" };
			for (int c = 0; c < 4; c++) for (int p = 0; p < 3; p++) {
				char sn[160], dn[96];
				snprintf(sn, sizeof(sn), "model.llm.layers.%d.controllers.%s.%s_weight", l, src[c], part[p]);
				snprintf(dn, sizeof(dn), "blk.%d.ctrl_%s_%s", l, dst[c], part[p]);
				const cats_st_tensor *t = cats_st_find(&st, sn);
				if (!t) { fprintf(stderr, "missing %s\n", sn); return 1; }
				put_same(dn, t);
			}
		}
		const int edge = l < hi_layers || l >= NL - hi_layers;		// first / last layers at a higher precision
		const int t13 = edge ? hi_type : body, t2 = edge ? hi_type : (w2_type >= 0 ? w2_type : body);
		put_experts(l, 13, t13, E, 2 * I, H, I);
		put_experts(l, 2, t2, E, H, I, I);
#undef NAME
#undef SRC
	}

	// tokenizer (optional) and config, embedded last so that every tensor before them stays aligned
	snprintf(path, sizeof(path), "%s/tokenizer.json", in);
	size_t tj_size;
	char *tj = cats_read_file(path, &tj_size);
	uint8_t *blob; size_t blob_size;
	if (!tj) fprintf(stderr, "warning: no %s, the model file will accept token ids only\n", path);
	else if (cats_bpe_from_json(tj, tj_size, &blob, &blob_size)) { fprintf(stderr, "cannot convert %s\n", path); return 1; }
	else { int64_t s1[1] = { (int64_t)blob_size }; ent_add(&L, "tokenizer", "U8", 1, s1, blob, blob_size); }
	free(tj);
	{ int64_t s1[1] = { (int64_t)cfg_size }; ent_add(&L, "config", "U8", 1, s1, cfg, cfg_size); }
	if (cats_st_write(out, L.e, L.n, "pt")) { fprintf(stderr, "write failed: %s\n", out); return 1; }
	fprintf(stderr, "wrote %s (%.2f GB, %d tensors)\n", out, total_bytes / 1073741824.0, L.n);
	return 0;
}
