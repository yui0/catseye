//---------------------------------------------------------
//	Cat's eye
//
//		©2026 Yuichiro Nakada
//---------------------------------------------------------

// Inkling-2-Mini (Thinking Machines, 26B MoE, looped transformer) on Cat's eye: text-only CPU inference of the file
// written by inkling_quantize. Only what is specific to the architecture lives here (block attention residuals,
// the looped middle layers with their cross-pass messages, relative-position attention with sinks, short
// convolutions, the softcapped SwiGLU experts, the chat format); the safetensors reader, quantized GEMM and the
// BPE tokenizer are in ../catseye.h.
//
// The model file is memory-mapped and the routed experts (most of its size) are paged in on demand, so a file
// larger than RAM works: the experts a token needs are announced with madvise() right after routing. Prompts are
// processed in chunks (--batch tokens at a time, layer by layer) so every expert is read once per chunk instead of
// once per token.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <ctype.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#define CATS_INFER_ONLY
#include "../catseye.h"
#include "../catseye_openai.h"

static double tick(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static FILE *trace_f;	// INKLING_TRACE=file: binary (layer, expert) pairs of every routed expert, for cache studies
static long pr_n, pr_hit, pr_hit2, pr_tot;	// next-layer expert prediction: recall of the top K / top 2K
static int pr_for = -1, pr_top[64];
static double t_pre, t_attn, t_moe, t_w13, t_w2, t_route;	// -v breakdown (ms)
#define MAX_BLOCKS	16
#define MAX_TOPK	32

typedef struct {
	int H, NL, nq, nkv, hd, swa_nq, swa_nkv, swa_hd;
	int I, E, topk, window, rel_extent, drel, V, block;
	int loop_start, loop_end, passes, gate_rank;
	float eps, route_scale, qk_clip, rel_clip, log_n, log_alpha, gate_cap, up_cap, logit_cap;
	int eos;
	char *local;			// [NL]: 1 = sliding window layer
} cfg_t;

typedef struct {
	int full, boundary, loop, xtype, xtype2;
	const cats_st_tensor *wq, *wk, *wv, *wo, *wgate;
	const uint8_t *w13, *w2;
	size_t w13_rb, w2_rb;
	float *wr, *sink, *qscale, *rscale, *relproj;
	float *kconv, *vconv, *kstatic, *kwproj, *vstatic, *vwproj;	// sliding: conv [C][K]; full: static [K][C], proj [nkv*K][H]
	float *qin, *kin, *vin;					// RMSNorm weights of the attention inputs (sliding: only qin)
	float *attn_sconv, *mlp_sconv, *post_norm;
	float *router, *rbias;
	float *attn_res_q, *mlp_res_q;
	float *ctrl[4][3];					// q, k, v, mlp x down, forget, gain
	int Kc, Kkv, relw;
} layer_t;

typedef struct {
	float *k, *v;			// [cap][nkv * hd]
	float *hist[4];			// raw inputs of the k, v, attention and mlp short convolutions: [K - 1][C]
	int cap;
} vcache_t;

typedef struct {
	cfg_t c;
	cats_st st;
	cats_bpe tok;
	int have_tok;
	layer_t *L;
	const cats_st_tensor *embed, *unembed;
	float *embed_norm, *final_norm, *out_query, *msg_query;
	vcache_t *vc;
	int nvis, ctx, passes, B, nth, topk, big, kvbig, nqmax, ctxmax;
	size_t page;
	// scratch, [B] rows of every per-token quantity
	float *h, *resid, *x3, *qi, *ki, *vi, *tmp, *aout, *part, *a, *mi, *xn, *moe_out, *pre_h, *prev_h;
	float *q, *gate, *kk, *vv, *rs, *sinkv, *attn, *kconv, *vconv, *rlog;
	float *blocks, *prev_blocks, *msgs, *stack, *gin, *thr_score, *thr_rel;
	float *xg, *egu, *eact, *eout, *logits, *logits_all, *ex_w;
	int *ex_idx, *ex_cnt, *ex_tok, *ex_slot;
	float *decode_gate, *decode_output;
	int8_t *decode_x, *decode_activations;
	// expert cache (--cache MB): LRU of whole experts read with pread() into private memory, so a model larger than RAM
	// streams in 2-4 MB reads instead of thousands of small page faults
	uint8_t *cbuf; size_t cslot, c13; int cslots, *cmap, *ckey; uint64_t *cstamp, cclock; long chit, cmiss;
	int eslot[1024], eslot_of[1024];
	int cur_ci;						// visit index of the layer being run (see layer_fwd)
	pthread_t pf_th; int pf_run, pf_l, pf_n, pf_want[64]; long cpf;	// speculative read of the next layer's experts
	int fd;	// the model file, for large explicit reads (see prefetch_experts)
} model_t;

//---------------------------------------------------------
static void rmsnorm_w(float *o, const float *x, const float *w, int n, float eps)
{
	double s = 0;
	for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
	const float f = 1.0f / sqrtf((float)(s / n) + eps);
	for (int i = 0; i < n; i++) o[i] = x[i] * f * w[i];
}
static float *getf(cats_st *st, const char *fmt, int a, int need)
{
	char nm[160];
	snprintf(nm, sizeof(nm), fmt, a);
	const cats_st_tensor *t = cats_st_find(st, nm);
	if (!t) { if (need) fprintf(stderr, "missing tensor: %s\n", nm); return NULL; }
	float *d = (float*)malloc((size_t)cats_st_numel(t) * sizeof(float));
	if (!d || cats_st_read_f32(st, t, d)) { fprintf(stderr, "cannot read %s\n", nm); exit(1); }
	return d;
}
static const cats_st_tensor *gett(cats_st *st, const char *fmt, int a)
{
	char nm[160];
	snprintf(nm, sizeof(nm), fmt, a);
	const cats_st_tensor *t = cats_st_find(st, nm);
	if (!t) { fprintf(stderr, "missing tensor: %s\n", nm); exit(1); }
	return t;
}
static int thr_id(void)
{
#ifdef _OPENMP
	return omp_get_thread_num();
#else
	return 0;
#endif
}
// Announces that [p, p + len) of the mapped model file will be needed (the kernel starts paging it in).
static void advise(const model_t *m, const void *p, size_t len)
{
#if !defined(_WIN32)
	const uintptr_t a = (uintptr_t)p & ~(uintptr_t)(m->page - 1), e = ((uintptr_t)p + len + m->page - 1) & ~(uintptr_t)(m->page - 1);
	madvise((void*)a, e - a, MADV_WILLNEED);
#else
	(void)m; (void)p; (void)len;
#endif
}

// Brings the experts a layer visit needs into the page cache with a few large reads done in parallel. A page fault
// reads only ~4-16 KB, which saturates a drive's random-read IOPS (30 MB/s measured on a USB SSD that streams 580 MB/s);
// one expert is a contiguous 2-4 MB range, so reading it whole is what the drive is good at.
static void prefetch_experts(const model_t *m, const layer_t *y, const int *want, int n)
{
#if defined(__linux__)
	if (m->fd >= 0) {
		const int I = m->c.I, H = m->c.H;
		#pragma omp parallel for schedule(static) if(n > 1)
		for (int j = 0; j < 2 * n; j++) {
			const int e = want[j >> 1];
			const uint8_t *p = (j & 1) ? y->w2 + (size_t)e * H * y->w2_rb : y->w13 + (size_t)e * 2 * I * y->w13_rb;
			const size_t len = (j & 1) ? (size_t)H * y->w2_rb : (size_t)2 * I * y->w13_rb;
			readahead(m->fd, (off64_t)(p - m->st.map), len);
		}
		return;
	}
#endif
	const int I = m->c.I, H = m->c.H;
	for (int j = 0; j < n; j++) {
		advise(m, y->w13 + (size_t)want[j] * 2 * I * y->w13_rb, (size_t)2 * I * y->w13_rb);
		advise(m, y->w2 + (size_t)want[j] * H * y->w2_rb, (size_t)H * y->w2_rb);
	}
}

static const char *dump_dir;	// INKLING_DUMP=dir: append every MoE input (x%02d.f32, [tokens][H]) and its routing (i%02d.i16, [tokens][topk]) per layer
static void dump_moe(const model_t *m, int l, int nt, const float *x)
{
	static FILE *fx[64], *fi[64];
	char p[512];
	if (!fx[l]) { snprintf(p, sizeof(p), "%s/x%02d.f32", dump_dir, l); fx[l] = fopen(p, "ab"); snprintf(p, sizeof(p), "%s/i%02d.i16", dump_dir, l); fi[l] = fopen(p, "ab"); }
	fwrite(x, sizeof(float), (size_t)nt * m->c.H, fx[l]);
	for (int b = 0; b < nt; b++) for (int k = 0; k < m->topk; k++) { const int16_t v = (int16_t)m->ex_idx[(size_t)b * MAX_TOPK + k]; fwrite(&v, 2, 1, fi[l]); }
	fflush(fx[l]); fflush(fi[l]);
}
static int verbose_pred, opt_prefetch;	// --prefetch N: read the N most likely experts of the next layer in the background
static int opt_cache_mb = -1;	// -1 = automatic, 0 = off (plain mmap), N = N MB

static int load_model(model_t *m, const char *path, int ctx, int passes, int B, int topk)
{
	memset(m, 0, sizeof(*m));
	if (cats_st_open(&m->st, path)) { fprintf(stderr, "cannot open %s\n", path); return -1; }
	cats_st *st = &m->st;
#if !defined(_WIN32)
	m->page = (size_t)sysconf(_SC_PAGESIZE);
	m->fd = open(path, O_RDONLY);
#else
	m->page = 4096;
#endif
	const cats_st_tensor *ct = cats_st_find(st, "config");
	if (!ct) { fprintf(stderr, "%s: not an inkling_quantize file (no config)\n", path); return -1; }
	char *cj = (char*)malloc(ct->end - ct->begin + 1);
	memcpy(cj, cats_st_raw(st, ct), ct->end - ct->begin);
	cj[ct->end - ct->begin] = 0;
	cfg_t *c = &m->c;
	double v;
#define CI(f, key, d) do { c->f = cats_json_num(cj, key, &v) ? (d) : (int)v; } while (0)
#define CF(f, key, d) do { c->f = cats_json_num(cj, key, &v) ? (float)(d) : (float)v; } while (0)
	CI(H, "hidden_size", 2304); CI(NL, "num_hidden_layers", 24); CI(nq, "num_attention_heads", 24); CI(nkv, "num_key_value_heads", 8);
	CI(hd, "head_dim", 128); CI(swa_nq, "swa_num_attention_heads", c->nq); CI(swa_nkv, "swa_num_key_value_heads", c->nkv);
	CI(swa_hd, "swa_head_dim", c->hd); CI(I, "intermediate_size", 1152); CI(E, "n_routed_experts", 128);
	CI(topk, "num_experts_per_tok", 8); CI(window, "sliding_window_size", 1024); CI(rel_extent, "rel_extent", 1024);
	CI(drel, "d_rel", 16); CI(V, "unpadded_vocab_size", 200058); CI(block, "n_layers_per_attn_res_block", 6);
	CI(loop_start, "core_start", 6); CI(loop_end, "core_end", 18); CI(passes, "num_passes", 3); CI(gate_rank, "gate_rank", 64);
	CI(eos, "eos_token_id", 200006);
	CF(eps, "rms_norm_eps", 1e-6); CF(route_scale, "route_scale", 8); CF(qk_clip, "qk_l2_clip_threshold", 22.627417f);
	CF(rel_clip, "rel_l2_clip_threshold", 8); CF(log_n, "log_scaling_n_floor", 8192); CF(log_alpha, "log_scaling_alpha", 0.1f);
	CF(gate_cap, "mlp_gate_softcap", 4); CF(up_cap, "mlp_up_softcap", 25); CF(logit_cap, "final_logit_softcapping", 30);
#undef CI
#undef CF
	if (c->topk > MAX_TOPK || c->NL / c->block + 2 > MAX_BLOCKS || c->E > 1024) { fprintf(stderr, "unsupported configuration\n"); return -1; }
	c->local = (char*)calloc((size_t)c->NL, 1);
	double ids[256];
	int nl = cats_json_nums(cj, "local_layer_ids", ids, 256);
	for (int i = 0; i < nl && i < 256; i++) if ((int)ids[i] >= 0 && (int)ids[i] < c->NL) c->local[(int)ids[i]] = 1;
	free(cj);
	m->passes = passes > 0 ? passes : c->passes;
	m->topk = topk > 0 && topk < c->topk ? topk : c->topk;
	m->B = B > 0 ? B : 1;
	const int nloop = c->loop_end - c->loop_start;
	m->nvis = c->NL + (m->passes > 1 ? (m->passes - 1) * nloop : 0);
	const int H = c->H;
#ifdef _OPENMP
	m->nth = omp_get_max_threads();
#else
	m->nth = 1;
#endif

	if ((ct = cats_st_find(st, "tokenizer")) && !cats_bpe_load(&m->tok, cats_st_raw(st, ct), ct->end - ct->begin)) m->have_tok = 1;
	m->embed = gett(st, "embed", 0);
	m->unembed = gett(st, "unembed", 0);
	m->embed_norm = getf(st, "embed_norm", 0, 1);
	m->final_norm = getf(st, "final_norm", 0, 1);
	m->out_query = getf(st, "out_query", 0, 1);
	m->msg_query = getf(st, "msg_query", 0, 0);
	if (m->passes > 1 && !m->msg_query) return -1;
	m->L = (layer_t*)calloc((size_t)c->NL, sizeof(layer_t));
	for (int l = 0; l < c->NL; l++) {
		layer_t *y = &m->L[l];
		y->full = !c->local[l];
		y->boundary = l % c->block == 0;
		y->loop = l >= c->loop_start && l < c->loop_end;
		y->wq = gett(st, "blk.%d.wq", l); y->wk = gett(st, "blk.%d.wk", l); y->wv = gett(st, "blk.%d.wv", l);
		y->wo = gett(st, "blk.%d.wo", l); y->wgate = gett(st, "blk.%d.wgate", l);
		const cats_st_tensor *w13 = gett(st, "blk.%d.w13", l), *w2 = gett(st, "blk.%d.w2", l);
		y->xtype = w13->dtype; y->xtype2 = w2->dtype;
		y->w13 = cats_st_raw(st, w13); y->w2 = cats_st_raw(st, w2);
		y->w13_rb = (size_t)cats_st_bytes(y->xtype, w13->shape[1]);
		y->w2_rb = (size_t)cats_st_bytes(y->xtype2, w2->shape[1]);
		y->wr = getf(st, "blk.%d.wr", l, 1); y->sink = getf(st, "blk.%d.sink", l, 1);
		y->qscale = getf(st, "blk.%d.qscale", l, 1); y->rscale = getf(st, "blk.%d.rscale", l, 1);
		y->relproj = getf(st, "blk.%d.relproj", l, 1);
		y->relw = (int)gett(st, "blk.%d.relproj", l)->shape[1];
		if (y->full) {
			y->kstatic = getf(st, "blk.%d.k_static", l, 1); y->kwproj = getf(st, "blk.%d.k_wproj", l, 1);
			y->vstatic = getf(st, "blk.%d.v_static", l, 1); y->vwproj = getf(st, "blk.%d.v_wproj", l, 1);
			y->qin = getf(st, "blk.%d.q_in_norm", l, 1); y->kin = getf(st, "blk.%d.k_in_norm", l, 1);
			y->vin = getf(st, "blk.%d.v_in_norm", l, 1);
			y->Kkv = (int)gett(st, "blk.%d.k_static", l)->shape[0];
		} else {
			y->kconv = getf(st, "blk.%d.k_conv", l, 1); y->vconv = getf(st, "blk.%d.v_conv", l, 1);
			y->qin = getf(st, "blk.%d.attn_norm", l, 1);
			y->Kkv = (int)gett(st, "blk.%d.k_conv", l)->shape[1];
		}
		y->attn_sconv = getf(st, "blk.%d.attn_sconv", l, 1); y->mlp_sconv = getf(st, "blk.%d.mlp_sconv", l, 1);
		y->Kc = (int)gett(st, "blk.%d.attn_sconv", l)->shape[1];
		y->post_norm = getf(st, "blk.%d.post_norm", l, 1);
		y->router = getf(st, "blk.%d.router", l, 1); y->rbias = getf(st, "blk.%d.router_bias", l, 1);
		y->attn_res_q = l > 0 ? getf(st, "blk.%d.attn_res_q", l, 1) : NULL;
		y->mlp_res_q = getf(st, "blk.%d.mlp_res_q", l, 1);
		if (y->loop && m->passes > 1) {
			static const char *cn[4] = { "q", "k", "v", "m" }, *pn[3] = { "down", "forget", "gain" };
			for (int a = 0; a < 4; a++) for (int b = 0; b < 3; b++) {
				char nm[96];
				snprintf(nm, sizeof(nm), "blk.%d.ctrl_%s_%s", l, cn[a], pn[b]);
				const cats_st_tensor *t = cats_st_find(st, nm);
				if (!t) { fprintf(stderr, "missing tensor: %s\n", nm); return -1; }
				y->ctrl[a][b] = (float*)malloc((size_t)cats_st_numel(t) * 4);
				cats_st_read_f32(st, t, y->ctrl[a][b]);
			}
		}
	}
	// Everything but the experts and the embedding table is needed for every token: page it in now. The experts are
	// read on demand (see moe_batch); random access means no read-ahead of the neighbours.
#if !defined(_WIN32)
	madvise(st->map, st->size, MADV_RANDOM);
	for (int i = 0; i < st->count; i++) {
		const cats_st_tensor *t = &st->tensor[i];
		const size_t n = strlen(t->name);
		if ((n > 4 && !strcmp(t->name + n - 4, ".w13")) || (n > 3 && !strcmp(t->name + n - 3, ".w2")) || !strcmp(t->name, "embed")) continue;
		advise(m, st->data + t->begin, (size_t)(t->end - t->begin));
	}
#endif
	// caches: one per layer visit (the looped layers get a private copy per pass). The sliding ring holds the
	// window plus one chunk, so a chunk can write its keys before its first token has read the oldest ones.
	m->ctx = ctx;
	m->ctxmax = ctx > c->window + m->B ? ctx : c->window + m->B;
	m->vc = (vcache_t*)calloc((size_t)m->nvis, sizeof(vcache_t));
	for (int i = 0; i < m->nvis; i++) {
		const int l = i < c->NL ? i : c->loop_start + (i - c->NL) % nloop;
		const layer_t *y = &m->L[l];
		const int kvdim = y->full ? c->nkv * c->hd : c->swa_nkv * c->swa_hd;
		vcache_t *k = &m->vc[i];
		k->cap = y->full ? ctx : (ctx < c->window + m->B ? ctx : c->window + m->B);
		k->k = (float*)calloc((size_t)k->cap * kvdim, sizeof(float));
		k->v = (float*)calloc((size_t)k->cap * kvdim, sizeof(float));
		k->hist[0] = (float*)calloc((size_t)(y->Kkv - 1) * kvdim + 1, sizeof(float));
		k->hist[1] = (float*)calloc((size_t)(y->Kkv - 1) * kvdim + 1, sizeof(float));
		k->hist[2] = (float*)calloc((size_t)(y->Kc - 1) * H + 1, sizeof(float));
		k->hist[3] = (float*)calloc((size_t)(y->Kc - 1) * H + 1, sizeof(float));
	}
	m->big = c->nq * c->hd > c->swa_nq * c->swa_hd ? c->nq * c->hd : c->swa_nq * c->swa_hd;
	m->kvbig = c->nkv * c->hd > c->swa_nkv * c->swa_hd ? c->nkv * c->hd : c->swa_nkv * c->swa_hd;
	m->nqmax = c->nq > c->swa_nq ? c->nq : c->swa_nq;
	const size_t Bs = (size_t)m->B;
#define A(p, n) m->p = (float*)calloc((size_t)(n) + 8, sizeof(float))
	A(h, Bs * H); A(resid, Bs * H); A(x3, Bs * 3 * H); A(qi, Bs * H); A(ki, Bs * H); A(vi, Bs * H); A(tmp, Bs * H); A(aout, Bs * H);
	A(part, Bs * H); A(a, Bs * H); A(mi, Bs * H); A(xn, Bs * H); A(moe_out, Bs * H); A(pre_h, Bs * H); A(prev_h, Bs * H);
	A(q, Bs * m->big); A(gate, Bs * m->big); A(attn, Bs * m->big); A(kk, Bs * m->kvbig); A(vv, Bs * m->kvbig);
	A(kconv, Bs * m->kvbig); A(vconv, Bs * m->kvbig); A(rs, Bs * m->nqmax * c->drel); A(sinkv, Bs * m->nqmax); A(rlog, Bs * c->E);
	A(blocks, (size_t)MAX_BLOCKS * Bs * H); A(prev_blocks, (size_t)MAX_BLOCKS * Bs * H); A(msgs, Bs * 4 * H);
	A(stack, Bs * (MAX_BLOCKS + 1) * H); A(gin, Bs * 2 * H);
	A(thr_score, (size_t)m->nth * (m->ctxmax + 1)); A(thr_rel, (size_t)m->nth * (c->rel_extent > c->window ? c->rel_extent : c->window));
	A(xg, Bs * H); A(egu, Bs * 2 * c->I); A(eact, Bs * c->I); A(eout, Bs * H); A(logits, c->V); A(ex_w, Bs * MAX_TOPK);
#undef A
	m->ex_idx = (int*)calloc(Bs * MAX_TOPK, sizeof(int));
	m->ex_cnt = (int*)calloc((size_t)c->E, sizeof(int));
	m->ex_tok = (int*)calloc((size_t)c->E * Bs, sizeof(int));
	m->ex_slot = (int*)calloc((size_t)c->E * Bs, sizeof(int));
	m->decode_gate = malloc((size_t)m->topk * 2 * c->I * sizeof(float));
	m->decode_output = malloc((size_t)m->topk * H * sizeof(float));
	m->decode_x = malloc(cats_q8_0_row_bytes(H));
	m->decode_activations = malloc((size_t)m->topk * cats_q8_0_row_bytes(c->I));
	return 0;
}

//---------------------------------------------------------
// gemv / gemm helpers. W is any supported tensor, x is [n][in], out is [n][rows]
static void gemm_t(float *out, const float *x, const cats_st *st, const cats_st_tensor *t, int n)
{
	if (cats_st_gemm_nt(out, x, st, t, (size_t)n, NULL)) { fprintf(stderr, "unsupported tensor %s\n", t->name); exit(1); }
}
// small F32 matrices: out[n][rows] = x[n][in] . w[rows][in]^T
static void gemm_f32(float *out, const float *x, const float *w, int in, int rows, int n)
{
	if (n > 1) { cats_gemm_nt(out, x, w, (size_t)n, (size_t)in, (size_t)rows, NULL); return; }
	#pragma omp parallel for schedule(static) if((long)in * rows > 65536)
	for (int r = 0; r < rows; r++) out[r] = cats_dot_f32(x, w + (size_t)r * in, (size_t)in);
}
// rows [row0, row0 + rows) of an expert matrix stored as quantized rows: out[nt][rows] = x[nt][in] . W^T
static void gemm_rows(int type, float *out, const float *x, const uint8_t *w, size_t rb, long row0, int in, int rows, int nt)
{
	w += (size_t)row0 * rb;
	if (type == CATS_ST_Q4_0) cats_gemm_nt_q4_0(out, x, w, (size_t)nt, (size_t)in, (size_t)rows, NULL);
	else if (type == CATS_ST_Q3_0) cats_gemm_nt_q3_0(out, x, w, (size_t)nt, (size_t)in, (size_t)rows, NULL);
	else if (type == CATS_ST_F16) {		// debugging: unquantized experts
		#pragma omp parallel for schedule(static)
		for (int r = 0; r < rows; r++) {
			float *row = (float*)malloc((size_t)in * sizeof(float));
			for (int i = 0; i < in; i++) row[i] = cats_half_to_float(cats_load_u16(w + (size_t)r * rb + 2 * (size_t)i));
			for (int t = 0; t < nt; t++) out[(size_t)t * rows + r] = cats_dot_f32(x + (size_t)t * in, row, (size_t)in);
			free(row);
		}
	} else cats_gemm_nt_q8_0(out, x, w, (size_t)nt, (size_t)in, (size_t)rows, NULL);
}

// Block attention residual for one token: nqry learned queries attend over the (RMS-normalized) completed blocks
// (block j of this token at blocks + j * bs) plus the partial block; the normalized states are also the values.
static void block_res(const cfg_t *c, float *out, const float *blocks, size_t bs, int nb, const float *partial, const float *query, int nqry, float *stack)
{
	const int H = c->H;
	for (int j = 0; j < nb; j++) cats_rmsnorm_noscale(stack + (size_t)j * H, blocks + (size_t)j * bs, (unsigned)H, 1e-6f);
	cats_rmsnorm_noscale(stack + (size_t)nb * H, partial, (unsigned)H, 1e-6f);
	const float sc = 1.0f / sqrtf((float)H);
	for (int a = 0; a < nqry; a++) {
		float w[MAX_BLOCKS + 1], mx = -INFINITY;
		for (int j = 0; j <= nb; j++) { w[j] = cats_dot_f32(query + (size_t)a * H, stack + (size_t)j * H, (size_t)H) * sc; if (w[j] > mx) mx = w[j]; }
		float sum = 0;
		for (int j = 0; j <= nb; j++) { w[j] = expf(w[j] - mx); sum += w[j]; }
		float *o = out + (size_t)a * H;
		memset(o, 0, (size_t)H * sizeof(float));
		for (int j = 0; j <= nb; j++) { const float f = w[j] / sum; for (int i = 0; i < H; i++) o[i] += f * stack[(size_t)j * H + i]; }
	}
}

// (1 - forget) * x + gain * msg with forget / gain = tanh of low-rank projections of both (RMS-normalized)
static void msg_gate(const cfg_t *c, float *out, const float *x, const float *msg, float *const ctrl[3], float *g)
{
	const int H = c->H, R = c->gate_rank;
	cats_rmsnorm_noscale(g, x, (unsigned)H, c->eps);
	cats_rmsnorm_noscale(g + H, msg, (unsigned)H, c->eps);
	float d[256];
	gemm_f32(d, g, ctrl[0], 2 * H, R, 1);
	for (int i = 0; i < H; i++) {
		const float f = tanhf(cats_dot_f32(d, ctrl[1] + (size_t)i * R, (size_t)R));
		const float a = tanhf(cats_dot_f32(d, ctrl[2] + (size_t)i * R, (size_t)R));
		out[i] = (1.0f - f) * x[i] + a * msg[i];
	}
}

// Causal depthwise convolution of one token. w: [C][K] (tap K-1 = this token); hist: previous K-1 inputs.
// out = (conv(x) + x) / sqrt(K + 1); the history is advanced.
static void sconv_static(float *out, const float *x, float *hist, const float *w, int C, int K)
{
	const float inv = 1.0f / sqrtf((float)(K + 1));
	for (int c = 0; c < C; c++) {
		float a = w[(size_t)c * K + K - 1] * x[c];
		for (int j = 0; j < K - 1; j++) a += w[(size_t)c * K + j] * hist[(size_t)j * C + c];
		out[c] = (a + x[c]) * inv;
	}
	if (K > 1) { memmove(hist, hist + C, (size_t)(K - 2) * C * sizeof(float)); memcpy(hist + (size_t)(K - 2) * C, x, (size_t)C * sizeof(float)); }
}
// Dynamic variant (full attention k / v): taps = static[lag][c] + proj(dyn)[head][lag], out = (conv + x) / sqrt(2K + 1)
static void sconv_dynamic(float *out, const float *x, float *hist, const float *stat, const float *wproj, const float *dyn,
			  int H, int C, int nh, int K)
{
	const int hd = C / nh;
	float dw[64];
	gemm_f32(dw, dyn, wproj, H, nh * K, 1);
	const float inv = 1.0f / sqrtf((float)(2 * K + 1));
	for (int c = 0; c < C; c++) {
		const int h = c / hd;
		float a = (stat[c] + dw[h * K]) * x[c];						// lag 0
		for (int lag = 1; lag < K; lag++) a += (stat[(size_t)lag * C + c] + dw[h * K + lag]) * hist[(size_t)(K - 1 - lag) * C + c];
		out[c] = (a + x[c]) * inv;
	}
	if (K > 1) { memmove(hist, hist + C, (size_t)(K - 2) * C * sizeof(float)); memcpy(hist + (size_t)(K - 2) * C, x, (size_t)C * sizeof(float)); }
}

// Attention of nt consecutive tokens (positions pos0 ...) of one layer visit. m->qi / ki / vi: [nt][H] normalized inputs.
static void attention(model_t *m, int l, int ci, int nt, float *out, int pos0)
{
	const cfg_t *c = &m->c;
	const layer_t *y = &m->L[l];
	vcache_t *vc = &m->vc[ci];
	const int nq = y->full ? c->nq : c->swa_nq, nkv = y->full ? c->nkv : c->swa_nkv, hd = y->full ? c->hd : c->swa_hd;
	const int kvdim = nkv * hd, qdim = nq * hd, rep = nq / nkv, H = c->H;

	gemm_t(m->q, m->qi, &m->st, y->wq, nt);
	gemm_t(m->gate, m->qi, &m->st, y->wgate, nt);
	gemm_t(m->kk, m->ki, &m->st, y->wk, nt);
	gemm_t(m->vv, m->vi, &m->st, y->wv, nt);
	gemm_f32(m->rs, m->qi, y->wr, H, nq * c->drel, nt);
	gemm_f32(m->sinkv, m->qi, y->sink, H, nq, nt);

	// query: learned scale, L2 clip (+ log scaling in the full layers); same for the relative states
	#pragma omp parallel for schedule(static) if(nt > 1)
	for (int b = 0; b < nt; b++) {
		const float tau = y->full ? 1.0f + c->log_alpha * logf(fmaxf((float)(pos0 + b + 1) / c->log_n, 1.0f)) : 1.0f;
		for (int h = 0; h < nq; h++) {
			float *q = m->q + (size_t)b * qdim + (size_t)h * hd;
			for (int d = 0; d < hd; d++) q[d] *= y->qscale[(size_t)h * hd + d];
			cats_l2_clip(q, (size_t)hd, c->qk_clip);
			if (y->full) for (int d = 0; d < hd; d++) q[d] *= tau;
			float *r = m->rs + ((size_t)b * nq + h) * c->drel;
			for (int d = 0; d < c->drel; d++) r[d] *= y->rscale[(size_t)h * c->drel + d];
			cats_l2_clip(r, (size_t)c->drel, c->rel_clip);
			if (y->full) for (int d = 0; d < c->drel; d++) r[d] *= tau;
		}
	}
	// key / value: short convolution (sequential over the chunk: it carries state), L2 clip (k) / RMS norm without
	// scale (v), then into the cache
	for (int b = 0; b < nt; b++) {
		float *kconv = m->kconv + (size_t)b * kvdim, *vconv = m->vconv + (size_t)b * kvdim;
		if (y->full) {
			sconv_dynamic(kconv, m->kk + (size_t)b * kvdim, vc->hist[0], y->kstatic, y->kwproj, m->ki + (size_t)b * H, H, kvdim, nkv, y->Kkv);
			sconv_dynamic(vconv, m->vv + (size_t)b * kvdim, vc->hist[1], y->vstatic, y->vwproj, m->vi + (size_t)b * H, H, kvdim, nkv, y->Kkv);
		} else {
			sconv_static(kconv, m->kk + (size_t)b * kvdim, vc->hist[0], y->kconv, kvdim, y->Kkv);
			sconv_static(vconv, m->vv + (size_t)b * kvdim, vc->hist[1], y->vconv, kvdim, y->Kkv);
		}
		const int slot = (pos0 + b) % vc->cap;
		float *kd = vc->k + (size_t)slot * kvdim, *vd = vc->v + (size_t)slot * kvdim;
		for (int h = 0; h < nkv; h++) {
			memcpy(kd + (size_t)h * hd, kconv + (size_t)h * hd, (size_t)hd * sizeof(float));
			cats_l2_clip(kd + (size_t)h * hd, (size_t)hd, c->qk_clip);
			cats_rmsnorm_noscale(vd + (size_t)h * hd, vconv + (size_t)h * hd, (unsigned)hd, c->eps);
		}
	}

	const int rext = y->relw, relstride = c->rel_extent > c->window ? c->rel_extent : c->window;
	const float sc = 1.0f / sqrtf((float)hd);
	#pragma omp parallel for schedule(dynamic, 1) if(nt * nq > 1)
	for (int idx = 0; idx < nt * nq; idx++) {
		const int b = idx / nq, h = idx % nq, pos = pos0 + b, kvh = h / rep;
		const int lo = y->full ? 0 : (pos - c->window + 1 > 0 ? pos - c->window + 1 : 0);
		const int n = pos - lo + 1, nd = rext < n ? rext : n;
		float *sb = m->thr_score + (size_t)thr_id() * (m->ctxmax + 1), *rel = m->thr_rel + (size_t)thr_id() * relstride;
		const float *q = m->q + (size_t)b * qdim + (size_t)h * hd, *r = m->rs + ((size_t)b * nq + h) * c->drel;
		const float sinkl = m->sinkv[(size_t)b * nq + h];
		memset(rel, 0, (size_t)nd * sizeof(float));
		for (int k = 0; k < c->drel; k++) {
			const float *proj = y->relproj + (size_t)k * rext;
			const float weight = r[k];
			for (int d = 0; d < nd; d++) rel[d] += weight * proj[d];
		}
		float mx = sinkl;
		for (int j = 0; j < n; j++) {
			const int p = lo + j, dist = pos - p;
			const float *k = vc->k + (size_t)(p % vc->cap) * kvdim + (size_t)kvh * hd;
			float a = cats_dot_f32(q, k, (size_t)hd) * sc;
			if (dist < rext) a += rel[dist];
			sb[j] = a;
			if (a > mx) mx = a;
		}
		float sum = expf(sinkl - mx);
		for (int j = 0; j < n; j++) { sb[j] = expf(sb[j] - mx); sum += sb[j]; }
		float *o = m->attn + (size_t)b * qdim + (size_t)h * hd;
		memset(o, 0, (size_t)hd * sizeof(float));
		for (int j = 0; j < n; j++) {
			const float f = sb[j] / sum;
			const float *v = vc->v + (size_t)((lo + j) % vc->cap) * kvdim + (size_t)kvh * hd;
			for (int d = 0; d < hd; d++) o[d] += f * v[d];
		}
		const float *g = m->gate + (size_t)b * qdim + (size_t)h * hd;
		for (int d = 0; d < hd; d++) o[d] *= cats_sigmoid(g[d]);
	}
	gemm_t(out, m->attn, &m->st, y->wo, nt);
}

// Routed experts for nt tokens: route, group the tokens by expert, announce all needed experts to the kernel, then
// run each expert once over its tokens.
// Memory this process may use: physical RAM, or the cgroup limit when one is set (containers, systemd scopes).
static double mem_budget_bytes(void)
{
	double mem = 0;
#if defined(_SC_PHYS_PAGES)
	mem = (double)sysconf(_SC_PHYS_PAGES) * (double)sysconf(_SC_PAGESIZE);
#endif
#if defined(__linux__)
	char path[512], line[256], cg[256] = "";
	FILE *f = fopen("/proc/self/cgroup", "r");
	while (f && fgets(line, sizeof(line), f)) if (!strncmp(line, "0::", 3)) { snprintf(cg, sizeof(cg), "%s", line + 3); cg[strcspn(cg, "\n")] = 0; }
	if (f) fclose(f);
	for (char *p = cg; *cg; ) {		// the scope itself, then each parent: the tightest limit wins
		snprintf(path, sizeof(path), "/sys/fs/cgroup%s/memory.max", cg);
		FILE *g = fopen(path, "r");
		if (g) { if (fgets(line, sizeof(line), g) && line[0] >= '0' && line[0] <= '9' && atof(line) < mem) mem = atof(line); fclose(g); }
		if (!(p = strrchr(cg, '/')) || p == cg) break;
		*p = 0;
	}
#endif
	return mem;
}

// Decides the expert cache size when it was not given: only a model whose experts do not fit in memory needs one (the
// others are better served by the page cache, which the OS can shrink). Returns MB, 0 = not needed.
static int auto_cache_mb(const model_t *m)
{
	const cfg_t *c = &m->c;
	double experts = 0;
	for (int l = 0; l < c->NL; l++) experts += (double)c->E * ((double)2 * c->I * m->L[l].w13_rb + (double)c->H * m->L[l].w2_rb);
	const double rest = (double)m->st.size - experts + 0.4e9;		// attention, tables, KV cache and scratch stay resident
	const double budget = mem_budget_bytes() * 0.7 - rest;
	if (experts <= budget) return 0;
	return budget > 256e6 ? (int)(budget / 1048576.0) : 256;
}

// Sets up the expert cache on first use (the layer tensors are known by then).
static void cache_init(model_t *m)
{
	const cfg_t *c = &m->c;
	size_t c13 = 0, c2 = 0;
	for (int l = 0; l < c->NL; l++) {
		const layer_t *y = &m->L[l];
		const size_t a = (size_t)2 * c->I * y->w13_rb, b = (size_t)c->H * y->w2_rb;
		if (a > c13) c13 = a;
		if (b > c2) c2 = b;
	}
	c13 = (c13 + 63) & ~(size_t)63; c2 = (c2 + 63) & ~(size_t)63;
	m->c13 = c13; m->cslot = c13 + c2;
	long n = (long)((size_t)opt_cache_mb * 1048576 / m->cslot);
	const int minn = c->E < m->B * m->topk ? c->E : m->B * m->topk;
	if (n < minn) n = minn;
	if (n > (long)c->NL * c->E) n = (long)c->NL * c->E;
	m->cslots = (int)n;
	if (posix_memalign((void**)&m->cbuf, 4096, (size_t)n * m->cslot)) { fprintf(stderr, "cannot allocate the expert cache\n"); exit(1); }
	m->cmap = (int*)malloc((size_t)c->NL * c->E * sizeof(int));
	for (long i = 0; i < (long)c->NL * c->E; i++) m->cmap[i] = -1;
	m->ckey = (int*)malloc((size_t)n * sizeof(int));
	m->cstamp = (uint64_t*)calloc((size_t)n, sizeof(uint64_t));
	for (long i = 0; i < n; i++) m->ckey[i] = -1;
	fprintf(stderr, "expert cache: %ld slots x %.1f MB = %.0f MB (of %d experts)\n", n, m->cslot / 1048576.0, n * m->cslot / 1048576.0, c->NL * c->E);
}

// Makes the experts want[0..n) of layer l resident (slot[j] = cache slot of want[j]); misses are read in parallel. Slots
// stamped with the current clock are never evicted, so the layer being computed is safe while the next one is read.
static int cache_ensure(model_t *m, int l, const layer_t *y, const int *want, int n, int *slot, long *hits)
{
	const int E = m->c.E, I = m->c.I, H = m->c.H;
	int miss[1024], nm = 0;
	for (int j = 0; j < n; j++) {
		const int key = l * E + want[j], s = m->cmap[key];
		if (s >= 0) { slot[j] = s; m->cstamp[s] = m->cclock; (*hits)++; }
		else miss[nm++] = j;
	}
	// victims: the least recently used slots that this visit has not touched
	for (int q = 0; q < nm; q++) {
		int best = -1;
		for (int s = 0; s < m->cslots; s++) if (m->cstamp[s] < m->cclock && (best < 0 || m->cstamp[s] < m->cstamp[best])) best = s;
		if (best < 0) { fprintf(stderr, "expert cache too small\n"); exit(1); }
		if (m->ckey[best] >= 0) m->cmap[m->ckey[best]] = -1;
		const int j = miss[q], key = l * E + want[j];
		m->ckey[best] = key; m->cmap[key] = best; m->cstamp[best] = m->cclock; slot[j] = best;
	}
	#pragma omp parallel for schedule(dynamic, 1) num_threads(16)
	for (int q = 0; q < 2 * nm; q++) {
		const int j = miss[q >> 1], e = want[j];
		uint8_t *dst = m->cbuf + (size_t)slot[j] * m->cslot;
		const uint8_t *src; size_t len;
		if (q & 1) { dst += m->c13; src = y->w2 + (size_t)e * H * y->w2_rb; len = (size_t)H * y->w2_rb; }
		else { src = y->w13 + (size_t)e * 2 * I * y->w13_rb; len = (size_t)2 * I * y->w13_rb; }
		const off_t off = (off_t)(src - m->st.map);
		for (size_t got = 0; got < len; ) {
			const ssize_t r = pread(m->fd, dst + got, len - got, off + (off_t)got);
			if (r <= 0) { fprintf(stderr, "read error on the model file\n"); exit(1); }
			got += (size_t)r;
		}
		posix_fadvise(m->fd, off, (off_t)len, POSIX_FADV_DONTNEED);	// the copy lives in the cache: don't hold it twice
	}
	return nm;
}
static void pf_join(model_t *m)
{
	if (m->pf_run) { pthread_join(m->pf_th, NULL); m->pf_run = 0; }
}
static void *pf_main(void *arg)
{
	model_t *m = (model_t*)arg;
	int slot[64]; long hits = 0;
	m->cpf += cache_ensure(m, m->pf_l, &m->L[m->pf_l], m->pf_want, m->pf_n, slot, &hits);
	return NULL;
}
static void cache_fetch(model_t *m, int l, const layer_t *y, const int *want, int n, int *slot)
{
	pf_join(m);				// the speculative read of this layer (if any) must be complete first
	m->cclock++;
	long hits = 0;
	const int nm = cache_ensure(m, l, y, want, n, slot, &hits);
	m->chit += hits; m->cmiss += nm;
}
// Starts reading the experts predicted for layer l2 in the background, while the current layer computes.
static void cache_speculate(model_t *m, int l2, const int *pred, int n)
{
	m->pf_l = l2; m->pf_n = n < 64 ? n : 64;
	memcpy(m->pf_want, pred, (size_t)m->pf_n * sizeof(int));
	if (pthread_create(&m->pf_th, NULL, pf_main, m) == 0) m->pf_run = 1;
}

static void moe_decode_q4(model_t *m, const layer_t *y, float *out, const float *x,
			  const int *experts, int n)
{
	const cfg_t *c = &m->c;
	const int H = c->H, I = c->I;
	const size_t ar = cats_q8_0_row_bytes(I);
	const uint8_t *w13[MAX_TOPK], *w2[MAX_TOPK];
	float weights[MAX_TOPK];
	for (int j = 0; j < n; j++) {
		const int e = experts[j];
		weights[j] = m->ex_w[m->ex_slot[(size_t)e * m->B]];
		w13[j] = y->w13 + (size_t)e * 2 * I * y->w13_rb;
		w2[j] = y->w2 + (size_t)e * H * y->w2_rb;
		if (m->cbuf) {
			w13[j] = m->cbuf + (size_t)m->eslot_of[e] * m->cslot;
			w2[j] = w13[j] + m->c13;
		}
	}
	cats_q8_0_quantize(m->decode_x, x, H);
	#pragma omp parallel
	{
		#pragma omp for schedule(static)
		for (int r = 0; r < n * 2 * I; r++) {
			const int j = r / (2 * I), row = r % (2 * I);
			m->decode_gate[r] = cats_dot_q4_0_q8_0(w13[j] + (size_t)row * y->w13_rb, m->decode_x, H);
		}
		#pragma omp for schedule(static)
		for (int j = 0; j < n; j++) {
			float *gu = m->decode_gate + (size_t)j * 2 * I;
			for (int i = 0; i < I; i++) {
				const float g = gu[i], u = gu[I + i];
				gu[i] = cats_softcap(g, c->gate_cap) * cats_sigmoid(g) * cats_softcap(u, c->up_cap);
			}
			cats_q8_0_quantize(m->decode_activations + (size_t)j * ar, gu, I);
		}
		#pragma omp for schedule(static)
		for (int r = 0; r < n * H; r++) {
			const int j = r / H, row = r % H;
			m->decode_output[r] = cats_dot_q4_0_q8_0(w2[j] + (size_t)row * y->w2_rb,
							      m->decode_activations + (size_t)j * ar, I);
		}
	}
	memset(out, 0, (size_t)H * sizeof(float));
	for (int j = 0; j < n; j++) {
		const float *value = m->decode_output + (size_t)j * H;
		for (int i = 0; i < H; i++) out[i] += weights[j] * value[i];
	}
}

static void moe_batch(model_t *m, int l, int nt, float *out, const float *x)
{
	const cfg_t *c = &m->c;
	const layer_t *y = &m->L[l];
	const int E = c->E, I = c->I, H = c->H, K = m->topk;
	gemm_f32(m->rlog, x, y->router, H, E, nt);
	#pragma omp parallel for schedule(static) if(nt > 1)
	for (int b = 0; b < nt; b++) {
		float prob[1024], sel[1024];
		const float *sc = m->rlog + (size_t)b * E;
		for (int e = 0; e < E; e++) { prob[e] = cats_sigmoid(sc[e]); sel[e] = prob[e] + y->rbias[e]; }
		float wsum = 0;
		int *idx = m->ex_idx + (size_t)b * MAX_TOPK;
		float *wt = m->ex_w + (size_t)b * MAX_TOPK;
		for (int k = 0; k < K; k++) {
			int best = -1;
			for (int e = 0; e < E; e++) if (sel[e] > -1e30f && (best < 0 || sel[e] > sel[best])) best = e;
			idx[k] = best; wt[k] = prob[best]; wsum += prob[best]; sel[best] = -2e30f;
		}
		for (int k = 0; k < K; k++) wt[k] = wt[k] / (wsum + 1e-20f) * c->route_scale;
	}
	if (nt == 1 && (verbose_pred || (opt_prefetch > 0 && opt_cache_mb > 0))) {
		if (pr_for == l) {		// score what the previous layer predicted for this one
			for (int k = 0; k < K; k++) { int in1 = 0, in2 = 0; for (int j = 0; j < 2 * K; j++) if (pr_top[j] == m->ex_idx[k]) { in2 = 1; if (j < K) in1 = 1; } pr_hit += in1; pr_hit2 += in2; }
			pr_tot += K; pr_n++;
		}
		pr_for = -1;
		int nl = l + 1;			// the layer visited next: the loop's body repeats in the extra passes
		if (l == c->loop_end - 1) {
			const int pass = m->cur_ci < c->NL ? 0 : (m->cur_ci - c->NL) / (c->loop_end - c->loop_start) + 1;
			nl = pass + 1 < m->passes ? c->loop_start : c->loop_end;
		}
		if (nl < c->NL) {		// predict the next layer: its router applied to this layer's MoE input
			const layer_t *z = &m->L[nl];
			float lg[1024], sel[1024];
			gemm_f32(lg, x, z->router, H, E, 1);
			for (int e = 0; e < E; e++) sel[e] = cats_sigmoid(lg[e]) + z->rbias[e];
			for (int k = 0; k < 2 * K && k < 64; k++) {
				int best = 0;
				for (int e = 1; e < E; e++) if (sel[e] > sel[best]) best = e;
				pr_top[k] = best; sel[best] = -1e30f;
			}
			pr_for = nl;
		}
	}
	if (dump_dir) dump_moe(m, l, nt, x);
	if (trace_f && nt == 1) for (int k = 0; k < K; k++) { const uint16_t r[2] = { (uint16_t)l, (uint16_t)m->ex_idx[k] }; fwrite(r, 2, 2, trace_f); }
	memset(m->ex_cnt, 0, (size_t)E * sizeof(int));
	for (int b = 0; b < nt; b++) for (int k = 0; k < K; k++) {
		const int e = m->ex_idx[(size_t)b * MAX_TOPK + k], i = m->ex_cnt[e]++;
		m->ex_tok[(size_t)e * m->B + i] = b; m->ex_slot[(size_t)e * m->B + i] = k;
	}
	{
		int want[1024], nw = 0;
		for (int e = 0; e < E; e++) if (m->ex_cnt[e]) want[nw++] = e;
		const double tp = tick();
		if (opt_cache_mb > 0 && m->fd >= 0) {
			if (!m->cbuf) cache_init(m);
			cache_fetch(m, l, y, want, nw, m->eslot);
			if (nt == 1 && opt_prefetch > 0 && pr_for >= 0) cache_speculate(m, pr_for, pr_top, opt_prefetch < 2 * K ? opt_prefetch : 2 * K);
			for (int j = 0; j < nw; j++) m->eslot_of[want[j]] = m->eslot[j];
		} else prefetch_experts(m, y, want, nw);
		t_pre += tick() - tp;
		if (nt == 1 && y->xtype == CATS_ST_Q4_0 && y->xtype2 == CATS_ST_Q4_0 &&
		    m->decode_gate && m->decode_output && m->decode_x && m->decode_activations) {
			moe_decode_q4(m, y, out, x, want, nw);
			return;
		}
	}
	memset(out, 0, (size_t)nt * H * sizeof(float));
	for (int e = 0; e < E; e++) {
		const int cnt = m->ex_cnt[e];
		if (!cnt) continue;
		const int *tk = m->ex_tok + (size_t)e * m->B, *sl = m->ex_slot + (size_t)e * m->B;
		for (int i = 0; i < cnt; i++) memcpy(m->xg + (size_t)i * H, x + (size_t)tk[i] * H, (size_t)H * sizeof(float));
		double tg_ = tick();
		const uint8_t *W13 = y->w13, *W2 = y->w2;
		long r13 = (long)e * 2 * I, r2 = (long)e * H;
		if (m->cbuf) { W13 = m->cbuf + (size_t)m->eslot_of[e] * m->cslot; W2 = W13 + m->c13; r13 = r2 = 0; }
		gemm_rows(y->xtype, m->egu, m->xg, W13, y->w13_rb, r13, H, 2 * I, cnt);
		t_w13 += tick() - tg_;
		#pragma omp parallel for schedule(static) if((long)cnt * I > 4096)
		for (int i = 0; i < cnt; i++) {
			const float *gu = m->egu + (size_t)i * 2 * I;
			for (int j = 0; j < I; j++) {
				const float g = gu[j], u = gu[I + j];
				m->eact[(size_t)i * I + j] = cats_softcap(g, c->gate_cap) * cats_sigmoid(g) * cats_softcap(u, c->up_cap);
			}
		}
		tg_ = tick();
		gemm_rows(y->xtype2, m->eout, m->eact, W2, y->w2_rb, r2, I, H, cnt);
		t_w2 += tick() - tg_;
		for (int i = 0; i < cnt; i++) {
			const float w = m->ex_w[(size_t)tk[i] * MAX_TOPK + sl[i]];
			float *o = out + (size_t)tk[i] * H;
			const float *eo = m->eout + (size_t)i * H;
			for (int j = 0; j < H; j++) o[j] += w * eo[j];
		}
	}
}

// One layer visit for nt tokens. h: [nt][H] running (partial block) states, updated in place. blocks: completed block
// states, block j of token b at blocks + (j * B + b) * H. msg: [nt][4][H] cross-pass messages (q, k, v, mlp) in the
// passes after the first, else NULL.
static void layer_fwd(model_t *m, int l, int ci, int nt, float *h, int *nb, const float *msg, int pos0)
{
	const cfg_t *c = &m->c;
	layer_t *y = &m->L[l];
	const int H = c->H;
	const size_t bs = (size_t)m->B * H, SH = (size_t)(MAX_BLOCKS + 1) * H;
	vcache_t *vc = &m->vc[ci];
	memcpy(m->resid, h, (size_t)nt * H * sizeof(float));

	#pragma omp parallel for schedule(static) if(nt > 1)
	for (int b = 0; b < nt; b++) {
		float *x = m->x3 + (size_t)b * 3 * H, *tmp = m->tmp + (size_t)b * H, *g = m->gin + (size_t)b * 2 * H;
		float *dst[3] = { m->qi + (size_t)b * H, m->ki + (size_t)b * H, m->vi + (size_t)b * H };
		const float *mg = msg ? msg + (size_t)b * 4 * H : NULL;
		if (l > 0) block_res(c, x, m->blocks + (size_t)b * H, bs, *nb, h + (size_t)b * H, y->attn_res_q, y->full ? 3 : 1, m->stack + (size_t)b * SH);
		else memcpy(x, h + (size_t)b * H, (size_t)H * sizeof(float));
		if (y->full) {
			const float *nw[3] = { y->qin, y->kin, y->vin };
			for (int a = 0; a < 3; a++) {
				if (mg) { msg_gate(c, tmp, x + (size_t)a * H, mg + (size_t)a * H, y->ctrl[a], g); rmsnorm_w(dst[a], tmp, nw[a], H, c->eps); }
				else rmsnorm_w(dst[a], x + (size_t)a * H, nw[a], H, c->eps);
			}
		} else if (mg) {
			for (int a = 0; a < 3; a++) { msg_gate(c, tmp, x, mg + (size_t)a * H, y->ctrl[a], g); rmsnorm_w(dst[a], tmp, y->qin, H, c->eps); }
		} else {
			rmsnorm_w(dst[0], x, y->qin, H, c->eps);
			memcpy(dst[1], dst[0], (size_t)H * sizeof(float));
			memcpy(dst[2], dst[0], (size_t)H * sizeof(float));
		}
	}
	double t0_ = tick();
	m->cur_ci = ci;
	attention(m, l, ci, nt, m->aout, pos0);
	t_attn += tick() - t0_;
	for (int b = 0; b < nt; b++) sconv_static(m->a + (size_t)b * H, m->aout + (size_t)b * H, vc->hist[2], y->attn_sconv, H, y->Kc);

	// partial block after the attention
	for (int b = 0; b < nt; b++) {
		float *part = m->part + (size_t)b * H;
		const float *resid = m->resid + (size_t)b * H, *a = m->a + (size_t)b * H;
		if (y->boundary) { memcpy(m->blocks + ((size_t)(*nb) * m->B + b) * H, resid, (size_t)H * sizeof(float)); memcpy(part, a, (size_t)H * sizeof(float)); }
		else for (int i = 0; i < H; i++) part[i] = resid[i] + a[i];
	}
	if (y->boundary) (*nb)++;

	#pragma omp parallel for schedule(static) if(nt > 1)
	for (int b = 0; b < nt; b++) {
		float *mi = m->mi + (size_t)b * H, *tmp = m->tmp + (size_t)b * H;
		block_res(c, mi, m->blocks + (size_t)b * H, bs, *nb, m->part + (size_t)b * H, y->mlp_res_q, 1, m->stack + (size_t)b * SH);
		if (msg) { msg_gate(c, tmp, mi, msg + (size_t)b * 4 * H + 3 * H, y->ctrl[3], m->gin + (size_t)b * 2 * H); memcpy(mi, tmp, (size_t)H * sizeof(float)); }
		rmsnorm_w(m->xn + (size_t)b * H, mi, y->post_norm, H, c->eps);
	}
	t0_ = tick();
	moe_batch(m, l, nt, m->moe_out, m->xn);
	t_moe += tick() - t0_;
	for (int b = 0; b < nt; b++) {
		sconv_static(m->a + (size_t)b * H, m->moe_out + (size_t)b * H, vc->hist[3], y->mlp_sconv, H, y->Kc);
		for (int i = 0; i < H; i++) h[(size_t)b * H + i] = m->part[(size_t)b * H + i] + m->a[(size_t)b * H + i];
	}
}

// Runs nt consecutive tokens (positions pos0 ...) through the whole stack, layer by layer.
// want: 0 = no logits, 1 = logits of the last token in m->logits, 2 = logits of every token in m->logits_all [nt][V].
static void forward(model_t *m, const int *tok, int nt, int pos0, int want)
{
	const cfg_t *c = &m->c;
	const int H = c->H, nloop = c->loop_end - c->loop_start;
	const size_t bs = (size_t)m->B * H;
	float *h = m->h;
	int nb = 0;
	for (int b = 0; b < nt; b++) {
		float *e = m->tmp + (size_t)b * H;
		cats_st_read_row_f32(&m->st, m->embed, tok[b], e);
		rmsnorm_w(h + (size_t)b * H, e, m->embed_norm, H, c->eps);
	}
	for (int l = 0; l < c->loop_start; l++) layer_fwd(m, l, l, nt, h, &nb, NULL, pos0);
	memcpy(m->pre_h, h, (size_t)nt * H * sizeof(float));
	const int pre_nb = nb;
	for (int l = c->loop_start; l < c->loop_end; l++) layer_fwd(m, l, l, nt, h, &nb, NULL, pos0);
	for (int pass = 1; pass < m->passes; pass++) {
		// messages come from the states the previous pass ended with (its blocks after the first, and its output)
		const int np = nb - pre_nb - 1 > 0 ? nb - pre_nb - 1 : 0;
		memcpy(m->prev_blocks, m->blocks + (size_t)(pre_nb + 1) * bs, (size_t)np * bs * sizeof(float));
		memcpy(m->prev_h, h, (size_t)nt * H * sizeof(float));
		memcpy(h, m->pre_h, (size_t)nt * H * sizeof(float));
		nb = pre_nb;
		for (int li = 0; li < nloop; li++) {
			#pragma omp parallel for schedule(static) if(nt > 1)
			for (int b = 0; b < nt; b++)
				block_res(c, m->msgs + (size_t)b * 4 * H, m->prev_blocks + (size_t)b * H, bs, np, m->prev_h + (size_t)b * H,
					  m->msg_query + (size_t)li * 4 * H, 4, m->stack + (size_t)b * (MAX_BLOCKS + 1) * H);
			layer_fwd(m, c->loop_start + li, c->NL + (pass - 1) * nloop + li, nt, h, &nb, m->msgs, pos0);
		}
	}
	for (int l = c->loop_end; l < c->NL; l++) layer_fwd(m, l, l, nt, h, &nb, NULL, pos0);
	if (!want) return;
	const int first = want == 2 ? 0 : nt - 1, cnt = nt - first;
	for (int b = first; b < nt; b++) {
		float *f = m->x3 + (size_t)b * 3 * H;
		block_res(c, f, m->blocks + (size_t)b * H, bs, nb, h + (size_t)b * H, m->out_query, 1, m->stack + (size_t)b * (MAX_BLOCKS + 1) * H);
		rmsnorm_w(m->xn + (size_t)b * H, f, m->final_norm, H, c->eps);
	}
	float *dst = want == 2 ? m->logits_all : m->logits;
	gemm_t(dst, m->xn + (size_t)first * H, &m->st, m->unembed, cnt);
	#pragma omp parallel for schedule(static) if((size_t)cnt * c->V >= 32768)
	for (size_t i = 0; i < (size_t)cnt * c->V; i++) dst[i] = cats_softcap(dst[i], c->logit_cap);
}

//---------------------------------------------------------
// sampling
static uint64_t rng_s = 88172645463325252ull;
static float rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (float)((rng_s >> 11) * (1.0 / 9007199254740992.0)); }
typedef struct { float p; int i; } pi_t;
static int cmp_pi(const void *a, const void *b)
{
	const pi_t *x = a, *y = b;
	if (x->p != y->p) return x->p < y->p ? 1 : -1;
	return (x->i > y->i) - (x->i < y->i);
}
static int sample(const float *logits, int n, float temp, float top_p, int top_k)
{
	if (temp <= 0) { int b = 0; for (int i = 1; i < n; i++) if (logits[i] > logits[b]) b = i; return b; }
	pi_t *a = (pi_t*)malloc((size_t)n * sizeof(pi_t));
	float mx = -INFINITY;
	for (int i = 0; i < n; i++) if (logits[i] > mx) mx = logits[i];
	double sum = 0;
	for (int i = 0; i < n; i++) { a[i].i = i; a[i].p = expf((logits[i] - mx) / temp); sum += a[i].p; }
	int k = top_k > 0 && top_k < n ? top_k : n;
	// The shortlist contains every token at or above the cutoff. It is
	// sufficient only if it covers the full top-k or reaches the nucleus.
	enum { SHORTLIST = 512 };
	pi_t likely[SHORTLIST];
	int count = 0;
	for (int i = 0; i < n && count < SHORTLIST; i++)
		if (a[i].p >= 1.0f / 1024) likely[count++] = a[i];
	if (count < SHORTLIST) {
		qsort(likely, (size_t)count, sizeof(pi_t), cmp_pi);
		if (count > 0 && count < k) {
			double mass = 0;
			for (int i = 0; i < count; i++) mass += likely[i].p / sum;
			if (mass >= top_p) k = count;
		}
	}
	const pi_t *ranked = likely;
	if (count < k || count == SHORTLIST) {
		qsort(a, (size_t)n, sizeof(pi_t), cmp_pi);
		ranked = a;
	}
	double cum = 0;
	for (int i = 0; i < k; i++) { cum += ranked[i].p / sum; if (cum >= top_p) { k = i + 1; break; } }
	double s2 = 0;
	for (int i = 0; i < k; i++) s2 += ranked[i].p;
	double r = rnd() * s2, acc = 0;
	int pick = ranked[k - 1].i;
	for (int i = 0; i < k; i++) { acc += ranked[i].p; if (r < acc) { pick = ranked[i].i; break; } }
	free(a);
	return pick;
}

//---------------------------------------------------------
// chat format
typedef struct { int *id; int n, cap; } ids_t;
static void ids_push(ids_t *t, int id) { if (t->n == t->cap) { t->cap = t->cap ? t->cap * 2 : 256; t->id = (int*)realloc(t->id, (size_t)t->cap * sizeof(int)); } t->id[t->n++] = id; }
static int special_id(const cats_bpe *b, const char *s)
{
	const size_t l = strlen(s);
	for (int i = 0; i < b->nsp; i++) if (b->tlen[b->sp[i]] == l && !memcmp(b->tok[b->sp[i]], s, l)) return b->sp[i];
	return -1;
}
static void ids_text(const cats_bpe *b, ids_t *t, const char *s)
{
	if (!*s) return;
	const size_t need = strlen(s) + t->n;
	if (need > (size_t)t->cap) {
		size_t cap = t->cap ? (size_t)t->cap : 256;
		while (cap < need) cap *= 2;
		int *ids = realloc(t->id, cap * sizeof(int));
		if (!ids || cap > INT_MAX) { fprintf(stderr, "cannot allocate token buffer\n"); exit(1); }
		t->id = ids;
		t->cap = (int)cap;
	}
	t->n += cats_bpe_encode(b, s, 0, t->id + t->n, t->cap - t->n);
}
static void ids_msg(const cats_bpe *b, ids_t *t, const char *role, const char *text)
{
	ids_push(t, special_id(b, role));
	ids_push(t, special_id(b, "<|content_text|>"));
	ids_text(b, t, text);
	ids_push(t, special_id(b, "<|end_message|>"));
}
static const char *effort_string(const char *e, char *buf)
{
	static const struct { const char *k, *v; } map[] = { { "none", "0" }, { "minimal", "0.1" }, { "low", "0.2" }, { "medium", "0.7" },
							      { "high", "0.9" }, { "xhigh", "0.99" }, { "max", "0.99" } };
	for (size_t i = 0; i < sizeof(map) / sizeof(*map); i++) if (!strcmp(e, map[i].k)) return map[i].v;
	char *end;
	double v = strtod(e, &end);
	if (end == e || *end || !isfinite(v) || v < 0 || v > 0.99) return NULL;
	snprintf(buf, 32, "%g", v);
	return buf;
}

#include "openai_adapter.h"

//---------------------------------------------------------
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static void usage(const char *a)
{
	fprintf(stderr,
		"Usage: %s -m MODEL.safetensors [-p PROMPT | -f FILE] [options]\n"
		"  -s TEXT        system message           -n N      max new tokens (default 512)\n"
		"  -t TEMP        temperature (0 = greedy, default 0.7)   --top-p P (0.95)   --top-k K (0 = off)   --seed N\n"
		"  --effort E     thinking effort: none minimal low medium high(default) xhigh max, or a number 0..0.99\n"
		"  --think        also print the model's thinking     --raw   print every token including control tokens\n"
		"  --serve PORT   OpenAI-compatible server on 127.0.0.1\n"
		"  -i             interactive chat (the conversation stays in the KV cache)\n"
		"  -c N           context length (default 4096)        --passes N   loop passes (default from the model; fewer = faster)\n"
		"  --topk K       experts per token (default: the model's 8; fewer = faster, lower quality)\n"
		"  --batch N      prompt tokens processed together (default 64; each expert is read once per chunk)\n"
		"  --threads N    -v   timing\n"
		"  --ppl FILE     perplexity of a text file (plain text, no chat format); --save FILE writes its logits,\n"
		"                 --ref FILE compares with logits saved by another model file: mean KL(ref||this) and top-1 agreement\n"
		"  --cache MB     expert cache for models larger than memory (default: automatic; 0 = plain mmap)\n"
		"  --prefetch N   read the N most likely experts of the next layer in the background (with the expert cache)\n"
		"  --ids 1,2,3   feed raw token ids (no chat format)    --logits FILE   write the logits of every fed id (float32)\n", a);
}

int main(int argc, char **argv)
{
	const char *mp = NULL, *prompt = NULL, *file = NULL, *sys = NULL, *effort = "high", *ids_arg = NULL, *logits_file = NULL, *ppl_file = NULL, *save_file = NULL, *ref_file = NULL;
	int maxn = 512, ctx = 4096, passes = 0, threads = 0, top_k = 0, interactive = 0, think = 0, raw = 0, verbose = 0, batch = 64, topk = 0;
	int port = 0;
	float temp = 0.7f, top_p = 0.95f;
	uint64_t seed = 0;
	for (int i = 1; i < argc; i++) {
		const char *a = argv[i];
#define ARG (i + 1 < argc ? argv[++i] : (usage(argv[0]), exit(1), (char*)0))
		if (!strcmp(a, "-m")) mp = ARG;
		else if (!strcmp(a, "-p")) prompt = ARG;
		else if (!strcmp(a, "-f")) file = ARG;
		else if (!strcmp(a, "-s")) sys = ARG;
		else if (!strcmp(a, "-n")) maxn = atoi(ARG);
		else if (!strcmp(a, "-t")) temp = (float)atof(ARG);
		else if (!strcmp(a, "--top-p")) top_p = (float)atof(ARG);
		else if (!strcmp(a, "--top-k")) top_k = atoi(ARG);
		else if (!strcmp(a, "--seed")) seed = strtoull(ARG, NULL, 10);
		else if (!strcmp(a, "--effort")) effort = ARG;
		else if (!strcmp(a, "-c")) ctx = atoi(ARG);
		else if (!strcmp(a, "--passes")) passes = atoi(ARG);
		else if (!strcmp(a, "--cache")) opt_cache_mb = atoi(ARG);
		else if (!strcmp(a, "--pred")) verbose_pred = 1;
		else if (!strcmp(a, "--prefetch")) opt_prefetch = atoi(ARG);
		else if (!strcmp(a, "--topk")) topk = atoi(ARG);
		else if (!strcmp(a, "--batch")) batch = atoi(ARG);
		else if (!strcmp(a, "--threads")) threads = atoi(ARG);
		else if (!strcmp(a, "--ids")) ids_arg = ARG;
		else if (!strcmp(a, "--logits")) logits_file = ARG;
		else if (!strcmp(a, "--ppl")) ppl_file = ARG;
		else if (!strcmp(a, "--save")) save_file = ARG;
		else if (!strcmp(a, "--ref")) ref_file = ARG;
		else if (!strcmp(a, "--serve")) {
			const char *value = ARG;
			char *end;
			long parsed = strtol(value, &end, 10);
			if (*end || parsed < 1 || parsed > 65535) { usage(argv[0]); return 1; }
			port = (int)parsed;
		}
		else if (!strcmp(a, "-i")) interactive = 1;
		else if (!strcmp(a, "--think")) think = 1;
		else if (!strcmp(a, "--raw")) raw = 1;
		else if (!strcmp(a, "-v")) verbose = 1;
		else { usage(argv[0]); return 1; }
#undef ARG
	}
	if (!mp || ctx < 2 || batch < 1 ||
	    (port && (maxn < 1 || prompt || file || sys || interactive || ids_arg || ppl_file || logits_file || raw || think))) {
		usage(argv[0]);
		return 1;
	}
	cats_set_threads(threads);
	if (seed) rng_s ^= seed * 0x9E3779B97F4A7C15ull;
	model_t M;
	dump_dir = getenv("INKLING_DUMP");
	if (getenv("INKLING_TRACE")) trace_f = fopen(getenv("INKLING_TRACE"), "wb");
	double t0 = now_ms();
	if (load_model(&M, mp, ctx, passes, batch, topk)) return 1;
	if (opt_cache_mb < 0) { opt_cache_mb = auto_cache_mb(&M); if (verbose && opt_cache_mb) fprintf(stderr, "experts exceed memory: automatic expert cache %d MB\n", opt_cache_mb); }
	if (verbose) fprintf(stderr, "loaded in %.0f ms: %d layers, %d experts, loop [%d,%d) x %d, ctx %d\n", now_ms() - t0, M.c.NL, M.c.E, M.c.loop_start, M.c.loop_end, M.passes, ctx);

	ids_t in = { 0 };
	if (ids_arg) {
		for (const char *p = ids_arg; *p; ) { char *e; long v = strtol(p, &e, 10); if (e == p) break; ids_push(&in, (int)v); p = *e ? e + 1 : e; }
	}
	FILE *lf = logits_file ? fopen(logits_file, "wb") : NULL;
	int pos = 0;
	if (ids_arg) {	// raw ids: run them in chunks, dump the logits of every position, print the greedy next token
		if (lf) M.logits_all = (float*)calloc((size_t)M.B * M.c.V + 8, sizeof(float));
		const float *last = M.logits;
		for (int i = 0; i < in.n && pos < ctx; ) {
			const int nt = in.n - i < M.B ? in.n - i : M.B, final = i + nt == in.n;
			forward(&M, in.id + i, nt, pos, lf ? 2 : final ? 1 : 0);
			if (lf) { fwrite(M.logits_all, sizeof(float), (size_t)nt * M.c.V, lf); last = M.logits_all + (size_t)(nt - 1) * M.c.V; }
			i += nt; pos += nt;
		}
		if (lf) fclose(lf);
		printf("next %d\n", sample(last, M.c.V, 0, 1, 0));
		return 0;
	}
	if (ppl_file) {
		if (!M.have_tok) { fprintf(stderr, "this model file has no tokenizer\n"); return 1; }
		char *txt = cats_read_file(ppl_file, NULL);
		if (!txt) { fprintf(stderr, "cannot read %s\n", ppl_file); return 1; }
		const int cap = ctx - 1;
		int *ti = (int*)malloc((size_t)cap * sizeof(int));
		int nt = cats_bpe_encode(&M.tok, txt, 0, ti, cap);
		if (nt > cap) nt = cap;
		const int V = M.c.V;
		M.logits_all = (float*)calloc((size_t)M.B * V + 8, sizeof(float));
		FILE *sf = save_file ? fopen(save_file, "wb") : NULL, *rf = ref_file ? fopen(ref_file, "rb") : NULL;
		float *rl = rf ? (float*)malloc((size_t)V * sizeof(float)) : NULL;
		double nll = 0, kl = 0;
		long cnt = 0, agree = 0;
		const double t1 = now_ms();
		for (int i = 0; i < nt; ) {
			const int n = nt - i < M.B ? nt - i : M.B;
			forward(&M, ti + i, n, i, 2);
			for (int b = 0; b < n; b++) {
				const float *lg = M.logits_all + (size_t)b * V;
				if (sf) fwrite(lg, sizeof(float), (size_t)V, sf);
				float mx = lg[0]; int am = 0;
				for (int v = 1; v < V; v++) if (lg[v] > mx) { mx = lg[v]; am = v; }
				double z = 0;
				for (int v = 0; v < V; v++) z += exp((double)lg[v] - mx);
				const double lz = log(z) + mx;
				if (rl && fread(rl, sizeof(float), (size_t)V, rf) == (size_t)V) {
					float rmx = rl[0]; int ram = 0;
					for (int v = 1; v < V; v++) if (rl[v] > rmx) { rmx = rl[v]; ram = v; }
					double rz = 0;
					for (int v = 0; v < V; v++) rz += exp((double)rl[v] - rmx);
					const double rlz = log(rz) + rmx;
					double k = 0;
					for (int v = 0; v < V; v++) { const double p = exp((double)rl[v] - rlz); if (p > 1e-12) k += p * ((rl[v] - rlz) - (lg[v] - lz)); }
					kl += k; agree += am == ram;
				}
				if (i + b + 1 < nt) { nll -= (double)lg[ti[i + b + 1]] - lz; cnt++; }
			}
			i += n;
		}
		printf("tokens %d  perplexity %.3f  (mean NLL %.4f)", nt, exp(nll / cnt), nll / cnt);
		if (rf) printf("  KL(ref||this) %.4f  top-1 agreement %.1f%%", kl / nt, 100.0 * agree / nt);
		printf("  [%.0f s]\n", (now_ms() - t1) / 1e3);
		return 0;
	}
	if (!M.have_tok) { fprintf(stderr, "this model file has no tokenizer\n"); return 1; }
	char *ftxt = NULL;
	if (file) { ftxt = cats_read_file(file, NULL); if (!ftxt) { fprintf(stderr, "cannot read %s\n", file); return 1; } prompt = ftxt; }
	char ebuf[32];
	const char *eff = effort_string(effort, ebuf);
	if (!eff) { fprintf(stderr, "bad --effort\n"); return 1; }
	const cats_bpe *B = &M.tok;
	const int id_end = special_id(B, "<|end_message|>"), id_think = special_id(B, "<|content_thinking|>"), id_text = special_id(B, "<|content_text|>"), id_eos = M.c.eos, id_model = special_id(B, "<|message_model|>");
	if (port) {
		inkling_oa server = {
			.m = &M, .defaults_n = maxn, .default_temp = temp, .default_p = top_p, .top_k = top_k,
			.effort = eff, .verbose = verbose,
			.system = special_id(B, "<|message_system|>"), .user = special_id(B, "<|message_user|>"),
			.model = id_model, .tool = special_id(B, "<|message_tool|>"), .text = id_text,
			.thinking = id_think, .xml = special_id(B, "<|content_xml|>"),
			.invoke = special_id(B, "<|content_invoke_tool_json|>"), .end = id_end
		};
		if (server.system < 0 || server.user < 0 || server.model < 0 || server.tool < 0 ||
		    server.text < 0 || server.thinking < 0 || server.xml < 0 || server.invoke < 0 || server.end < 0) {
			fprintf(stderr, "model tokenizer lacks Inkling chat tokens\n");
			return 1;
		}
		server.cached = malloc((size_t)M.ctx * sizeof(int));
		server.checkpoint = malloc(inkling_history(&server, 0) * sizeof(float) + 1);
		server.checkpoint_logits = malloc((size_t)M.c.V * sizeof(float));
		if (!server.cached || !server.checkpoint || !server.checkpoint_logits) {
			perror("prompt cache");
			return 1;
		}
		const cats_oa_backend api = { "inkling", &server, inkling_prepare, inkling_generate };
		const int result = cats_oa_serve(&api, port);
		free(server.cached);
		free(server.checkpoint);
		free(server.checkpoint_logits);
		free(server.input.id);
		return result;
	}
	int first = 1;
	char line[1 << 16];
	for (;;) {
		ids_t turn = { 0 };
		const char *user = prompt;
		if (interactive) {
			if (isatty(0)) { fprintf(stderr, "\n> "); fflush(stderr); }
			if (!fgets(line, sizeof(line), stdin)) break;
			line[strcspn(line, "\r\n")] = 0;
			if (!*line) continue;
			user = line;
		} else if (!user) {
			size_t n = 0, cap = 1 << 16;
			char *b = (char*)malloc(cap);
			int ch;
			while ((ch = getchar()) != EOF) { if (n + 1 >= cap) b = (char*)realloc(b, cap *= 2); b[n++] = (char)ch; }
			b[n] = 0;
			user = b;
		}
		if (first) {
			if (sys) ids_msg(B, &turn, "<|message_system|>", sys);
			char em[96];
			snprintf(em, sizeof(em), "Thinking effort level: %s", eff);
			ids_msg(B, &turn, "<|message_system|>", em);
			first = 0;
		}
		ids_msg(B, &turn, "<|message_user|>", user);
		if (pos + turn.n + 1 >= ctx) { fprintf(stderr, "context full (%d tokens)\n", ctx); return 1; }
		t0 = now_ms();
		for (int i = 0; i < turn.n; ) {
			const int nt = turn.n - i < M.B ? turn.n - i : M.B;
			forward(&M, turn.id + i, nt, pos, i + nt == turn.n);
			i += nt; pos += nt;
		}
		const double tp = now_ms() - t0;
		t0 = now_ms();
		int gen = 0, mode = 0;	// 0: between messages, 1: thinking, 2: text, 3: other
		for (; gen < maxn && pos < ctx; gen++) {
			const int next = sample(M.logits, M.c.V, temp, top_p, top_k);
			if (next == id_eos) { forward(&M, &next, 1, pos++, 0); break; }
			char piece[1024];
			if (raw) { const size_t n = cats_bpe_piece(B, next, 1, piece, sizeof(piece)); fwrite(piece, 1, n, stdout); }
			else if (next == id_think) mode = 1;
			else if (next == id_text) mode = 2;
			else if (next == id_end) { if (mode == 1 && think) fputs("\n\n", stdout); mode = 0; }
			else if (B->special[next]) { if (mode == 0 && next != id_model) mode = 3; }	// header / tool markers stay hidden
			else if (mode == 2 || (mode == 1 && think)) {
				const size_t n = cats_bpe_piece(B, next, 0, piece, sizeof(piece));
				fwrite(piece, 1, n, stdout);
			}
			fflush(stdout);
			forward(&M, &next, 1, pos++, 1);
			if (verbose && (gen + 1) % 32 == 0) { static double tw; const double tn = now_ms(); fprintf(stderr, "  [tokens %d-%d: %.2f tok/s]\n", gen - 30, gen + 1, 32e3 / (tn - (tw ? tw : t0))); tw = tn; }
		}
		fputc('\n', stdout);
		pf_join(&M);
		if (verbose) {
			double rb = 0;
#if defined(__linux__)
			FILE *pf = fopen("/proc/self/io", "r");
			char ln[128];
			while (pf && fgets(ln, sizeof(ln), pf)) if (!strncmp(ln, "read_bytes:", 11)) rb = atof(ln + 11);
			if (pf) fclose(pf);
#endif
			fprintf(stderr, "next-layer prediction: top-K recall %.1f%%, top-2K recall %.1f%% (%ld layer visits)\nexpert cache hits %ld misses %ld, speculative loads %ld\ntime: prefetch %.0f ms, attention %.0f ms, experts %.0f ms (w13 %.0f, w2 %.0f), storage reads %.2f GB\n", pr_tot ? 100.0 * pr_hit / pr_tot : 0, pr_tot ? 100.0 * pr_hit2 / pr_tot : 0, pr_n, M.chit, M.cmiss, M.cpf, t_pre, t_attn, t_moe, t_w13, t_w2, rb / 1e9);
		}
		if (verbose) fprintf(stderr, "prompt %d tokens %.0f ms (%.1f tok/s), generated %d tokens %.0f ms (%.2f tok/s)\n",
				     turn.n, tp, turn.n * 1e3 / (tp > 0 ? tp : 1), gen, now_ms() - t0, gen * 1e3 / (now_ms() - t0 > 0 ? now_ms() - t0 : 1));
		free(turn.id);
		if (!interactive) break;
	}
	return 0;
}
