//---------------------------------------------------------
//	Cat's eye - eg2e: EmbeddingGemma 2 vectors and similarity search (images and text)
//
//		eg2e -m MODEL_DIR vec  FILE_OR_"text"            print the 768-d unit vector
//		eg2e -m MODEL_DIR sim  QUERY CANDIDATE...        rank candidates by cosine similarity
//
//	QUERY / CANDIDATE: an image file (jpg/png/...), or text when it is not an existing file or
//	-t is used (text needs -k TOKENIZER.eg2t; make one with tools/eg2_embed_tokenizer.py).
//	Text is embedded as a search query / document with the model's prompts (-q to disable).
//	Embeddings of image files are cached in -C FILE (path + size + mtime + soft-token budget).
//---------------------------------------------------------

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#include "../stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../stb_image_resize.h"
#include <sys/stat.h>
#include <unistd.h>
#ifdef CATS_GPU
#include "../cats_gpu.h"
#endif
#include "eg2_embed.h"
#include "eg2_tokenizer.h"

#define PROMPT_QUERY	"task: search result | query: "
#define PROMPT_DOC	"title: none | text: "

static int bicubic(const unsigned char *src, int sw, int sh, unsigned char *dst, int dw, int dh)
{
	// Catmull-Rom (a = -0.5) with the support widened when shrinking, like torchvision's bicubic + antialias.
	return !stbir_resize_uint8_generic(src, sw, sh, sw * 3, dst, dw, dh, dw * 3, 3, -1, 0, STBIR_EDGE_CLAMP,
					   STBIR_FILTER_CATMULLROM, STBIR_COLORSPACE_LINEAR, NULL);
}

typedef struct {
	char *path;
	int is_text;
	float vec[EG2E_OUT_DIM];
	int ok;
} item;

static struct {
	eg2e_model model;
	eg2e_tok tok;
	int have_tok, soft, raw_text, loaded_vision;
	const char *model_dir, *cache_path;
} g = { .soft = 140 };

// ---- cache: [magic][count] then { path_len, path, size, mtime, soft, vec[768] } -------------------

typedef struct { char *path; long long size, mtime; int soft; float vec[EG2E_OUT_DIM]; } centry;
static centry *cache; static int ncache;

static void cache_load(void)
{
	FILE *f = g.cache_path ? fopen(g.cache_path, "rb") : NULL;
	if (!f) return;
	char magic[4]; int n;
	if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "EG2C", 4) || fread(&n, 4, 1, f) != 1) { fclose(f); return; }
	cache = (centry*)calloc((size_t)n + 1, sizeof(centry));
	for (int i = 0; i < n; i++) {
		int len;
		centry *e = &cache[ncache];
		if (fread(&len, 4, 1, f) != 1 || len <= 0 || len > 4096) break;
		e->path = (char*)malloc((size_t)len + 1);
		if (fread(e->path, 1, (size_t)len, f) != (size_t)len) break;
		e->path[len] = 0;
		if (fread(&e->size, 8, 1, f) != 1 || fread(&e->mtime, 8, 1, f) != 1 || fread(&e->soft, 4, 1, f) != 1 ||
		    fread(e->vec, 4, EG2E_OUT_DIM, f) != EG2E_OUT_DIM) break;
		ncache++;
	}
	fclose(f);
}
static void cache_save(void)
{
	if (!g.cache_path) return;
	FILE *f = fopen(g.cache_path, "wb");
	if (!f) return;
	fwrite("EG2C", 1, 4, f);
	fwrite(&ncache, 4, 1, f);
	for (int i = 0; i < ncache; i++) {
		int len = (int)strlen(cache[i].path);
		fwrite(&len, 4, 1, f); fwrite(cache[i].path, 1, (size_t)len, f);
		fwrite(&cache[i].size, 8, 1, f); fwrite(&cache[i].mtime, 8, 1, f); fwrite(&cache[i].soft, 4, 1, f);
		fwrite(cache[i].vec, 4, EG2E_OUT_DIM, f);
	}
	fclose(f);
}
static centry *cache_find(const char *path, long long size, long long mtime)
{
	for (int i = 0; i < ncache; i++)
		if (!strcmp(cache[i].path, path) && cache[i].size == size && cache[i].mtime == mtime && cache[i].soft == g.soft)
			return &cache[i];
	return NULL;
}
static void cache_add(const char *path, long long size, long long mtime, const float *vec)
{
	cache = (centry*)realloc(cache, ((size_t)ncache + 1) * sizeof(centry));
	centry *e = &cache[ncache++];
	e->path = strdup(path); e->size = size; e->mtime = mtime; e->soft = g.soft;
	memcpy(e->vec, vec, sizeof(e->vec));
}

// ---- embedding ----------------------------------------------------------------------------------

static double now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + 1e-9 * t.tv_nsec;
}
static int verbose;
static int ensure_model(void)
{
	if (g.model.st.map) return 0;
	double t0 = now();
	int rc = eg2e_load(&g.model, g.model_dir, g.loaded_vision);	// loaded_vision = any image among the arguments
	if (verbose) fprintf(stderr, "eg2e: model loaded in %.1f s\n", now() - t0);
	return rc;
}
static int embed_text(const char *text, const char *prompt, float *out)
{
	if (!g.have_tok) { fprintf(stderr, "eg2e: text needs -k TOKENIZER.eg2t\n"); return -1; }
	if (ensure_model()) return -1;
	size_t n = strlen(text) + strlen(prompt) + 1;
	char *full = (char*)malloc(n);
	snprintf(full, n, "%s%s", prompt, text);
	int ids[4096];
	int cnt = eg2t_encode(&g.tok, full, ids, 4096);
	free(full);
	return cnt < 0 ? -1 : eg2e_embed_text_ids(&g.model, ids, (size_t)cnt, out);
}
static int embed_image(const char *path, float *out)
{
	struct stat st;
	long long size = 0, mtime = 0;
	if (!stat(path, &st)) { size = st.st_size; mtime = st.st_mtime; }
	centry *c = cache_find(path, size, mtime);
	if (c) { memcpy(out, c->vec, sizeof(c->vec)); return 0; }
	int w, h, ch;
	unsigned char *rgb = stbi_load(path, &w, &h, &ch, 3);
	if (!rgb) { fprintf(stderr, "eg2e: cannot read image %s\n", path); return -1; }
	if (ensure_model()) { stbi_image_free(rgb); return -1; }
	double t0 = now();
	int rc = eg2e_embed_image(&g.model, rgb, w, h, g.soft, bicubic, out);
	if (verbose) fprintf(stderr, "eg2e: %s (%dx%d) embedded in %.2f s\n", path, w, h, now() - t0);
	stbi_image_free(rgb);
	if (!rc) cache_add(path, size, mtime, out);
	return rc;
}
static int embed_item(item *it, const char *prompt)
{
	it->ok = !(it->is_text ? embed_text(it->path, g.raw_text ? "" : prompt, it->vec) : embed_image(it->path, it->vec));
	return it->ok ? 0 : -1;
}
static int looks_like_file(const char *s) { return access(s, R_OK) == 0; }

static int by_score(const void *a, const void *b)
{
	float x = ((const float*)a)[0], y = ((const float*)b)[0];
	return x < y ? 1 : x > y ? -1 : 0;
}

// ---- quantize: model.safetensors (bf16) -> Q4_0 safetensors for text + image ------------------------

static const char *dtype_name(int d) { return d == CATS_ST_F32 ? "F32" : d == CATS_ST_F16 ? "F16" : "BF16"; }

// Matrices (2-D, input dim a multiple of 32) become Q4_0; norms, scalars and the position table keep their
// dtype. The audio tower is dropped. refine > 0 searches a better scale per block (slower, lower error).
static int quantize_model(const char *dir, const char *out_path, int refine, const char *keep)
{
	char in_path[1024];
	snprintf(in_path, sizeof in_path, "%s/model.safetensors", dir);
	cats_st st;
	if (cats_st_open(&st, in_path)) { fprintf(stderr, "eg2e: cannot open %s\n", in_path); return -1; }
	cats_st_entry *e = (cats_st_entry*)calloc((size_t)st.count, sizeof(*e));
	void **owned = (void**)calloc((size_t)st.count, sizeof(void*));
	int n = 0, nq = 0;
	uint64_t before = 0, after = 0;
	for (int i = 0; i < st.count; i++) {
		const cats_st_tensor *t = &st.tensor[i];
		if (!strncmp(t->name, "audio_tower.", 12) || !strncmp(t->name, "embed_audio.", 12)) continue;
		if (t->dtype < 0 || t->dtype == CATS_ST_Q4_0) { fprintf(stderr, "eg2e: %s: unsupported dtype\n", t->name); return -1; }
		cats_st_entry *x = &e[n];
		x->name = t->name; x->ndim = t->ndim; memcpy(x->shape, t->shape, sizeof x->shape);
		before += t->end - t->begin;
		int matrix = t->ndim == 2 && t->shape[1] % CATS_Q4_0_BLOCK == 0;
		if (matrix && keep) {	// -K a,b : leave tensors whose name contains any of these words in their original dtype
			char list[512];
			snprintf(list, sizeof list, "%s", keep);
			for (char *w = strtok(list, ","); w; w = strtok(NULL, ",")) if (strstr(t->name, w)) matrix = 0;
		}
		if (matrix) {
			size_t rows = (size_t)t->shape[0], cols = (size_t)t->shape[1], rb = cats_q4_0_row_bytes(cols);
			float *f = (float*)malloc(rows * cols * sizeof(float));
			uint8_t *q = (uint8_t*)malloc(rows * rb);
			if (!f || !q || cats_st_read_f32(&st, t, f)) { fprintf(stderr, "eg2e: out of memory / bad tensor %s\n", t->name); return -1; }
			#pragma omp parallel for schedule(static)
			for (long r = 0; r < (long)rows; r++) cats_q4_0_quantize_row(f + (size_t)r * cols, cols, q + (size_t)r * rb, refine);
			free(f);
			x->dtype = "Q4_0"; x->data = q; x->bytes = rows * rb; owned[n] = q;
			nq++;
		} else {
			x->dtype = dtype_name(t->dtype); x->data = cats_st_raw(&st, t); x->bytes = t->end - t->begin;
		}
		after += x->bytes;
		n++;
	}
	int rc = cats_st_write(out_path, e, n, "cats-q4_0");
	fprintf(stderr, "eg2e: %d tensors (%d quantized to Q4_0), %.0f MB -> %.0f MB\n", n, nq, before / 1e6, after / 1e6);
	for (int i = 0; i < n; i++) free(owned[i]);
	free(owned); free(e);
	cats_st_close(&st);
	return rc;
}

static void usage(const char *p)
{
	fprintf(stderr,
		"usage: %s -m MODEL_DIR [options] vec  FILE|TEXT\n"
		"       %s -m MODEL_DIR [options] sim  QUERY CANDIDATE...\n"
		"       %s -m MODEL_DIR -C CACHE index FILE...\n"
		"       %s -m MODEL_DIR [-F] [-K words] quantize OUT.safetensors   (Q4_0 model; load it with -m OUT_DIR)\n"
		"  -m DIR   directory with model.safetensors (google/embeddinggemma-2)\n"
		"  -k FILE  tokenizer (.eg2t) for text queries\n"
		"  -s N     soft tokens per image: 70, 140 (default), 280, 560, 1120 (more = finer, slower)\n"
		"  -n N     print only the first N vector values (vec)\n"
		"  -N N     show the best N matches (sim, default 10)\n"
		"  -C FILE  cache image embeddings\n"
		"  -t       treat the argument as text even if a file of that name exists\n"
		"  -r       raw text: do not add the search prompts\n"
		"  -F       quantize: plain GGML Q4_0 scales (faster, about 10%% larger error)\n"
		"  -K a,b   quantize: keep tensors whose name contains a or b unquantized (e.g. embed_tokens,vision_tower)\n"
		"  -j N     threads (default: physical cores, or OMP_NUM_THREADS)\n"
		"  -G auto|N  GPU matrix products (needs make GPU=1; weights are held as fp16 on the GPU, see README)\n"
		"  -v       print timings\n", p, p, p, p);
	exit(2);
}

int main(int argc, char **argv)
{
	int show = EG2E_OUT_DIM, top = 10, force_text = 0, threads = 0, fast_q = 0, c;
	const char *keep_list = NULL, *gpu_spec = NULL;
	const char *tokpath = NULL;
	opterr = 1;
	while ((c = getopt(argc, argv, "m:k:s:n:N:C:j:K:G:trvF")) != -1) {
		switch (c) {
		case 'm': g.model_dir = optarg; break;
		case 'k': tokpath = optarg; break;
		case 's': g.soft = atoi(optarg); break;
		case 'n': show = atoi(optarg); break;
		case 'N': top = atoi(optarg); break;
		case 'C': g.cache_path = optarg; break;
		case 't': force_text = 1; break;
		case 'r': g.raw_text = 1; break;
		case 'v': verbose = 1; eg2e_prof_on = 1; break;
		case 'F': fast_q = 1; break;
		case 'K': keep_list = optarg; break;
		case 'G': gpu_spec = optarg; break;
		case 'j': threads = atoi(optarg); break;
		default: usage(argv[0]);
		}
	}
	if (!g.model_dir || optind >= argc) usage(argv[0]);
	if (g.soft < 1 || g.soft > 1120) usage(argv[0]);
	if (tokpath) {
		if (eg2t_load(&g.tok, tokpath)) { fprintf(stderr, "eg2e: cannot read tokenizer %s\n", tokpath); return 1; }
		g.have_tok = 1;
	}
	cats_set_threads(threads);
#ifdef CATS_GPU
	if (gpu_spec && !cs_gpu_init(gpu_spec)) fprintf(stderr, "eg2e: GPU GEMM on %s\n", cs_gpu_name());
#else
	if (gpu_spec) fprintf(stderr, "eg2e: built without GPU support (make GPU=1); using the CPU\n");
#endif
	const char *cmd = argv[optind++];
	if (!strcmp(cmd, "quantize") && optind < argc) return quantize_model(g.model_dir, argv[optind], !fast_q, keep_list) ? 1 : 0;
	cache_load();
	for (int i = optind; i < argc; i++) if (!force_text && looks_like_file(argv[i])) g.loaded_vision = 1;
	int rc = 0;
	if (!strcmp(cmd, "vec") && optind < argc) {
		item it = { .path = argv[optind] };
		it.is_text = force_text || !looks_like_file(it.path);
		if (embed_item(&it, PROMPT_DOC)) return 1;
		if (show > EG2E_OUT_DIM) show = EG2E_OUT_DIM;
		for (int i = 0; i < show; i++) printf("%.7g%c", it.vec[i], i + 1 < show ? ' ' : '\n');
	} else if (!strcmp(cmd, "index") && optind < argc) {
		// embed every file once and keep the vectors in the -C cache (saved every 25 images)
		int total = argc - optind, done = 0, failed = 0;
		double t0 = now();
		for (int i = optind; i < argc; i++) {
			float v[EG2E_OUT_DIM];
			if (embed_image(argv[i], v)) failed++;
			if (++done % 25 == 0 || done == total) {
				cache_save();
				fprintf(stderr, "eg2e: %d/%d indexed (%d failed), %.0f s elapsed\n", done, total, failed, now() - t0);
			}
		}
	} else if (!strcmp(cmd, "sim") && optind + 1 < argc) {
		item q = { .path = argv[optind++] };
		q.is_text = force_text || !looks_like_file(q.path);
		if (embed_item(&q, PROMPT_QUERY)) return 1;
		int n = argc - optind;
		float (*rank)[2] = (float(*)[2])malloc((size_t)n * sizeof(*rank));
		item *cand = (item*)calloc((size_t)n, sizeof(item));
		int m = 0;
		for (int i = 0; i < n; i++) {
			cand[i].path = argv[optind + i];
			cand[i].is_text = looks_like_file(cand[i].path) ? 0 : 1;
			if (embed_item(&cand[i], PROMPT_DOC)) { fprintf(stderr, "skipped %s\n", cand[i].path); continue; }
			rank[m][0] = eg2e_cosine(q.vec, cand[i].vec);
			rank[m][1] = (float)i;
			m++;
		}
		qsort(rank, (size_t)m, sizeof(*rank), by_score);
		if (top > m) top = m;
		for (int i = 0; i < top; i++) printf("%.4f  %s\n", rank[i][0], cand[(int)rank[i][1]].path);
		free(rank); free(cand);
	} else {
		usage(argv[0]);
	}
	cache_save();
	if (verbose) {
		double tot = 0;
		for (int i = 0; i < EG2E_P_COUNT; i++) tot += eg2e_prof[i];
		for (int i = 0; i < EG2E_P_COUNT; i++) if (eg2e_prof[i] > 0) fprintf(stderr, "  %-18s %7.1f ms  %4.1f%%\n", eg2e_prof_name[i], eg2e_prof[i] * 1e3, 100 * eg2e_prof[i] / tot);
	}
	if (g.model.st.map) eg2e_free(&g.model);
#ifdef CATS_GPU
	cs_gpu_term();
#endif
	return rc;
}
