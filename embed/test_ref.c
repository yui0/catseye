// Layer-by-layer check of the C port against the Python reference dumps (eg2_embed_reference.py).
//   test_ref MODEL_DIR REF_DIR [name ...]     e.g. test_ref eg2_hf ref text_0 image_1
#include "eg2_embed.h"

typedef struct { float *f; int *i; int nd; long shape[4]; size_t n; } npy;
static int npy_load(const char *path, npy *a)
{
	memset(a, 0, sizeof(*a));
	FILE *fp = fopen(path, "rb");
	if (!fp) return -1;
	unsigned char hd[10];
	if (fread(hd, 1, 10, fp) != 10 || memcmp(hd, "\x93NUMPY", 6)) { fclose(fp); return -1; }
	size_t hl = hd[8] | (hd[9] << 8);
	char *hdr = (char*)malloc(hl + 1);
	if (fread(hdr, 1, hl, fp) != hl) { fclose(fp); free(hdr); return -1; }
	hdr[hl] = 0;
	int is_int = strstr(hdr, "<i4") != NULL;
	char *sp = strstr(hdr, "shape");
	sp = strchr(sp, '(') + 1;
	a->n = 1;
	while (*sp && *sp != ')') {
		while (*sp == ' ' || *sp == ',') sp++;
		if (*sp == ')') break;
		a->shape[a->nd] = strtol(sp, &sp, 10);
		a->n *= (size_t)a->shape[a->nd++];
	}
	free(hdr);
	void *buf = malloc(a->n * 4);
	if (fread(buf, 4, a->n, fp) != a->n) { fclose(fp); free(buf); return -1; }
	fclose(fp);
	if (is_int) a->i = (int*)buf; else a->f = (float*)buf;
	return 0;
}
static void npy_free(npy *a) { free(a->f); free(a->i); memset(a, 0, sizeof(*a)); }

static int load_in(const char *dir, const char *name, const char *file, npy *a)
{
	char p[1024];
	snprintf(p, sizeof p, "%s/%s/%s.npy", dir, name, file);
	return npy_load(p, a);
}
// relative L2 error and max abs difference between two equally sized arrays
static double compare(const char *label, const float *a, const float *b, size_t n)
{
	double num = 0, den = 0, mx = 0;
	for (size_t i = 0; i < n; i++) {
		double d = (double)a[i] - b[i];
		num += d * d; den += (double)b[i] * b[i];
		if (fabs(d) > mx) mx = fabs(d);
	}
	double rel = sqrt(num / (den > 0 ? den : 1));
	printf("  %-18s n=%-9zu rel_L2=%.3e max_abs=%.3e\n", label, n, rel, mx);
	return rel;
}

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: %s MODEL_DIR REF_DIR name...\n", argv[0]); return 2; }
	cats_set_threads(0);
	eg2e_model m;
	if (eg2e_load(&m, argv[1], 1)) return 1;
	int bad = 0;
	for (int a = 3; a < argc; a++) {
		const char *name = argv[a];
		printf("%s\n", name);
		npy ids, ref;
		if (load_in(argv[2], name, "input_ids", &ids)) { printf("  missing input_ids\n"); bad++; continue; }
		size_t t = ids.n;
		float *emb = (float*)malloc(t * 512 * sizeof(float)), *vec = (float*)malloc(768 * sizeof(float));
		const int is_image = !strncmp(name, "image", 5);
		float *soft = NULL;
		if (is_image) {
			npy pix, pos, vp, vs;
			if (load_in(argv[2], name, "pixel_values", &pix) || load_in(argv[2], name, "patch_positions", &pos)) { bad++; continue; }
			int gw = 0, gh = 0;
			for (long i = 0; i < pos.shape[0]; i++) { if (pos.i[2*i] + 1 > gw) gw = pos.i[2*i] + 1; if (pos.i[2*i+1] + 1 > gh) gh = pos.i[2*i+1] + 1; }
			size_t P = (size_t)gw * gh;
			size_t n = P / 9;
			soft = (float*)malloc(n * 512 * sizeof(float));
			float *dump = (float*)malloc((size_t)EG2E_VIS_LAYERS * P * 768 * sizeof(float));
			struct timespec t0, t1;
			clock_gettime(CLOCK_MONOTONIC, &t0);
			if (eg2e_vision(&m, pix.f, gw, gh, soft, dump)) { printf("  vision failed\n"); bad++; continue; }
			clock_gettime(CLOCK_MONOTONIC, &t1);
			printf("  vision %dx%d patches -> %zu soft tokens in %.1f s\n", gw, gh, n, time_diff(&t0, &t1));
			for (int li = 0; li < EG2E_VIS_LAYERS; li += (li == 0 ? 1 : li == 1 ? 6 : 5)) {
				char nm[32]; snprintf(nm, sizeof nm, "vis_layer_%02d", li);
				if (!load_in(argv[2], name, nm, &ref)) { compare(nm, dump + (size_t)li * P * 768, ref.f, ref.n); npy_free(&ref); }
			}
			if (!load_in(argv[2], name, "vis_layer_15", &ref)) { compare("vis_layer_15", dump + 15 * P * 768, ref.f, ref.n); npy_free(&ref); }
			if (!load_in(argv[2], name, "vis_soft_tokens", &vs)) { compare("vis_soft_tokens", soft, vs.f, vs.n); npy_free(&vs); }
			free(dump); npy_free(&pix); npy_free(&pos);
		}
		if (eg2e_embed_ids(&m, ids.i, t, soft, emb)) { printf("  embed failed\n"); bad++; continue; }
		if (!load_in(argv[2], name, "text_inputs_embeds", &ref)) { compare("inputs_embeds", emb, ref.f, ref.n); npy_free(&ref); }
		float *hid = (float*)malloc((size_t)EG2E_TEXT_LAYERS * t * 512 * sizeof(float));
		struct timespec t0, t1;
		clock_gettime(CLOCK_MONOTONIC, &t0);
		eg2e_backbone(&m, emb, t, vec, hid);
		clock_gettime(CLOCK_MONOTONIC, &t1);
		printf("  backbone %zu tokens in %.2f s\n", t, time_diff(&t0, &t1));
		for (int li = 0; li < EG2E_TEXT_LAYERS; li += (li == 0 ? 1 : 5)) {
			char nm[32]; snprintf(nm, sizeof nm, "text_layer_%02d", li);
			if (!load_in(argv[2], name, nm, &ref)) { compare(nm, hid + (size_t)li * t * 512, ref.f, ref.n); npy_free(&ref); }
		}
		if (!load_in(argv[2], name, "text_layer_23", &ref)) { compare("text_layer_23", hid + 23 * t * 512, ref.f, ref.n); npy_free(&ref); }
		if (!load_in(argv[2], name, "embedding", &ref)) {
			double r = compare("embedding", vec, ref.f, 768);
			printf("  cosine(C, reference) = %.7f\n", eg2e_cosine(vec, ref.f));
			if (r > 1e-3) bad++;
			npy_free(&ref);
		}
		free(hid); free(emb); free(vec); free(soft); npy_free(&ids);
	}
	eg2e_free(&m);
	printf(bad ? "FAILED\n" : "OK\n");
	return bad != 0;
}
