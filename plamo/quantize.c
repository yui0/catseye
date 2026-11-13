//---------------------------------------------------------
//	Cat's eye
//
//		©2026 Yuichiro Nakada
//---------------------------------------------------------

// PLaMo 3 Translate: shrink a Hugging Face checkpoint (BF16 safetensors) into
// Q8_0 / Q4_0 / F16 safetensors that plamo reads directly.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <sys/stat.h>
#define CATS_INFER_ONLY
#include "../catseye.h"

static int parse_type(const char *s)
{
	if (!strcmp(s, "f16")) return CATS_ST_F16;
	if (!strcmp(s, "q8")) return CATS_ST_Q8_0;
	if (!strcmp(s, "q4")) return CATS_ST_Q4_0;
	return -1;
}
static const char *type_name(int t) { return t == CATS_ST_F16 ? "F16" : t == CATS_ST_Q8_0 ? "Q8_0" : "Q4_0"; }

static int copy_file(const char *from, const char *to)
{
	FILE *a = fopen(from, "rb");
	if (!a) return -1;
	FILE *b = fopen(to, "wb");
	if (!b) { fclose(a); return -1; }
	char buf[1 << 16];
	size_t n;
	while ((n = fread(buf, 1, sizeof(buf), a)) > 0) fwrite(buf, 1, n, b);
	fclose(a);
	return fclose(b);
}

int main(int argc, char **argv)
{
	const char *in = NULL, *out = NULL;
	int body = CATS_ST_Q4_0, emb = -1, head = -1, refine = 1;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-i") && i + 1 < argc) in = argv[++i];
		else if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
		else if (!strcmp(argv[i], "-b") && i + 1 < argc) body = parse_type(argv[++i]);
		else if (!strcmp(argv[i], "-e") && i + 1 < argc) emb = parse_type(argv[++i]);
		else if (!strcmp(argv[i], "-h") && i + 1 < argc) head = parse_type(argv[++i]);
		else if (!strcmp(argv[i], "--fast")) refine = 0;
		else in = NULL;
	}
	if (emb < 0) emb = body;
	if (head < 0) head = body;
	if (!in || !out || body < 0 || emb < 0 || head < 0) {
		fprintf(stderr,
			"Usage: %s -i HF_DIR -o OUT_DIR [-b TYPE] [-e TYPE] [-h TYPE] [--fast]\n"
			"  TYPE: f16 | q8 | q4.  -b layer matrices (default q4), -e embedding table, -h lm_head (both default to -b)\n"
			"  Q4_0 uses a per-block scale search; --fast is plain GGML Q4_0 (quicker, a little worse).\n"
			"  Typical: q8 = ~0.95 GB, q4 = ~0.5 GB, q4 body with q8 embedding / head = ~0.6 GB\n", argv[0]);
		return 1;
	}
	char path[1024];
	snprintf(path, sizeof(path), "%s/model.safetensors", in);
	cats_st st;
	if (cats_st_open(&st, path)) { fprintf(stderr, "cannot open %s\n", path); return 1; }
	mkdir(out, 0755);

	cats_st_entry *e = (cats_st_entry*)calloc((size_t)st.count, sizeof(*e));
	char **names = (char**)calloc((size_t)st.count, sizeof(char*));
	size_t total = 0;
	for (int i = 0; i < st.count; i++) {
		const cats_st_tensor *t = &st.tensor[i];
		names[i] = strdup(t->name);
		e[i].name = names[i];
		e[i].ndim = t->ndim;
		for (int d = 0; d < t->ndim; d++) e[i].shape[d] = t->shape[d];
		const int matrix = t->ndim == 2 && t->shape[1] % CATS_Q4_0_BLOCK == 0 && t->shape[1] >= 256;
		const int is_emb = !strcmp(t->name, "model.embed_tokens.weight"), is_head = !strcmp(t->name, "lm_head.weight");
		int type = !matrix ? CATS_ST_F32 : is_emb ? emb : is_head ? head : body;
		const int64_t rows = matrix ? t->shape[0] : 1, cols = matrix ? t->shape[1] : cats_st_numel(t);
		const size_t rb = (size_t)cats_st_bytes(type, cols);
		uint8_t *data = (uint8_t*)malloc(rb * (size_t)rows);
		if (!data) { fprintf(stderr, "out of memory\n"); return 1; }
		if (!matrix) {
			if (cats_st_read_f32(&st, t, (float*)data)) { fprintf(stderr, "cannot read %s\n", t->name); return 1; }
		} else {
			#pragma omp parallel
			{
				float *row = (float*)malloc((size_t)cols * sizeof(float));
				#pragma omp for schedule(static)
				for (long r = 0; r < rows; r++) {
					cats_st_read_row_f32(&st, t, r, row);
					uint8_t *o = data + (size_t)r * rb;
					if (type == CATS_ST_Q4_0) cats_q4_0_quantize_row(row, (size_t)cols, o, refine);
					else if (type == CATS_ST_Q8_0) cats_q8_0w_quantize_row(row, (size_t)cols, o);
					else for (int64_t k = 0; k < cols; k++) { uint16_t h = cats_float_to_half(row[k]); o[2*k] = (uint8_t)h; o[2*k+1] = (uint8_t)(h >> 8); }
				}
				free(row);
			}
		}
		e[i].dtype = type == CATS_ST_F32 ? "F32" : type_name(type);
		e[i].data = data;
		e[i].bytes = (uint64_t)rb * (uint64_t)rows;
		total += (size_t)e[i].bytes;
		if (matrix) fprintf(stderr, "%-58s %-4s %10.1f MB\n", t->name, e[i].dtype, e[i].bytes / 1048576.0);
	}
	snprintf(path, sizeof(path), "%s/model.safetensors", out);
	if (cats_st_write(path, e, st.count, "pt")) { fprintf(stderr, "write failed: %s\n", path); return 1; }
	const char *side[] = { "config.json", "tokenizer.jsonl", "LICENSE", "generation_config.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json" };
	for (size_t i = 0; i < sizeof(side) / sizeof(*side); i++) {
		char a[1024], b[1024];
		snprintf(a, sizeof(a), "%s/%s", in, side[i]);
		snprintf(b, sizeof(b), "%s/%s", out, side[i]);
		if (copy_file(a, b) && i < 2) { fprintf(stderr, "cannot copy %s\n", a); return 1; }
	}
	fprintf(stderr, "wrote %s (%.1f MB)\n", path, total / 1048576.0);
	return 0;
}
