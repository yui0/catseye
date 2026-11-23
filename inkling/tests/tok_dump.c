// Encodes every line of a file (\n inside a line is written as the two characters "\n") with cats_bpe and prints
// the ids; check_tokenizer.py compares them with Hugging Face tokenizers.
#define CATS_INFER_ONLY
#include "../../catseye.h"

static char *read_all(const char *p, size_t *n)
{
	FILE *f = fopen(p, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END); long l = ftell(f); fseek(f, 0, SEEK_SET);
	char *b = (char*)malloc((size_t)l + 1);
	if (fread(b, 1, (size_t)l, f) != (size_t)l) return NULL;
	b[l] = 0; fclose(f);
	if (n) *n = (size_t)l;
	return b;
}
int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: %s tokenizer.json lines.txt\n", argv[0]); return 1; }
	size_t n; char *j = read_all(argv[1], &n);
	uint8_t *blob; size_t bs;
	if (!j || cats_bpe_from_json(j, n, &blob, &bs)) { fprintf(stderr, "bad tokenizer.json\n"); return 1; }
	cats_bpe t;
	if (cats_bpe_load(&t, blob, bs)) { fprintf(stderr, "bad blob\n"); return 1; }
	fprintf(stderr, "vocab %d merges %d blob %.1f MB\n", t.n, t.nmerge, bs / 1048576.0);
	char *txt = read_all(argv[2], NULL);
	for (char *l = strtok(txt, "\n"); l; l = strtok(NULL, "\n")) {
		char *u = (char*)malloc(strlen(l) + 1); size_t k = 0;	// unescape \n and \t
		for (size_t i = 0; l[i]; i++) {
			if (l[i] == '\\' && l[i + 1] == 'n') { u[k++] = '\n'; i++; }
			else if (l[i] == '\\' && l[i + 1] == 't') { u[k++] = '\t'; i++; }
			else u[k++] = l[i];
		}
		u[k] = 0;
		int ids[8192];
		int c = cats_bpe_encode(&t, u, 1, ids, 8192);
		for (int i = 0; i < c; i++) printf("%d%c", ids[i], i + 1 < c ? ' ' : '\n');
		if (!c) printf("\n");
		free(u);
	}
	return 0;
}
