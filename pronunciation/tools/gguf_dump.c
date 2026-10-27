/* gguf_dump: list metadata keys and tensors (name, type, dims) of a GGUF file */
#include "../gguf.h"
int main(int argc, char **argv)
{
	cg_file g;
	if (argc < 2 || cg_open(&g, argv[1])) { fprintf(stderr, "usage: gguf_dump file.gguf\n"); return 1; }
	for (int i = 0; i < g.nkv; i++) {
		printf("kv %s type=%d", g.kv[i].key, g.kv[i].type);
		if (g.kv[i].type == CG_U32 || g.kv[i].type == CG_I32) { uint32_t v; memcpy(&v, g.kv[i].val, 4); printf(" = %u", v); }
		printf("\n");
	}
	for (int i = 0; i < g.nt; i++) {
		printf("t %s type=%d [", g.t[i].name, g.t[i].type);
		for (int d = 0; d < g.t[i].nd; d++) printf(d ? ",%llu" : "%llu", (unsigned long long)g.t[i].dim[d]);
		printf("]\n");
	}
	return 0;
}
