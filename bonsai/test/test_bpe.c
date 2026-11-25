// test_bpe: GGUF vocabulary -> cats_bpe (catseye.h); encode/decode round trips and the special tokens of the chat template
#include "../bonsai.h"
int main(int argc, char **argv)
{
	cg_file g;
	if (argc < 2 || cg_open(&g, argv[1])) { fprintf(stderr, "usage: test_bpe MODEL.gguf\n"); return 2; }
	char **tok, **mrg;
	int nt = cg_str_array(&g, "tokenizer.ggml.tokens", &tok), nm = cg_str_array(&g, "tokenizer.ggml.merges", &mrg);
	const cg_kv *ty = cg_kvfind(&g, "tokenizer.ggml.token_type");
	int8_t *tt = (int8_t*)malloc(nt);
	{ const uint8_t *p = ty->val; cg_rd32(&p); cg_rd64(&p); for (int i = 0; i < nt; i++) { int32_t v; memcpy(&v, p, 4); p += 4; tt[i] = (int8_t)v; } }
	cats_bpe b; uint8_t *blob;
	if (cats_bpe_from_gguf(&b, &blob, tok, nt, tt, mrg, nm, 1)) return 1;
	const char *tests[] = { "Hello, world!", "The capital of France is Paris.", "日本の首都は東京です。", "  indented\n\n\tcode(1234567) x=3.14159;", "I'll don't can't 'quoted' ÄÖÜ café", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n", "emoji 😀 and mixed 漢字かな交じり文" };
	int bad = 0;
	for (int k = 0; k < (int)(sizeof tests / sizeof *tests); k++) {
		int ids[512];
		int n = cats_bpe_encode(&b, tests[k], 1, ids, 512);
		char out[4096]; size_t o = 0;
		for (int i = 0; i < n; i++) o += cats_bpe_piece(&b, ids[i], 1, out + o, sizeof out - o);
		out[o] = 0;
		int ok = !strcmp(out, tests[k]);
		printf("%s %d tokens:", ok ? "ok  " : "FAIL", n);
		for (int i = 0; i < n && i < 24; i++) printf(" %d", ids[i]);
		printf("\n");
		if (!ok) { printf("   got [%s]\n", out); bad++; }
	}
	return bad;
}
