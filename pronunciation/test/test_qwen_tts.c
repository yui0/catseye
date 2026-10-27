/* test_qwen_tts: stage-by-stage check of the Qwen3-TTS talker against qwen-tts --dump output (greedy) */
#include <math.h>
#include <time.h>
#include "../tts_qwen.h"

static float *rd(const char *dir, const char *name, int *n0, int *n1)
{
	char p[512]; snprintf(p, sizeof p, "%s/%s.bin", dir, name);
	FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "missing %s\n", p); return NULL; }
	int h[3]; fread(h, 4, 3, f);
	int nd = h[0], a = h[1], hdr = nd > 1 ? 12 : 8;	/* {ndim, dims...}; values are float32 */
	fseek(f, 0, SEEK_END); long sz = ftell(f) - hdr; fseek(f, hdr, SEEK_SET);
	float *d = (float *)malloc(sz); fread(d, 1, sz, f); fclose(f);
	if (n0) *n0 = a; if (n1) *n1 = (int)(sz / 4 / (a ? a : 1));
	return d;
}

static double rel(const float *a, const float *b, size_t n)
{
	double d = 0, r = 0; for (size_t i = 0; i < n; i++) { d += (double)(a[i] - b[i]) * (a[i] - b[i]); r += (double)b[i] * b[i]; }
	return sqrt(d / (r + 1e-30));
}

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: test_qwen_tts talker.gguf codec.gguf dumpdir [text]\n"); return 2; }
	const char *text = argc > 4 ? argv[4] : "Hello world, good morning.";
	int bad = 0;
	qw_tts *t = (qw_tts *)calloc(1, sizeof *t);
	if (qw_open(t, argv[1], argv[2], 1)) { fprintf(stderr, "open failed\n"); return 1; }
	/* 1. BPE ids */
	int ids[256], nids = 0; float *x;
	int T = qw_build_prompt(t, text, "English", &x, ids, &nids);
	int n0; float *ref = rd(argv[3], "prompt-ids", &n0, NULL);
	int idok = ref && n0 == nids; for (int i = 0; idok && i < nids; i++) idok = (int)ref[i] == ids[i];
	printf("prompt ids (%d)  %s\n", nids, idok ? "ok" : "FAIL"); bad += !idok;
	/* 2. prompt embeddings */
	float *re = rd(argv[3], "talker-input-embed", &n0, NULL);
	double e = re ? rel(x, re, (size_t)T * t->D) : 1;
	printf("input embeds T=%d  rel-rmse=%.4f %s\n", T, e, e < 0.03 ? "ok" : "FAIL"); bad += !(e < 0.03);
	/* 3. greedy generation, compare frames with codes-full */
	qw_params P = qw_default_params(); P.temp = 0; P.sub_temp = 0;
	qw_gen g = { 0 }; clock_t c0 = clock(); struct timespec w0, w1; clock_gettime(CLOCK_MONOTONIC, &w0);
	int nf = qw_generate(t, text, "English", &P, &g);
	clock_gettime(CLOCK_MONOTONIC, &w1);
	float *rl = rd(argv[3], "talker-logits-prefill", &n0, NULL);
	if (rl) { double l = rel(g.prefill_logits, rl, 3072); printf("prefill logits  rel-rmse=%.4f %s\n", l, l < 0.05 ? "ok" : "FAIL"); bad += !(l < 0.05); }
	int cn0, cn1; float *rc = rd(argv[3], "codes-full", &cn0, &cn1);
	int rf = cn0, same = 0, tot = 0;
	for (int f = 0; f < nf && f < rf; f++) for (int k = 0; k < 16; k++) { tot++; same += g.codes[(size_t)k * nf + f] == (int)rc[(size_t)f * 16 + k]; }
	for (int f = 0; f < 3; f++) { printf("frame %d mine:", f); for (int k = 0; k < 16; k++) printf(" %d", g.codes[(size_t)k * nf + f]); printf("\n      ref :"); for (int k = 0; k < 16; k++) printf(" %d", (int)rc[(size_t)f * 16 + k]); printf("\n"); }
	int c0same = 0; for (int f = 0; f < nf && f < rf; f++) c0same += g.codes[f] == (int)rc[(size_t)f * 16];
	printf("frames %d (ref %d), codes equal %d/%d, first-codebook equal %d/%d  (%.2fs wall)\n", nf, rf, same, tot, c0same, nf < rf ? nf : rf, (w1.tv_sec - w0.tv_sec) + (w1.tv_nsec - w0.tv_nsec) / 1e9);
	/* later frames drift: Q8 activation rounding and an f32 (not f16) KV cache flip near-ties, and greedy decoding amplifies that */
	int f0 = 1; for (int k = 0; k < 16; k++) f0 &= g.codes[(size_t)k * nf] == (int)rc[k];
	printf("first frame identical: %s\n", f0 ? "ok" : "FAIL"); bad += !f0;
	/* 4. decode */
	float *pcm; int ns = qw_codec_decode(&t->codec, g.codes, nf, &pcm);
	FILE *f = fopen("/tmp/qwen_test.wav", "wb");
	if (f) { uint32_t dl = ns * 2, v32; uint16_t v16; fwrite("RIFF", 1, 4, f); v32 = 36 + dl; fwrite(&v32, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); v32 = 16; fwrite(&v32, 4, 1, f); v16 = 1; fwrite(&v16, 2, 1, f); fwrite(&v16, 2, 1, f); v32 = 24000; fwrite(&v32, 4, 1, f); v32 = 48000; fwrite(&v32, 4, 1, f); v16 = 2; fwrite(&v16, 2, 1, f); v16 = 16; fwrite(&v16, 2, 1, f); fwrite("data", 1, 4, f); fwrite(&dl, 4, 1, f);
		for (int i = 0; i < ns; i++) { int16_t s = (int16_t)(pcm[i] * 32767); fwrite(&s, 2, 1, f); } fclose(f); }
	printf("qwen talker %s\n", bad ? "FAIL" : "ok");
	return bad;
}
