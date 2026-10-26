#include <stdio.h>
#include <math.h>
#include "latphon.h"
#include "tts_piper.h"

int main(void)
{
	cpp_voice v;
	if (cpp_open(&v, "models/en.onnx", "models/en.onnx.json")) { puts("piper skipped (models/en.onnx missing)"); return 0; }
	int ids[64], n = cpp_ids(&v, "hɛˈloʊ", ids, 64);
	/* ^ (phoneme _)* $ : 5 phonemes + stress mark = 6 symbols */
	if (n != 2 + 2 * 6 || ids[0] != v.bos || ids[n - 1] != v.eos || ids[1] == v.pad || ids[2] != v.pad) { printf("FAIL ids (n=%d)\n", n); return 1; }
	float *pcm; size_t cnt;
	if (cpp_synth(&v, "hɛˈloʊ", 1.0f, &pcm, &cnt)) { puts("FAIL synth"); return 1; }
	double e = 0; for (size_t i = 0; i < cnt; i++) { if (!isfinite(pcm[i])) { puts("FAIL nan"); return 1; } e += pcm[i] * pcm[i]; }
	double sec = (double)cnt / v.rate;
	if (sec < 0.3 || sec > 2.0 || e / cnt < 1e-5) { printf("FAIL audio (%.2fs, power %g)\n", sec, e / cnt); return 1; }
	puts("piper ok");
	return 0;
}
