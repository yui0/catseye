/* Compares the C Kokoro pipeline with PyTorch dumps from tools/kokoro_ref.py.
 *   test_kokoro MODEL.gguf VOICE.bin REFDIR        (skips when no arguments / files are missing) */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "tts_kokoro.h"

static float *rd(const char *dir, const char *name, size_t *n)
{
	char p[512]; snprintf(p, sizeof p, "%s/%s.bin", dir, name);
	FILE *f = fopen(p, "rb"); if (!f) return NULL;
	fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
	float *x = (float *)malloc(sz); if (fread(x, 1, sz, f) != (size_t)sz) { fclose(f); free(x); return NULL; }
	fclose(f); *n = sz / 4; return x;
}
static int cmp(const char *name, const float *a, const char *dir, const char *ref, double tol)
{
	size_t n; float *b = rd(dir, ref, &n);
	if (!b) { printf("%-8s no reference\n", name); return 1; }
	double md = 0, mx = 0, num = 0, den = 0;
	for (size_t i = 0; i < n; i++) { double d = fabs(a[i] - b[i]); if (d > md) md = d; if (fabs(b[i]) > mx) mx = fabs(b[i]); num += d * d; den += (double)b[i] * b[i]; }
	double rel = sqrt(num / (den + 1e-30));
	printf("%-8s n=%zu maxabs=%.4g (ref max %.4g) rel-rmse=%.4g %s\n", name, n, md, mx, rel, rel < tol ? "ok" : "FAIL");
	free(b);
	return rel < tol ? 0 : 1;
}

/* waveforms are phase sensitive, so compare short-time log magnitude spectra instead */
static int cmp_spec(const char *name, const float *a, const char *dir, const char *ref, double tol)
{
	size_t n; float *b = rd(dir, ref, &n);
	if (!b) return 1;
	enum { W = 512, H = 256, K = 129 };
	static float win[W], cs[W][K], sn[W][K];
	for (int i = 0; i < W; i++) { win[i] = 0.5f - 0.5f * cosf(2 * (float)M_PI * i / W); for (int k = 0; k < K; k++) { cs[i][k] = cosf(2 * (float)M_PI * i * k / W); sn[i][k] = sinf(2 * (float)M_PI * i * k / W); } }
	double sum = 0; long cnt = 0;
	for (size_t o = 0; o + W <= n; o += H * 2) {
		for (int k = 0; k < K; k++) {
			double ra = 0, ia = 0, rb = 0, ib = 0;
			for (int i = 0; i < W; i++) { ra += a[o + i] * win[i] * cs[i][k]; ia += a[o + i] * win[i] * sn[i][k]; rb += b[o + i] * win[i] * cs[i][k]; ib += b[o + i] * win[i] * sn[i][k]; }
			sum += fabs(log10(hypot(ra, ia) + 1e-3) - log10(hypot(rb, ib) + 1e-3)); cnt++;
		}
	}
	double m = sum / cnt;
	printf("%-8s log-spectrum mean abs diff = %.3f decades %s\n", name, m, m < tol ? "ok" : "FAIL");
	free(b);
	return m < tol ? 0 : 1;
}

int main(int argc, char **argv)
{
	if (argc < 4) { puts("kokoro skipped (usage: test_kokoro MODEL.gguf VOICE.bin REFDIR [q8])"); return 0; }
	int q8 = argc > 4;
	kk_model m;
	if (kk_open(&m, argv[1], q8)) { puts("kokoro skipped (model not loadable)"); return 0; }
	m.deterministic = 1;
	FILE *vf = fopen(argv[2], "rb"); if (!vf) { puts("kokoro skipped (no voice)"); return 0; }
	float *pack = (float *)malloc(510 * 256 * 4); if (fread(pack, 4, 510 * 256, vf) != 510 * 256) return 1; fclose(vf);
	/* ids from the reference meta.json */
	char p[512]; snprintf(p, sizeof p, "%s/meta.json", argv[3]);
	FILE *mf = fopen(p, "rb"); char js[4096]; size_t jl = fread(js, 1, sizeof js - 1, mf); js[jl] = 0; fclose(mf);
	int ids[600], n = 0; char *q = strstr(js, "\"ids\""); q = strchr(q, '[') + 1;
	while (*q && *q != ']') { ids[n++] = (int)strtol(q, &q, 10); while (*q == ',' || *q == ' ') q++; }
	const float *style = pack + (size_t)(n - 2 - 1) * 256;
	kk_dump d; memset(&d, 0, sizeof d);
	float *pcm; size_t cnt;
	struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
	if (kk_run(&m, ids, n, style, 1.0f, &pcm, &cnt, &d)) { puts("FAIL run"); return 1; }
	clock_gettime(CLOCK_MONOTONIC, &t1);
	printf("T=%d F=%d samples=%zu (%.2fs audio) in %.2fs\n", d.T, d.F, cnt, (double)cnt / KK_RATE, (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
	double tol = 0.025;	/* GGUF weights are Q8_0, the reference is fp32: ~2-5% rms is the quantization itself */
	int bad = 0;
	bad += cmp("d_en", d.d_en, argv[3], "d_en", tol * 2);
	bad += cmp("d", d.d, argv[3], "d", tol * 2);
	bad += cmp("dur", d.dur, argv[3], "duration", tol * 2);
	bad += cmp("en", d.en, argv[3], "en", tol * 2);
	bad += cmp("F0", d.F0, argv[3], "F0", tol * 3);
	bad += cmp("N", d.N, argv[3], "N", tol * 3);
	bad += cmp("t_en", d.t_en, argv[3], "t_en", tol * 2);
	bad += cmp("asr", d.asr, argv[3], "asr", tol * 2);
	bad += cmp_spec("audio", d.audio, argv[3], "audio", q8 ? 0.35 : 0.30);
	printf(bad ? "kokoro FAIL\n" : "kokoro ok\n");
	return bad ? 1 : 0;
}
