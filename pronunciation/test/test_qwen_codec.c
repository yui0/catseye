/* test_qwen_codec: decode a reference .rvq (qwen-codec) and compare with the reference wav (log-spectrum distance) */
#include <math.h>
#include <time.h>
#include "../tts_qwen.h"

static void *slurp(const char *p, size_t *n) { FILE *f = fopen(p, "rb"); if (!f) return NULL; fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET); void *b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b; }

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: test_qwen_codec codec.gguf ref.rvq ref.wav\n"); return 2; }
	size_t n; uint8_t *r = slurp(argv[2], &n);
	if (!r) return 2;
	int total = (int)(n * 8 / 11), T = total / QW_NCB;
	int32_t *codes = (int32_t *)malloc(total * 4);
	uint64_t acc = 0; int bits = 0; size_t pos = 0;
	for (int i = 0; i < total; i++) { while (bits < 11) { acc |= (uint64_t)r[pos++] << bits; bits += 8; } codes[i] = acc & 2047; acc >>= 11; bits -= 11; }
	qw_codec c;
	if (qw_codec_open(&c, argv[1], 1)) { fprintf(stderr, "open failed\n"); return 1; }
	float *pcm; clock_t t0 = clock();
	int ns = qw_codec_decode(&c, codes, T, &pcm); if (getenv("QW_TWICE")) { free(pcm); fprintf(stderr, "second run:\n"); ns = qw_codec_decode(&c, codes, T, &pcm); }
	fprintf(stderr, "T=%d samples=%d (%.2fs audio) in %.2fs cpu\n", T, ns, ns / 24000.0, (double)(clock() - t0) / CLOCKS_PER_SEC);
	size_t wn; uint8_t *w = slurp(argv[3], &wn);
	int16_t *ref = (int16_t *)(w + 44); size_t rn = (wn - 44) / 2;
	enum { W = 512, K = 129 };
	double sum = 0, rms = 0, rr = 0; long cnt = 0;
	for (size_t i = 0; i < rn && i < (size_t)ns; i++) { double d = pcm[i] - ref[i] / 32768.0; rms += d * d; rr += (ref[i] / 32768.0) * (ref[i] / 32768.0); }
	for (size_t o = 0; o + W <= rn && o + W <= (size_t)ns; o += 512) for (int k = 0; k < K; k++) {
		double ra = 0, ia = 0, rb = 0, ib = 0;
		for (int i = 0; i < W; i++) { double win = 0.5 - 0.5 * cos(2 * M_PI * i / W), ph = 2 * M_PI * i * k / W; ra += pcm[o + i] * win * cos(ph); ia += pcm[o + i] * win * sin(ph); rb += ref[o + i] / 32768.0 * win * cos(ph); ib += ref[o + i] / 32768.0 * win * sin(ph); }
		sum += fabs(log10(hypot(ra, ia) + 1e-3) - log10(hypot(rb, ib) + 1e-3)); cnt++;
	}
	printf("samples %d vs %zu, waveform rel-rmse %.4f, log-spectrum diff %.3f decades %s\n", ns, rn, sqrt(rms / rr), sum / cnt, sum / cnt < 0.05 ? "ok" : "FAIL");
	return sum / cnt < 0.05 ? 0 : 1;
}
