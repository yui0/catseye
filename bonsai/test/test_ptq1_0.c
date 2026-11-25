// test_ptq1_0: PTQ1_0 packing/decoding/dot kernels in catseye.h against a reference quantizer written from the
// llama.cpp definition (quantize_row_ptq1_0_ref / dequantize_row_ptq1_0), and the Hadamard transform properties.
#define CATS_INFER_ONLY
#include "../../catseye.h"
#include <assert.h>

static const size_t stages[3] = { 32, 16, 8 };
static void ref_quantize(const float *x, uint8_t *y /* 28 bytes per 128 */, int k)
{
	for (int i = 0; i < k / 128; i++, y += 28) {
		float amax = 0; for (int j = 0; j < 128; j++) amax = fmaxf(amax, fabsf(x[j]));
		const float id = amax ? 1.0f / amax : 0.0f;
		uint16_t h = 0; /* d as fp16: caller stores via conversion */ (void)h;
		uint8_t *qs = y, *qh = y + 24;
		size_t j = 0;
		for (int s = 0; s < 3; s++) {
			const size_t c = stages[s];
			for (; j + c <= 24; j += c) {
				for (size_t m = 0; m < c; m++) {
					uint8_t q = 0;
					for (size_t n = 0; n < 5; n++) { int xi = lroundf(x[m + n * c] * id) + 1; q *= 3; q += xi; }
					qs[j + m] = ((uint16_t)q * 256 + 242) / 243;
				}
				x += 5 * c;
			}
		}
		for (size_t hh = 0; hh < 2; hh++) {
			uint8_t q = 0;
			for (size_t m = 0; m < 4; m++) { int xi = lroundf(x[hh + m * 2] * id) + 1; q *= 3; q += xi; }
			q *= 3;
			qh[hh] = ((uint16_t)q * 256 + 242) / 243;
		}
		x += 8;
		/* fp16 scale: amax as half (round via float->half emulation using _Float16 when available) */
		_Float16 dh = (_Float16)amax; memcpy(y + 26, &dh, 2);
	}
}

int main(void)
{
	srand(1);
	enum { N = 1024, T = 7, ROWS = 7 };
	static float w[ROWS][N], tr[ROWS][N];
	static uint8_t pk[ROWS][N / 128 * 28];
	/* ternary-valued rows with random scales */
	for (int r = 0; r < ROWS; r++) for (int b = 0; b < N / 128; b++) {
		float sc = 0.05f + (rand() % 100) / 500.0f;
		for (int i = 0; i < 128; i++) { int t = rand() % 3 - 1; w[r][b * 128 + i] = t * sc; }
	}
	for (int r = 0; r < ROWS; r++) ref_quantize(w[r], pk[r], N);
	int bad = 0;
	for (int r = 0; r < ROWS; r++) {
		cats_ptq1_0_dequantize_row(pk[r], N, tr[r]);
		for (int i = 0; i < N; i++) {
			/* scale roundtrips through fp16 */
			if (fabsf(tr[r][i] - w[r][i]) > 1e-3f * fabsf(w[r][i]) + 1e-6f) { if (bad++ < 5) printf("mismatch r=%d i=%d %f vs %f\n", r, i, tr[r][i], w[r][i]); }
		}
	}
	printf("dequantize: %s\n", bad ? "FAIL" : "ok");
	/* decode: AVX2 path equals scalar path */
#if CATS_SIMD_X86
	if (cats_cpu_simd_level() >= 1) {
		int8_t a[128], b[128]; int dbad = 0;
		for (int r = 0; r < ROWS; r++) for (int blk = 0; blk < N / 128; blk++) {
			cats_ptq1_0_decode_scalar(pk[r] + blk * 28, a); cats_ptq1_0_decode_avx(pk[r] + blk * 28, b);
			if (memcmp(a, b, 128)) dbad++;
		}
		printf("decode avx == scalar: %s\n", dbad ? "FAIL" : "ok"); bad += dbad;
	}
#endif
#if CATS_SIMD_X86
	if (cats_cpu_simd_level() >= 2) {	/* AVX2 decode == scalar decode */
		int8_t a[128], b[128]; int dbad = 0;
		for (int r = 0; r < ROWS; r++) for (int blk = 0; blk < N / 128; blk++) {
			cats_ptq1_0_decode_scalar(pk[r] + blk * 28, a); cats_ptq1_0_decode_avx2(pk[r] + blk * 28, b);
			if (memcmp(a, b, 128)) dbad++;
		}
		printf("decode avx2 == scalar: %s\n", dbad ? "FAIL" : "ok"); bad += dbad;
	} else printf("decode avx2: skipped (CPU has no AVX2)\n");
#endif
	/* matmul against float reference */
	static float x[T][N], out[T * ROWS], ref[T * ROWS]; (void)ref;
	for (int t = 0; t < T; t++) for (int i = 0; i < N; i++) x[t][i] = (rand() % 2001 - 1000) / 500.0f;
	cats_gemm_nt_ptq1_0(out, &x[0][0], &pk[0][0], T, N, ROWS);
	double maxerr = 0;
	for (int t = 0; t < T; t++) for (int r = 0; r < ROWS; r++) {
		double s = 0; for (int i = 0; i < N; i++) s += (double)tr[r][i] * x[t][i];
		ref[t * ROWS + r] = (float)s; maxerr = fmax(maxerr, fabs(s - out[t * ROWS + r]) / (1 + fabs(s)));
	}
	/* the difference to the float reference is the Q8_0 rounding of the activations */
	printf("gemm: max rel err %.2e (%s)\n", maxerr, maxerr < 8e-2 ? "ok" : "FAIL"); if (maxerr >= 8e-2) bad++;
	/* repacked 2-bit planes: same result as the original-format kernel, all kernels (scalar / avx / avx2) */
	{
		cats_ptq1_0_r2 r2;
		cats_ptq1_0_r2_pack(&r2, &pk[0][0], ROWS, N);
		const size_t arow = cats_q8_0_row_bytes(N);
		int8_t *xq = malloc(T * arow);
		for (int t = 0; t < T; t++) cats_q8_0_quantize(xq + t * arow, x[t], N);
		static float o2[T * ROWS];
		const char *lv[] = { "scalar", "avx", "avx2" };
		for (int l = 0; l < 3; l++) {
			if (l > cats_cpu_simd_level()) { printf("r2 %-6s: skipped (CPU)\n", lv[l]); continue; }
			cats_r2_ctx c; float *corr = malloc(T * (N / 128) * 4);
			for (int t = 0; t < T; t++) for (int b = 0; b < N / 128; b++) { const int8_t *y = xq + t * arow + b * 4 * 36; float a = 0; for (int k = 0; k < 4; k++, y += 36) { int s = 0; float ds; for (int i = 0; i < 32; i++) s += y[i]; memcpy(&ds, y + 32, 4); a += ds * s; } corr[t * (N / 128) + b] = a; }
			c = (cats_r2_ctx){ o2, &r2, xq, corr, T };
			if (l == 0) cats_r2_body_scalar(&c, 0, ROWS);
#if CATS_SIMD_X86
			if (l == 1) cats_r2_body_avx(&c, 0, ROWS);
			if (l == 2) cats_r2_body_avx2(&c, 0, ROWS);
#endif
			double me = 0; for (int i = 0; i < T * ROWS; i++) me = fmax(me, fabs(o2[i] - out[i]) / (1 + fabs(out[i])));
			printf("r2 %-6s: max diff vs original kernel %.2e (%s)\n", lv[l], me, me < 1e-4 ? "ok" : "FAIL"); if (me >= 1e-4) bad++;
			free(corr);
		}
		free(xq); cats_ptq1_0_r2_free(&r2);
	}
	/* Hadamard: orthonormal, involutive, inverse() undoes forward */
	float v[2048], u[2048]; int8_t sg[2048];
	for (int i = 0; i < 2048; i++) { v[i] = u[i] = (rand() % 2001 - 1000) / 500.0f; sg[i] = (rand() & 1) ? 1 : -1; }
	double n0 = 0; for (int i = 0; i < 2048; i++) n0 += v[i] * v[i];
	cats_hadamard_rows(v, 1, 2048, 1024, sg, 0);
	double n1 = 0; for (int i = 0; i < 2048; i++) n1 += v[i] * v[i];
	cats_hadamard_rows(v, 1, 2048, 1024, sg, 0); /* not an inverse (signs first) – just checking norm */
	float e[2048]; memcpy(e, u, sizeof e);
	cats_hadamard_rows(e, 1, 2048, 1024, sg, 0);
	/* row-vector identity: (w D H) . (H D x) == w . x */
	float wv[1024], xv[1024]; double dot0 = 0, dot1 = 0; int8_t s1[1024];
	for (int i = 0; i < 1024; i++) { wv[i] = (rand() % 2001 - 1000) / 500.0f; xv[i] = (rand() % 2001 - 1000) / 500.0f; s1[i] = (rand() & 1) ? 1 : -1; dot0 += wv[i] * xv[i]; }
	float wf[1024]; memcpy(wf, wv, sizeof wf);
	for (int i = 0; i < 1024; i++) wf[i] *= s1[i];	/* w D */
	cats_fwht(wf, 1024);				/* (w D) H */
	cats_hadamard_rows(xv, 1, 1024, 1024, s1, 0);	/* H D x */
	for (int i = 0; i < 1024; i++) dot1 += wf[i] * xv[i];
	printf("norm %.3f -> %.3f; folded dot %.4f vs %.4f (%s)\n", n0, n1, dot0, dot1, fabs(dot0 - dot1) < 1e-2 * (1 + fabs(dot0)) ? "ok" : "FAIL");
	if (fabs(dot0 - dot1) >= 1e-2 * (1 + fabs(dot0)) || fabs(n0 - n1) > 1e-2 * n0) bad++;
	/* inverse of a stored (folded) row recovers the original: e' = e D H -> e = H(e') D */
	float ro[1024], rf[1024]; memcpy(ro, wv, sizeof ro); memcpy(rf, wv, sizeof rf);
	for (int i = 0; i < 1024; i++) rf[i] *= s1[i];
	cats_fwht(rf, 1024);
	cats_hadamard_rows(rf, 1, 1024, 1024, s1, 1);
	double ie = 0; for (int i = 0; i < 1024; i++) ie = fmax(ie, fabs(rf[i] - ro[i]));
	printf("inverse roundtrip max err %.2e (%s)\n", ie, ie < 1e-4 ? "ok" : "FAIL"); if (ie >= 1e-4) bad++;
	return bad ? 1 : 0;
}
