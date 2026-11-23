// Compares every SIMD dot kernel with its scalar reference on random data (run on each CPU family: x86 AVX, ARM NEON).
#define CATS_INFER_ONLY
#include "../../catseye.h"

static float rnd(void) { return (float)rand() / RAND_MAX - 0.5f; }
int main(void)
{
	srand(3);
	const int n = 2304;
	int bad = 0;
	float *x = malloc(n * 4), *a = malloc(n * 4);
	uint8_t *w4 = malloc(cats_q4_0_row_bytes(n)), *w3 = malloc(cats_q3_0_row_bytes(n)), *w8 = malloc(cats_q8_0w_row_bytes(n));
	int8_t *aq = malloc(cats_q8_0_row_bytes(n));
	for (int it = 0; it < 50; it++) {
		for (int i = 0; i < n; i++) { x[i] = rnd() * (1 + 3 * (i % 101 == 0)); a[i] = rnd() * 4; }
		cats_q4_0_quantize_row(x, n, w4, 1); cats_q3_0_quantize_row(x, n, w3, 1); cats_q8_0w_quantize_row(x, n, w8);
		cats_q8_0_quantize(aq, a, n);
		const float r4 = cats_dot_q4_0_q8_0_scalar(w4, aq, n), r3 = cats_dot_q3_0_q8_0_scalar(w3, aq, n), r8 = cats_dot_q8_0w_q8_0_scalar(w8, aq, n);
		const float f4 = cats_dot_q4_0_q8_0(w4, aq, n), f3 = cats_dot_q3_0_q8_0(w3, aq, n), f8 = cats_dot_q8_0w_q8_0(w8, aq, n);
		double ref = 0; for (int i = 0; i < n; i++) ref += (double)x[i] * a[i];
		const float ff = cats_dot_f32(x, a, n);
#define CHK(name, got, want, tol) if (fabsf((got) - (want)) > (tol) * (1 + fabsf(want))) { printf("%s mismatch: %g vs %g\n", name, (double)(got), (double)(want)); bad++; }
		CHK("q4_0", f4, r4, 1e-4f) CHK("q3_0", f3, r3, 1e-4f) CHK("q8_0", f8, r8, 1e-4f) CHK("f32", ff, (float)ref, 1e-4f)
		// the quantized dots must approximate the dot of the dequantized weights with the float activations
		float *d = malloc(n * 4);
		double na = 0; for (int i = 0; i < n; i++) na += (double)a[i] * a[i];
		for (int k = 0; k < 3; k++) {
			if (k == 0) cats_q4_0_dequantize_row(w4, n, d); else if (k == 1) cats_q3_0_dequantize_row(w3, n, d); else cats_q8_0w_dequantize_row(w8, n, d);
			double dr = 0, nd = 0; for (int i = 0; i < n; i++) { dr += (double)d[i] * a[i]; nd += (double)d[i] * d[i]; }
			const float got = k == 0 ? f4 : k == 1 ? f3 : f8, tol = (float)(0.012 * sqrt(nd * na));
			if (fabsf(got - (float)dr) > tol) { printf("%s vs dequantized float: %g vs %g (tol %g)\n", k == 0 ? "q4_0" : k == 1 ? "q3_0" : "q8_0", got, dr, tol); bad++; }
		}
		free(d);
	}
	printf("simd level %d: %s\n", cats_cpu_simd_level(), bad ? "FAILED" : "all dot kernels agree with the scalar references");
	return bad != 0;
}
