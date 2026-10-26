#include <stdio.h>
#include <math.h>
#include "latphon.h"
#include "klatt.h"

/* energy (via direct DFT) between lo..hi Hz of the middle half of the signal */
static double band(const float *x, size_t n, double lo, double hi)
{
	size_t a = n / 4, b = n * 3 / 4, m = b - a; double e = 0;
	for (double f = lo; f < hi; f += 25) {
		double re = 0, im = 0;
		for (size_t i = 0; i < m; i += 2) { double w = 0.5 - 0.5 * cos(2 * M_PI * i / m); double ph = 2 * M_PI * f * (a + i) / CK_RATE; re += w * x[a + i] * cos(ph); im += w * x[a + i] * sin(ph); }
		e += re * re + im * im;
	}
	return e;
}

int main(void)
{
	int fails = 0;
	float *a, *i, *u; size_t na, ni, nu;
	ck_opt o = ck_default();
	ck_synth("ˈa", o, &a, &na); ck_synth("ˈi", o, &i, &ni); ck_synth("ˈu", o, &u, &nu);
	if (!a || !i || !u) { puts("FAIL: no audio"); return 1; }
	for (size_t k = 0; k < na; k++) if (!isfinite(a[k]) || fabsf(a[k]) > 1.0f) { puts("FAIL: bad sample"); fails++; break; }
	/* /i/ has F2 near 2.2 kHz, /u/ near 0.9 kHz, /a/ F1 near 0.75 kHz */
	double iH = band(i, ni, 1800, 2700), iL = band(i, ni, 700, 1300);
	double uH = band(u, nu, 1800, 2700), uL = band(u, nu, 700, 1300);
	double aM = band(a, na, 600, 1000), aH = band(a, na, 3500, 4500);
	if (!(iH / iL > 3 * uH / uL)) { printf("FAIL: /i/ vs /u/ F2 (%g vs %g)\n", iH / iL, uH / uL); fails++; }
	if (!(aM > aH * 5)) { puts("FAIL: /a/ spectral tilt"); fails++; }
	/* duration scales with speed */
	float *s1, *s2; size_t n1, n2;
	ck_synth("hɛˈloʊ", o, &s1, &n1); o.speed = 2.0f; ck_synth("hɛˈloʊ", o, &s2, &n2);
	if (!(n2 < n1 * 0.65)) { printf("FAIL: speed (%zu vs %zu)\n", n1, n2); fails++; }
	/* fricative /s/ is high frequency noise */
	float *s; size_t ns; ck_synth("s", ck_default(), &s, &ns);
	if (!(band(s, ns, 5000, 9000) > 5 * band(s, ns, 200, 1500))) { puts("FAIL: /s/ spectrum"); fails++; }
	if (fails) return 1;
	puts("klatt ok");
	return 0;
}
