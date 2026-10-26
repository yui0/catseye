/*
 * klatt.h - rule-based formant synthesizer (Klatt-style cascade/parallel).
 *
 * ck_synth(ipa, opt, &pcm, &n) turns the IPA string produced by cp_g2p() into
 * mono float PCM. No model files. Voice: glottal pulse source -> cascade of
 * five formant resonators (+ nasal resonator) for voiced sounds, noise through
 * a parallel frication resonator for fricatives and stop bursts.
 */
#ifndef CATS_KLATT_H
#define CATS_KLATT_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "phon.h"

#define CK_RATE 22050
#define CK_FRAME (CK_RATE / 200)	/* 5 ms parameter frames */

enum { CK_V, CK_VN, CK_NAS, CK_STOP, CK_VSTOP, CK_FRIC, CK_VFRIC, CK_AFF, CK_VAFF, CK_APPR, CK_TAP, CK_ASP };

typedef struct {
	const char *ipa; int type;
	float f1, f2, f3, e1, e2, e3;	/* formant targets (start / end) */
	float dur;			/* ms */
	float fc, fbw, famp;		/* frication / burst resonator */
} ck_ph;

static const ck_ph ck_tab[] = {
	/* vowels */
	{"i",CK_V,280,2250,3000,280,2250,3000,90},    {"\xc9\xaa",CK_V,400,1920,2560,400,1920,2560,70},
	{"e",CK_V,420,2100,2700,420,2100,2700,100},   {"\xc9\x9b",CK_V,550,1770,2490,550,1770,2490,100},
	{"\xc3\xa6",CK_V,690,1660,2490,690,1660,2490,120}, {"\xc9\x91",CK_V,710,1100,2540,710,1100,2540,120},
	{"\xc9\x94",CK_V,590,880,2540,590,880,2540,110},   {"o",CK_V,450,880,2540,450,880,2540,110},
	{"\xca\x8a",CK_V,450,1030,2380,450,1030,2380,70},  {"u",CK_V,310,870,2250,310,870,2250,100},
	{"\xca\x8c",CK_V,600,1200,2500,600,1200,2500,80},  {"\xc9\x99",CK_V,500,1500,2500,500,1500,2500,50},
	{"\xc9\x9a",CK_V,480,1350,1700,480,1350,1700,100}, {"a",CK_V,750,1300,2500,750,1300,2500,110},
	{"\xc9\x90",CK_V,650,1300,2500,650,1300,2500,70},
	/* diphthongs (English) */
	{"a\xc9\xaa",CK_V,710,1100,2540,400,1920,2560,170}, {"a\xca\x8a",CK_V,710,1100,2540,450,1030,2380,170},
	{"\xc9\x94\xc9\xaa",CK_V,590,880,2540,400,1920,2560,170}, {"e\xc9\xaa",CK_V,480,1900,2600,300,2200,2900,150},
	{"o\xca\x8a",CK_V,450,880,2540,350,800,2300,150},
	/* nasal vowels (Portuguese) */
	{"\xc3\xa3",CK_VN,650,1300,2500,650,1300,2500,120}, {"\xe1\xba\xbd",CK_VN,420,2000,2650,420,2000,2650,110},
	{"\xc4\xa9",CK_VN,280,2250,3000,280,2250,3000,100}, {"\xc3\xb5",CK_VN,450,880,2540,450,880,2540,110},
	{"\xc5\xa9",CK_VN,310,870,2250,310,870,2250,100},
	/* Japanese */
	{"\xc9\xaf",CK_V,320,1350,2250,320,1350,2250,80},
	{"\xc9\xb4",CK_NAS,250,1300,2300,250,1300,2300,75},
	{"\xc9\x95",CK_FRIC,300,2000,2800,300,2000,2800,110,5200,1800,0.9f}, {"\xca\x91",CK_VFRIC,300,2000,2800,300,2000,2800,85,5200,1800,0.6f},
	{"t\xc9\x95",CK_AFF,200,2000,2800,200,2000,2800,120,5000,1800,0.9f}, {"d\xca\x91",CK_VAFF,200,2000,2800,200,2000,2800,100,4800,1800,0.6f},
	{"\xc3\xa7",CK_FRIC,300,2100,2900,300,2100,2900,95,4000,2200,0.5f}, {"\xc9\xb8",CK_FRIC,300,1200,2400,300,1200,2400,95,3500,5000,0.3f},
	{"ts",CK_AFF,200,1700,2800,200,1700,2800,115,6500,2000,0.9f},
	/* sonorants */
	{"m",CK_NAS,250,1100,2100,250,1100,2100,70}, {"n",CK_NAS,250,1700,2600,250,1700,2600,60},
	{"\xc5\x8b",CK_NAS,250,2000,2700,250,2000,2700,70}, {"\xc9\xb2",CK_NAS,250,2150,2800,250,2150,2800,75},
	{"l",CK_APPR,360,1300,2700,360,1300,2700,70}, {"\xca\x8e",CK_APPR,300,2000,2800,300,2000,2800,85},
	{"\xc9\xb9",CK_APPR,350,1200,1600,350,1200,1600,80}, {"w",CK_APPR,300,700,2300,300,700,2300,65},
	{"j",CK_APPR,280,2250,3000,280,2250,3000,60}, {"\xc9\xbe",CK_TAP,400,1500,2500,400,1500,2500,28},
	/* stops: fc = burst centre */
	{"p",CK_STOP,200,1000,2200,200,1000,2200,90,800,1500,0.5f},  {"b",CK_VSTOP,200,1000,2200,200,1000,2200,75,800,1500,0.35f},
	{"t",CK_STOP,200,1700,2800,200,1700,2800,90,4200,2000,0.8f}, {"d",CK_VSTOP,200,1700,2800,200,1700,2800,70,4000,2000,0.5f},
	{"k",CK_STOP,200,1800,2300,200,1800,2300,95,2200,1200,0.7f}, {"g",CK_VSTOP,200,1800,2300,200,1800,2300,75,2000,1200,0.45f},
	/* affricates */
	{"t\xca\x83",CK_AFF,200,1900,2700,200,1900,2700,130,3500,1500,0.9f}, {"d\xca\x92",CK_VAFF,200,1900,2700,200,1900,2700,110,3300,1500,0.6f},
	/* fricatives */
	{"f",CK_FRIC,300,1200,2400,300,1200,2400,100,4500,5000,0.35f}, {"v",CK_VFRIC,300,1200,2400,300,1200,2400,70,4500,5000,0.25f},
	{"\xce\xb8",CK_FRIC,300,1400,2500,300,1400,2500,100,4500,5000,0.3f}, {"\xc3\xb0",CK_VFRIC,300,1400,2500,300,1400,2500,55,4500,5000,0.2f},
	{"s",CK_FRIC,300,1600,2600,300,1600,2600,110,7200,2000,1.0f}, {"z",CK_VFRIC,300,1600,2600,300,1600,2600,90,7200,2000,0.7f},
	{"\xca\x83",CK_FRIC,300,1800,2500,300,1800,2500,115,3300,1800,1.0f}, {"\xca\x92",CK_VFRIC,300,1800,2500,300,1800,2500,95,3300,1800,0.7f},
	{"h",CK_ASP,500,1500,2500,500,1500,2500,70,1800,2500,0.0f}, {"\xca\x81",CK_VFRIC,400,1400,2400,400,1400,2400,85,1800,2500,0.35f},
};
#define CK_NTAB ((int)(sizeof ck_tab / sizeof ck_tab[0]))

typedef struct {
	float f0;	/* mean pitch, Hz */
	float speed;	/* 1.0 = normal */
	float pitch_range;	/* 1.0 = normal intonation depth */
} ck_opt;

static inline ck_opt ck_default(void) { ck_opt o = {120.0f, 1.0f, 1.0f}; return o; }

typedef struct { float f1, f2, f3, av, ah, af, fc, fbw, nas, f0; int inherit, stress, vowel; } ck_frame;

typedef struct { double a, b, c, y1, y2; } ck_res;

static inline void ck_res_set(ck_res *r, double f, double bw)
{
	double c = -exp(-M_PI * bw / CK_RATE);
	double b = 2.0 * exp(-M_PI * bw / CK_RATE) * cos(2.0 * M_PI * f / CK_RATE);
	r->c = c; r->b = b; r->a = 1.0 - b - c;
}
static inline double ck_res_run(ck_res *r, double x)
{
	double y = r->a * x + r->b * r->y1 + r->c * r->y2;
	r->y2 = r->y1; r->y1 = y;
	return y;
}

static inline int ck_find(const char *s, int *len)
{
	int best = -1, bl = 0;
	for (int i = 0; i < CK_NTAB; i++) {
		int l = (int)strlen(ck_tab[i].ipa);
		if (l > bl && !strncmp(s, ck_tab[i].ipa, l)) { best = i; bl = l; }
	}
	*len = bl;
	return best;
}

typedef struct { ck_frame *f; int n, cap; } ck_buf;

static inline ck_frame *ck_add(ck_buf *b)
{
	if (b->n == b->cap) { b->cap = b->cap ? b->cap * 2 : 1024; b->f = (ck_frame *)realloc(b->f, b->cap * sizeof(ck_frame)); }
	ck_frame *f = &b->f[b->n++];
	memset(f, 0, sizeof *f);
	return f;
}

static inline void ck_silence(ck_buf *b, float ms, float speed)
{
	int n = (int)(ms / speed / 5.0f + 0.5f);
	for (int i = 0; i < n; i++) { ck_frame *f = ck_add(b); f->inherit = 1; }
}

/* IPA -> PCM. Returns 0 on success; *pcm is malloc'ed (caller frees). */
static inline int ck_synth(const char *ipa, ck_opt opt, float **pcm, size_t *count)
{
	ck_buf b = {0};
	float sp = opt.speed > 0.2f ? opt.speed : 1.0f;
	int stress = 0, lengthen = 0;
	for (const char *p = ipa; *p;) {
		if (*p == ' ') { ck_silence(&b, 45, sp); p++; continue; }
		if (*p == ',') { ck_silence(&b, 170, sp); p++; continue; }
		if (*p == '.' || *p == '?' || *p == '!') {
			ck_silence(&b, 320, sp);
			if (*p == '?' && b.n) b.f[b.n - 1].stress = -1;
			p++; continue;
		}
		if (!strncmp(p, "\xcb\x88", 2)) { stress = 1; p += 2; continue; }
		if (!strncmp(p, "\xcb\x8c", 2)) { p += 2; continue; }
		if (!strncmp(p, "\xcb\x90", 2)) { lengthen = 1; p += 2; continue; }
		int l; int id = ck_find(p, &l);
		if (id < 0) { uint32_t c; p += cp_utf8_decode((const unsigned char *)p, &c); continue; }
		p += l;
		const ck_ph *ph = &ck_tab[id];
		float dur = ph->dur;
		int isv = ph->type == CK_V || ph->type == CK_VN;
		if (isv) dur *= stress ? 1.15f : 0.85f;
		if (lengthen) dur *= 1.4f;
		lengthen = 0;
		int nf = (int)(dur / sp / 5.0f + 0.5f); if (nf < 2) nf = 2;
		int closure = 0;
		switch (ph->type) {
		case CK_V: case CK_VN: case CK_NAS: case CK_APPR: case CK_TAP:
			for (int i = 0; i < nf; i++) {
				ck_frame *f = ck_add(&b);
				float t = nf > 1 ? (float)i / (nf - 1) : 0;
				f->f1 = ph->f1 + (ph->e1 - ph->f1) * t; f->f2 = ph->f2 + (ph->e2 - ph->f2) * t; f->f3 = ph->f3 + (ph->e3 - ph->f3) * t;
				f->av = ph->type == CK_NAS ? 0.6f : ph->type == CK_APPR ? 0.8f : ph->type == CK_TAP ? (i == nf / 2 ? 0.15f : 0.9f) : 1.0f;
				if (ph->type == CK_V || ph->type == CK_VN) f->av = 1.0f;
				f->nas = ph->type == CK_NAS ? 1.0f : ph->type == CK_VN ? 0.7f : 0.0f;
				f->vowel = isv; f->stress = (isv && stress) ? 1 : 0;
			}
			break;
		case CK_STOP: case CK_VSTOP: case CK_AFF: case CK_VAFF: {
			int voiced = ph->type == CK_VSTOP || ph->type == CK_VAFF;
			int aff = ph->type == CK_AFF || ph->type == CK_VAFF;
			closure = (int)(nf * (aff ? 0.35f : 0.55f)); if (closure < 1) closure = 1;
			for (int i = 0; i < closure; i++) {
				ck_frame *f = ck_add(&b);
				f->f1 = 200; f->f2 = ph->f2; f->f3 = ph->f3; f->av = voiced ? 0.15f : 0; f->inherit = !voiced;
			}
			int rel = nf - closure;
			for (int i = 0; i < rel; i++) {
				ck_frame *f = ck_add(&b);
				f->f1 = ph->f1; f->f2 = ph->f2; f->f3 = ph->f3;
				f->fc = ph->fc; f->fbw = ph->fbw;
				f->af = (i == 0 || aff) ? ph->famp * (aff ? 0.7f : 1.0f) * (i < 2 ? 1.0f : 0.6f) : 0;
				if (!aff && i > 0 && !voiced) { f->ah = 0.45f; f->af = i == 1 ? ph->famp * 0.4f : 0; }
				if (aff && i >= rel - 1) f->af *= 0.5f;
				f->av = voiced ? 0.8f : 0; f->inherit = !voiced;
			}
			break;
		}
		case CK_FRIC: case CK_VFRIC: {
			int voiced = ph->type == CK_VFRIC;
			for (int i = 0; i < nf; i++) {
				ck_frame *f = ck_add(&b);
				float env = i < 2 ? 0.6f : (i > nf - 2 ? 0.7f : 1.0f);
				f->f1 = ph->f1; f->f2 = ph->f2; f->f3 = ph->f3; f->fc = ph->fc; f->fbw = ph->fbw;
				f->af = ph->famp * env; f->av = voiced ? 0.45f : 0; f->inherit = !voiced;
			}
			break;
		}
		case CK_ASP:
			for (int i = 0; i < nf; i++) { ck_frame *f = ck_add(&b); f->ah = 0.7f; f->f1 = 500; f->f2 = 1500; f->f3 = 2500; f->inherit = 1; }
			break;
		}
		if (isv) stress = 0; else if (ph->type != CK_NAS && ph->type != CK_APPR) { /* keep pending stress across onset consonants */ }
	}
	if (!b.n) { free(b.f); *pcm = NULL; *count = 0; return -1; }
	/* leading/trailing silence */
	{ ck_buf pre = {0}; (void)pre; }

	/* inherited formants: copy from the next non-inherit frame, else previous */
	for (int i = b.n - 1; i >= 0; i--) {
		if (!b.f[i].inherit) continue;
		int j = i + 1; while (j < b.n && b.f[j].inherit) j++;
		if (j < b.n) { b.f[i].f1 = b.f[j].f1; b.f[i].f2 = b.f[j].f2; b.f[i].f3 = b.f[j].f3; }
		else { int k = i - 1; while (k >= 0 && b.f[k].inherit) k--; if (k >= 0) { b.f[i].f1 = b.f[k].f1; b.f[i].f2 = b.f[k].f2; b.f[i].f3 = b.f[k].f3; } else { b.f[i].f1 = 500; b.f[i].f2 = 1500; b.f[i].f3 = 2500; } }
	}
	/* smooth formants and nasalization across frames */
	{
		ck_frame *s = (ck_frame *)malloc(sizeof(ck_frame) * b.n);
		if (!s) { free(b.f); return -1; }
		memcpy(s, b.f, sizeof(ck_frame) * b.n);
		for (int i = 0; i < b.n; i++) {
			float a1 = 0, a2 = 0, a3 = 0, an = 0; int c = 0;
			for (int k = -3; k <= 3; k++) { int j = i + k; if (j < 0 || j >= b.n) continue; a1 += s[j].f1; a2 += s[j].f2; a3 += s[j].f3; an += s[j].nas; c++; }
			b.f[i].f1 = a1 / c; b.f[i].f2 = a2 / c; b.f[i].f3 = a3 / c; b.f[i].nas = an / c;
		}
		/* smooth voicing amplitude over 2 frames each side to avoid clicks */
		for (int i = 0; i < b.n; i++) {
			float a = 0; int c = 0;
			for (int k = -1; k <= 1; k++) { int j = i + k; if (j < 0 || j >= b.n) continue; a += s[j].av; c++; }
			b.f[i].av = a / c;
		}
		free(s);
	}
	/* F0 contour: declination + accent on stressed vowels, final fall */
	{
		int last_voiced = 0; for (int i = 0; i < b.n; i++) if (b.f[i].av > 0.1f) last_voiced = i;
		int q = 0; for (int i = 0; i < b.n; i++) if (b.f[i].stress == -1) q = 1;
		float lastv = 0;
		for (int i = 0; i < b.n; i++) {
			float t = (float)i / (b.n > 1 ? b.n - 1 : 1);
			float f0 = opt.f0 * (1.12f - 0.28f * t * opt.pitch_range);
			int st = 0; for (int k = -5; k <= 5; k++) { int j = i + k; if (j >= 0 && j < b.n && b.f[j].stress == 1) st = 1; }
			if (st) f0 *= 1.0f + 0.12f * opt.pitch_range;
			if (i > last_voiced - 25 && last_voiced > 25) {
				float r = (float)(i - (last_voiced - 25)) / 25.0f;
				f0 *= q ? 1.0f + 0.25f * r * opt.pitch_range : 1.0f - 0.15f * r * opt.pitch_range;
			}
			b.f[i].f0 = f0; lastv = f0;
		}
		(void)lastv;
	}

	/* synthesis */
	size_t total = (size_t)(b.n + 12) * CK_FRAME;
	float *out = (float *)calloc(total, sizeof(float));
	if (!out) { free(b.f); return -1; }
	ck_res r1 = {0}, r2 = {0}, r3 = {0}, r4 = {0}, r5 = {0}, rn = {0}, rf = {0};
	uint32_t rng = 12345;
	double ph = 0, prev_pulse = 0, lp = 0, nz_lp = 0;
	size_t pos = (size_t)6 * CK_FRAME;
	for (int fi = 0; fi < b.n; fi++) {
		const ck_frame *c = &b.f[fi], *nx = &b.f[fi + 1 < b.n ? fi + 1 : fi];
		for (int s = 0; s < CK_FRAME; s++) {
			if (s % 16 == 0) {
				float t = (float)s / CK_FRAME;
				#define LERP(x) (c->x + (nx->x - c->x) * t)
				double f1 = LERP(f1), f2 = LERP(f2), f3 = LERP(f3);
				ck_res_set(&r1, f1, 60 + 0.1 * f1); ck_res_set(&r2, f2, 90 + 0.05 * f2); ck_res_set(&r3, f3, 150 + 0.05 * f3);
				ck_res_set(&r4, 3300, 250); ck_res_set(&r5, 3750, 200); ck_res_set(&rn, 280, 90);
				if (c->fc > 0 || nx->fc > 0) ck_res_set(&rf, c->fc > 0 ? LERP(fc) : nx->fc, (c->fbw > 0 ? c->fbw : nx->fbw));
			}
			float t = (float)s / CK_FRAME;
			double f0 = LERP(f0), av = LERP(av), ah = LERP(ah), af = LERP(af), nas = LERP(nas);
			rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
			double noise = ((double)(rng & 0xFFFF) / 32768.0) - 1.0;
			/* glottal flow (Rosenberg) differentiated */
			ph += f0 / CK_RATE; if (ph >= 1.0) ph -= 1.0;
			double g = ph < 0.6 ? 0.5 * (1.0 - cos(M_PI * ph / 0.6)) : cos(0.5 * M_PI * (ph - 0.6) / 0.4);
			double d = (g - prev_pulse) * 40.0; prev_pulse = g;
			double src = av * d + ah * 0.25 * noise + av * 0.02 * noise;
			double y = ck_res_run(&r1, src);
			if (nas > 0.01) y = y * (1.0 - 0.45 * nas) + 0.8 * nas * ck_res_run(&rn, y);
			y = ck_res_run(&r2, y); y = ck_res_run(&r3, y); y = ck_res_run(&r4, y); y = ck_res_run(&r5, y);
			double fr = 0;
			if (af > 0.001) {
				nz_lp = noise;
				fr = ck_res_run(&rf, nz_lp) * af * 3.5;
				if (c->av > 0.05f && c->fc > 0) fr *= 0.8;
			} else { rf.y1 *= 0.9; rf.y2 *= 0.9; }
			lp = y;
			if (pos + s < total) out[pos + s] = (float)(lp * 0.9 + fr);
			#undef LERP
		}
		pos += CK_FRAME;
	}
	free(b.f);
	/* DC removal, normalize */
	double mean = 0, hp = 0, peak = 1e-9; (void)mean;
	for (size_t i = 0; i < total; i++) { double x = out[i]; hp = x - (x - 0.995 * 0); (void)hp; }
	{ double px = 0, py = 0; for (size_t i = 0; i < total; i++) { double x = out[i]; double y = x - px + 0.995 * py; px = x; py = y; out[i] = (float)y; } }
	for (size_t i = 0; i < total; i++) if (fabs(out[i]) > peak) peak = fabs(out[i]);
	for (size_t i = 0; i < total; i++) out[i] = (float)(out[i] / peak * 0.8);
	*pcm = out; *count = total;
	return 0;
}

static inline int ck_write_wav(const char *path, const float *pcm, size_t n, int rate)
{
	FILE *f = fopen(path, "wb");
	if (!f) return -1;
	uint32_t data = (uint32_t)(n * 2), riff = 36 + data, sr = rate, br = rate * 2, fmt = 16;
	uint16_t pcmfmt = 1, ch = 1, ba = 2, bits = 16;
	fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmt, 4, 1, f);
	fwrite(&pcmfmt, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&sr, 4, 1, f); fwrite(&br, 4, 1, f); fwrite(&ba, 2, 1, f); fwrite(&bits, 2, 1, f);
	fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f);
	for (size_t i = 0; i < n; i++) {
		float v = pcm[i]; if (v > 1) v = 1; if (v < -1) v = -1;
		int16_t s = (int16_t)lrintf(v * 32767.0f); fwrite(&s, 2, 1, f);
	}
	fclose(f);
	return 0;
}

#endif
