/*
 * phon.h - shared helpers for the pronunciation tools: UTF-8, IPA classes and
 * the text normalizer (case folding, digits -> words) used by every TTS engine.
 */
#ifndef CATS_PHON_H
#define CATS_PHON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* language ids: add new languages here and in cp_langs[] (latphon.h) */
enum { CP_EN = 0, CP_PT, CP_JA, CP_NLANG };

/* decode one code point; returns its byte length (>=1) */
static inline int cp_utf8_decode(const unsigned char *s, uint32_t *cp)
{
	if (s[0] < 0x80) { *cp = s[0]; return 1; }
	if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) { *cp = (s[0] & 0x1F) << 6 | (s[1] & 0x3F); return 2; }
	if ((s[0] & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
		*cp = (s[0] & 0x0F) << 12 | (s[1] & 0x3F) << 6 | (s[2] & 0x3F); return 3;
	}
	if ((s[0] & 0xF8) == 0xF0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
		*cp = (s[0] & 0x07) << 18 | (s[1] & 0x3F) << 12 | (s[2] & 0x3F) << 6 | (s[3] & 0x3F); return 4;
	}
	*cp = s[0]; return 1;
}

static inline int cp_is_ipa_vowel(uint32_t c)
{
	switch (c) {
	case 'i': case 0x26A: case 'e': case 0x25B: case 0xE6: case 0x251: case 0x254: case 'o': case 0x28A:
	case 'u': case 0x28C: case 0x259: case 0x25A: case 'a': case 0x250: case 0x268:
	case 0xE3: case 0x1EBD: case 0x129: case 0xF5: case 0x169:
		return 1;
	}
	return 0;
}

/* lower-case ASCII and Latin-1 supplement letters (UTF-8) in place; returns s */
static inline char *cp_lower(char *s)
{
	for (unsigned char *p = (unsigned char *)s; *p; p++) {
		if (*p >= 'A' && *p <= 'Z') *p += 32;
		else if (*p == 0xC3 && p[1] >= 0x80 && p[1] <= 0x9E && p[1] != 0x97) { p[1] += 32; p++; }
		else if (*p >= 0x80) { while (p[1] >= 0x80 && p[1] < 0xC0) p++; }
	}
	return s;
}

/* ---- numbers -> words (0 .. 999999) ---- */
static const char *cp_en_ones[] = {"zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"};
static const char *cp_en_tens[] = {"","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"};
static const char *cp_pt_ones[] = {"zero","um","dois","tr\xc3\xaas","quatro","cinco","seis","sete","oito","nove","dez","onze","doze","treze","catorze","quinze","dezesseis","dezessete","dezoito","dezenove"};
static const char *cp_pt_tens[] = {"","","vinte","trinta","quarenta","cinquenta","sessenta","setenta","oitenta","noventa"};
static const char *cp_pt_hund[] = {"","cento","duzentos","trezentos","quatrocentos","quinhentos","seiscentos","setecentos","oitocentos","novecentos"};

static inline void cp_cat(char *out, size_t cap, const char *s)
{
	size_t n = strlen(out);
	if (n && n + 1 < cap) out[n++] = ' ';
	size_t l = strlen(s);
	if (n + l >= cap) l = cap - n - 1;
	memcpy(out + n, s, l); out[n + l] = 0;
}

static inline void cp_number_below_1000(int lang, int n, char *out, size_t cap)
{
	if (lang == CP_EN) {
		if (n >= 100) { cp_cat(out, cap, cp_en_ones[n / 100]); cp_cat(out, cap, "hundred"); n %= 100; if (!n) return; }
		if (n >= 20) { cp_cat(out, cap, cp_en_tens[n / 10]); n %= 10; if (!n) return; }
		cp_cat(out, cap, cp_en_ones[n]);
	} else {
		if (n == 100) { cp_cat(out, cap, "cem"); return; }
		if (n >= 100) { cp_cat(out, cap, cp_pt_hund[n / 100]); n %= 100; if (!n) return; cp_cat(out, cap, "e"); }
		if (n >= 20) { cp_cat(out, cap, cp_pt_tens[n / 10]); n %= 10; if (!n) return; cp_cat(out, cap, "e"); }
		cp_cat(out, cap, cp_pt_ones[n]);
	}
}

static inline void cp_number_words(int lang, long n, char *out, size_t cap)
{
	out[0] = 0;
	if (n >= 1000000) { cp_cat(out, cap, lang == CP_EN ? "many" : "muitos"); return; }
	if (n == 0) { cp_cat(out, cap, lang == CP_EN ? "zero" : "zero"); return; }
	if (n >= 1000) {
		int th = (int)(n / 1000);
		if (!(lang == CP_PT && th == 1)) cp_number_below_1000(lang, th, out, cap);
		cp_cat(out, cap, lang == CP_EN ? "thousand" : "mil");
		n %= 1000;
		if (!n) return;
		if (lang == CP_PT && (n < 100 || n % 100 == 0)) cp_cat(out, cap, "e");
	}
	cp_number_below_1000(lang, (int)n, out, cap);
}

/*
 * Normalize text: lower-case, digits spelled out, anything that is neither a
 * letter, apostrophe nor one of , . ? ! becomes a space. Sentence punctuation
 * is kept as separate tokens so the engines can insert pauses.
 */
static inline int cp_normalize_default(int lang, const char *in, char *out, size_t cap)
{
	size_t o = 0;
	#define PUT(c) do { if (o + 1 < cap) out[o++] = (c); } while (0)
	for (const unsigned char *p = (const unsigned char *)in; *p;) {
		if (*p >= '0' && *p <= '9') {
			long n = 0;
			while (*p >= '0' && *p <= '9') { if (n < 10000000) n = n * 10 + (*p - '0'); p++; }
			char w[512]; cp_number_words(lang, n, w, sizeof w);
			PUT(' '); for (char *q = w; *q; q++) PUT(*q); PUT(' ');
		} else if ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') || *p >= 0x80 || *p == '\'') {
			if (*p >= 'A' && *p <= 'Z') PUT(*p + 32);
			else if (*p == 0xC3 && p[1] >= 0x80 && p[1] <= 0x9E && p[1] != 0x97) { PUT(*p); PUT(p[1] + 32); p++; }
			else PUT(*p);
			p++;
		} else if (*p == ',' || *p == '.' || *p == '?' || *p == '!' || *p == ';' || *p == ':') {
			PUT(' '); PUT(*p == ';' || *p == ':' ? ',' : *p); PUT(' '); p++;
		} else { PUT(' '); p++; }
	}
	out[o] = 0;
	#undef PUT
	return (int)o;
}

#endif
