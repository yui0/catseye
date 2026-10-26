/*
 * lang_ja.h - Japanese grapheme-to-phoneme (Tokyo standard, no pitch accent).
 *
 * Kana (hiragana, katakana, ー) are converted to IPA by rule. Kanji need a
 * dictionary: set cp_ja_reader (tts.c wires it to tools/ja_reading.py, which
 * uses pykakasi) and kanji text is first turned into hiragana, with particles
 * は / へ / を respelled わ / え / お. Without a reader, kanji are skipped.
 *
 * Output is mora-by-mora: ん assimilates (m n ŋ ɴ), っ doubles the next
 * consonant, ー / おう / えい / repeated vowels lengthen with ː.
 */
#ifndef CATS_LANG_JA_H
#define CATS_LANG_JA_H

#include "phon.h"

/* optional: hira = reading of text (kanji -> kana); return 0 on success */
static int (*cp_ja_reader)(const char *text, char *hira, size_t cap) = NULL;

typedef struct { const char *k; const char *p[5]; } cp_ja_row;
static const cp_ja_row cp_ja_rows[] = {
	{"あいうえお", {"a","i","ɯ","e","o"}},
	{"かきくけこ", {"ka","ki","kɯ","ke","ko"}},
	{"がぎぐげご", {"ga","gi","gɯ","ge","go"}},
	{"さしすせそ", {"sa","ɕi","sɯ","se","so"}},
	{"ざじずぜぞ", {"za","dʑi","zɯ","ze","zo"}},
	{"たちつてと", {"ta","tɕi","tsɯ","te","to"}},
	{"だぢづでど", {"da","dʑi","zɯ","de","do"}},
	{"なにぬねの", {"na","ɲi","nɯ","ne","no"}},
	{"はひふへほ", {"ha","çi","ɸɯ","he","ho"}},
	{"ばびぶべぼ", {"ba","bi","bɯ","be","bo"}},
	{"ぱぴぷぺぽ", {"pa","pi","pɯ","pe","po"}},
	{"まみむめも", {"ma","mi","mɯ","me","mo"}},
	{"らりるれろ", {"ɾa","ɾi","ɾɯ","ɾe","ɾo"}},
	{"やいゆえよ", {"ja","i","jɯ","e","jo"}},
	{"わゐうゑを", {"wa","i","ɯ","e","o"}},
	{"ぁぃぅぇぉ", {"a","i","ɯ","e","o"}},
};

/* i-row kana + ゃ ゅ ょ : palatal onset */
static const char *cp_ja_pal[][2] = {
	{"き","kj"},{"ぎ","gj"},{"し","ɕ"},{"じ","dʑ"},{"ぢ","dʑ"},{"ち","tɕ"},{"に","ɲ"},{"ひ","ç"},{"び","bj"},{"ぴ","pj"},{"み","mj"},{"り","ɾj"},
};
/* kana + small vowel (foreign sounds): onset */
static const char *cp_ja_small[][2] = {
	{"ふ","ɸ"},{"つ","ts"},{"ゔ","v"},{"て","t"},{"で","d"},{"う","w"},{"し","ɕ"},{"ち","tɕ"},{"じ","dʑ"},{"ど","d"},{"と","t"},{"く","kw"},{"ぐ","gw"},
};

static inline int cp_ja_put(char *o, uint32_t c)
{
	if (c < 0x80) { o[0] = (char)c; return 1; }
	if (c < 0x800) { o[0] = (char)(0xC0 | c >> 6); o[1] = (char)(0x80 | (c & 63)); return 2; }
	o[0] = (char)(0xE0 | c >> 12); o[1] = (char)(0x80 | ((c >> 6) & 63)); o[2] = (char)(0x80 | (c & 63)); return 3;
}

static inline int cp_ja_is_cjk(uint32_t c) { return (c >= 0x4E00 && c <= 0x9FFF) || c == 0x3005; }

/* katakana -> hiragana (ー stays) */
static inline uint32_t cp_ja_hira(uint32_t c) { return (c >= 0x30A1 && c <= 0x30F6) ? c - 0x60 : c; }

/* numbers 0..9999 as hiragana */
static inline void cp_ja_number(long n, char *out, size_t cap)
{
	static const char *ones[] = {"ぜろ","いち","に","さん","よん","ご","ろく","なな","はち","きゅう"};
	out[0] = 0;
	if (n < 0 || n > 9999) return;
	if (n == 0) { snprintf(out, cap, "ぜろ"); return; }
	int th = (int)(n / 1000), h = (int)(n / 100 % 10), t = (int)(n / 10 % 10), o = (int)(n % 10);
	#define A(s) strncat(out, (s), cap - strlen(out) - 1)
	if (th) { if (th == 3) A("さんぜん"); else if (th == 8) A("はっせん"); else { if (th > 1) A(ones[th]); A("せん"); } }
	if (h) { if (h == 3) A("さんびゃく"); else if (h == 6) A("ろっぴゃく"); else if (h == 8) A("はっぴゃく"); else { if (h > 1) A(ones[h]); A("ひゃく"); } }
	if (t) { if (t > 1) A(ones[t]); A("じゅう"); }
	if (o) A(ones[o]);
	#undef A
}

static inline int cp_ja_normalize(const char *in, char *out, size_t cap)
{
	char *rd = NULL;
	for (const unsigned char *p = (const unsigned char *)in; *p; ) {
		uint32_t c; int l = cp_utf8_decode(p, &c);
		if (cp_ja_is_cjk(c) && cp_ja_reader) {
			size_t n = strlen(in) * 6 + 256;
			rd = (char *)malloc(n);
			if (rd && cp_ja_reader(in, rd, n) == 0) in = rd; else { free(rd); rd = NULL; }
			break;
		}
		p += l;
	}
	size_t o = 0;
	#define PUT(c) do { if (o + 4 < cap) out[o++] = (c); } while (0)
	for (const unsigned char *p = (const unsigned char *)in; *p;) {
		uint32_t c; p += cp_utf8_decode(p, &c);
		if (c >= 0xFF10 && c <= 0xFF19) c = '0' + (c - 0xFF10);
		if (c >= '0' && c <= '9') {
			long n = c - '0'; int digits = 1;
			while (*p >= '0' && *p <= '9') { n = n * 10 + (*p - '0'); p++; digits++; }
			char w[128];
			if (digits <= 4) cp_ja_number(n, w, sizeof w); else w[0] = 0;
			PUT(' '); for (char *q = w; *q; q++) PUT(*q); PUT(' ');
			continue;
		}
		if (c == 0x3001 || c == 0xFF0C || c == ',') { PUT(' '); PUT(','); PUT(' '); }
		else if (c == 0x3002 || c == 0xFF0E || c == '.') { PUT(' '); PUT('.'); PUT(' '); }
		else if (c == 0xFF01 || c == '!') { PUT(' '); PUT('!'); PUT(' '); }
		else if (c == 0xFF1F || c == '?') { PUT(' '); PUT('?'); PUT(' '); }
		else if ((c >= 0x3041 && c <= 0x30FF && c != 0x30FB) || cp_ja_is_cjk(c)) { char b[4]; int l = cp_ja_put(b, c); for (int k = 0; k < l; k++) PUT(b[k]); }
		else PUT(' ');
	}
	out[o] = 0;
	#undef PUT
	free(rd);
	return (int)o;
}

/* onset helpers for ん assimilation and っ gemination */
static inline const char *cp_ja_nasal_for(const char *next)
{
	if (!next || !*next) return "\xc9\xb4";			/* ɴ */
	switch (next[0]) {
	case 'p': case 'b': case 'm': return "m";
	case 't': case 'd': case 'n': case 's': case 'z': return "n";
	case 'k': case 'g': return "\xc5\x8b";			/* ŋ */
	}
	if (!strncmp(next, "\xc9\xbe", 2) || !strncmp(next, "\xc9\x95", 2) || !strncmp(next, "\xca\x91", 2)) return "n";	/* ɾ ɕ ʑ */
	if (!strncmp(next, "\xc9\xb2", 2)) return "\xc9\xb2";	/* ɲ */
	return "\xc9\xb4";
}

static void cp_ja_word(const char *word, char *out, size_t cap)
{
	/* kana string (hiragana-normalized) */
	char s[512]; size_t sl = 0;
	for (const unsigned char *p = (const unsigned char *)word; *p && sl + 4 < sizeof s;) {
		uint32_t c; p += cp_utf8_decode(p, &c);
		if (cp_ja_is_cjk(c)) continue;
		sl += cp_ja_put(s + sl, cp_ja_hira(c));
	}
	s[sl] = 0;
	/* greetings whose final は is the particle wa */
	{
		static const char *greet[] = {"こんにちは", "こんばんは"};
		for (int g = 0; g < 2; g++) if (!strcmp(s, greet[g])) { s[sl - 3] = 0; strcat(s, "わ"); break; }
	}
	char moras[96][12]; int nm = 0;
	for (const char *p = s; *p && nm < 90;) {
		uint32_t c; int l = cp_utf8_decode((const unsigned char *)p, &c);
		char cur[8]; memcpy(cur, p, l); cur[l] = 0;
		const char *next = p + l; uint32_t nc = 0; int nl = 0;
		if (*next) nl = cp_utf8_decode((const unsigned char *)next, &nc);
		char *m = moras[nm]; m[0] = 0;
		if (c == 0x3063) { strcpy(m, "Q"); nm++; p += l; continue; }	/* っ */
		if (c == 0x3093) { strcpy(m, "N"); nm++; p += l; continue; }	/* ん */
		if (c == 0x30FC) { strcpy(m, "ː"); nm++; p += l; continue; }
		/* palatal: i-row + ゃゅょ */
		if (nc == 0x3083 || nc == 0x3085 || nc == 0x3087) {
			int done = 0;
			for (size_t k = 0; k < sizeof cp_ja_pal / sizeof *cp_ja_pal && !done; k++)
				if (!strcmp(cur, cp_ja_pal[k][0])) {
					snprintf(m, 12, "%s%s", cp_ja_pal[k][1], nc == 0x3083 ? "a" : nc == 0x3085 ? "\xc9\xaf" : "o");
					done = 1;
				}
			if (done) { nm++; p += l + nl; continue; }
		}
		/* kana + small vowel (ふぁ てぃ うぃ ...) */
		if (nc == 0x3041 || nc == 0x3043 || nc == 0x3045 || nc == 0x3047 || nc == 0x3049) {
			int done = 0;
			for (size_t k = 0; k < sizeof cp_ja_small / sizeof *cp_ja_small && !done; k++)
				if (!strcmp(cur, cp_ja_small[k][0])) {
					const char *v = nc == 0x3041 ? "a" : nc == 0x3043 ? "i" : nc == 0x3045 ? "\xc9\xaf" : nc == 0x3047 ? "e" : "o";
					snprintf(m, 12, "%s%s", cp_ja_small[k][1], v);
					done = 1;
				}
			if (done) { nm++; p += l + nl; continue; }
		}
		/* plain row lookup */
		int found = 0;
		if (c == 0x3094) { strcpy(m, "b\xc9\xaf"); found = 1; }	/* ゔ */
		for (size_t r = 0; r < sizeof cp_ja_rows / sizeof *cp_ja_rows && !found; r++) {
			const char *k = cp_ja_rows[r].k;
			for (int j = 0; j < 5; j++) {
				uint32_t kc; int kl = cp_utf8_decode((const unsigned char *)k, &kc);
				if (kc == c) { snprintf(m, 12, "%s", cp_ja_rows[r].p[j]); found = 1; break; }
				k += kl;
			}
		}
		if (found) nm++;
		p += l;
	}
	/* assemble */
	char buf[512]; size_t bl = 0; buf[0] = 0;
	#define APP(str) do { size_t l_ = strlen(str); if (bl + l_ < sizeof buf) { memcpy(buf + bl, str, l_); bl += l_; buf[bl] = 0; } } while (0)
	int geminate = 0;
	for (int i = 0; i < nm; i++) {
		const char *m = moras[i];
		if (!strcmp(m, "Q")) { geminate = 1; continue; }
		if (!strcmp(m, "N")) {
			const char *nx = NULL;
			for (int j = i + 1; j < nm; j++) if (strcmp(moras[j], "Q") && strcmp(moras[j], "N") && strcmp(moras[j], "\xcb\x90")) { nx = moras[j]; break; }
			APP(cp_ja_nasal_for(nx)); continue;
		}
		if (!strcmp(m, "\xcb\x90")) { if (bl && strncmp(buf + bl - 2, "\xcb\x90", 2)) APP(m); continue; }
		if (geminate) {
			if (!strncmp(m, "ts", 2) || !strncmp(m, "t\xc9\x95", 3)) APP("t");
			else if (!strncmp(m, "d\xca\x91", 3)) APP("d");
			else if (!(m[0] == 'a' || m[0] == 'e' || m[0] == 'o' || m[0] == 'i' || !strncmp(m, "\xc9\xaf", 2))) { uint32_t c; int l = cp_utf8_decode((const unsigned char *)m, &c); char t[8]; memcpy(t, m, l); t[l] = 0; APP(t); }
			geminate = 0;
		}
		/* おう / えい / same-vowel repeats -> long vowel */
		int vowel_only = (m[0] == 'a' || m[0] == 'e' || m[0] == 'o' || m[0] == 'i' || !strncmp(m, "\xc9\xaf", 2)) && strlen(m) <= 2;
		if (vowel_only && bl) {
			const char *last = buf + bl - 1;
			if (!strncmp(buf + (bl >= 2 ? bl - 2 : 0), "\xc9\xaf", 2)) last = buf + bl - 2;
			if ((*last == 'o' && (!strcmp(m, "\xc9\xaf") || !strcmp(m, "o"))) || (*last == 'e' && (!strcmp(m, "i") || !strcmp(m, "e")))
			    || (*last == 'a' && !strcmp(m, "a")) || (*last == 'i' && !strcmp(m, "i")) || (!strncmp(last, "\xc9\xaf", 2) && !strcmp(m, "\xc9\xaf"))) {
				APP("\xcb\x90"); continue;
			}
		}
		APP(m);
	}
	#undef APP
	snprintf(out, cap, "%s", buf);
}

#endif
