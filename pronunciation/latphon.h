/*
 * latphon.h - grapheme-to-phoneme registry for the pronunciation tools.
 *
 * cp_g2p(lang, text, out, cap) is the single entry point every TTS engine and
 * the pronunciation checker use: text -> IPA words separated by spaces, with
 * sentence punctuation (, . ? !) passed through for pauses.
 *
 * Adding a language = three steps (see README "Adding a language"):
 *   1. write lang_xx.h with `static void cp_xx_word(const char *w, char *ipa, size_t cap)`
 *      (and, if word splitting differs, `static int cp_xx_normalize(in, out, cap)`)
 *   2. add an enum value and one row to cp_langs[] below
 *   3. add the phonemes it needs to ck_tab[] in klatt.h
 * A neural backend (LatPhon, 7.5M parameters) can be plugged in through
 * cp_g2p_hook, which is consulted first and may decline a word.
 */
#ifndef CATS_LATPHON_H
#define CATS_LATPHON_H

#include "phon.h"
#include "lang_en.h"
#include "lang_pt.h"
#include "lang_ja.h"

/* optional neural backend: write IPA for one normalized word, return 1 if handled */
static int (*cp_g2p_hook)(int lang, const char *word, char *ipa, size_t cap) = NULL;

typedef struct {
	const char *code;	/* ISO 639-1, matched as a prefix of -L (so pt-BR and pt both work) */
	const char *name;
	void (*word)(const char *word, char *ipa, size_t cap);
	int (*normalize)(const char *in, char *out, size_t cap);	/* NULL = cp_normalize_default */
	const char *piper;	/* default Piper voice file stem in models/, or NULL */
	const char *kokoro;	/* default Kokoro voice (models/<stem>.bin), or NULL */
} cp_langdef;

static void cp_en_word_(const char *w, char *o, size_t c) { cp_en_word(w, o, c); }
static int cp_en_normalize_(const char *in, char *out, size_t cap) { return cp_normalize_default(CP_EN, in, out, cap); }
static int cp_pt_normalize_(const char *in, char *out, size_t cap) { return cp_normalize_default(CP_PT, in, out, cap); }

static const cp_langdef cp_langs[CP_NLANG] = {
	[CP_EN] = {"en", "English",    cp_en_word_, cp_en_normalize_, "en", "af_heart"},
	[CP_PT] = {"pt", "Portuguese", cp_pt_word,  cp_pt_normalize_, "pt", "pf_dora"},
	[CP_JA] = {"ja", "Japanese",   cp_ja_word,  cp_ja_normalize,  NULL, "jf_alpha"},
};

static inline int cp_lang(const char *s)
{
	if (!s) return CP_EN;
	for (int i = 0; i < CP_NLANG; i++) if (!strncmp(s, cp_langs[i].code, strlen(cp_langs[i].code))) return i;
	return -1;
}

static inline int cp_normalize(int lang, const char *in, char *out, size_t cap)
{
	return cp_langs[lang].normalize(in, out, cap);
}

/* ------------------------------------------------------------------ driver */

/* text -> IPA words separated by spaces; sentence punctuation (, . ? !) is passed through */
static inline int cp_g2p(int lang, const char *text, char *out, size_t cap)
{
	char *norm = (char *)malloc(strlen(text) * 4 + 64);
	if (!norm) return -1;
	cp_normalize(lang, text, norm, strlen(text) * 4 + 64);
	out[0] = 0; size_t o = 0;
	for (char *p = norm; *p;) {
		while (*p == ' ') p++;
		if (!*p) break;
		char *e = p; while (*e && *e != ' ') e++;
		char word[512]; size_t wl = (size_t)(e - p); if (wl > 500) wl = 500;
		memcpy(word, p, wl); word[wl] = 0;
		char ipa[1024]; ipa[0] = 0;
		if (!strcmp(word, ",") || !strcmp(word, ".") || !strcmp(word, "?") || !strcmp(word, "!")) snprintf(ipa, sizeof ipa, "%s", word);
		else if (cp_g2p_hook && cp_g2p_hook(lang, word, ipa, sizeof ipa)) { }
		else cp_langs[lang].word(word, ipa, sizeof ipa);
		size_t l = strlen(ipa);
		if (l && o + l + 2 < cap) { if (o) out[o++] = ' '; memcpy(out + o, ipa, l); o += l; out[o] = 0; }
		p = e;
	}
	free(norm);
	return (int)o;
}

#endif
