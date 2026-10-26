/*
 * lang_pt.h - Portuguese (Brazilian) grapheme-to-phoneme: rules + clitic lexicon
 * Part of the language registry in latphon.h: one lower-case word in, IPA out.
 */
#ifndef CATS_LANG_PT_H
#define CATS_LANG_PT_H

#include "phon.h"
#include <ctype.h>


typedef struct { char c[4]; int v; int acc; int nasal; char k; } cp_pt_tok;
/* v: vowel nucleus (k = a e i o u); acc: 0 none, 1 acute, 2 circumflex, 3 tilde */

static int cp_pt_isv(uint32_t c)
{
	switch (c) {
	case 'a': case 'e': case 'i': case 'o': case 'u':
	case 0xE1: case 0xE9: case 0xED: case 0xF3: case 0xFA: case 0xE2: case 0xEA: case 0xF4: case 0xE3: case 0xF5: case 0xE0:
		return 1;
	}
	return 0;
}

/* unstressed clitics */
static const char *cp_pt_lex[][2] = {
	{"o","u"}, {"a","ɐ"}, {"os","us"}, {"as","ɐs"}, {"e","i"}, {"de","dʒi"}, {"da","dɐ"}, {"do","du"}, {"em","ẽ"}, {"um","ũ"},
	{"que","ki"}, {"se","si"}, {"com","kõ"}, {"no","nu"}, {"na","nɐ"}, {"para","pɐɾɐ"}, {"por","poɾ"}, {"me","mi"}, {"te","tʃi"},
};

static void cp_pt_word(const char *word, char *out, size_t cap)
{
	for (size_t i = 0; i < sizeof cp_pt_lex / sizeof cp_pt_lex[0]; i++)
		if (!strcmp(word, cp_pt_lex[i][0])) { snprintf(out, cap, "%s", cp_pt_lex[i][1]); return; }
	uint32_t g[64]; int n = 0;
	for (const unsigned char *p = (const unsigned char *)word; *p && n < 60;) { uint32_t c; p += cp_utf8_decode(p, &c); g[n++] = c; }
	g[n] = 0;
	struct { char s[8]; int v, acc, nasal, k, glide; } t[96]; int nt = 0;
	#define PUSH(str) do { memset(&t[nt], 0, sizeof t[nt]); snprintf(t[nt].s, sizeof t[nt].s, "%s", str); nt++; } while (0)
	#define VOW(kk, ac) do { PUSH(""); t[nt-1].v = 1; t[nt-1].k = (kk); t[nt-1].acc = (ac); } while (0)
	for (int i = 0; i < n; i++) {
		uint32_t c = g[i], nx = g[i + 1], pv = i ? g[i - 1] : ' ', nx2 = nx ? g[i + 2] : 0;
		int atstart = (i == 0);
		switch (c) {
		case 0xE1: VOW('a', 1); break; case 0xE0: VOW('a', 1); break; case 0xE2: VOW('a', 2); break; case 0xE3: VOW('a', 3); break;
		case 0xE9: VOW('e', 1); break; case 0xEA: VOW('e', 2); break; case 0xED: VOW('i', 1); break;
		case 0xF3: VOW('o', 1); break; case 0xF4: VOW('o', 2); break; case 0xF5: VOW('o', 3); break; case 0xFA: VOW('u', 1); break;
		case 'a': case 'e': case 'o': VOW((char)c, 0); break;
		case 'i': case 'u': {
			/* glide after a vowel when not followed by another vowel nucleus: ai, ei, ou, au... */
			int after_v = i && cp_pt_isv(pv) && pv != 0xED && pv != 0xFA;
			if (after_v && !(cp_pt_isv(nx) && nx != 'o')) { PUSH(c == 'i' ? "j" : "w"); }
			else if (c == 'u' && (pv == 'q' || (pv == 'g' && i > 1 && 0)) && (nx == 'e' || nx == 'i' || nx == 0xE9 || nx == 0xEA || nx == 0xED)) { /* silent u in que/qui */ }
			else if (c == 'u' && pv == 'g' && (nx == 'e' || nx == 'i' || nx == 0xE9 || nx == 0xEA)) { /* gue/gui */ }
			else VOW((char)c, 0);
			break;
		}
		case 'b': PUSH("b"); break; case 'f': PUSH("f"); break; case 'j': PUSH("\xca\x92"); break; case 'k': PUSH("k"); break;
		case 'p': PUSH("p"); break; case 'v': PUSH("v"); break; case 'w': PUSH("w"); break;
		case 0xE7: PUSH("s"); break;
		case 'c':
			if (nx == 'h') { PUSH("\xca\x83"); i++; }
			else if (nx == 'e' || nx == 'i' || nx == 0xE9 || nx == 0xEA || nx == 0xED) PUSH("s");
			else if (nx == 'c' && (nx2 == 'e' || nx2 == 'i')) { PUSH("k"); }
			else PUSH("k");
			break;
		case 'd': PUSH("d"); t[nt - 1].k = 'd'; break;
		case 't': PUSH("t"); t[nt - 1].k = 't'; break;
		case 'g':
			if (nx == 'e' || nx == 'i' || nx == 0xE9 || nx == 0xEA || nx == 0xED) PUSH("\xca\x92"); else PUSH("g");
			break;
		case 'h': break;
		case 'l':
			if (nx == 'h') { PUSH("\xca\x8e"); i++; }
			else if (!nx || !cp_pt_isv(nx)) PUSH("w");
			else PUSH("l");
			break;
		case 'm': case 'n': {
			if (c == 'n' && nx == 'h') { PUSH("\xc9\xb2"); i++; break; }
			/* nasalize the preceding vowel before a consonant or at word end */
			if (nt && t[nt - 1].v && !cp_pt_isv(nx)) { t[nt - 1].nasal = 1; break; }
			if (nt && t[nt - 1].glide && !cp_pt_isv(nx)) break;
			PUSH(c == 'm' ? "m" : "n");
			break;
		}
		case 'q': PUSH("k"); if (nx == 'u' && (nx2 == 'a' || nx2 == 'o' || nx2 == 0xE1 || nx2 == 0xE2 || nx2 == 0xF3)) { PUSH("w"); i++; } break;
		case 'r':
			if (nx == 'r') { PUSH("\xca\x81"); i++; }
			else if (atstart || pv == 'n' || pv == 'l' || pv == 's') PUSH("\xca\x81");
			else PUSH("\xc9\xbe");
			break;
		case 's':
			if (nx == 's') { PUSH("s"); i++; }
			else if (nx == 'c' && (nx2 == 'e' || nx2 == 'i')) { PUSH("s"); i++; }
			else if (!atstart && cp_pt_isv(pv) && cp_pt_isv(nx)) PUSH("z");
			else if (nx && !cp_pt_isv(nx) && strchr("bdgvzjlmnr", (int)(nx < 128 ? nx : 0)) && nx) PUSH("z");
			else PUSH("s");
			break;
		case 'x': PUSH(atstart || !cp_pt_isv(pv) ? "\xca\x83" : (nx == 0 ? "ks" : "\xca\x83")); break;
		case 'y': VOW('i', 0); break;
		case 'z': PUSH(nx ? "z" : "s"); break;
		default: break;
		}
	}
	/* stressed nucleus */
	int nv = 0, stress = -1, last = -1, prev = -1;
	for (int i = 0; i < nt; i++) if (t[i].v) { nv++; prev = last; last = i; if (stress < 0 && t[i].acc) stress = i; }
	if (stress < 0 && nv) {
		int wl = n; uint32_t e1 = wl > 0 ? g[wl - 1] : 0, e2 = wl > 1 ? g[wl - 2] : 0;
		int pen = (e1 == 'a' || e1 == 'e' || e1 == 'o' || (e1 == 's' && (e2 == 'a' || e2 == 'e' || e2 == 'o')) || e1 == 'm' || (e1 == 's' && e2 == 'n'));
		if (!(e1 == 'm' && !(e2 == 'a' || e2 == 'e' || e2 == 'o' || e2 == 'u')) ) { }
		/* words ending in -ão/-ões/-i/-u/-l/-r/-z are oxytone; -a/-e/-o/-am/-em/-as/-es/-os/-ens paroxytone */
		if (e1 == 'i' || e1 == 'u' || e1 == 'l' || e1 == 'r' || e1 == 'z' || (e1 == 's' && (e2 == 'i' || e2 == 'u' || e2 == 'l' || e2 == 'r' || e2 == 'z')) || (e1 == 'm' && e2 == 'i')) pen = 0;
		if (nv > 1 && pen && prev >= 0) stress = prev; else stress = last;
	}
	char buf[256]; buf[0] = 0; size_t bl = 0;
	#define APPEND(str) do { size_t l_ = strlen(str); if (bl + l_ < sizeof buf) { memcpy(buf + bl, str, l_); bl += l_; buf[bl] = 0; } } while (0)
	for (int i = 0; i < nt; i++) {
		if (!t[i].v) {
			/* palatalization of t/d before i (incl. final unstressed e -> i) */
			if (t[i].k == 't' || t[i].k == 'd') {
				int j = i + 1; int pal = 0;
				if (j < nt && t[j].v) {
					int fin = 1; for (int q = j + 1; q < nt; q++) if (t[q].v) fin = 0;
					if (t[j].k == 'i' || (t[j].k == 'e' && j != stress && fin && !t[j].nasal)) pal = 1;
				}
				if (pal) { APPEND(t[i].k == 't' ? "t\xca\x83" : "d\xca\x92"); continue; }
			}
			APPEND(t[i].s); continue;
		}
		int fin = 1; for (int q = i + 1; q < nt; q++) if (t[q].v) fin = 0;
		int st = (i == stress);
		const char *v = "";
		if (st) APPEND("\xcb\x88");
		if (t[i].nasal || t[i].acc == 3) {
			switch (t[i].k) {
			case 'a': v = "\xc3\xa3"; break; case 'e': v = "\xe1\xba\xbd"; break; case 'i': v = "\xc4\xa9"; break;
			case 'o': v = "\xc3\xb5"; break; default: v = "\xc5\xa9"; break;
			}
			APPEND(v);
			if (t[i].acc == 3 && i + 1 < nt && !t[i + 1].v && strcmp(t[i + 1].s, "w") && strcmp(t[i + 1].s, "j")) { /* plain ã */ }
			continue;
		}
		switch (t[i].k) {
		case 'a': v = (t[i].acc == 2 || (!st && fin)) ? "\xc9\x90" : "a"; break;
		case 'e': v = t[i].acc == 1 ? "\xc9\x9b" : (!st && fin) ? "i" : "e"; break;
		case 'i': v = "i"; break;
		case 'o': v = t[i].acc == 1 ? "\xc9\x94" : (!st && fin) ? "u" : "o"; break;
		default: v = "u"; break;
		}
		APPEND(v);
	}
	snprintf(out, cap, "%s", buf);
}


#endif
