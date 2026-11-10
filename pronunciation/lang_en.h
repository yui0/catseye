/*
 * lang_en.h - English (American) grapheme-to-phoneme: NRL letter-to-sound rules + exception lexicon
 * Part of the language registry in latphon.h: one lower-case word in, IPA out.
 */
#ifndef CATS_LANG_EN_H
#define CATS_LANG_EN_H

#include "phon.h"
#include <ctype.h>


static const char *cp_en_rules[][4] = {
	/* A */
	{" ","A"," ","ə"}, {" ","ARE"," ","ɑɹ"}, {"","AR","O","əɹ"}, {"","AR","#","ɛɹ"}, {"^","AS","#","eɪs"},
	{"","A","WA","ə"}, {"","AW","","ɔ"}, {" :","ANY","","ɛni"}, {"","A","^+#","eɪ"}, {"#:","ALLY","","əli"},
	{" ","AL","#","əl"}, {"","AGAIN","","əgɛn"}, {"#:","AG","E","ɪdʒ"}, {"","A","^+:#","æ"}, {" :","A","^+ ","eɪ"},
	{"","A","^%","eɪ"}, {" ","ARR","","əɹ"}, {"","ARR","","æɹ"}, {" :","AR"," ","ɑɹ"}, {"","AR"," ","ɚ"},
	{"","AR","","ɑɹ"}, {"","AIR","","ɛɹ"}, {"","AI","","eɪ"}, {"","AY","","eɪ"}, {"","AU","","ɔ"},
	{"#:","AL"," ","əl"}, {"","ALK","","ɔk"}, {"","AL","^","ɔl"}, {" :","ABLE","","eɪbəl"}, {"","ABLE","","əbəl"},
	{"","ANG","+","eɪndʒ"}, {"","A","","æ"},
	/* B */
	{" ","BE","^#","bɪ"}, {"","BEING","","biɪŋ"}, {" ","BOTH"," ","boʊθ"}, {" ","BUS","#","bɪz"}, {"","BUIL","","bɪl"}, {"","B","","b"},
	/* C */
	{" ","CH","^","k"}, {"^E","CH","","k"}, {"","CH","","tʃ"}, {" S","CI","#","saɪ"}, {"","CI","A","ʃ"}, {"","CI","O","ʃ"},
	{"","CI","EN","ʃ"}, {"","C","+","s"}, {"","CK","","k"}, {"","COM","%","kʌm"}, {"","C","","k"},
	/* D */
	{"#:","DED"," ","dɪd"}, {".E","D"," ","d"}, {"#^:E","D"," ","t"}, {" ","DE","^#","dɪ"}, {" ","DO"," ","du"},
	{" ","DOES","","dʌz"}, {" ","DOING","","duɪŋ"}, {" ","DOW","","daʊ"}, {"","DU","A","dʒu"}, {"","D","","d"},
	/* E */
	{"#:","E"," ",""}, {"':^","E"," ",""}, {" :","E"," ","i"}, {"#","ED"," ","d"}, {"#:","E","D ",""}, {"","EV","ER","ɛv"},
	{"","E","^%","i"}, {"","ERI","#","ɪɹi"}, {"","ERI","","ɛɹɪ"}, {"#:","ER","#","ɚ"}, {"","ER","#","ɛɹ"}, {"","ER","","ɚ"},
	{" ","EVEN","","ivɛn"}, {"#:","E","W",""}, {"@","EW","","u"}, {"","EW","","ju"}, {"","E","O","i"}, {"#:&","ES"," ","ɪz"},
	{"#:","E","S ",""}, {"#:","ELY"," ","li"}, {"#:","EMENT","","mɛnt"}, {"","EFUL","","fʊl"}, {"","EE","","i"},
	{"","EARN","","ɚn"}, {" ","EAR","^","ɚ"}, {"","EAD","","ɛd"}, {"#:","EA"," ","iə"}, {"","EA","SU","ɛ"}, {"","EA","","i"},
	{"","EIGH","","eɪ"}, {"","EI","","i"}, {" ","EYE","","aɪ"}, {"","EY","","i"}, {"","EU","","ju"}, {"","E","","ɛ"},
	/* F */
	{"","FUL","","fʊl"}, {"","F","","f"},
	/* G */
	{"","GIV","","gɪv"}, {" ","G","I^","g"}, {"","GE","T","gɛ"}, {"SU","GGES","","gdʒɛs"}, {"","GG","","g"}, {" B#","G","","g"},
	{"","G","+","dʒ"}, {"","GREAT","","gɹeɪt"}, {"#","GH","",""}, {"","G","","g"},
	/* H */
	{" ","HAV","","hæv"}, {" ","HERE","","hɪɹ"}, {" ","HOUR","","aʊɚ"}, {"","HOW","","haʊ"}, {"","H","#","h"}, {"","H","",""},
	/* I */
	{" ","IN","","ɪn"}, {" ","I"," ","aɪ"}, {"","IN","D","aɪn"}, {"","IER","","iɚ"}, {"#:R","IED","","id"}, {"","IED"," ","aɪd"},
	{"","IEN","","iɛn"}, {"","IE","T","aɪɛ"}, {" :","I","%","aɪ"}, {"","I","%","i"}, {"","IE","","i"}, {"","I","^+:#","ɪ"},
	{"","IR","#","aɪɹ"}, {"","IZ","%","aɪz"}, {"","IS","%","aɪz"}, {"","I","D%","aɪ"}, {"+^","I","^+","ɪ"}, {"","I","T%","aɪ"},
	{"#^:","I","^+","ɪ"}, {"","I","^+","aɪ"}, {"","IR","","ɚ"}, {"","IGH","","aɪ"}, {"","ILD","","aɪld"}, {"","IGN"," ","aɪn"},
	{"","IGN","^","aɪn"}, {"","IGN","%","aɪn"}, {"","IQUE","","ik"}, {"","I","","ɪ"},
	/* J K L M N */
	{"","J","","dʒ"}, {" ","K","N",""}, {"","K","","k"},
	{"","LO","C#","loʊ"}, {"L","L","",""}, {"#^:","L","%","əl"}, {"","LEAD","","lid"}, {"","L","","l"},
	{"","MOV","","muv"}, {"","M","","m"},
	{"E","NG","+","ndʒ"}, {"","NG","R","ŋg"}, {"","NG","#","ŋg"}, {"","NGL","%","ŋgəl"}, {"","NG","","ŋ"}, {"","NK","","ŋk"},
	{" ","NOW"," ","naʊ"}, {"","N","","n"},
	/* O */
	{"","OF"," ","əv"}, {"","OROUGH","","ɚoʊ"}, {"#:","OR"," ","ɚ"}, {"#:","ORS"," ","ɚz"}, {"","OR","","ɔɹ"}, {" ","ONE","","wʌn"},
	{"","OW","","oʊ"}, {" ","OVER","","oʊvɚ"}, {"","OV","","ʌv"}, {"","O","^%","oʊ"}, {"","O","^EN","oʊ"}, {"","O","^I#","oʊ"},
	{"","OL","D","oʊl"}, {"","OUGHT","","ɔt"}, {"","OUGH","","ʌf"}, {" ","OU","","aʊ"}, {"H","OU","S#","aʊ"}, {"","OUS","","əs"},
	{"","OUR","","ɔɹ"}, {"","OULD","","ʊd"}, {"^","OU","^L","ʌ"}, {"","OUP","","up"}, {"","OU","","aʊ"}, {"","OY","","ɔɪ"},
	{"","OING","","oʊɪŋ"}, {"","OI","","ɔɪ"}, {"","OOR","","ɔɹ"}, {"","OOK","","ʊk"}, {"","OOD","","ʊd"}, {"","OO","","u"},
	{"","O","E","oʊ"}, {"","O"," ","oʊ"}, {"","OA","","oʊ"}, {" ","ONLY","","oʊnli"}, {" ","ONCE","","wʌns"}, {"","ON'T","","oʊnt"},
	{"C","O","N","ɑ"}, {"","O","NG","ɔ"}, {" :^","O","N","ʌ"}, {"I","ON","","ən"}, {"#:","ON"," ","ən"}, {"#^","ON","","ən"},
	{"","O","ST ","oʊ"}, {"","OF","^","ɔf"}, {"","OTHER","","ʌðɚ"}, {"","OSS"," ","ɔs"}, {"#^:","OM","","ʌm"}, {"","O","","ɑ"},
	/* P Q R */
	{"","PH","","f"}, {"","PEOP","","pip"}, {"","POW","","paʊ"}, {"","PUT"," ","pʊt"}, {"","P","","p"},
	{"","QUAR","","kwɔɹ"}, {"","QU","","kw"}, {"","Q","","k"},
	{" ","RE","^#","ɹi"}, {"","R","","ɹ"},
	/* S */
	{"","SH","","ʃ"}, {"#","SION","","ʒən"}, {"","SOME","","sʌm"}, {"#","SUR","#","ʒɚ"}, {"","SUR","#","ʃɚ"}, {"#","SU","#","ʒu"},
	{"#","SSU","#","ʃu"}, {"#","SED"," ","zd"}, {"#","S","#","z"}, {"","SAID","","sɛd"}, {"^","SION","","ʃən"}, {"","S","S",""},
	{".","S"," ","z"}, {"#:.E","S"," ","z"}, {"U","S"," ","s"}, {" :#","S"," ","z"}, {"","SCH","","sk"}, {"","S","C+",""},
	{"#","SM","","zm"}, {"","S","","s"},
	/* T */
	{" ","THE"," ","ðə"}, {"","TO"," ","tu"}, {"","THAT"," ","ðæt"}, {" ","THIS"," ","ðɪs"}, {" ","THEY","","ðeɪ"},
	{" ","THERE","","ðɛɹ"}, {"","THER","","ðɚ"}, {"","THEIR","","ðɛɹ"}, {" ","THAN"," ","ðæn"}, {" ","THEM"," ","ðɛm"},
	{"","THESE"," ","ðiz"}, {" ","THEN","","ðɛn"}, {"","THROUGH","","θɹu"}, {"","THOSE","","ðoʊz"}, {"","THOUGH"," ","ðoʊ"},
	{" ","THUS","","ðʌs"}, {"","TH","","θ"}, {"#:","TED"," ","tɪd"}, {"S","TI","#N","tʃ"}, {"","TI","O","ʃ"}, {"","TI","A","ʃ"},
	{"","TIEN","","ʃən"}, {"","TUR","#","tʃɚ"}, {"","TU","A","tʃu"}, {" ","TWO","","tu"}, {"","T","","t"},
	/* U V W X Y Z */
	{" ","UN","I","jun"}, {" ","UN","","ʌn"}, {" ","UPON","","əpɔn"}, {"@","UR","#","ʊɹ"}, {"","UR","#","jʊɹ"}, {"","UR","","ɚ"},
	{"","U","^ ","ʌ"}, {"","U","^^","ʌ"}, {"","UY","","aɪ"}, {" G","U","#",""}, {"G","U","%",""}, {"G","U","#","w"},
	{"#N","U","","ju"}, {"@","U","","u"}, {"","U","","ju"},
	{"","VIEW","","vju"}, {"","V","","v"},
	{" ","WERE","","wɚ"}, {"","WA","S","wɑ"}, {"","WA","T","wɑ"}, {"","WHERE","","wɛɹ"}, {"","WHAT","","wɑt"}, {"","WHOL","","hoʊl"},
	{"","WHO","","hu"}, {"","WH","","w"}, {"","WAR","","wɔɹ"}, {"","WOR","^","wɚ"}, {"","WR","","ɹ"}, {"","W","","w"},
	{" ","X","","z"}, {"","X","","ks"},
	{"","YOUNG","","jʌŋ"}, {" ","YOU","","ju"}, {" ","YES","","jɛs"}, {" ","Y","","j"}, {"#^:","Y"," ","i"}, {"#^:","Y","I","i"},
	{" :","Y"," ","aɪ"}, {" :","Y","#","aɪ"}, {" :","Y","^+:#","ɪ"}, {" :","Y","^#","aɪ"}, {"","Y","","ɪ"},
	{"","Z","","z"},
};
#define CP_EN_NRULES ((int)(sizeof cp_en_rules / sizeof cp_en_rules[0]))

/* words the rules get wrong or that are weak forms (no stress mark added) */
static const char *cp_en_lex[][2] = {
	{"a","ə"}, {"an","ən"}, {"the","ðə"}, {"of","əv"}, {"to","tu"}, {"and","ænd"}, {"in","ɪn"}, {"is","ɪz"}, {"it","ɪt"},
	{"at","æt"}, {"on","ɑn"}, {"for","fɚ"}, {"as","æz"}, {"was","wʌz"}, {"be","bi"}, {"by","baɪ"}, {"or","ɔɹ"}, {"you","ju"},
	{"i","aɪ"}, {"are","ɑɹ"}, {"we","wi"}, {"he","hi"}, {"she","ʃi"}, {"me","mi"}, {"my","maɪ"}, {"do","du"}, {"no","noʊ"},
	{"so","soʊ"}, {"go","goʊ"}, {"the.","ðə"},
	{"hello","hɛˈloʊ"}, {"world","wˈɚld"}, {"one","wˈʌn"}, {"two","tˈu"}, {"four","fˈɔɹ"}, {"eight","ˈeɪt"}, {"zero","zˈiɹoʊ"},
	{"nine","nˈaɪn"}, {"seven","sˈɛvən"}, {"twenty","twˈɛnti"}, {"thirty","θˈɚti"}, {"forty","fˈɔɹti"}, {"fifty","fˈɪfti"},
	{"hundred","hˈʌndɹɪd"}, {"thousand","θˈaʊzənd"}, {"water","wˈɔtɚ"}, {"people","pˈipəl"}, {"said","sˈɛd"}, {"have","hˈæv"},
	{"were","wˈɚ"}, {"what","wˈʌt"}, {"there","ðˈɛɹ"}, {"your","jˈɔɹ"}, {"through","θɹˈu"}, {"thought","θˈɔt"}, {"because","bɪkˈʌz"},
	{"very","vˈɛɹi"}, {"brown","bɹˈaʊn"}, {"down","dˈaʊn"}, {"town","tˈaʊn"}, {"now","nˈaʊ"}, {"how","hˈaʊ"}, {"out","ˈaʊt"}, {"English","ˈɪŋglɪʃ"}, {"Portuguese","pˈɔɹtʃəgˌiz"}, {"portuguese","pˈɔɹtʃəgˌiz"}, {"english","ˈɪŋglɪʃ"},
	{"speech","spˈitʃ"}, {"pronunciation","pɹənˌʌnsiˈeɪʃən"}, {"hear","hˈɪɹ"}, {"good","gˈʊd"}, {"morning","mˈɔɹnɪŋ"},
	{"thank","θˈæŋk"}, {"thanks","θˈæŋks"}, {"please","plˈiz"}, {"sorry","sˈɑɹi"}, {"yes","jˈɛs"}, {"this","ðˈɪs"}, {"that","ðˈæt"},
};

static inline int cp_en_isvowel(int c) { return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'Y'; }
static inline int cp_en_iscons(int c) { return c >= 'A' && c <= 'Z' && !cp_en_isvowel(c); }

static int cp_en_class(int p, int c)
{
	switch (p) {
	case '^': return cp_en_iscons(c);
	case '.': return c && strchr("BDGJLMNRVWZ", c);
	case '+': return c == 'E' || c == 'I' || c == 'Y';
	case '@': return c && strchr("TSRDLZNJ", c);
	case '&': return c && strchr("SCGZXJ", c);
	case ' ': return c == ' ';
	default:  return c == p;
	}
}

/* does pattern pat[0..n) match the text ending at index i (inclusive), going left */
static int cp_en_left(const char *w, int i, const char *pat, int n)
{
	if (!n) return 1;
	int p = pat[n - 1];
	if (p == '#') {
		if (i < 0 || !cp_en_isvowel(w[i])) return 0;
		while (i >= 0 && cp_en_isvowel(w[i])) i--;
		return cp_en_left(w, i, pat, n - 1);
	}
	if (p == ':') {
		while (i >= 0 && cp_en_iscons(w[i])) i--;
		return cp_en_left(w, i, pat, n - 1);
	}
	if (i < 0 || !cp_en_class(p, w[i])) return 0;
	return cp_en_left(w, i - 1, pat, n - 1);
}

static int cp_en_right(const char *w, int j, const char *pat)
{
	if (!*pat) return 1;
	int p = *pat;
	if (p == '#') {
		if (!cp_en_isvowel(w[j])) return 0;
		while (cp_en_isvowel(w[j])) j++;
		return cp_en_right(w, j, pat + 1);
	}
	if (p == ':') {
		while (cp_en_iscons(w[j])) j++;
		return cp_en_right(w, j, pat + 1);
	}
	if (p == '%') {
		static const char *suf[] = {"ING", "ELY", "ER", "ES", "ED", "E"};
		for (int s = 0; s < 6; s++) {
			int l = (int)strlen(suf[s]);
			if (!strncmp(w + j, suf[s], l) && (w[j + l] == ' ' || w[j + l] == '\'') && cp_en_right(w, j + l, pat + 1)) return 1;
		}
		return 0;
	}
	if (!w[j] || !cp_en_class(p, w[j])) return 0;
	return cp_en_right(w, j + 1, pat + 1);
}

/* insert the primary stress mark before the first non-schwa vowel of a bare IPA word */
static void cp_stress(const char *ipa, char *out, size_t cap)
{
	int first = -1, firstany = -1;
	for (const unsigned char *p = (const unsigned char *)ipa; *p;) {
		uint32_t c; int l = cp_utf8_decode(p, &c);
		if (cp_is_ipa_vowel(c)) {
			if (firstany < 0) firstany = (int)(p - (const unsigned char *)ipa);
			if (c != 0x259 && first < 0) first = (int)(p - (const unsigned char *)ipa);
		}
		p += l;
	}
	int at = first >= 0 ? first : firstany;
	if (at < 0 || strchr(ipa, '\xcb')) { snprintf(out, cap, "%s", ipa); return; }
	snprintf(out, cap, "%.*s\xcb\x88%s", at, ipa, ipa + at);
}


/* ------------------------------------------------------------ pronunciation lexicon */

/* models/lexicon_en.txt: "word<TAB>IPA" per line, sorted (tools/make_lexicon_en.py). Loaded on request; the NRL rules below
 * and the small exception table are the fallback when it is absent or a word is not in it. */
typedef struct { const char *w, *ipa; } cp_en_ent;
static cp_en_ent *cp_en_tab; static size_t cp_en_ntab;

static int cp_en_lexicon_load(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) return -1;
	fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
	char *buf = (char *)malloc(n + 1);
	if (!buf || fread(buf, 1, n, f) != (size_t)n) { fclose(f); free(buf); return -1; }
	fclose(f); buf[n] = 0;
	size_t lines = 1; for (long i = 0; i < n; i++) lines += buf[i] == '\n';
	cp_en_tab = (cp_en_ent *)malloc(sizeof(cp_en_ent) * lines); cp_en_ntab = 0;
	for (char *p = buf; *p;) {
		char *e = strchr(p, '\n'); if (e) *e = 0;
		char *t = strchr(p, '\t');
		if (t) { *t = 0; cp_en_tab[cp_en_ntab].w = p; cp_en_tab[cp_en_ntab++].ipa = t + 1; }
		if (!e) break;
		p = e + 1;
	}
	return (int)cp_en_ntab;
}

/* try <dir>/lexicon_en.txt, then ./models/ */
static int cp_en_lexicon_autoload(const char *dir)
{
	char p[600];
	if (cp_en_ntab) return 0;
	if (dir) { snprintf(p, sizeof p, "%s/lexicon_en.txt", dir); if (cp_en_lexicon_load(p) > 0) return 0; }
	return cp_en_lexicon_load("models/lexicon_en.txt") > 0 ? 0 : -1;
}

static const char *cp_en_find(const char *w)
{
	size_t lo = 0, hi = cp_en_ntab;
	while (lo < hi) {
		size_t mid = (lo + hi) / 2; int c = strcmp(cp_en_tab[mid].w, w);
		if (!c) return cp_en_tab[mid].ipa;
		if (c < 0) lo = mid + 1; else hi = mid;
	}
	return NULL;
}

static int cp_en_ends(const char *s, const char *suf) { size_t a = strlen(s), b = strlen(suf); return a >= b && !strcmp(s + a - b, suf); }

/* inflected forms missing from the lexicon: plural / possessive / 3sg, -ed, -ing, -ly built on a known base */
static int cp_en_inflect(const char *word, char *out, size_t cap)
{
	size_t L = strlen(word);
	char base[130]; const char *ip;
	if (L < 4 || L > 120) return 0;
	static const char *sib[] = {"s", "z", "ʃ", "ʒ", "tʃ", "dʒ"}, *vl[] = {"p", "t", "k", "f", "θ"};
	#define TRY(strip, add) do { size_t bl = L - (strip); memcpy(base, word, bl); base[bl] = 0; strcat(base, add); ip = cp_en_find(base); } while (0)
	if (cp_en_ends(word, "'s") || (word[L - 1] == 's' && L > 3)) {
		int poss = cp_en_ends(word, "'s"), es = cp_en_ends(word, "es") && !poss;
		ip = NULL;
		if (poss) TRY(2, "");
		else { TRY(1, ""); if (!ip && es) TRY(2, ""); if (!ip && cp_en_ends(word, "ies")) TRY(3, "y"); }
		if (ip) {
			const char *suf = "z";
			for (int k = 0; k < 6; k++) if (cp_en_ends(ip, sib[k])) suf = "ɪz";
			if (!strcmp(suf, "z")) for (int k = 0; k < 5; k++) if (cp_en_ends(ip, vl[k])) suf = "s";
			snprintf(out, cap, "%s%s", ip, suf); return 1;
		}
	}
	if (cp_en_ends(word, "ed")) {
		ip = NULL; TRY(2, ""); if (!ip) TRY(1, ""); if (!ip && L > 5 && word[L - 3] == word[L - 4]) TRY(3, "");
		if (ip) {
			const char *suf = "d";
			if (cp_en_ends(ip, "t") || cp_en_ends(ip, "d")) suf = "ɪd";
			else for (int k = 0; k < 5; k++) if (cp_en_ends(ip, vl[k])) suf = "t";
			if (cp_en_ends(ip, "s") || cp_en_ends(ip, "ʃ") || cp_en_ends(ip, "tʃ")) suf = "t";
			snprintf(out, cap, "%s%s", ip, suf); return 1;
		}
	}
	if (cp_en_ends(word, "ing")) {
		ip = NULL; TRY(3, ""); if (!ip) TRY(3, "e"); if (!ip && L > 6 && word[L - 4] == word[L - 5]) TRY(4, "");
		if (ip) { snprintf(out, cap, "%sɪŋ", ip); return 1; }
	}
	if (cp_en_ends(word, "ly")) { ip = NULL; TRY(2, ""); if (ip) { snprintf(out, cap, "%sli", ip); return 1; } }
	#undef TRY
	return 0;
}

/* connected-speech tweaks espeak makes: "the" / "to" before a vowel are ðɪ / tʊ */
static void cp_en_post(char *s)
{
	char *o = (char *)malloc(strlen(s) * 2 + 8); size_t k = 0;
	for (char *p = s; *p;) {
		char *e = p; while (*e && *e != ' ') e++;
		size_t wl = (size_t)(e - p);
		const char *rep = NULL;
		if (*e == ' ' && (wl == 3 || wl == 4)) {		/* "ðə" (4 bytes) / "tə" (3 bytes) */
			int the = wl == 4 && !strncmp(p, "ðə", 4), to = wl == 3 && !strncmp(p, "tə", 3);
			if (the || to) {
				const unsigned char *q = (const unsigned char *)e + 1;
				if (!strncmp((const char *)q, "\xcb\x88", 2) || !strncmp((const char *)q, "\xcb\x8c", 2)) q += 2;
				uint32_t c = 0; if (*q) cp_utf8_decode(q, &c);
				if (c && cp_is_ipa_vowel(c)) rep = the ? "ðɪ" : "tʊ";
			}
		}
		if (rep) { size_t rl = strlen(rep); memcpy(o + k, rep, rl); k += rl; }
		else { memcpy(o + k, p, wl); k += wl; }
		if (*e == ' ') o[k++] = ' ';
		p = *e ? e + 1 : e;
	}
	o[k] = 0; strcpy(s, o); free(o);
}

static void cp_en_word(const char *word, char *out, size_t cap)
{
	if (cp_en_ntab) {
		const char *ip = cp_en_find(word);
		if (ip) { snprintf(out, cap, "%s", ip); return; }
	}
	for (size_t i = 0; i < sizeof cp_en_lex / sizeof cp_en_lex[0]; i++)
		if (!strcmp(word, cp_en_lex[i][0])) { snprintf(out, cap, "%s", cp_en_lex[i][1]); return; }
	if (cp_en_ntab) { char inf[256]; if (cp_en_inflect(word, inf, sizeof inf)) { snprintf(out, cap, "%s", inf); return; } }
	char w[130]; int n = 0;
	w[n++] = ' ';
	for (const char *p = word; *p && n < 126; p++) w[n++] = isalpha((unsigned char)*p) ? (char)toupper((unsigned char)*p) : *p == '\'' ? '\'' : 0;
	w[n++] = ' '; w[n] = 0;
	for (int k = 0; k < n; k++) if (!w[k]) w[k] = ' ';
	char buf[256]; buf[0] = 0; size_t bl = 0;
	int i = 1;
	while (i < n - 1) {
		int hit = 0;
		for (int r = 0; r < CP_EN_NRULES; r++) {
			const char *const *R = cp_en_rules[r];
			int ml = (int)strlen(R[1]);
			if (strncmp(w + i, R[1], ml)) continue;
			if (!cp_en_left(w, i - 1, R[0], (int)strlen(R[0]))) continue;
			if (!cp_en_right(w, i + ml, R[2])) continue;
			size_t ol = strlen(R[3]);
			if (bl + ol < sizeof buf) { memcpy(buf + bl, R[3], ol); bl += ol; buf[bl] = 0; }
			i += ml; hit = 1; break;
		}
		if (!hit) i++;
	}
	cp_stress(buf, out, cap);
}


#endif
