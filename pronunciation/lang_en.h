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

static void cp_en_word(const char *word, char *out, size_t cap)
{
	for (size_t i = 0; i < sizeof cp_en_lex / sizeof cp_en_lex[0]; i++)
		if (!strcmp(word, cp_en_lex[i][0])) { snprintf(out, cap, "%s", cp_en_lex[i][1]); return; }
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
