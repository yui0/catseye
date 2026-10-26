#include <stdio.h>
#include <string.h>
#include "latphon.h"

static int fails;
static void chk(int lang, const char *in, const char *want)
{
	char o[1024];
	cp_g2p(lang, in, o, sizeof o);
	if (strcmp(o, want)) { printf("FAIL [%s] -> %s (want %s)\n", in, o, want); fails++; }
}

int main(void)
{
	chk(CP_EN, "hello world", "hɛˈloʊ wˈɚld");
	chk(CP_EN, "the quick brown fox", "ðə kwˈɪk bɹˈaʊn fˈɑks");
	chk(CP_EN, "make that cake", "mˈeɪk ðˈæt kˈeɪk");
	chk(CP_EN, "knight", "nˈaɪt");
	chk(CP_EN, "42", "fˈɔɹti tˈu");
	chk(CP_EN, "Yes, please.", "jˈɛs , plˈiz .");
	chk(CP_PT, "olá mundo", "olˈa mˈũdu");
	chk(CP_PT, "obrigado", "obɾigˈadu");
	chk(CP_PT, "não", "nˈãu");
	chk(CP_PT, "ninho filho", "nˈiɲu fˈiʎu");
	chk(CP_PT, "cidade", "sidˈadʒi");
	chk(CP_PT, "o rato roeu a roupa", "u ʁˈatu ʁoˈew ɐ ʁˈowpɐ");
	chk(CP_PT, "21", "vˈĩtʃi i ũ");
	chk(CP_JA, "こんにちは", "koɲɲitɕiwa");
	chk(CP_JA, "ありがとうございます", "aɾigatoːgozaimasɯ");
	chk(CP_JA, "しんぶん", "ɕimbɯɴ");
	chk(CP_JA, "コーヒー", "koːçiː");
	chk(CP_JA, "ちょっと、まって。", "tɕotto , matte .");
	chk(CP_JA, "2024", "ɲiseɲɲidʑɯːjoɴ");
	if (cp_lang("ja") != CP_JA || cp_lang("pt-BR") != CP_PT || cp_lang("xx") != -1) { puts("FAIL cp_lang"); fails++; }
	if (fails) { printf("%d failure(s)\n", fails); return 1; }
	puts("g2p ok");
	return 0;
}
