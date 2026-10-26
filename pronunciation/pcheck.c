/*
 * pcheck - pronunciation check against an expected text.
 *
 *   pcheck -L en|pt|ja -t TEXT -z vocab.txt --post learner.cpost [--json]
 *   pcheck -L en|pt|ja -t TEXT -z vocab.txt -w learner.wav        (runs tools/phoneme_logits.py)
 *
 * .cpost: "CPST", u32 T, u32 V, then T*V float32 logits (CTC phoneme recognizer output).
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "pcheck.h"
#include "ja_reader.h"

int main(int argc, char **argv)
{
	const char *lang = "en", *text = NULL, *vocab = NULL, *post = NULL, *wav = NULL;
	int json = 0;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-L") && i + 1 < argc) lang = argv[++i];
		else if (!strcmp(argv[i], "-t") && i + 1 < argc) text = argv[++i];
		else if (!strcmp(argv[i], "-z") && i + 1 < argc) vocab = argv[++i];
		else if (!strcmp(argv[i], "--post") && i + 1 < argc) post = argv[++i];
		else if (!strcmp(argv[i], "-w") && i + 1 < argc) wav = argv[++i];
		else if (!strcmp(argv[i], "--json")) json = 1;
		else { text = NULL; break; }
	}
	if (!text || !vocab || (!post && !wav)) {
		fprintf(stderr, "usage: pcheck -L en|pt|ja -t TEXT -z vocab.txt (--post learner.cpost | -w learner.wav) [--json]\n");
		return 2;
	}
	char tmp[] = "/tmp/pcheckXXXXXX";
	if (!post) {
		int fd = mkstemp(tmp);
		if (fd < 0) { perror("mkstemp"); return 1; }
		close(fd);
		char cmd[2048];
		/* locate the tool next to this executable */
		char *dir = strdup(argv[0]); char *sl = strrchr(dir, '/'); if (sl) *sl = 0; else strcpy(dir, ".");
		snprintf(cmd, sizeof cmd, "python3 '%s/tools/phoneme_logits.py' --wav '%s' --out '%s' --vocab '%s'", dir, wav, tmp, vocab);
		free(dir);
		if (system(cmd)) { fprintf(stderr, "phoneme_logits.py failed\n"); remove(tmp); return 1; }
		post = tmp;
	}
	int lg = cp_lang(lang);
	if (lg < 0) { fprintf(stderr, "unknown language '%s'\n", lang); return 2; }
	cp_ja_reader_enable(argv[0]);
	cpa_vocab v;
	if (cpa_vocab_load(&v, vocab)) { perror(vocab); return 1; }
	FILE *f = fopen(post, "rb");
	if (!f) { perror(post); return 1; }
	char magic[4]; unsigned T, V;
	if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CPST", 4) || fread(&T, 4, 1, f) != 1 || fread(&V, 4, 1, f) != 1 || (int)V != v.n) {
		fprintf(stderr, "%s: bad posterior file (vocab size %d)\n", post, v.n); return 1;
	}
	float *lp = (float *)malloc(sizeof(float) * (size_t)T * V);
	if (fread(lp, sizeof(float), (size_t)T * V, f) != (size_t)T * V) { fprintf(stderr, "short read\n"); return 1; }
	fclose(f);
	if (post == tmp) remove(tmp);
	cpc_report r;
	if (cpc_check(lg, text, &v, lp, (int)T, &r)) { fprintf(stderr, "no phonemes in text\n"); return 1; }
	if (json) cpc_json(&r, &v, stdout); else cpc_print(&r, &v, stdout);
	cpc_free(&r); free(lp); cpa_vocab_free(&v);
	return 0;
}
