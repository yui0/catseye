/* tts - English / Portuguese text-to-speech. Engines: klatt (built in), piper (VITS ONNX). */
#include <stdio.h>
#include <string.h>
#include "latphon.h"
#include "ja_reader.h"
#include "klatt.h"
#include "tts_piper.h"
#include "tts_kokoro.h"

static void usage(void)
{
	fprintf(stderr,
		"usage: tts [-e klatt|piper|kokoro] -L en|pt|ja (-t TEXT | -f FILE) [-o out.wav] [-s SPEED] [-p PITCH_HZ] [--ipa]\n"
		"  -m DIR  model directory (piper: en.onnx / pt.onnx; kokoro: kokoro-q8.gguf + VOICE.bin)\n  -v NAME kokoro voice (default per language: af_heart / pf_dora / jf_alpha)\n  --f32   kokoro: dequantize weights to float32 (slower, closer to the original)\n  --ipa   print the phoneme string only\n");
}

int main(int argc, char **argv)
{
	const char *mdir = "models", *engine = "klatt", *lang = "en", *text = NULL, *file = NULL, *outp = "out.wav";
	const char *voice = NULL; int ipa_only = 0, q8 = 1; ck_opt opt = ck_default();
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-e") && i + 1 < argc) engine = argv[++i];
		else if (!strcmp(argv[i], "-L") && i + 1 < argc) lang = argv[++i];
		else if (!strcmp(argv[i], "-t") && i + 1 < argc) text = argv[++i];
		else if (!strcmp(argv[i], "-f") && i + 1 < argc) file = argv[++i];
		else if (!strcmp(argv[i], "-m") && i + 1 < argc) mdir = argv[++i];
		else if (!strcmp(argv[i], "-o") && i + 1 < argc) outp = argv[++i];
		else if (!strcmp(argv[i], "-s") && i + 1 < argc) opt.speed = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "-p") && i + 1 < argc) opt.f0 = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "-v") && i + 1 < argc) voice = argv[++i];
		else if (!strcmp(argv[i], "--f32")) q8 = 0;
		else if (!strcmp(argv[i], "--ipa")) ipa_only = 1;
		else { usage(); return 2; }
	}
	char *buf = NULL;
	if (file) {
		FILE *f = fopen(file, "rb");
		if (!f) { perror(file); return 1; }
		fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
		buf = (char *)malloc(n + 1);
		if (fread(buf, 1, n, f) != (size_t)n) { fclose(f); return 1; }
		buf[n] = 0; fclose(f); text = buf;
	}
	if (!text) { usage(); return 2; }
	if (strcmp(engine, "klatt") && strcmp(engine, "piper") && strcmp(engine, "kokoro")) { fprintf(stderr, "engine '%s' is not available (klatt, piper, kokoro)\n", engine); return 2; }
	int l = cp_lang(lang);
	if (l < 0) { fprintf(stderr, "unknown language '%s'\n", lang); return 2; }
	cp_ja_reader_enable(argv[0]);
	size_t cap = strlen(text) * 24 + 256;
	char *ipa = (char *)malloc(cap);
	cp_g2p(l, text, ipa, cap);
	if (ipa_only) { puts(ipa); return 0; }
	float *pcm; size_t n; int rate = CK_RATE;
	if (!strcmp(engine, "piper")) {
		char onnx[512], json[520];
		if (!cp_langs[l].piper) { fprintf(stderr, "no Piper voice for %s\n", cp_langs[l].name); return 2; }
		snprintf(onnx, sizeof onnx, "%s/%s.onnx", mdir, cp_langs[l].piper); snprintf(json, sizeof json, "%s.json", onnx);
		cpp_voice voice;
		if (cpp_open(&voice, onnx, json)) { fprintf(stderr, "cannot open Piper voice %s\n", onnx); return 1; }
		if (cpp_synth(&voice, ipa, opt.speed, &pcm, &n)) { fprintf(stderr, "piper synthesis failed\n"); return 1; }
		rate = voice.rate; cpp_close(&voice);
	} else if (!strcmp(engine, "kokoro")) {
		char gguf[512], vb[512];
		const char *stem = voice ? voice : cp_langs[l].kokoro;
		snprintf(gguf, sizeof gguf, "%s/kokoro-q8.gguf", mdir); snprintf(vb, sizeof vb, "%s/%s.bin", mdir, stem);
		static kk_voice kv;
		if (kk_voice_open(&kv, gguf, vb, q8)) { fprintf(stderr, "cannot open Kokoro model %s / voice %s\n", gguf, vb); return 1; }
		if (kk_synth(&kv, ipa, opt.speed, &pcm, &n)) { fprintf(stderr, "kokoro synthesis failed\n"); return 1; }
		rate = KK_RATE;
	} else if (ck_synth(ipa, opt, &pcm, &n)) { fprintf(stderr, "nothing to say\n"); return 1; }
	if (ck_write_wav(outp, pcm, n, rate)) { perror(outp); return 1; }
	fprintf(stderr, "%s -> %s (%.2fs)\n", ipa, outp, (double)n / rate);
	free(pcm); free(ipa); free(buf);
	return 0;
}
