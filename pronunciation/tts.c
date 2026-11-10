/* tts - English / Portuguese text-to-speech. Engines: klatt (built in), piper (VITS ONNX), kokoro and qwen (GGUF). */
#include <stdio.h>
#include <string.h>
#include "latphon.h"
#include "ja_reader.h"
#include "klatt.h"
#include "tts_piper.h"
#include "tts_kokoro.h"
#include "tts_qwen.h"

static void usage(void)
{
	fprintf(stderr,
		"usage: tts [-e klatt|piper|kokoro|qwen] -L en|pt|ja (-t TEXT | -f FILE) [-o out.wav] [-s SPEED] [-p PITCH_HZ] [--ipa]\n"
		"  -m DIR  model directory (piper: en.onnx / pt.onnx; kokoro: kokoro-q8.gguf + VOICE.bin)\n  -v NAME piper: voice file stem in -m DIR (e.g. en_high); kokoro: voice (default per language: af_heart / pf_dora / jf_alpha)\n  --f32   kokoro: dequantize weights to float32 (slower, closer to the original)\n  --ipa   print the phoneme string only\n  --noise X --noise-w Y  piper sampling noise (defaults 0.667 / 0.8 from the voice json; lower = steadier, less lively)\n  --from-ipa  the text is already IPA (skip G2P; e.g. espeak output from tools/espeak_ipa.py)\n  qwen: raw text in, no G2P; needs qwen-talker-0.6b-base-Q8_0.gguf + qwen-tokenizer-12hz-Q8_0.gguf in -m DIR; --seed N, --greedy\n");
}

int main(int argc, char **argv)
{
	const char *mdir = "models", *engine = "klatt", *lang = "en", *text = NULL, *file = NULL, *outp = "out.wav";
	const char *voice = NULL; float noise = -1, noise_w = -1; int ipa_only = 0, from_ipa = 0, q8 = 1, greedy = 0; unsigned long long seed = 0; ck_opt opt = ck_default();
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
		else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
		else if (!strcmp(argv[i], "--greedy")) greedy = 1;
		else if (!strcmp(argv[i], "--noise") && i + 1 < argc) noise = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--noise-w") && i + 1 < argc) noise_w = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--ipa")) ipa_only = 1;
		else if (!strcmp(argv[i], "--from-ipa")) from_ipa = 1;
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
	if (strcmp(engine, "klatt") && strcmp(engine, "piper") && strcmp(engine, "kokoro") && strcmp(engine, "qwen")) { fprintf(stderr, "engine '%s' is not available (klatt, piper, kokoro, qwen)\n", engine); return 2; }
	int l = cp_lang(lang);
	if (l < 0) { fprintf(stderr, "unknown language '%s'\n", lang); return 2; }
	cp_ja_reader_enable(argv[0]);
	if (strcmp(engine, "qwen")) cp_en_lexicon_autoload(mdir);	/* optional, see tools/make_lexicon_en.py */
	if (!strcmp(engine, "qwen")) {
		static const char *names[] = { "english", "portuguese", "japanese" };
		const char *qlang = !strcmp(lang, "en") ? names[0] : !strcmp(lang, "pt") ? names[1] : !strcmp(lang, "ja") ? names[2] : lang;
		char tk[512], cd[512];
		snprintf(tk, sizeof tk, "%s/qwen-talker-0.6b-base-Q8_0.gguf", mdir); snprintf(cd, sizeof cd, "%s/qwen-tokenizer-12hz-Q8_0.gguf", mdir);
		qw_tts *qt = (qw_tts *)calloc(1, sizeof *qt);
		if (qw_open(qt, tk, cd, q8)) { fprintf(stderr, "cannot open Qwen3-TTS models %s / %s\n", tk, cd); return 1; }
		if (seed) qt->rng = seed * 0x9E3779B97F4A7C15ull | 1;
		qw_params P = qw_default_params();
		if (greedy) P.temp = P.sub_temp = 0;
		qw_gen g = { 0 };
		int nf = qw_generate(qt, text, qlang, &P, &g);
		if (nf == -2) { fprintf(stderr, "language '%s' is not known to the model\n", qlang); return 2; }
		if (nf <= 0) { fprintf(stderr, "qwen generated no audio\n"); return 1; }
		float *wav; int ns = qw_codec_decode(&qt->codec, g.codes, nf, &wav);
		if (ck_write_wav(outp, wav, ns, QW_RATE)) { perror(outp); return 1; }
		fprintf(stderr, "%d frames -> %s (%.2fs)\n", nf, outp, (double)ns / QW_RATE);
		return 0;
	}
	size_t cap = strlen(text) * 24 + 256;
	char *ipa = (char *)malloc(cap);
	if (from_ipa) { snprintf(ipa, cap, "%s", text); } else cp_g2p(l, text, ipa, cap);
	if (ipa_only) { puts(ipa); return 0; }
	float *pcm; size_t n; int rate = CK_RATE;
	if (!strcmp(engine, "piper")) {
		char onnx[512], json[520];
		if (!cp_langs[l].piper) { fprintf(stderr, "no Piper voice for %s\n", cp_langs[l].name); return 2; }
		snprintf(onnx, sizeof onnx, "%s/%s.onnx", mdir, voice ? voice : cp_langs[l].piper); snprintf(json, sizeof json, "%s.json", onnx);
		cpp_voice pv;
		if (cpp_open(&pv, onnx, json)) { fprintf(stderr, "cannot open Piper voice %s\n", onnx); return 1; }
		if (noise >= 0) pv.noise = noise;
		if (noise_w >= 0) pv.noise_w = noise_w;
		if (cpp_synth(&pv, ipa, opt.speed, &pcm, &n)) { fprintf(stderr, "piper synthesis failed\n"); return 1; }
		rate = pv.rate; cpp_close(&pv);
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
