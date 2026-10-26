/*
 * tts_piper.h - Piper (VITS) voices on the native ONNX runtime (../transcript/cats_onnx.h).
 *
 * cpp_open() reads voice.onnx and voice.onnx.json; cpp_synth() turns the IPA
 * string from cp_g2p() into PCM. Phoneme ids follow Piper's espeak layout:
 * ^ then (phoneme, pad)* then $. Both voices were trained on espeak phonemes,
 * so a few of our symbols are respelled (g -> ɡ, nasal vowels -> base + U+0303).
 */
#ifndef CATS_TTS_PIPER_H
#define CATS_TTS_PIPER_H

#include "phon.h"
#include "../transcript/cats_onnx.h"

typedef struct {
	cx_graph g;
	uint32_t *key; int *val; int nmap;	/* single code point -> id */
	int pad, bos, eos;
	float noise, length, noise_w;
	int rate, speakers;
} cpp_voice;

static inline int cpp_json_num(const char *js, const char *key, float *out)
{
	char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
	const char *p = strstr(js, pat);
	if (!p) return -1;
	p = strchr(p + strlen(pat), ':');
	if (!p) return -1;
	*out = (float)atof(p + 1);
	return 0;
}

static inline int cpp_open(cpp_voice *v, const char *onnx, const char *json)
{
	memset(v, 0, sizeof *v);
	FILE *f = fopen(json, "rb");
	if (!f) return -1;
	fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
	char *js = (char *)malloc(n + 1);
	if (fread(js, 1, n, f) != (size_t)n) { fclose(f); free(js); return -1; }
	js[n] = 0; fclose(f);
	v->noise = 0.667f; v->length = 1.0f; v->noise_w = 0.8f; v->rate = 22050; v->speakers = 1;
	float t;
	if (!cpp_json_num(js, "noise_scale", &t)) v->noise = t;
	if (!cpp_json_num(js, "length_scale", &t)) v->length = t;
	if (!cpp_json_num(js, "noise_w", &t)) v->noise_w = t;
	if (!cpp_json_num(js, "sample_rate", &t)) v->rate = (int)t;
	if (!cpp_json_num(js, "num_speakers", &t)) v->speakers = (int)t;
	const char *m = strstr(js, "\"phoneme_id_map\"");
	if (!m) { free(js); return -1; }
	m = strchr(m, '{');
	int cap = 256; v->key = (uint32_t *)malloc(cap * sizeof(uint32_t)); v->val = (int *)malloc(cap * sizeof(int));
	for (const char *p = m + 1; *p;) {
		while (*p && *p != '"' && *p != '}') p++;
		if (*p != '"') break;
		p++;
		char key[16]; int kl = 0;
		while (*p && *p != '"') {
			if (*p == '\\') {
				p++;
				if (*p == 'u') {
					unsigned c = 0; sscanf(p + 1, "%4x", &c); p += 5;
					if (c < 0x80) key[kl++] = (char)c;
					else if (c < 0x800) { key[kl++] = (char)(0xC0 | c >> 6); key[kl++] = (char)(0x80 | (c & 63)); }
					else { key[kl++] = (char)(0xE0 | c >> 12); key[kl++] = (char)(0x80 | ((c >> 6) & 63)); key[kl++] = (char)(0x80 | (c & 63)); }
					continue;
				}
				key[kl++] = *p == 'n' ? '\n' : *p; p++;
			} else if (kl < 12) key[kl++] = *p++; else p++;
		}
		key[kl] = 0; p++;
		p = strchr(p, '[');
		if (!p) break;
		int id = atoi(p + 1);
		uint32_t cp; cp_utf8_decode((const unsigned char *)key, &cp);
		if (v->nmap == cap) { cap *= 2; v->key = (uint32_t *)realloc(v->key, cap * sizeof(uint32_t)); v->val = (int *)realloc(v->val, cap * sizeof(int)); }
		v->key[v->nmap] = cp; v->val[v->nmap++] = id;
		p = strchr(p, ']'); if (!p) break;
		p++;
	}
	free(js);
	v->pad = v->bos = v->eos = -1;
	for (int i = 0; i < v->nmap; i++) { if (v->key[i] == '_') v->pad = v->val[i]; if (v->key[i] == '^') v->bos = v->val[i]; if (v->key[i] == '$') v->eos = v->val[i]; }
	if (v->pad < 0 || v->bos < 0 || v->eos < 0) return -1;
	if (cx_graph_open(&v->g, onnx)) return -1;
	return 0;
}

static inline void cpp_close(cpp_voice *v) { cx_graph_close(&v->g); free(v->key); free(v->val); memset(v, 0, sizeof *v); }

static inline int cpp_lookup(const cpp_voice *v, uint32_t cp)
{
	for (int i = 0; i < v->nmap; i++) if (v->key[i] == cp) return v->val[i];
	return -1;
}

/* our IPA -> Piper phoneme ids. Returns count. */
static inline int cpp_ids(const cpp_voice *v, const char *ipa, int *ids, int cap)
{
	int n = 0;
	ids[n++] = v->bos;
	for (const unsigned char *p = (const unsigned char *)ipa; *p && n + 3 < cap;) {
		uint32_t c; p += cp_utf8_decode(p, &c);
		uint32_t seq[2] = {c, 0};
		switch (c) {
		case 'g': seq[0] = 0x261; break;
		case 0xE3: seq[0] = 'a'; seq[1] = 0x303; break;
		case 0x1EBD: seq[0] = 'e'; seq[1] = 0x303; break;
		case 0x129: seq[0] = 'i'; seq[1] = 0x303; break;
		case 0xF5: seq[0] = 'o'; seq[1] = 0x303; break;
		case 0x169: seq[0] = 'u'; seq[1] = 0x303; break;
		}
		for (int k = 0; k < 2 && seq[k]; k++) {
			int id = cpp_lookup(v, seq[k]);
			if (id < 0) continue;
			ids[n++] = id; ids[n++] = v->pad;
		}
	}
	ids[n++] = v->eos;
	return n;
}

static inline int cpp_synth(cpp_voice *v, const char *ipa, float speed, float **pcm, size_t *count)
{
	int ids[1024], n = cpp_ids(v, ipa, ids, 1024);
	cx_env env = {0};
	int64_t d2[2] = {1, n}, d1[1] = {1}, ds[1] = {3};
	cx_tensor *in = cx_env_new(&env, v->g.input[0], d2, 2, 7);
	cx_tensor *ln = cx_env_new(&env, v->g.input[1], d1, 1, 7);
	cx_tensor *sc = cx_env_new(&env, v->g.input[2], ds, 1, 1);
	if (!in || !ln || !sc) { cx_env_free(&env); return -1; }
	for (int i = 0; i < n; i++) in->v[i] = (float)ids[i];
	ln->v[0] = (float)n;
	sc->v[0] = v->noise; sc->v[1] = v->length / (speed > 0.1f ? speed : 1.0f); sc->v[2] = v->noise_w;
	if (v->g.ninput > 3) { cx_tensor *sid = cx_env_new(&env, v->g.input[3], d1, 1, 7); if (!sid) { cx_env_free(&env); return -1; } sid->v[0] = 0; }
	if (cx_graph_run_compact(&v->g, &env)) { cx_env_free(&env); return -1; }
	cx_tensor *y = cx_env_find(&env, v->g.output[0]);
	if (!y || !y->n) { cx_env_free(&env); return -1; }
	*pcm = (float *)malloc(y->n * sizeof(float));
	memcpy(*pcm, y->v, y->n * sizeof(float));
	*count = y->n;
	cx_env_free(&env);
	return 0;
}

#endif
