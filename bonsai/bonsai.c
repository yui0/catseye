/*
 * bonsai - chat with Ternary-Bonsai-2.5-27B (PrismML, PTQ1_0 GGUF) on the CPU.
 *
 *   make && ./bonsai models/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf -p "Explain quantum computing in simple terms."
 *   ./bonsai MODEL.gguf            # interactive chat (empty line or Ctrl-D quits)
 *
 * The generic parts (PTQ1_0 kernels, Hadamard rotation, BPE tokenizer, sampler) are in ../catseye.h,
 * the qwen35 graph (Gated DeltaNet + gated attention) is in bonsai.h.
 */
#include "vision.h"

static void usage(const char *a0)
{
	fprintf(stderr,
		"usage: %s MODEL.gguf [options]\n"
		"  -p TEXT      one-shot prompt (otherwise interactive chat on stdin)\n"
		"  -s TEXT      system prompt (default: \"You are a helpful assistant\")\n"
		"  -n N         max new tokens per reply (default 2048)\n"
		"  -c N         context size in tokens (default 8192)\n"
		"  -j N         threads (default: physical cores)\n"
		"  --temp X --top-k N --top-p X --min-p X --seed N   sampling (defaults: 1.0 20 0.95 0, the thinking-mode settings)\n"
		"  --no-think   answer without the reasoning trace (also use --temp 0.7 --top-p 0.8)\n"
		"  --greedy     argmax decoding\n"
		"  --draft N    speculative decoding with the MTP draft head, N draft tokens per step (1..3); needs the MTP head:\n"
		"               either the -mtp.gguf as MODEL, or --mtp FILE (the head is read from FILE)\n"
		"  --mtp FILE   GGUF containing the MTP head (blk.64.*)\n"
		"  --mmproj FILE  mmproj GGUF (vision tower); then a line / -p text starting with @image.jpg attaches that image\n"
		"  --image-tokens N  upper limit of LLM tokens per image (default 256; each is 32x32 pixels)\n"
		"  --top        print the 10 most likely first tokens of the prompt and exit (sanity check)\n", a0);
}

static int emit(const bn_model *m, int id)
{
	char b[512];
	size_t n = cats_bpe_piece(&m->bpe, id, 1, b, sizeof b);
	if (n) { fwrite(b, 1, n, stdout); fflush(stdout); }
	return (int)n;
}

/* state of the MTP head: the final-norm hidden state of the last token fed to the main model */
static float *hlast;

/* feeds tokens in BN_MAXT chunks; logits for the last one if wanted. With the MTP head the head's KV cache is filled
 * for the same positions (pairs of a token and the hidden state of the position before it). */
static int feed(bn_model *m, const int *ids, int n, float *logits)
{
	const int D = m->D;
	float *hn = m->mtp.ok ? (float*)malloc((size_t)BN_MAXT * D * sizeof(float)) : NULL, *hp = m->mtp.ok ? (float*)malloc((size_t)(BN_MAXT + 1) * D * sizeof(float)) : NULL;
	for (int i = 0; i < n; i += BN_MAXT) {
		const int c = n - i < BN_MAXT ? n - i : BN_MAXT, pos0 = m->pos;
		if (bn_forward_ex(m, ids + i, c, i + c >= n ? logits : NULL, 0, hn)) return -1;
		if (hn) {
			memcpy(hp, hlast, (size_t)D * sizeof(float));
			memcpy(hp + D, hn, (size_t)(c - 1) * D * sizeof(float));	/* hp[t] = hidden state before token t */
			for (int t = 0; t < c; t += 3) bn_mtp_step(m, ids + i + t, hp + (size_t)t * D, c - t < 3 ? c - t : 3, pos0 + t, NULL, NULL);
			memcpy(hlast, hn + (size_t)(c - 1) * D, (size_t)D * sizeof(float));
		}
	}
	free(hn); free(hp);
	return 0;
}

static int is_eos(const bn_model *m, int id) { return id == m->eos || id == m->eos2; }

/* Speculative generation: the MTP head drafts `nd` tokens, the main model verifies them in one batch. The token that is
 * emitted is always one sampled from the main model's distribution; a draft is only accepted when it equals that sample,
 * so the output distribution is the same as without drafting. Returns the number of generated tokens. */
static int generate_spec(bn_model *m, int nd, int maxnew, float temp, int top_k, float top_p, float min_p, uint64_t *seed, int *drafted, int *accepted)
{
	const int D = m->D, V = m->vocab;
	float *vlog = (float*)malloc((size_t)(nd + 1) * V * sizeof(float)), *vhn = (float*)malloc((size_t)(nd + 1) * D * sizeof(float));
	float *mlog = (float*)malloc((size_t)V * sizeof(float)), *hm = (float*)malloc((size_t)D * sizeof(float)), *hp = (float*)malloc((size_t)D * sizeof(float));
	int ng = 0, x = cats_sample_logits(m->logits, V, temp, top_k, top_p, min_p, seed);
	while (ng < maxnew && m->pos + nd + 2 < m->ctx && !is_eos(m, x)) {
		emit(m, x); ng++;
		int d[3], cur = x;
		memcpy(hp, hlast, (size_t)D * sizeof(float));
		for (int k = 0; k < nd; k++) {	/* draft: greedy chain through the MTP head (pair k sits at position pos + k) */
			bn_mtp_step(m, &cur, hp, 1, m->pos + k, mlog, hm);
			int best = 0;
			for (int i = 1; i < V; i++) if (mlog[i] > mlog[best]) best = i;
			d[k] = best; cur = best;
			memcpy(hp, hm, (size_t)D * sizeof(float));
		}
		int batch[4] = { x, d[0], d[1], d[2] };
		const int pos0 = m->pos;
		m->spec = nd;
		if (bn_forward_ex(m, batch, nd + 1, vlog, 1, vhn)) break;
		int acc = 0, y = cats_sample_logits(vlog, V, temp, top_k, top_p, min_p, seed);
		while (acc < nd && y == d[acc] && !is_eos(m, y) && ng < maxnew) {
			emit(m, y); ng++; acc++;
			y = cats_sample_logits(vlog + (size_t)acc * V, V, temp, top_k, top_p, min_p, seed);
		}
		*drafted += nd; *accepted += acc;
		bn_spec_commit(m, acc);
		m->pos = m->rpos = pos0 + 1 + acc;
		if (acc) bn_mtp_step(m, d, vhn, acc, pos0 + 1, NULL, NULL);	/* the head's KV entries for the accepted drafts */
		memcpy(hlast, vhn + (size_t)acc * D, (size_t)D * sizeof(float));
		memcpy(m->logits, vlog + (size_t)acc * V, (size_t)V * sizeof(float));
		x = y;
	}
	free(vlog); free(vhn); free(mlog); free(hm); free(hp);
	/* the pending token was sampled but not fed: it is dropped (or is the end of turn), like in the plain loop */
	return ng;
}

int main(int argc, char **argv)
{
	const char *path = NULL, *prompt = NULL, *sys = "You are a helpful assistant";
	const char *mtp_path = NULL, *mmproj = NULL;
	int image_tokens = 256;
	int draft = 0;
	int maxnew = 2048, ctx = 8192, threads = 0, top_k = 20, think = 1, greedy = 0, show_top = 0;
	float temp = 1.0f, top_p = 0.95f, min_p = 0.0f;
	uint64_t seed = (uint64_t)time(NULL) * 2654435761u + 88172645463325252ull;
	for (int i = 1; i < argc; i++) {
		const char *a = argv[i];
		#define ARG (i + 1 < argc ? argv[++i] : (usage(argv[0]), exit(1), (char*)0))
		if (!strcmp(a, "-p")) prompt = ARG;
		else if (!strcmp(a, "-s")) sys = ARG;
		else if (!strcmp(a, "-n")) maxnew = atoi(ARG);
		else if (!strcmp(a, "-c")) ctx = atoi(ARG);
		else if (!strcmp(a, "-j")) threads = atoi(ARG);
		else if (!strcmp(a, "--temp")) temp = (float)atof(ARG);
		else if (!strcmp(a, "--top-k")) top_k = atoi(ARG);
		else if (!strcmp(a, "--top-p")) top_p = (float)atof(ARG);
		else if (!strcmp(a, "--min-p")) min_p = (float)atof(ARG);
		else if (!strcmp(a, "--seed")) seed = strtoull(ARG, NULL, 10) * 2654435761u + 1;
		else if (!strcmp(a, "--mmproj")) mmproj = ARG;
		else if (!strcmp(a, "--image-tokens")) image_tokens = atoi(ARG);
		else if (!strcmp(a, "--draft")) draft = atoi(ARG);
		else if (!strcmp(a, "--mtp")) mtp_path = ARG;
		else if (!strcmp(a, "--no-think")) think = 0;
		else if (!strcmp(a, "--greedy")) greedy = 1;
		else if (!strcmp(a, "--top")) show_top = 1;
		else if (a[0] != '-' && !path) path = a;
		else { usage(argv[0]); return 1; }
	}
	if (!path) { usage(argv[0]); return 1; }
	cats_set_threads(threads);
	bn_model m;
	double t0 = bn_now();
	if (bn_open(&m, path, ctx)) return 1;
	fprintf(stderr, "bonsai: %d layers (%d gated delta-net, %d attention), hidden %d, vocab %d, hadamard block %d, %d threads, loaded in %.1fs\n",
		m.nl, m.ngdn, m.nfull, m.D, m.vocab, m.hblock, omp_get_max_threads(), bn_now() - t0);
	if (draft > 3 || draft < 0) { fprintf(stderr, "bonsai: --draft must be 0..3\n"); return 1; }
	if (draft) {
		if (!mtp_path && !m.nextn) { fprintf(stderr, "bonsai: --draft needs the MTP head: use the -mtp.gguf as MODEL or pass --mtp FILE\n"); return 1; }
		if (bn_mtp_open(&m, mtp_path) || bn_spec_alloc(&m, draft)) return 1;
		hlast = (float*)calloc(m.D, sizeof(float));
		fprintf(stderr, "bonsai: speculative decoding with the MTP head, %d draft token(s) per step\n", draft);
	}
	bn_vis vis;
	if (mmproj) {
		if (draft) { fprintf(stderr, "bonsai: --mmproj and --draft cannot be combined (the MTP head does not know image positions)\n"); return 1; }
		double tv = bn_now();
		if (bv_open(&vis, mmproj, m.D)) return 1;
		fprintf(stderr, "bonsai: vision tower loaded in %.1fs\n", bn_now() - tv);
	}
	int *ids = (int*)malloc(sizeof(int) * (ctx + 64));
	char line[8192];
	int first = 1;
	for (;;) {
		char user[8192];
		if (prompt) { snprintf(user, sizeof user, "%s", prompt); }
		else {
			fprintf(stderr, "\n> ");
			if (!fgets(line, sizeof line, stdin) || line[0] == '\n') break;
			line[strcspn(line, "\n")] = 0;
			snprintf(user, sizeof user, "%s", line);
		}
		/* an image is attached by starting the message with @path */
		const char *utext = user;
		float *emb = NULL;
		int inx = 0, iny = 0;
		if (user[0] == '@') {
			char ip[1024];
			size_t l = strcspn(user + 1, " \t");
			snprintf(ip, sizeof ip, "%.*s", (int)(l < sizeof ip - 1 ? l : sizeof ip - 1), user + 1);
			utext = user + 1 + l + (user[1 + l] ? 1 : 0);
			if (!mmproj) { fprintf(stderr, "bonsai: an image needs --mmproj FILE\n"); break; }
			int iw, ih, ic;
			unsigned char *px = stbi_load(ip, &iw, &ih, &ic, 3);
			if (!px) { fprintf(stderr, "bonsai: cannot read image %s\n", ip); break; }
			double te = bn_now();
			emb = bv_encode(&vis, px, iw, ih, image_tokens, &inx, &iny);
			stbi_image_free(px);
			if (!emb) { fprintf(stderr, "bonsai: image encoding failed\n"); break; }
			fprintf(stderr, "bonsai: image encoded in %.1fs\n", bn_now() - te);
		}
		char text[16384];
		int n = 0, np = 0;
		/* ChatML; generation starts inside <think> (thinking mode) or with an empty think block */
		if (first) n += snprintf(text + n, sizeof text - n, "<|im_start|>system\n%s<|im_end|>\n", sys);
		n += snprintf(text + n, sizeof text - n, "<|im_start|>user\n");
		first = 0;
		double t1 = bn_now();
		if (emb) {
			n += snprintf(text + n, sizeof text - n, "<|vision_start|>");
			np = cats_bpe_encode(&m.bpe, text, 1, ids, ctx);
			if (m.pos + np + inx * iny + 64 > ctx) { fprintf(stderr, "bonsai: out of context (use -c, or --image-tokens)\n"); free(emb); break; }
			if (feed(&m, ids, np, NULL) || bn_feed_embeddings(&m, emb, inx, iny)) return 1;
			np += inx * iny;
			free(emb);
			n = snprintf(text, sizeof text, "<|vision_end|>");
		}
		n += snprintf(text + n, sizeof text - n, "%s<|im_end|>\n<|im_start|>assistant\n%s", utext, think ? "<think>\n" : "<think>\n\n</think>\n\n");
		const int nt = cats_bpe_encode(&m.bpe, text, 1, ids, ctx);
		np += nt;
		if (nt > ctx - 16 || m.pos + nt + 16 > ctx) { fprintf(stderr, "bonsai: out of context (use -c)\n"); break; }
		if (show_top) {
			if (feed(&m, ids, nt, m.logits)) return 1;
			for (int k = 0; k < 10; k++) {
				int best = 0;
				for (int i = 1; i < m.vocab; i++) if (m.logits[i] > m.logits[best]) best = i;
				char b[128]; cats_bpe_piece(&m.bpe, best, 1, b, sizeof b);
				printf("%2d  id=%-7d logit=%8.3f  \"%s\"\n", k + 1, best, m.logits[best], b);
				m.logits[best] = -1e30f;
			}
			return 0;
		}
		if (feed(&m, ids, nt, m.logits)) return 1;
		double t2 = bn_now();
		int ng = 0, drafted = 0, accepted = 0;
		if (draft) ng = generate_spec(&m, draft, maxnew, greedy ? 0.0f : temp, top_k, top_p, min_p, &seed, &drafted, &accepted);
		else for (; ng < maxnew && m.pos < ctx; ng++) {
			int id = cats_sample_logits(m.logits, m.vocab, greedy ? 0.0f : temp, top_k, top_p, min_p, &seed);
			if (id == m.eos || id == m.eos2) break;
			emit(&m, id);
			if (bn_forward(&m, &id, 1, m.logits)) break;
		}
		printf("\n");
		double t3 = bn_now();
		/* close the turn so the next one continues the same conversation */
		{ int tail[2]; int nt = cats_bpe_encode(&m.bpe, "<|im_end|>\n", 1, tail, 2); if (nt && !prompt && m.pos + nt < ctx) { if (feed(&m, tail, nt, NULL)) break; } }
		fprintf(stderr, "[prompt %d tok %.1f t/s | generated %d tok %.2f t/s", np, np / (t2 - t1), ng, ng / (t3 - t2 > 1e-9 ? t3 - t2 : 1e-9));
		if (draft) fprintf(stderr, " | drafts accepted %d/%d (%.0f%%)", accepted, drafted, drafted ? 100.0 * accepted / drafted : 0.0);
		fprintf(stderr, "]\n");
		if (prompt) break;
	}
	bn_close(&m);
	free(ids);
	return 0;
}
