//---------------------------------------------------------
//	Cat's eye - eg2e_web: EmbeddingGemma 2 text / image vectors in the browser (Emscripten, single thread)
//
//	Switchable models: eg2w_load() replaces the current model, eg2w_load_tokenizer() the current tokenizer.
//	JS passes the safetensors image through a malloc'd buffer (ownership moves to C) and the tokenizer through
//	the Emscripten file system. Build: make -C embed web   (see eg2e_web.html)
//---------------------------------------------------------

#include <emscripten.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../stb_image_resize.h"
#include "eg2_embed.h"
#include "eg2_tokenizer.h"

static eg2e_model model;
static eg2e_tok tok;
static uint8_t *model_buf;
static int have_model, have_tok;

static int bicubic(const unsigned char *src, int sw, int sh, unsigned char *dst, int dw, int dh)
{
	return !stbir_resize_uint8_generic(src, sw, sh, sw * 3, dst, dw, dh, dw * 3, 3, -1, 0, STBIR_EDGE_CLAMP,
					   STBIR_FILTER_CATMULLROM, STBIR_COLORSPACE_LINEAR, NULL);
}

EMSCRIPTEN_KEEPALIVE int eg2w_dim(void) { return EG2E_OUT_DIM; }
// Threads to use (pthread build only; the single-thread build ignores it). Returns the number actually available.
EMSCRIPTEN_KEEPALIVE int eg2w_set_threads(int n)
{
#ifdef __EMSCRIPTEN_PTHREADS__
	cats_set_threads(n);
	return cats_pool.started + 1;
#else
	(void)n;
	return 1;
#endif
}

// Takes ownership of buf (from Module._malloc). Replaces any previously loaded model. Returns 0 on success.
EMSCRIPTEN_KEEPALIVE int eg2w_load(uint8_t *buf, size_t size, int with_vision)
{
	if (have_model) { eg2e_free(&model); have_model = 0; }
	free(model_buf);
	model_buf = buf;
	if (eg2e_load_mem(&model, buf, size, with_vision)) { free(model_buf); model_buf = NULL; return -1; }
	have_model = 1;
	return 0;
}
EMSCRIPTEN_KEEPALIVE int eg2w_has_vision(void) { return have_model && model.has_vision; }

// path inside the Emscripten FS (JS writes the .eg2t there first)
EMSCRIPTEN_KEEPALIVE int eg2w_load_tokenizer(const char *path)
{
	if (have_tok) { eg2t_free(&tok); have_tok = 0; }
	if (eg2t_load(&tok, path)) return -1;
	have_tok = 1;
	return 0;
}

// kind 0 = document ("title: none | text: "), 1 = search query ("task: search result | query: "), 2 = raw text.
EMSCRIPTEN_KEEPALIVE int eg2w_embed_text(const char *text, int kind, float *out)
{
	if (!have_model || !have_tok) return -1;
	static const char *const prompt[] = { "title: none | text: ", "task: search result | query: ", "" };
	if (kind < 0 || kind > 2) kind = 0;
	size_t n = strlen(text) + strlen(prompt[kind]) + 1;
	char *full = (char*)malloc(n);
	if (!full) return -1;
	snprintf(full, n, "%s%s", prompt[kind], text);
	int *ids = (int*)malloc(4096 * sizeof(int));
	int cnt = ids ? eg2t_encode(&tok, full, ids, 4096) : -1;
	free(full);
	int rc = cnt < 0 ? -1 : eg2e_embed_text_ids(&model, ids, (size_t)cnt, out);
	free(ids);
	return rc;
}

// rgba: w*h*4 bytes straight from canvas.getImageData().
EMSCRIPTEN_KEEPALIVE int eg2w_embed_rgba(const uint8_t *rgba, int w, int h, int soft, float *out)
{
	if (!have_model || !model.has_vision || w < 1 || h < 1) return -1;
	uint8_t *rgb = (uint8_t*)malloc((size_t)w * h * 3);
	if (!rgb) return -1;
	for (size_t i = 0, n = (size_t)w * h; i < n; i++) { rgb[3*i] = rgba[4*i]; rgb[3*i+1] = rgba[4*i+1]; rgb[3*i+2] = rgba[4*i+2]; }
	int rc = eg2e_embed_image(&model, rgb, w, h, soft, bicubic, out);
	free(rgb);
	return rc;
}

EMSCRIPTEN_KEEPALIVE float eg2w_cosine(const float *a, const float *b) { return eg2e_cosine(a, b); }
