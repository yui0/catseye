//---------------------------------------------------------
//	Cat's eye
//
//		©2026 Yuichiro Nakada
//---------------------------------------------------------

// Inkling chat template and generation adapter for catseye_openai.h.
typedef struct {
	model_t *m;
	ids_t input;
	cats_oa_json request;
	float *checkpoint, *checkpoint_logits;
	int *cached, pos, prompt_pos, have_logits, maxn, top_k, defaults_n, verbose, use_tools;
	float temp, top_p, default_temp, default_p;
	const char *effort;
	int system, user, model, tool, text, thinking, xml, invoke, end;
} inkling_oa;

static char *inkling_json_string(cats_oa_json j, const char *key)
{
	return cats_oa_string(cats_oa_get(j, key));
}

static int inkling_has_tool(cats_oa_json tools, const char *name)
{
	if (!tools.p || !cats_jr_eat(&tools, '[')) return 0;
	cats_oa_json entry;
	while ((entry = cats_oa_next(&tools)).p) {
		char *s = inkling_json_string(cats_oa_get(entry, "function"), "name");
		int match = s && !strcmp(s, name);
		free(s);
		if (match) return 1;
	}
	return 0;
}

static void inkling_segment(inkling_oa *s, int role, const char *name, int content, const char *text)
{
	const cats_bpe *b = &s->m->tok;
	ids_push(&s->input, role);
	if (name) ids_text(b, &s->input, name);
	ids_push(&s->input, content);
	ids_text(b, &s->input, text);
	ids_push(&s->input, s->end);
}

static int inkling_declare(inkling_oa *s, cats_oa_json tools)
{
	if (!cats_jr_eat(&tools, '[')) return 0;
	char *json = NULL;
	size_t len = 0;
	FILE *f = open_memstream(&json, &len);
	if (!f) return 0;
	int count = 0, valid = 1;
	fputc('[', f);
	cats_oa_json entry;
	while ((entry = cats_oa_next(&tools)).p) {
		cats_oa_json fn = cats_oa_get(entry, "function"), params = cats_oa_get(fn, "parameters");
		char *name = inkling_json_string(fn, "name"), *desc = inkling_json_string(fn, "description");
		valid = cats_oa_is(cats_oa_get(entry, "type"), "\"function\"") && name && *name &&
			(!params.p || *params.p == '{');
		if (valid) {
			if (count++) fputc(',', f);
			fputs("{\"description\":", f);
			cats_oa_quote(f, desc ? desc : "", desc ? strlen(desc) : 0);
			fputs(",\"name\":", f);
			cats_oa_quote(f, name, strlen(name));
			fputs(",\"parameters\":", f);
			if (params.p) fwrite(params.p, 1, params.e - params.p, f);
			else fputs("{}", f);
			fputs(",\"type\":\"function\"}", f);
		}
		free(name);
		free(desc);
		if (!valid) break;
	}
	fputc(']', f);
	if (fclose(f)) valid = 0;
	if (valid && count) inkling_segment(s, s->system, "tool_declare", s->xml, json);
	s->use_tools = count > 0;
	free(json);
	return valid;
}

static char *inkling_tool_name(cats_oa_json history, const char *id)
{
	cats_oa_json msg, call;
	while ((msg = cats_oa_next(&history)).p) {
		cats_oa_json calls = cats_oa_get(msg, "tool_calls");
		if (!calls.p || !cats_jr_eat(&calls, '[')) continue;
		while ((call = cats_oa_next(&calls)).p) {
			char *key = inkling_json_string(call, "id");
			const int match = key && !strcmp(key, id);
			free(key);
			if (match) return inkling_json_string(cats_oa_get(call, "function"), "name");
		}
	}
	return NULL;
}

static int inkling_call_history(inkling_oa *s, cats_oa_json calls)
{
	if (!calls.p || cats_oa_is(calls, "null")) return 1;
	if (!cats_jr_eat(&calls, '[')) return 0;
	cats_oa_json call;
	while ((call = cats_oa_next(&calls)).p) {
		cats_oa_json fn = cats_oa_get(call, "function");
		char *name = inkling_json_string(fn, "name"), *args = inkling_json_string(fn, "arguments");
		cats_oa_json reader = args ? (cats_oa_json){args, args + strlen(args)} : (cats_oa_json){0};
		cats_oa_json object = args ? cats_oa_value(&reader) : (cats_oa_json){0};
		if (args) cats_jr_ws(&reader);
		int ok = name && *name && object.p && *object.p == '{' && reader.p == reader.e;
		if (ok) {
			char *json = NULL;
			size_t len = 0;
			FILE *f = open_memstream(&json, &len);
			if (f) {
				fputs("{\"name\":", f);
				cats_oa_quote(f, name, strlen(name));
				fputs(",\"args\":", f);
				fwrite(object.p, 1, object.e - object.p, f);
				fputc('}', f);
				ok = fclose(f) == 0;
				if (ok) inkling_segment(s, s->model, name, s->invoke, json);
				free(json);
			} else ok = 0;
		}
		free(name);
		free(args);
		if (!ok) return 0;
	}
	return 1;
}

static int inkling_content(inkling_oa *s, cats_oa_json content, int role, const char *name)
{
	if (!content.p || cats_oa_is(content, "null")) return 1;
	if (cats_jr_eat(&content, '[')) {
		cats_oa_json part;
		while ((part = cats_oa_next(&content)).p) {
			char *kind = NULL;
			char *text = cats_oa_string(part);
			if (!text) {
				kind = inkling_json_string(part, "type");
				if (kind && (!strcmp(kind, "text") || !strcmp(kind, "input_text")))
					text = inkling_json_string(part, "text");
			}
			free(kind);
			if (!text) return 0;
			inkling_segment(s, role, name, s->text, text);
			free(text);
		}
		return 1;
	}
	char *text = cats_oa_string(content);
	if (!text) return 0;
	if (*text || role != s->model) inkling_segment(s, role, name, s->text, text);
	free(text);
	return 1;
}

static const char *inkling_prepare(void *opaque, cats_oa_json request)
{
	inkling_oa *s = opaque;
	s->input.n = 0;
	s->request = request;
	s->maxn = s->defaults_n;
	s->temp = s->default_temp;
	s->top_p = s->default_p;
	s->use_tools = 0;
	double x;
	cats_oa_json j = cats_oa_get(request, "max_completion_tokens");
	if (!j.p || cats_oa_is(j, "null")) j = cats_oa_get(request, "max_tokens");
	if (j.p && !cats_oa_is(j, "null")) {
		if (!cats_oa_number(j, &x) || x < 1 || x > INT_MAX || x != (int)x) return "Invalid token limit";
		s->maxn = (int)x;
	}
	j = cats_oa_get(request, "temperature");
	if (j.p && !cats_oa_is(j, "null")) {
		if (!cats_oa_number(j, &x) || x < 0 || x > 2) return "Invalid temperature";
		s->temp = (float)x;
	}
	j = cats_oa_get(request, "top_p");
	if (j.p && !cats_oa_is(j, "null")) {
		if (!cats_oa_number(j, &x) || x <= 0 || x > 1) return "Invalid top_p";
		s->top_p = (float)x;
	}
	j = cats_oa_get(request, "n");
	if (j.p && !cats_oa_is(j, "null") && !cats_oa_is(j, "1")) return "Only n=1 is supported";
	j = cats_oa_get(request, "tool_choice");
	if (j.p && !cats_oa_is(j, "null") && !cats_oa_is(j, "\"auto\"") && !cats_oa_is(j, "\"none\""))
		return "Tool choice must be auto or none";
	cats_oa_json tools = cats_oa_get(request, "tools");
	if (tools.p && !cats_oa_is(tools, "null") && !cats_oa_is(j, "\"none\"") && !inkling_declare(s, tools))
		return "Invalid function definitions";
	j = cats_oa_get(request, "reasoning_effort");
	char *effort = cats_oa_string(j), buf[32], instruction[96];
	const char *eff = s->effort;
	if (j.p && !cats_oa_is(j, "null")) {
		if (effort) eff = effort_string(effort, buf);
		else if (cats_oa_number(j, &x) && x >= 0 && x <= 0.99) {
			snprintf(buf, sizeof(buf), "%g", x);
			eff = buf;
		} else eff = NULL;
	}
	if (eff) snprintf(instruction, sizeof(instruction), "Thinking effort level: %s", eff);
	free(effort);
	if (!eff) return "Invalid reasoning_effort";

	cats_oa_json messages = cats_oa_get(request, "messages");
	if (!messages.p || !cats_jr_eat(&messages, '[')) return "messages must be an array";
	cats_oa_json history = messages, msg;
	int emitted = 0, count = 0;
	while ((msg = cats_oa_next(&messages)).p) {
		char *role = inkling_json_string(msg, "role");
		const int id = !role ? -1 : !strcmp(role, "user") ? s->user : !strcmp(role, "assistant") ? s->model :
			(!strcmp(role, "system") || !strcmp(role, "developer")) ? s->system : !strcmp(role, "tool") ? s->tool : -1;
		free(role);
		if (id < 0) return "Unsupported message role";
		if (!emitted && id != s->system) {
			inkling_segment(s, s->system, NULL, s->text, instruction);
			emitted = 1;
		}
		char *name = inkling_json_string(msg, "name");
		if (id == s->tool && !name) {
			char *call = inkling_json_string(msg, "tool_call_id");
			if (call) name = inkling_tool_name(history, call);
			free(call);
			if (!name) return "Unknown tool_call_id";
		}
		if (id == s->model) {
			char *reason = inkling_json_string(msg, "reasoning_content");
			if (reason && *reason) inkling_segment(s, id, name, s->thinking, reason);
			free(reason);
		}
		const int ok = inkling_content(s, cats_oa_get(msg, "content"), id, name);
		free(name);
		if (!ok) return "Only text message content is supported";
		if (id == s->model) {
			if (!inkling_call_history(s, cats_oa_get(msg, "tool_calls"))) return "Invalid tool call history";
			ids_push(&s->input, s->m->c.eos);
		}
		if (s->input.n >= s->m->ctx) return "Context length exceeded";
		count++;
	}
	if (!count) return "messages must not be empty";
	if (!emitted) inkling_segment(s, s->system, NULL, s->text, instruction);
	return s->input.n >= s->m->ctx ? "Context length exceeded" : NULL;
}

// Only convolution state needs saving. Global KV positions are append-only.
// Sliding KV is also reusable while the decode stays within the ring's spare batch.
static size_t inkling_history(inkling_oa *s, int action)
{
	model_t *m = s->m;
	const cfg_t *c = &m->c;
	size_t total = 0;
	for (int i = 0; i < m->nvis; i++) {
		const int l = i < c->NL ? i : c->loop_start + (i - c->NL) % (c->loop_end - c->loop_start);
		const layer_t *y = &m->L[l];
		const int dim = y->full ? c->nkv * c->hd : c->swa_nkv * c->swa_hd;
		for (int k = 0; k < 4; k++) {
			const size_t n = k < 2 ? (size_t)(y->Kkv - 1) * dim : (size_t)(y->Kc - 1) * c->H;
			float *hist = m->vc[i].hist[k];
			if (action == 1) memcpy(s->checkpoint + total, hist, n * sizeof(float));
			if (action == 2) memcpy(hist, s->checkpoint + total, n * sizeof(float));
			if (action == 3) memset(hist, 0, n * sizeof(float));
			total += n;
		}
	}
	return total;
}

static void inkling_reuse(inkling_oa *s)
{
	if (s->pos <= s->input.n && (s->pos != s->input.n || s->have_logits) &&
	    !memcmp(s->cached, s->input.id, (size_t)s->pos * sizeof(int))) return;
	if (s->prompt_pos && s->prompt_pos <= s->input.n &&
	    (s->pos - s->prompt_pos <= s->m->B || s->m->ctx <= s->m->c.window + s->m->B) &&
	    !memcmp(s->cached, s->input.id, (size_t)s->prompt_pos * sizeof(int))) {
		inkling_history(s, 2);
		memcpy(s->m->logits, s->checkpoint_logits, (size_t)s->m->c.V * sizeof(float));
		s->pos = s->prompt_pos;
		s->have_logits = 1;
	} else {
		inkling_history(s, 3);
		s->pos = 0;
		s->have_logits = 0;
	}
	pr_for = -1;
}

static int inkling_emit(cats_oa_response *r, int mode, char pending[4], int *count, const char *text, size_t n)
{
	const char replacement[] = "\xef\xbf\xbd";
	for (size_t i = 0; i < n; ) {
		const unsigned char c = text[i];
		if (!*count && c < 128) {
			size_t begin = i++;
			while (i < n && (unsigned char)text[i] < 128) i++;
			if (!cats_oa_piece(r, mode, text + begin, i - begin)) return 0;
			continue;
		}
		if (!*count) {
			if (c >= 0xc2 && c <= 0xf4) pending[(*count)++] = (char)c;
			else if (!cats_oa_piece(r, mode, replacement, 3)) return 0;
			i++;
			continue;
		}
		const unsigned char first = pending[0];
		const int width = first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;
		if ((c & 0xc0) != 0x80) {
			*count = 0;
			if (!cats_oa_piece(r, mode, replacement, 3)) return 0;
			continue;
		}
		pending[(*count)++] = (char)c;
		i++;
		if (*count == width) {
			const unsigned char second = pending[1];
			const int valid = !(first == 0xe0 && second < 0xa0) && !(first == 0xed && second >= 0xa0) &&
				!(first == 0xf0 && second < 0x90) && !(first == 0xf4 && second >= 0x90);
			*count = 0;
			if (!cats_oa_piece(r, mode, valid ? pending : replacement, valid ? (size_t)width : 3)) return 0;
		}
	}
	return 1;
}

static const char *inkling_generate(void *opaque, cats_oa_response *r)
{
	inkling_oa *s = opaque;
	model_t *m = s->m;
	const cats_bpe *b = &m->tok;
	inkling_reuse(s);
	const int reused = s->pos;
	const double t0 = tick();
	double progress = t0;
	if (s->verbose) fprintf(stderr, "prefill %d tokens (%d cached)\n", s->input.n, reused);
	while (s->pos < s->input.n) {
		const int left = s->input.n - s->pos, nt = left < m->B ? left : m->B;
		forward(m, s->input.id + s->pos, nt, s->pos, nt == left);
		s->pos += nt;
		s->have_logits = nt == left;
		if (r->stream && tick() - progress > 1000) {
			progress = tick();
			if (!cats_oa_progress(r)) {
				// The cache tokens still describe the previous request.
				s->pos = 0;
				s->prompt_pos = 0;
				s->have_logits = 0;
				inkling_history(s, 3);
				pf_join(m);
				return NULL;
			}
		}
	}
	memcpy(s->cached, s->input.id, (size_t)s->pos * sizeof(int));
	s->prompt_pos = s->pos;
	inkling_history(s, 1);
	memcpy(s->checkpoint_logits, m->logits, (size_t)m->c.V * sizeof(float));
	r->prompt = s->input.n;
	if (s->verbose) fprintf(stderr, "prompt %d tokens (%d reused), %.0f ms\n", r->prompt, reused, tick() - t0);

	cats_oa_json tools = s->use_tools ? cats_oa_get(s->request, "tools") : (cats_oa_json){0};
	ids_t payload = {0};
	int mode = 0, pending_n = 0, stop = 0, connected = 1;
	char pending[4];
	for (int i = 0; i < s->maxn && s->pos < m->ctx && connected; i++) {
		const int next = sample(m->logits, m->c.V, s->temp, s->top_p, s->top_k);
		r->completion++;
		if (next == m->c.eos) {
			forward(m, &next, 1, s->pos, 0);
			s->cached[s->pos++] = next;
			s->have_logits = 0;
			stop = 1;
			break;
		}
		if (next >= b->n || b->special[next]) {
			if (pending_n) {
				connected = cats_oa_piece(r, mode == 2, "\xef\xbf\xbd", 3);
				pending_n = 0;
			}
			if (next == s->end && mode == 3 && payload.n) {
				char *json = NULL;
				size_t n = 0;
				FILE *f = open_memstream(&json, &n);
				if (f) {
					for (int k = 0; k < payload.n; k++) {
						int id = payload.id[k];
						fwrite(b->tok[id], 1, b->tlen[id], f);
					}
					if (!fclose(f)) {
						cats_oa_json reader = {json, json + n}, call = cats_oa_value(&reader);
						cats_jr_ws(&reader);
						cats_oa_json args = cats_oa_get(call, "args");
						char *name = inkling_json_string(call, "name");
						if (reader.p == reader.e && name && args.p && *args.p == '{' && inkling_has_tool(tools, name))
							connected = cats_oa_tool(r, name, args);
						free(name);
					}
					free(json);
				}
			}
			mode = next == s->text ? 1 : next == s->thinking ? 2 : next == s->invoke && tools.p ? 3 : 0;
			payload.n = 0;
		} else if (mode == 3) ids_push(&payload, next);
		else if (mode) connected = inkling_emit(r, mode == 2, pending, &pending_n, (const char *)b->tok[next], b->tlen[next]);

		// Keep logits for every cached token. The last sampled token needs no forward pass.
		if (connected && i + 1 < s->maxn && s->pos + 1 < m->ctx) {
			forward(m, &next, 1, s->pos, 1);
			s->cached[s->pos++] = next;
		} else break;
	}
	if (connected && pending_n) connected = cats_oa_piece(r, mode == 2, "\xef\xbf\xbd", 3);
	free(payload.id);
	pf_join(m);
	if (s->verbose) fprintf(stderr, "completion %d tokens, request %.0f ms\n", r->completion, tick() - t0);
	return !connected ? NULL : r->calls ? "tool_calls" : stop ? "stop" : "length";
}
