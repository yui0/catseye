#ifndef CATSEYE_OPENAI_H
#define CATSEYE_OPENAI_H

// Include after catseye.h. The adapter prepares one request, then supplies
// UTF-8 text, reasoning, and complete function calls through the response API.
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <netinet/in.h>
#include <signal.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

typedef cats_jr cats_oa_json;

static int cats_oa_skip(cats_oa_json *j, int depth)
{
	cats_jr_ws(j);
	if (j->p == j->e || depth > 64) return 0;
	const char c = *j->p;
	if (c == '"') {
		j->p++;
		while (j->p < j->e) {
			unsigned char ch = *j->p++;
			if (ch == '"') return 1;
			if (ch < 32) return 0;
			if (ch != '\\') continue;
			if (j->p == j->e || !strchr("\"\\/bfnrtu", *j->p) || !*j->p) return 0;
			if (*j->p++ == 'u') {
				for (int i = 0; i < 4; i++) {
					if (j->p == j->e || !isxdigit((unsigned char)*j->p)) return 0;
					j->p++;
				}
			}
		}
		return 0;
	}
	if (c == '[' || c == '{') {
		const char close = c == '[' ? ']' : '}';
		j->p++;
		if (cats_jr_eat(j, close)) return 1;
		do {
			cats_jr_ws(j);
			if (c == '{' && (j->p == j->e || *j->p != '"' ||
					!cats_oa_skip(j, depth + 1) || !cats_jr_eat(j, ':'))) return 0;
			if (!cats_oa_skip(j, depth + 1)) return 0;
		} while (cats_jr_eat(j, ','));
		return cats_jr_eat(j, close);
	}
	const char *p = j->p;
	for (; j->p < j->e && !strchr(" \t\r\n,]}", *j->p); j->p++) {}
	const size_t n = j->p - p;
	if ((n == 4 && (!memcmp(p, "true", 4) || !memcmp(p, "null", 4))) ||
		(n == 5 && !memcmp(p, "false", 5))) return 1;
	if (p < j->p && *p == '-') p++;
	if (p == j->p || !isdigit((unsigned char)*p)) return 0;
	if (*p++ != '0') while (p < j->p && isdigit((unsigned char)*p)) p++;
	if (p < j->p && *p == '.') {
		if (++p == j->p || !isdigit((unsigned char)*p)) return 0;
		while (p < j->p && isdigit((unsigned char)*p)) p++;
	}
	if (p < j->p && (*p == 'e' || *p == 'E')) {
		p++;
		if (p < j->p && (*p == '+' || *p == '-')) p++;
		if (p == j->p || !isdigit((unsigned char)*p)) return 0;
		while (p < j->p && isdigit((unsigned char)*p)) p++;
	}
	return p == j->p;
}

static cats_oa_json cats_oa_value(cats_oa_json *j)
{
	cats_jr_ws(j);
	const char *p = j->p;
	if (!cats_oa_skip(j, 0)) return (cats_oa_json){0};
	return (cats_oa_json){p, j->p};
}

static cats_oa_json cats_oa_get(cats_oa_json j, const char *key)
{
	if (!j.p || !cats_jr_eat(&j, '{')) return (cats_oa_json){0};
	do {
		char *name = cats_jr_str(&j, NULL);
		if (!name) break;
		const int match = !strcmp(name, key);
		free(name);
		if (!cats_jr_eat(&j, ':')) break;
		cats_oa_json value = cats_oa_value(&j);
		if (match) return value;
	} while (cats_jr_eat(&j, ','));
	return (cats_oa_json){0};
}

static cats_oa_json cats_oa_next(cats_oa_json *j)
{
	if (!j->p || cats_jr_eat(j, ']')) return (cats_oa_json){0};
	cats_oa_json value = cats_oa_value(j);
	cats_jr_eat(j, ',');
	return value;
}

static char *cats_oa_string(cats_oa_json j)
{
	size_t n;
	char *s = j.p ? cats_jr_str(&j, &n) : NULL;
	if (s && strlen(s) != n) { free(s); return NULL; }
	return s;
}

static int cats_oa_is(cats_oa_json j, const char *literal)
{
	return j.p && (size_t)(j.e - j.p) == strlen(literal) && !memcmp(j.p, literal, j.e - j.p);
}

static int cats_oa_number(cats_oa_json j, double *n)
{
	if (!j.p || (*j.p != '-' && !isdigit((unsigned char)*j.p))) return 0;
	char *end;
	*n = strtod(j.p, &end);
	return end == j.e && isfinite(*n);
}

static void cats_oa_quote(FILE *f, const char *s, size_t n)
{
	fputc('"', f);
	for (size_t i = 0; i < n; i++) {
		const unsigned char c = (unsigned char)s[i];
		if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
		else if (c < 32) fprintf(f, "\\u%04x", c);
		else fputc(c, f);
	}
	fputc('"', f);
}

typedef struct {
	FILE *out;
	const char *model;
	unsigned long id;
	time_t created;
	int stream, usage, prompt, completion, calls;
	char *text, *reason, *tools;
	size_t text_n, reason_n, tools_n;
	FILE *text_f, *reason_f, *tools_f;
} cats_oa_response;

static void cats_oa_head(FILE *f, const cats_oa_response *r, const char *object)
{
	fprintf(f, "{\"id\":\"chatcmpl-%lu\",\"object\":\"%s\",\"created\":%lld,\"model\":",
		r->id, object, (long long)r->created);
	cats_oa_quote(f, r->model, strlen(r->model));
}

static void cats_oa_delta(cats_oa_response *r)
{
	fputs("data: ", r->out);
	cats_oa_head(r->out, r, "chat.completion.chunk");
	fputs(",\"choices\":[{\"index\":0,\"delta\":", r->out);
}

static int cats_oa_progress(cats_oa_response *r)
{
	if (!r->stream) return 1;
	fputs(": prefill\n\n", r->out);
	return fflush(r->out) == 0;
}

static int cats_oa_piece(cats_oa_response *r, int reasoning, const char *s, size_t n)
{
	if (!n) return 1;
	if (!r->stream) {
		FILE *f = reasoning ? r->reason_f : r->text_f;
		return f && fwrite(s, 1, n, f) == n;
	}
	cats_oa_delta(r);
	fprintf(r->out, "{\"%s\":", reasoning ? "reasoning_content" : "content");
	cats_oa_quote(r->out, s, n);
	fputs("},\"finish_reason\":null}]}\n\n", r->out);
	return fflush(r->out) == 0;
}

static int cats_oa_tool(cats_oa_response *r, const char *name, cats_oa_json args)
{
	FILE *f = r->stream ? r->out : r->tools_f;
	if (!f) return 0;
	if (r->stream) {
		cats_oa_delta(r);
		fprintf(f, "{\"tool_calls\":[{\"index\":%d,", r->calls);
	} else {
		if (r->calls) fputc(',', f);
		fputc('{', f);
	}
	fprintf(f, "\"id\":\"call_%lu_%d\",\"type\":\"function\",\"function\":{\"name\":", r->id, r->calls++);
	cats_oa_quote(f, name, strlen(name));
	fputs(",\"arguments\":", f);
	cats_oa_quote(f, args.p, args.e - args.p);
	fputs("}}", f);
	if (r->stream) fputs("]},\"finish_reason\":null}]}\n\n", f);
	return fflush(f) == 0;
}

static void cats_oa_usage(FILE *f, const cats_oa_response *r)
{
	fprintf(f, ",\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%lld}",
		r->prompt, r->completion, (long long)r->prompt + r->completion);
}

static void cats_oa_http(FILE *f, const char *status, const char *type)
{
	fprintf(f, "HTTP/1.1 %s\r\nContent-Type: %s\r\nConnection: close\r\n\r\n", status, type);
}

static void cats_oa_error(FILE *f, const char *status, const char *message)
{
	cats_oa_http(f, status, "application/json");
	fputs("{\"error\":{\"type\":\"invalid_request_error\",\"message\":", f);
	cats_oa_quote(f, message, strlen(message));
	fputs("}}\n", f);
}

typedef struct {
	const char *model;
	void *state;
	// Return an error message or NULL. prepare runs before HTTP response headers.
	const char *(*prepare)(void *, cats_oa_json);
	const char *(*generate)(void *, cats_oa_response *); // finish_reason, or NULL if disconnected
} cats_oa_backend;

static void cats_oa_complete(FILE *f, cats_oa_json request, const cats_oa_backend *api, unsigned long id)
{
	char *model = cats_oa_string(cats_oa_get(request, "model"));
	const int match = model && !strcmp(model, api->model);
	free(model);
	if (!match) { cats_oa_error(f, "404 Not Found", "Unknown model"); return; }
	const char *error = api->prepare(api->state, request);
	if (error) { cats_oa_error(f, "400 Bad Request", error); return; }

	cats_oa_response r = { .out = f, .model = api->model, .id = id, .created = time(NULL),
		.stream = cats_oa_is(cats_oa_get(request, "stream"), "true"),
		.usage = cats_oa_is(cats_oa_get(cats_oa_get(request, "stream_options"), "include_usage"), "true") };
	if (r.stream) {
		cats_oa_http(f, "200 OK", "text/event-stream");
		cats_oa_delta(&r);
		fputs("{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n", f);
		if (fflush(f)) return;
	} else {
		r.text_f = open_memstream(&r.text, &r.text_n);
		r.reason_f = open_memstream(&r.reason, &r.reason_n);
		r.tools_f = open_memstream(&r.tools, &r.tools_n);
	}
	const int ready = r.stream || (r.text_f && r.reason_f && r.tools_f);
	const char *finish = ready ? api->generate(api->state, &r) : NULL;
	if (r.text_f) fclose(r.text_f);
	if (r.reason_f) fclose(r.reason_f);
	if (r.tools_f) fclose(r.tools_f);
	if (finish && r.stream) {
		cats_oa_delta(&r);
		fprintf(f, "{},\"finish_reason\":\"%s\"}]}\n\n", finish);
		if (r.usage) {
			fputs("data: ", f);
			cats_oa_head(f, &r, "chat.completion.chunk");
			fputs(",\"choices\":[]", f);
			cats_oa_usage(f, &r);
			fputs("}\n\n", f);
		}
		fputs("data: [DONE]\n\n", f);
	} else if (finish) {
		cats_oa_http(f, "200 OK", "application/json");
		cats_oa_head(f, &r, "chat.completion");
		fputs(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":", f);
		if (r.calls && !r.text_n) fputs("null", f);
		else cats_oa_quote(f, r.text, r.text_n);
		if (r.reason_n) {
			fputs(",\"reasoning_content\":", f);
			cats_oa_quote(f, r.reason, r.reason_n);
		}
		if (r.calls) fprintf(f, ",\"tool_calls\":[%s]", r.tools);
		fprintf(f, "},\"finish_reason\":\"%s\"}]", finish);
		cats_oa_usage(f, &r);
		fputs("}\n", f);
	} else if (!r.stream) cats_oa_error(f, "500 Internal Server Error", "Generation failed");
	free(r.text);
	free(r.reason);
	free(r.tools);
}

static void cats_oa_client(FILE *in, FILE *out, const cats_oa_backend *api, unsigned long id)
{
	char line[8192], method[8], path[256], http[16];
	if (!fgets(line, sizeof(line), in) || !strchr(line, '\n') ||
		sscanf(line, "%7s %255s %15s", method, path, http) != 3) return;
	long size = -1;
	size_t headers = strlen(line);
	while (fgets(line, sizeof(line), in)) {
		headers += strlen(line);
		if (headers > 65536 || !strchr(line, '\n')) break;
		if (!strcmp(line, "\r\n") || !strcmp(line, "\n")) {
			if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
				cats_oa_http(out, "200 OK", "application/json");
				fputs("{\"object\":\"list\",\"data\":[{\"object\":\"model\",\"created\":0,\"owned_by\":\"local\",\"id\":", out);
				cats_oa_quote(out, api->model, strlen(api->model));
				fputs("}]}\n", out);
				return;
			}
			if (strcmp(method, "POST") || strcmp(path, "/v1/chat/completions")) {
				cats_oa_error(out, "404 Not Found", "Unknown endpoint");
				return;
			}
			if (size < 0) { cats_oa_error(out, "411 Length Required", "Content-Length required"); return; }
			if (size > 16 * 1024 * 1024) { cats_oa_error(out, "413 Content Too Large", "Request too large"); return; }
			char *body = malloc((size_t)size + 1);
			if (!body) { cats_oa_error(out, "500 Internal Server Error", "Allocation failed"); return; }
			if (fread(body, 1, (size_t)size, in) == (size_t)size) {
				body[size] = 0;
				cats_oa_json reader = {body, body + size};
				cats_oa_json request = cats_oa_value(&reader);
				cats_jr_ws(&reader);
				if (request.p && *request.p == '{' && reader.p == reader.e)
					cats_oa_complete(out, request, api, id);
				else cats_oa_error(out, "400 Bad Request", "Invalid JSON object");
			}
			free(body);
			return;
		}
		if (!strncasecmp(line, "Transfer-Encoding:", 18)) break;
		if (!strncasecmp(line, "Content-Length:", 15)) {
			char *end;
			errno = 0;
			if (size >= 0) break;
			size = strtol(line + 15, &end, 10);
			if (errno || size < 0 || end == line + 15 || strspn(end, " \t\r\n") != strlen(end)) break;
		}
	}
	cats_oa_error(out, "400 Bad Request", "Invalid request headers");
}

static int cats_oa_serve(const cats_oa_backend *api, int port)
{
	const int server = socket(AF_INET, SOCK_STREAM, 0);
	if (server < 0) { perror("socket"); return 1; }
	const int reuse = 1;
	setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
	struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons((unsigned short)port),
		.sin_addr.s_addr = htonl(INADDR_LOOPBACK) };
	if (bind(server, (struct sockaddr *)&addr, sizeof(addr)) || listen(server, 16)) {
		perror("Inkling API");
		close(server);
		return 1;
	}
	signal(SIGPIPE, SIG_IGN);
	fprintf(stderr, "%s API http://127.0.0.1:%d/v1\n", api->model, port);
	for (unsigned long id = 1; ; id++) {
		const int client = accept(server, NULL, NULL);
		if (client < 0) {
			if (errno == EINTR) continue;
			perror("accept");
			close(server);
			return 1;
		}
		const struct timeval timeout = { .tv_sec = 30 };
		setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
		setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
		const int output = dup(client);
		FILE *in = fdopen(client, "r"), *out = output < 0 ? NULL : fdopen(output, "w");
		if (in && out) cats_oa_client(in, out, api, id);
		if (in) fclose(in); else close(client);
		if (out) fclose(out); else if (output >= 0) close(output);
	}
}
#endif
