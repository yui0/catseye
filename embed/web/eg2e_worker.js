// Web Worker: loads eg2e_web.wasm, switches models, embeds text / images. Messages: {id, op, ...} -> {id, ok, ...}
// Multi-thread build when the page is cross-origin isolated (needs COOP/COEP headers: web/serve.py), else single thread.
const MT = self.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined';
const BASE = MT ? '../eg2e_web_mt.js' : '../eg2e_web.js';
importScripts(BASE);
// Emscripten starts its pthread workers from this very script (name 'em-pthread'): the engine script alone handles them.
const IS_PTHREAD = self.name === 'em-pthread';
let M = null, api = null, current = null, threads = 1;

async function init() {
  if (M) return;
  const opt = { locateFile: (f) => '../' + f };
  if (MT) opt.mainScriptUrlOrBlob = new URL(BASE, self.location.href).href;   // pthread workers start from this script
  M = await Eg2eModule(opt);
  if (MT) threads = M.cwrap('eg2w_set_threads', 'number', ['number'])(Math.max(1, Math.min(12, self.navigator.hardwareConcurrency || 4)));
  api = {
    load: M.cwrap('eg2w_load', 'number', ['number', 'number', 'number']),
    loadTok: M.cwrap('eg2w_load_tokenizer', 'number', ['string']),
    text: M.cwrap('eg2w_embed_text', 'number', ['string', 'number', 'number']),
    rgba: M.cwrap('eg2w_embed_rgba', 'number', ['number', 'number', 'number', 'number', 'number']),
  };
}

// Downloads url straight into a malloc'd WASM buffer (cached in the Cache API), reporting progress.
async function fetchInto(url, label) {
  let cache = null, res = null;
  try { cache = await caches.open('eg2e-models'); res = await cache.match(url); } catch (e) {}
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(url + ': HTTP ' + res.status);
    if (cache) { try { await cache.put(url, res.clone()); } catch (e) {} }
  }
  const total = +res.headers.get('content-length') || 0;
  if (!total) { const b = new Uint8Array(await res.arrayBuffer()); const p = M._malloc(b.length); M.HEAPU8.set(b, p); return [p, b.length]; }
  const ptr = M._malloc(total);
  if (!ptr) throw new Error('out of memory');
  const rd = res.body.getReader();
  let got = 0, last = 0;
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    M.HEAPU8.set(value, ptr + got);      // re-read HEAPU8 each time: memory growth replaces the view
    got += value.length;
    if (got - last > (4 << 20)) { last = got; postMessage({ progress: got / total, label }); }
  }
  return [ptr, got];
}

const ops = {
  async load({ model, tokenizer, key }) {
    await init();
    if (current === key) return { reused: true };
    current = null;
    postMessage({ progress: 0, label: 'tokenizer' });
    const tb = new Uint8Array(await (await fetch(tokenizer)).arrayBuffer());
    M.FS.writeFile('/tok.eg2t', tb);
    if (api.loadTok('/tok.eg2t')) throw new Error('bad tokenizer ' + tokenizer);
    const [ptr, size] = await fetchInto(model, 'model');
    if (api.load(ptr, size, 1)) throw new Error('bad model ' + model);
    current = key;
    return { size, threads, mt: MT };
  },
  async info() { return { mt: MT, threads }; },
  async text({ text, kind }) {
    const out = M._malloc(768 * 4);
    const rc = api.text(text, kind, out);
    const v = rc ? null : M.HEAPF32.slice(out >> 2, (out >> 2) + 768);
    M._free(out);
    if (!v) throw new Error('text embedding failed');
    return { vec: v };
  },
  async image({ blob, soft }) {
    const bmp = await createImageBitmap(blob);
    const c = new OffscreenCanvas(bmp.width, bmp.height), g = c.getContext('2d');
    g.drawImage(bmp, 0, 0);
    const d = g.getImageData(0, 0, bmp.width, bmp.height).data;
    const p = M._malloc(d.length); M.HEAPU8.set(d, p);
    const out = M._malloc(768 * 4);
    const rc = api.rgba(p, bmp.width, bmp.height, soft || 140, out);
    const v = rc ? null : M.HEAPF32.slice(out >> 2, (out >> 2) + 768);
    M._free(p); M._free(out); bmp.close();
    if (!v) throw new Error('image embedding failed');
    return { vec: v };
  },
};

if (!IS_PTHREAD) onmessage = async (e) => {
  const { id, op } = e.data;
  try { postMessage({ id, ok: true, ...(await ops[op](e.data)) }); }
  catch (err) { postMessage({ id, ok: false, error: String(err && err.message || err) }); }
};
