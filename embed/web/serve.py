#!/usr/bin/env python3
"""Static server with the headers that enable SharedArrayBuffer (multi-threaded WASM).
   python3 web/serve.py [port]   ->  http://localhost:8000/web/   (run from the embed directory)"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class H(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    ThreadingHTTPServer(('', port), H).serve_forever()
