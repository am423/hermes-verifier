"""Local sentence-embedding endpoint for semantic dedupe.

Replaces the d1-omni-600M decision model for this purpose. Measured on the same
10 labelled pairs (sidecar/embed_compare.py):

    all-MiniLM-L6-v2   22.7M params   ~90 MB   sep +0.451   pairwise 100%   recall 5/5
    d1-omni-600M        587M params   2.35 GB   sep +0.401   pairwise  76%   recall 3/5

Smaller on every axis and better on every metric. It also needs no
`trust_remote_code`, no question design, and no criteria -- cosine similarity is
the canonical approach for "are these the same thing", and it is deterministic.

Two routes:
    POST /similarity {"a": "...", "b": "..."}  -> {"similarity": 0.78}
    GET  /health                                -> {"ok": true, ...}

Cosine is a similarity, NOT a probability. The operating point is ~0.3: the five
paraphrase pairs scored 0.355-0.780 and the five distinct pairs 0.064-0.191, so
0.30 catches 5/5 with 0/5 wrong. Do not carry over the 0.9 threshold used for the
decision model's noul output -- the scales are unrelated.

Bound to 127.0.0.1 only; there is no authentication.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed-sidecar")

EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_tok = None
_lock = threading.Lock()


def get_model():
    """Load once. Mean-pooled last hidden state, the standard readout for this model."""
    global _model, _tok
    with _lock:
        if _model is None:
            import torch  # noqa: F401  (import for side effect of validating the env)
            from transformers import AutoModel, AutoTokenizer
            log.info("loading %s ...", EMBED_ID)
            _tok = AutoTokenizer.from_pretrained(EMBED_ID)
            _model = AutoModel.from_pretrained(EMBED_ID)
            _model.eval()
            log.info("model loaded")
    return _model, _tok


def similarity(a: str, b: str) -> float:
    import torch
    model, tok = get_model()
    batch = tok([a, b], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**batch)
    mask = batch["attention_mask"].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    cos = torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0)
    return float(cos)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._json(200, {"ok": True, "model": EMBED_ID, "loaded": _model is not None})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/similarity":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return
        a, b = body.get("a"), body.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            self._json(400, {"error": "need string 'a' and 'b'"})
            return
        try:
            self._json(200, {"similarity": similarity(a, b)})
        except Exception as exc:  # the plugin fails open anyway
            log.warning("similarity failed: %s", exc)
            self._json(200, {"error": str(exc)[:200]})

    def _json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # noqa: A002 - base class signature
        log.debug("%s - %s", self.address_string(), format % args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("binding to %s: this endpoint has no authentication", args.host)
    get_model()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("embedding sidecar on http://%s:%d/similarity", args.host, args.port)
    srv.serve_forever()


if __name__ == "__main__":
    main()
