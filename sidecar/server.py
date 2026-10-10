"""Local d1-omni decision endpoint.

One route: `POST /decide` with `{"state": "..."}`, returning

    {"answers": {"needs_review": {...}, "effort": {...}}, "usage": {...}}

Bound to 127.0.0.1 only -- this serves a local model to a local plugin and has
no authentication, so it must never be reachable off-host.

Run:  ./run.sh          (or: python server.py --port 8765)
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("d1-sidecar")

MODEL_ID = "LiquidAI/d1-omni-600M"

# Asked on every request. Kept to two questions so one forward pass answers all
# of them; each extra question costs compute but not extra output tokens.
QUESTIONS = {
    "needs_review": {
        "type": "noul",
        "instructions": "Is there anything here a reviewer should examine?",
    },
    "effort": {
        "type": "score",
        "instructions": "How much review effort does this warrant?",
        "criteria": ["Routine", "Worth a look", "Needs a hard look"],
    },
}

_model = None


def get_model():
    """Load once, lazily. Float32 on CPU per the model card."""
    global _model
    if _model is None:
        from transformers import AutoModel
        log.info("loading %s ...", MODEL_ID)
        _model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype="float32")
        log.info("model loaded")
    return _model


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._json(200, {"ok": True, "model_loaded": _model is not None})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/decide":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return

        state = body.get("state")
        if state is None:
            self._json(400, {"error": "missing 'state'"})
            return
        try:
            answers = get_model().system_one(str(state), QUESTIONS)
        except Exception as exc:  # never 500 a gate: the plugin fails open anyway
            log.warning("decision failed: %s", exc)
            self._json(200, {"error": str(exc)[:200]})
            return
        self._json(200, answers)

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
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (keep the default; there is no auth)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("binding to %s: this endpoint has no authentication", args.host)
    get_model()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("d1-omni sidecar on http://%s:%d/decide", args.host, args.port)
    srv.serve_forever()


if __name__ == "__main__":
    main()
