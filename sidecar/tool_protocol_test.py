"""Can the advisor actually emit a TOOL_CALL under the host's output contract?

runtime._review_one sends json_mode=True + REVIEW_SCHEMA, so the HOST appends:
  "Respond with a single JSON object that matches the requested shape.
   Do not include prose or markdown fences."
and sets response_format. The plugin's tool block separately instructs a bare
line: "TOOL_CALL: <name> <json-args>". Those cannot both be obeyed.

parse_tool_call() demands the bare line, so if the model follows the host the
tool loop breaks on its first iteration and tool verification never runs -- the
feature is dead while 20+ tests pass, because the tests script the raw text.

This reproduces the REAL composed prompt against the REAL provider.
"""
import json, re, sys, urllib.request
from llm_gate import load_key

# Load the LIVE brief, not a copy. This file imported `verifier_pkg_brief`, a
# sidecar duplicate that had drifted 111 lines behind brief.py -- so the test
# reproduced a prompt the plugin no longer ships, and a passing run said nothing
# about the shipped reviewer. Same _load trick evals/run.py uses for the same
# reason. The duplicate is deleted.
import importlib.util, pathlib

_REPO = pathlib.Path(__file__).resolve().parent.parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_VERIFIER_PKG = {"verifier_pkg": type(sys)("verifier_pkg")}
sys.modules.setdefault("verifier_pkg", _VERIFIER_PKG["verifier_pkg"])
_brief = _load(_REPO / "brief.py", "hv_brief")
SYSTEM_PROMPT, REVIEW_TASK = _brief.SYSTEM_PROMPT, _brief.REVIEW_TASK

TOOL_BLOCK = (
    "You may verify claims with these read-only tools:\n"
    "- read_file(path) — read a file\n"
    "- search_files(pattern) — search\n"
    "To use one, reply with a line exactly of the form:\n"
    'TOOL_CALL: <name> <json-args>\n'
    'Example: TOOL_CALL: read_file {"path": "src/jobs/queue.py"}\n'
    "You get at most 3 tool calls per review. Never call anything else."
)
HOST_JSON_LINE = ("Respond with a single JSON object that matches the requested "
                  "shape. Do not include prose or markdown fences.")

system = "\n\n".join([SYSTEM_PROMPT, TOOL_BLOCK, HOST_JSON_LINE])
delta = ('[user] make the write path durable\n'
         '[assistant] Done — I switched src/jobs/write.py to enqueue directly\n'
         'terminal({"command": "git add -A && git commit -m x"})\n'
         '[assistant final] The write path now enqueues through the queue.')

body = json.dumps({
    "model": "deepseek-flash",
    "messages": [{"role": "system", "content": system},
                 {"role": "user", "content": f"{REVIEW_TASK}\n\n{delta}"}],
    "temperature": 0, "max_tokens": 900,
}).encode()
req = urllib.request.Request("https://api.deepseek.com/v1/chat/completions",
                             data=body,
                             headers={"Authorization": f"Bearer {load_key()}",
                                      "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    text = json.load(r)["choices"][0]["message"]["content"] or ""

print("raw response:", repr(text[:220]))
print()
emits_tool_call = "TOOL_CALL" in text
parses_as_json = bool(re.search(r"\{.*\}", text, re.S))
print(f"  emits TOOL_CALL: {emits_tool_call}")
print(f"  emits JSON:      {parses_as_json}")
print()
print("  VERDICT:", "tool path REACHABLE" if emits_tool_call
      else "*** TOOL PATH DEAD: the advisor never emits the protocol ***")
