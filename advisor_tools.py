"""Read-only investigative tools an advisor may use to verify claims.

OMP's advisor reads and searches the project so it can check a claim instead
of reviewing prose alone. This module is that capability, deliberately narrow:

* an allowlist of read-only Hermes tools — nothing that mutates, shells out,
  or spends money;
* dispatch goes through ``ctx.dispatch_tool``, so the host's approval,
  redaction, and result-budget pipelines still apply exactly as they would for
  a model-initiated call;
* every result is truncated and the whole loop is bounded, because it runs on
  the background review thread and must never stall a session.

Pure stdlib at module level: imported by tests without a Hermes runtime.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes-verifier.advisor-tools")

# Read-only investigative grant. Anything not named here is refused, so a
# mistyped or over-broad roster entry cannot widen the advisor's reach.
ALLOWED_TOOLS: Dict[str, Dict[str, Any]] = {
    "read_file": {
        "description": "Read a file from the workspace to verify a claim about its contents.",
        "args": {"path": {"type": "string", "description": "Absolute or workspace-relative file path."}},
        "required": ["path"],
    },
    "search_files": {
        "description": "Search file contents (regex) or find files by name to verify a claim.",
        "args": {
            "pattern": {"type": "string", "description": "Regex pattern, or a glob when target='files'."},
            "target": {"type": "string", "description": "'content' (default) or 'files'."},
        },
        "required": ["pattern"],
    },
}

MAX_TOOL_CALLS = 3          # per review — bounds both cost and latency
MAX_RESULT_CHARS = 6000     # per tool result fed back to the advisor


def resolve_grant(requested: Optional[List[str]]) -> List[str]:
    """Intersect a roster's requested tools with the read-only allowlist.

    Unknown or mutating names are dropped (mirroring OMP's "unknown tool names
    are dropped with a warning"), so the grant can only ever shrink.
    """
    if not requested:
        return []
    granted: List[str] = []
    for name in requested:
        if name in ALLOWED_TOOLS:
            granted.append(name)
        else:
            logger.warning("advisor tool grant refused (not read-only allowlisted): %s", name)
    return granted[:MAX_TOOL_CALLS]


def render_available(granted: List[str]) -> str:
    """Human-readable grant block for the advisor's system prompt."""
    if not granted:
        return ""
    lines = ["You may verify claims with these read-only tools:"]
    for name in granted:
        spec = ALLOWED_TOOLS[name]
        args = ", ".join(spec["args"])
        lines.append(f"- {name}({args}) — {spec['description']}")
    lines.append(
        "To use one, reply with a line exactly of the form:\n"
        "TOOL_CALL: <name> <json-args>\n"
        "Example: TOOL_CALL: read_file {\"path\": \"src/jobs/queue.py\"}\n"
        f"You get at most {MAX_TOOL_CALLS} tool calls per review. Never call anything else."
    )
    return "\n".join(lines)


TOOL_CALL_PREFIX = "TOOL_CALL:"


def parse_tool_call(text: str, granted: List[str]) -> Optional[Dict[str, Any]]:
    """Extract the first well-formed, allowlisted tool call from advisor text.

    Returns None when there is no call, the JSON is malformed, the tool is not
    granted, or required args are missing — every failure is a no-op so the
    review simply proceeds without verification.
    """
    import json

    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith(TOOL_CALL_PREFIX):
            continue
        body = stripped[len(TOOL_CALL_PREFIX):].strip()
        if not body:
            return None
        name, _, raw_args = body.partition(" ")
        name = name.strip()
        if name not in granted:
            logger.warning("advisor requested non-granted tool: %s", name)
            return None
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            logger.warning("advisor tool call had malformed JSON args")
            return None
        if not isinstance(args, dict):
            return None
        spec = ALLOWED_TOOLS[name]
        missing = [k for k in spec["required"] if k not in args]
        if missing:
            logger.warning("advisor tool call %s missing args: %s", name, missing)
            return None
        return {"tool": name, "args": args}
    return None


def dispatch(ctx: Any, call: Dict[str, Any]) -> str:
    """Run one allowlisted tool through the host registry and bound the result."""
    name = call["tool"]
    try:
        raw = ctx.dispatch_tool(name, call["args"])
    except Exception as exc:  # dispatch_tool already returns {"error": ...}; belt and braces
        logger.warning("advisor tool %s failed: %s", name, exc)
        return f"[tool {name} failed: {type(exc).__name__}]"
    text = raw if isinstance(raw, str) else str(raw)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "\n…[truncated]"
    return text
