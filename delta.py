"""Transcript delta extraction, rendering, and redaction.

The verifier reviews only what changed since its last look. This module turns
the ``post_llm_call`` payload (full conversation history + final response)
into a bounded, redacted delta string, and maintains the per-session cursor
that makes the next delta cheap.

Pure stdlib: unit-testable in isolation.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Finding-strip: the verifier must never review its own output. Injected
# findings rendered as <finding ...>...</finding> blocks are removed before
# the delta is built.
# ---------------------------------------------------------------------------

_FINDING_BLOCK = re.compile(r"<finding\b[^>]*>.*?</finding>", re.DOTALL)

# ---------------------------------------------------------------------------
# Redaction. The session's full obfuscation pipeline runs upstream of us for
# provider-bound messages; this is a belt-and-braces scrub for tool results
# and user-visible text before the delta reaches the verifier model.
# ---------------------------------------------------------------------------

_REDACTIONS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"sk-[A-Za-z0-9]{12,}"), "[REDACTED:openai-key]"),
    (re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"), "[REDACTED:aws-key]"),
    (re.compile(r"AIza[0-9A-Za-z_-]{20,}"), "[REDACTED:google-key]"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "[REDACTED:github-token]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED:github-token]"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "[REDACTED:slack-token]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
     "[REDACTED:jwt]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9._-]{12,}"), "Bearer [REDACTED]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                re.DOTALL), "[REDACTED:private-key]"),
    (re.compile(r"(?i)\b(api_key|apikey|api-key|secret|password|passwd|token)"
                r"\s*[:=]\s*[\"']?[^\s\"'<>]{6,}[\"']?"),
     r"\1 [REDACTED]"),
]

_LONG_TOKEN = re.compile(r"\S{200,}")


def strip_findings(text: str) -> str:
    return _FINDING_BLOCK.sub(" ", text)


def redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    text = _LONG_TOKEN.sub("[long-token]", text)
    return text


def _content_to_text(content: Any, cap: int = 4000) -> str:
    """Best-effort flatten of a message content field to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = item.get("type")
                if kind == "text":
                    text = item.get("text")
                    parts.append(text if isinstance(text, str) else "")
                elif kind == "image_url":
                    parts.append("[image]")
                elif kind == "image":
                    parts.append("[image]")
                elif kind == "tool_call" or "name" in item and "arguments" in item:
                    name = item.get("name") or item.get("function", {}).get("name")
                    parts.append(f"[tool_call: {name}]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False)[:cap])
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)[:cap]
    return str(content)


def render_message(message: Dict[str, Any], cap: int = 6000) -> str:
    """Render one history message as a compact transcript row."""
    role = message.get("role", "?")
    content = _content_to_text(message.get("content"), cap)

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        calls = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            name = fn.get("name", "?")
            try:
                args = json.dumps(fn.get("arguments", {}), ensure_ascii=False)
            except (TypeError, ValueError):
                args = str(fn.get("arguments", ""))
            calls.append(f"{name}({args[:300]})")
        content = (content + " " if content else "") + " | ".join(calls)

    tool_call_id = message.get("tool_call_id")
    role_label = role
    if role == "tool" and tool_call_id:
        role_label = f"tool({tool_call_id[:12]})"

    text = redact(strip_findings(content))
    return f"[{role_label}] {text}"[:cap]


def extract_delta(
    *,
    cursor_len: Optional[int],
    conversation_history: List[Dict[str, Any]],
    assistant_response: str,
    max_chars: int = 20000,
) -> Tuple[Optional[int], Optional[str]]:
    """Compute the new-messages delta and return ``(new_cursor, delta_text)``.

    ``cursor_len`` is the number of history rows the verifier has already
    reviewed. A history shorter than the cursor means the transcript was
    rewritten (compaction / session resume): the whole bounded history is
    replayed and the cursor rewinds.

    Returns ``(cursor_len, None)`` when nothing new is worth reviewing.
    """
    history_len = len(conversation_history)
    if cursor_len is None or history_len < cursor_len:
        rows = conversation_history[-24:]
        new_cursor = history_len
        tail = "\n".join(render_message(m) for m in rows)
        if assistant_response:
            tail += f"\n[assistant final] {redact(strip_findings(assistant_response))}"
    else:
        rows = conversation_history[cursor_len:]
        new_cursor = history_len
        tail = "\n".join(render_message(m) for m in rows)
        if assistant_response:
            tail += f"\n[assistant final] {redact(strip_findings(assistant_response))}"

    tail = tail.strip()
    if not tail:
        return new_cursor, None
    if len(tail) > max_chars:
        tail = tail[:max_chars] + "\n… [delta truncated]"
    return new_cursor, tail


def user_anchor(user_message: Any) -> str:
    """Stable short hash of the current user message (compaction detection)."""
    text = _content_to_text(user_message)
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]
