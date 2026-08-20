"""Per-session verifier runtime: review loop, delivery, persistence.

The verifier is a passive reviewer. It reviews each primary turn delta in a
background thread (never blocking the primary), raises at most one finding
per review, and delivers accepted findings back through the next turn's
context injection (``pre_llm_call``) or — for blockers on coding turns — the
``pre_verify`` steer gate.

Hermes imports are lazy (inside functions) so this module imports cleanly in
tests and under the plugin loader.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import brief, delta as delta_mod, guard as guard_mod

logger = logging.getLogger("hermes-verifier")

PLUGIN_ID = "hermes-verifier"
PLUGIN_NAME = "hermes-verifier"
DEFAULT_VERIFIER = "default"

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {
            "type": "string",
            "enum": ["none", "note", "caution", "blocker"],
            "description": "Severity of the finding, or 'none' when there is nothing to raise.",
        },
        "finding": {
            "type": "string",
            "description": "The finding text. Empty when severity is 'none'. Concrete and actionable.",
        },
        "reasoning": {
            "type": "string",
            "description": "One or two sentences of private reasoning; shown only in the verifier transcript.",
        },
    },
    "required": ["severity", "finding", "reasoning"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_LOCK = threading.RLock()
_SESSIONS: Dict[str, "SessionVerifier"] = {}
_MAX_SESSIONS = 64
_MAX_INFLIGHT = 2
_INFLIGHT = threading.BoundedSemaphore(_MAX_INFLIGHT)

# Facade bound at register() time: PluginContext (get_config / llm / state).
_CTX: Any = None
# Injectable overrides for tests.
_DATA_DIR_OVERRIDE: Optional[Path] = None
_REVIEW_FN_OVERRIDE = None
_RUNTIME_OVERRIDES: Dict[str, Any] = {}  # /verify on|off session-scoped toggles
_RECENT_SESSION: Optional[str] = None


@dataclass
class SessionVerifier:
    session_id: str
    cursor_len: Optional[int] = None
    guard: guard_mod.EmissionGuard = field(default_factory=guard_mod.EmissionGuard)
    pending: Optional[Dict[str, Any]] = None
    immunity_left: int = 0
    stats: Dict[str, Any] = field(
        default_factory=lambda: {
            "reviews": 0, "findings": 0, "blockers": 0, "cautions": 0,
            "notes": 0, "input_tokens": 0, "output_tokens": 0,
            "cost_usd": 0.0, "model": "", "last_error": "",
        }
    )
    last_error: str = ""
    last_review_ts: float = 0.0

    def reset(self) -> None:
        """Re-prime after compaction / session reset / /new."""
        self.cursor_len = None
        self.guard.reset()
        self.pending = None
        self.immunity_left = 0


def _session_key(session_id: Any) -> str:
    return str(session_id or "default")


def _get_state(session_id: Any, create: bool = True) -> Optional[SessionVerifier]:
    key = _session_key(session_id)
    with _LOCK:
        state = _SESSIONS.get(key)
        if state is None and create:
            if len(_SESSIONS) >= _MAX_SESSIONS:
                # Evict the least-recently-created tracked session.
                oldest = next(iter(_SESSIONS))
                _SESSIONS.pop(oldest, None)
            state = SessionVerifier(session_id=key)
            _SESSIONS[key] = state
        return state


def bind_ctx(ctx: Any) -> None:
    global _CTX
    _CTX = ctx


def subsystem_enabled(session_id: Any) -> bool:
    """Config default OR a sticky session-scoped override for this session."""
    override = _RUNTIME_OVERRIDES.get(_session_key(session_id))
    if override is not None:
        return override
    if _CTX is None:
        return False
    try:
        return bool(_CTX.get_config("enabled", False))
    except Exception:
        return False


def _config(ctx: Any, key: str, default: Any) -> Any:
    try:
        return ctx.get_config(key, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _slug_session(session_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", session_id)[:80]
    return slug or "default"


def data_dir() -> Path:
    if _DATA_DIR_OVERRIDE is not None:
        return _DATA_DIR_OVERRIDE
    from plugins.plugin_storage import plugin_data_dir

    return plugin_data_dir(PLUGIN_ID)


def _log_path(session_id: str) -> Path:
    return data_dir() / f"verify.{_slug_session(session_id)}.jsonl"


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("failed to append %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Review loop
# ---------------------------------------------------------------------------

def _render_finding(finding: Dict[str, Any]) -> str:
    severity = finding.get("severity", "note")
    text = str(finding.get("finding", "")).strip()
    return (
        f'<finding verifier="{DEFAULT_VERIFIER}" severity="{severity}" '
        f'guidance="weigh, don\'t blindly obey">\n{text}\n</finding>'
    )


def _resolve_model(ctx: Any) -> tuple:
    """Return (provider, model) overrides, or (None, None) for primary model."""
    raw = str(_config(ctx, "model", "") or "").strip()
    if not raw:
        return None, None
    if "/" in raw:
        provider, _, model = raw.partition("/")
        return provider.strip(), model.strip()
    return None, raw


def _run_review(state: SessionVerifier, delta_text: str) -> None:
    """One review cycle. Runs in a background thread; never raises."""
    ctx = _CTX
    if ctx is None:
        return
    if not _INFLIGHT.acquire(blocking=False):
        # Verifier already busy elsewhere; skip this delta rather than queue.
        logger.debug("verifier busy; skipping delta for %s", state.session_id)
        return
    try:
        state.guard.begin_cycle()
        provider, model = _resolve_model(ctx)
        try:
            from agent.plugin_llm import PluginLlmTrustError  # noqa: F401
        except Exception:
            PluginLlmTrustError = None  # type: ignore[assignment]

        try:
            system_prompt = brief.build_system_prompt()
            llm = ctx.llm
            kwargs: Dict[str, Any] = dict(
                instructions=brief.REVIEW_TASK,
                input=[{"type": "text", "text": delta_text}],
                json_schema=REVIEW_SCHEMA,
                json_mode=True,
                schema_name="verifier_review",
                system_prompt=system_prompt,
                purpose="hermes-verifier review",
                timeout=120.0,
            )
            if provider:
                kwargs["provider"] = provider
            if model:
                kwargs["model"] = model
            result = llm.complete_structured(**kwargs)
        except Exception as exc:
            # Trust-gate failures and provider errors both land here: record,
            # drop the batch, never stall the primary.
            cls = type(exc).__name__
            if PluginLlmTrustError is not None and isinstance(exc, PluginLlmTrustError):
                hint = (
                    "model override denied — set plugins.entries.hermes-verifier."
                    "llm.allow_model_override: true (or grant the llm.model_override "
                    "capability) to use the configured verifier model"
                )
            else:
                hint = str(exc)
            state.last_error = f"{cls}: {hint}"[:300]
            state.stats["last_error"] = state.last_error
            logger.warning("verifier review failed for %s: %s", state.session_id, hint)
            return

        parsed = getattr(result, "parsed", None)
        usage = getattr(result, "usage", None)
        if usage is not None:
            state.stats["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
            state.stats["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
            cost = getattr(usage, "cost_usd", None)
            if cost:
                state.stats["cost_usd"] += float(cost)
        state.stats["model"] = getattr(result, "model", "") or ""
        state.stats["reviews"] += 1
        state.last_review_ts = time.time()
        state.last_error = ""

        if not isinstance(parsed, dict):
            logger.debug("verifier returned non-JSON for %s", state.session_id)
            return

        severity = str(parsed.get("severity") or "none").strip().lower()
        finding_text = str(parsed.get("finding") or "").strip()
        reasoning = str(parsed.get("reasoning") or "").strip()
        if severity == "none" or not finding_text:
            return

        accepted = state.guard.accept(severity, finding_text)
        record = {
            "ts": time.time(),
            "session_id": state.session_id,
            "verifier": DEFAULT_VERIFIER,
            "severity": severity,
            "finding": finding_text,
            "reasoning": reasoning,
            "accepted": accepted,
            "model": state.stats["model"],
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cost_usd": getattr(usage, "cost_usd", None),
            } if usage is not None else {},
        }
        _append_jsonl(_log_path(state.session_id), record)

        if not accepted:
            return

        state.stats["findings"] += 1
        state.stats.setdefault(severity + "s", 0)
        state.stats[severity + "s"] += 1
        finding = {
            "severity": severity,
            "finding": finding_text,
            "reasoning": reasoning,
        }
        with _LOCK:
            state.pending = finding
        logger.info(
            "verifier raised %s finding for %s: %s",
            severity, state.session_id, finding_text[:120],
        )
    finally:
        _INFLIGHT.release()


def _schedule_review(state: SessionVerifier, delta_text: str) -> None:
    if _REVIEW_FN_OVERRIDE is not None:
        _REVIEW_FN_OVERRIDE(state, delta_text)
        return
    thread = threading.Thread(
        target=_run_review, args=(state, delta_text),
        name=f"hermes-verifier:{state.session_id[:24]}", daemon=True,
    )
    thread.start()


# ---------------------------------------------------------------------------
# Hook handlers
# ---------------------------------------------------------------------------

def handle_post_llm_call(
    session_id: Any = "",
    user_message: Any = "",
    conversation_history: Any = None,
    assistant_response: Any = "",
    **kwargs: Any,
) -> None:
    """Primary turn ended: extract the delta and schedule a review."""
    global _RECENT_SESSION
    if not subsystem_enabled(session_id):
        return
    try:
        history = conversation_history if isinstance(conversation_history, list) else []
        state = _get_state(session_id)
        if state is None:  # pragma: no cover - create=True above
            return
        with _LOCK:
            _RECENT_SESSION = state.session_id
        max_chars = int(_config(_CTX, "max_delta_chars", 20000) or 20000)
        new_cursor, delta_text = delta_mod.extract_delta(
            cursor_len=state.cursor_len,
            conversation_history=history,
            assistant_response=str(assistant_response or ""),
            max_chars=max_chars,
        )
        state.cursor_len = new_cursor
        if not delta_text:
            return
        _schedule_review(state, delta_text)
    except Exception as exc:  # pragma: no cover - defense in depth
        logger.warning("post_llm_call handler failed: %s", exc)


def handle_pre_llm_call(session_id: Any = "", **kwargs: Any) -> Optional[Dict[str, str]]:
    """Deliver a pending finding into the next primary turn's context."""
    if not subsystem_enabled(session_id):
        return None
    state = _get_state(session_id, create=False)
    if state is None:
        return None
    with _LOCK:
        finding = state.pending
        state.pending = None
    if finding is None:
        return None
    severity = finding.get("severity", "note")
    # Interrupting delivery arms the immunity cooldown.
    if severity in {"caution", "blocker"}:
        state.immunity_left = int(_config(_CTX, "immunity_turns", 3) or 3)
    else:
        state.immunity_left = max(0, state.immunity_left - 1)
    return {"context": _render_finding(finding)}


def handle_pre_verify(session_id: Any = "", **kwargs: Any) -> Optional[Dict[str, str]]:
    """Steer gate: a pending blocker keeps a coding turn alive.

    Only blockers steer here (cautions at a terminal answer become cards via
    the next-turn injection path). The immunity cooldown gates interruption
    frequency, and the host bounds nudges via agent.max_verify_nudges.
    """
    if not _config(_CTX, "blocker_steer", True):
        return None
    state = _get_state(session_id, create=False)
    if state is None:
        return None
    with _LOCK:
        finding = state.pending
        if finding is None or finding.get("severity") != "blocker":
            return None
        if state.immunity_left > 0:
            # Not allowed to interrupt; finding stays pending and rides the
            # next pre_llm_call injection instead.
            return None
        state.pending = None
        state.immunity_left = int(_config(_CTX, "immunity_turns", 3) or 3)
    text = str(finding.get("finding", "")).strip()
    return {
        "action": "continue",
        "message": (
            f"[verifier blocker] {text}\n"
            f"Acknowledge this finding and address it before finishing the turn."
        ),
    }


def handle_session_reset(session_id: Any = "", **kwargs: Any) -> None:
    state = _get_state(session_id, create=False)
    if state is not None:
        state.reset()


# ---------------------------------------------------------------------------
# Status / dump helpers (shared with the slash command)
# ---------------------------------------------------------------------------

def snapshot() -> Dict[str, Any]:
    with _LOCK:
        sessions = {
            key: {
                "session_id": st.session_id,
                "active": subsystem_enabled(st.session_id),
                "stats": dict(st.stats),
                "pending": st.pending,
                "immunity_left": st.immunity_left,
            }
            for key, st in _SESSIONS.items()
        }
        recent = _RECENT_SESSION
        overrides = dict(_RUNTIME_OVERRIDES)
    return {"recent": recent, "sessions": sessions, "overrides": overrides}


def set_session_override(session_id: Any, enabled: bool) -> None:
    with _LOCK:
        _RUNTIME_OVERRIDES[_session_key(session_id)] = enabled


def clear_session_override(session_id: Any) -> None:
    with _LOCK:
        _RUNTIME_OVERRIDES.pop(_session_key(session_id), None)


def recent_session() -> Optional[str]:
    with _LOCK:
        return _RECENT_SESSION
