"""hermes-verifier dashboard backend.

Mounted at /api/plugins/hermes-verifier/ by the Hermes gateway. Every route
degrades gracefully when the agent-side plugin module is not loaded (plugin
disabled, or an older process): state reads fall back to the JSONL logs on
disk, and writes report unavailable.

The live module is reached through the PluginManager's import namespace
(``hermes_plugins.<slug>``), so the endpoints observe the same runtime state
the hooks update — no file-based handshakes in the common path.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover - import smoke tests without fastapi
    class APIRouter:  # type: ignore[no-redef]
        def get(self, *_a, **_k):  # noqa: D102
            return lambda fn: fn

        def post(self, *_a, **_k):  # noqa: D102
            return lambda fn: fn

        def put(self, *_a, **_k):  # noqa: D102
            return lambda fn: fn

        def websocket(self, *_a, **_k):  # noqa: D102
            return lambda fn: fn

    class WebSocket:  # type: ignore[no-redef]
        pass

    class WebSocketDisconnect(Exception):  # type: ignore[no-redef]
        pass

    def JSONResponse(*_a, **_k):  # type: ignore[no-redef]
        return None


router = APIRouter()

_PLUGIN_ID = "hermes-verifier"
_MODULE_CANDIDATES = (
    "hermes_plugins.hermes_verifier.runtime",
    "hermes_plugins.hermes_verifier",
)


def _safe_int(value: Any, default: int = 0) -> int:
    """Delegate to the live runtime's helper so the two surfaces cannot drift."""
    runtime = _runtime()
    if runtime is not None and hasattr(runtime, "safe_int"):
        return runtime.safe_int(value, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _runtime() -> Any:
    """Return the live runtime module, or None when it is not loaded."""
    for name in _MODULE_CANDIDATES:
        mod = sys.modules.get(name)
        if mod is not None:
            return mod
    return None


def _data_dir() -> Path:
    try:
        from plugins.plugin_storage import plugin_data_dir

        return plugin_data_dir(_PLUGIN_ID)
    except Exception:
        return Path.home() / ".hermes" / "plugin-data" / _PLUGIN_ID


def _config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass
    return {}


def _plugin_settings() -> Dict[str, Any]:
    cfg = _config()
    entries = cfg.get("plugins", {}).get("entries", {}) if isinstance(cfg.get("plugins"), dict) else {}
    entry = entries.get(_PLUGIN_ID, {}) if isinstance(entries, dict) else {}
    settings = entry.get("settings", {}) if isinstance(entry, dict) else {}
    return settings if isinstance(settings, dict) else {}


def _primary_model() -> Dict[str, str]:
    cfg = _config()
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    fallback = cfg.get("fallback_model", {}) if isinstance(cfg.get("fallback_model"), dict) else {}
    return {
        "provider": str(model_cfg.get("provider") or ""),
        "model": str(model_cfg.get("default") or ""),
        "fallback_provider": str(fallback.get("provider") or ""),
        "fallback_model": str(fallback.get("model") or ""),
    }


def _transcript_from_disk(session_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
    data_dir = _data_dir()
    if not data_dir.is_dir():
        return []
    files = sorted(data_dir.glob("verify.*.jsonl"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    if session_id:
        files = [f for f in files if session_id in f.stem]
    records: List[Dict[str, Any]] = []
    for path in files[:8]:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        except OSError:
            continue
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(records) >= limit:
                return records[:limit]
    return records[:limit]


def _state_payload() -> Dict[str, Any]:
    runtime = _runtime()
    if runtime is not None:
        snap = runtime.snapshot()
        payload: Dict[str, Any] = {
            "ok": True,
            "backend": "live",
            "recent_session": snap.get("recent"),
            "sessions": snap.get("sessions", {}),
            "overrides": snap.get("overrides", {}),
        }
    else:
        payload = {
            "ok": True,
            "backend": "degraded",
            "note": "verifier runtime not loaded in this process; showing on-disk logs only",
            "recent_session": None,
            "sessions": {},
            "overrides": {},
        }
    settings = _plugin_settings()
    payload["settings"] = {
        "enabled": bool(settings.get("enabled", False)),
        "model": str(settings.get("verifier_model") or ""),
        "sync_backlog": str(settings.get("sync_backlog") or "off"),
        "immunity_turns": _safe_int(settings.get("immunity_turns", 3), 3),
        "max_delta_chars": _safe_int(settings.get("max_delta_chars", 20000), 20000),
        "blocker_steer": bool(settings.get("blocker_steer", True)),
        # Advisor-mode controls (must mirror /verify status and plugin.yaml).
        "advisor_tools": bool(settings.get("advisor_tools", False)),
        "default_tools": list(settings.get("default_tools") or []),
        "max_reviews_per_session": _safe_int(settings.get("max_reviews_per_session", 0)),
        "advisor_max_tokens": _safe_int(settings.get("advisor_max_tokens", 0)),
        "nudge_first_turns": _safe_int(settings.get("nudge_first_turns", 0)),
    }
    payload["primary"] = _primary_model()
    payload["transcript"] = _transcript_from_disk(None, 60)
    # Cost roll-up: mirrors what /verify status reports, so the two surfaces
    # cannot disagree. est_tokens is the measured-ratio estimate; input_tokens
    # is what the provider billed.
    _sessions = payload.get("sessions") or {}
    payload["cost"] = {
        "reviews": sum(int(((v.get("stats") or {}).get("reviews", 0)) or 0)
                       for s in _sessions.values()
                       for v in (s.get("verifiers") or {}).values()),
        "input_tokens": sum(int(((v.get("stats") or {}).get("input_tokens", 0)) or 0)
                            for s in _sessions.values()
                            for v in (s.get("verifiers") or {}).values()),
        "empty_responses": sum(int((s.get("empty_responses", 0)) or 0)
                               for s in _sessions.values()),
        "last_est_tokens": max(
            [int(((v.get("stats") or {}).get("est_tokens", 0)) or 0)
             for s in _sessions.values()
             for v in (s.get("verifiers") or {}).values()] or [0]),
        "last_input_tokens": max(
            [int(((v.get("stats") or {}).get("last_input_tokens", 0)) or 0)
             for s in _sessions.values()
             for v in (s.get("verifiers") or {}).values()] or [0]),
    }
    payload["last_event_seq"] = (
        runtime.last_event_seq() if runtime is not None else 0
    )
    payload["verifier_session"] = (
        runtime.designated_session()
        if runtime is not None
        else str(settings.get("verifier_session") or "") or None
    )
    return payload


def _events_after(seq: int) -> List[Dict[str, Any]]:
    runtime = _runtime()
    if runtime is not None:
        return runtime.events_after(seq)
    return []


@router.get("/state")
async def state() -> Dict[str, Any]:
    return _state_payload()


@router.get("/transcript")
async def transcript(session: str = "", limit: int = 60) -> Dict[str, Any]:
    limit = max(1, min(limit, 200))
    records = _transcript_from_disk(session or None, limit)
    return {"ok": True, "transcript": records}


@router.get("/findings")
async def findings(session: str = "", limit: int = 60) -> Dict[str, Any]:
    limit = max(1, min(limit, 200))
    records = _transcript_from_disk(session or None, limit * 4)
    accepted = [r for r in records if r.get("accepted", True) and r.get("kind") != "ask"]
    return {"ok": True, "findings": accepted[:limit]}


@router.get("/verifier-session")
async def verifier_session() -> Dict[str, Any]:
    """The designated Verifier Mode session, or null before the first open."""
    runtime = _runtime()
    if runtime is not None:
        session_id = runtime.designated_session()
        if session_id:
            return {"ok": True, "session_id": session_id}
        return {"ok": True, "session_id": None}
    settings = _plugin_settings()
    return {"ok": True, "session_id": str(settings.get("verifier_session") or "") or None}


@router.post("/verifier-session")
async def mark_verifier_session(body: dict) -> JSONResponse:
    """Designate the session the Verifier Mode screen drives."""
    session_id = str(body.get("session_id") or "").strip()
    if not session_id:
        return JSONResponse(status_code=400, content={"ok": False, "error": "session_id required"})
    runtime = _runtime()
    if runtime is None:
        return JSONResponse(status_code=503, content={"ok": False, "error": "verifier backend not loaded"})
    result = runtime.mark_verifier_session(session_id)
    persisted = result.get("persisted") if isinstance(result, dict) else True
    if not persisted:
        # Report the failure instead of claiming a durable designation. The
        # value is live for this process but will NOT survive a restart, and the
        # UI would otherwise show it as saved.
        return JSONResponse(
            status_code=200,
            content={"ok": False, "session_id": session_id,
                     "error": "designation applied for this session but could not "
                              "be persisted; it will not survive a restart"})
    return JSONResponse(content={"ok": True, "session_id": session_id})


@router.post("/ask")
async def ask(body: dict) -> JSONResponse:
    question = str(body.get("question") or "").strip()
    if not question:
        return JSONResponse(status_code=400, content={"ok": False, "error": "question required"})
    runtime = _runtime()
    if runtime is None:
        return JSONResponse(status_code=503, content={"ok": False, "error": "verifier backend not loaded"})
    session_id = body.get("session_id") or runtime.recent_session()
    if not session_id:
        return JSONResponse(status_code=404, content={"ok": False, "error": "no active verifier session"})
    # ask_verifier makes a BLOCKING model call (up to a 120s timeout) and this is
    # a coroutine, so awaiting it inline stalled the whole event loop: every
    # other dashboard request AND the events websocket froze for the duration of
    # the answer. Hand it to a worker thread instead.
    result = await asyncio.to_thread(runtime.ask_verifier, session_id, question)
    return JSONResponse(status_code=200 if result.get("ok") else 409, content=result)


@router.post("/pause")
async def pause(body: dict) -> Dict[str, Any]:
    runtime = _runtime()
    if runtime is None:
        return {"ok": False, "error": "verifier backend not loaded"}
    session_id = body.get("session_id") or runtime.recent_session()
    if not session_id:
        return {"ok": False, "error": "no active verifier session"}
    runtime.set_session_override(session_id, False)
    runtime.publish_event("status", {"session_id": session_id, "enabled": False})
    return {"ok": True}


@router.post("/resume")
async def resume(body: dict) -> Dict[str, Any]:
    runtime = _runtime()
    if runtime is None:
        return {"ok": False, "error": "verifier backend not loaded"}
    session_id = body.get("session_id") or runtime.recent_session()
    if not session_id:
        return {"ok": False, "error": "no active verifier session"}
    runtime.set_session_override(session_id, True)
    runtime.publish_event("status", {"session_id": session_id, "enabled": True})
    return {"ok": True}


@router.put("/model")
async def set_model(body: dict) -> JSONResponse:
    model = str(body.get("model") or "").strip()
    runtime = _runtime()
    if runtime is not None:
        ok = runtime.set_model(model)
        if ok:
            runtime.publish_event("status", {"model": model})
            return JSONResponse(content={"ok": True, "model": model})
        return JSONResponse(status_code=500, content={"ok": False, "error": "failed to persist model"})
    return JSONResponse(status_code=503, content={"ok": False, "error": "verifier backend not loaded"})


@router.websocket("/events")
async def events_ws(websocket: WebSocket) -> None:
    """Push verifier events to the UI; client disconnect ends the loop.

    The receive side is timeout-bounded so event flushing never waits on the
    client. Sends raise on disconnect, which terminates the loop.
    """
    await websocket.accept()
    client_seq = 0
    try:
        while True:
            runtime = _runtime()
            if runtime is not None:
                for event in runtime.events_after(client_seq):
                    client_seq = event["seq"]
                    await websocket.send_json(event)
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=0.2)
                try:
                    requested = int(json.loads(message or "{}").get("seq", client_seq))
                    client_seq = max(client_seq, requested)
                except (ValueError, TypeError, AttributeError,
                        json.JSONDecodeError):
                    # A frame that is not an object (`[]`) raises AttributeError on
                    # .get, and a null seq raises TypeError on int(None). Neither
                    # was caught, so a single malformed client frame propagated
                    # out of the loop and killed the events socket for the whole
                    # dashboard rather than being ignored.
                    pass
            except (asyncio.TimeoutError, WebSocketDisconnect):
                pass
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, RuntimeError):
        return
