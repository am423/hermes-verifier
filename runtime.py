"""Per-session verifier runtime: review loop, delivery, persistence.

The verifier is a passive reviewer. It reviews each primary turn delta in a
background thread (never blocking the primary, unless the catch-up window
``sync_backlog`` is armed), raises at most one finding per verifier per
review, and delivers accepted findings back through the next turn's context
injection (``pre_llm_call``) or — for blockers on coding turns — the
``pre_verify`` steer gate.

Verifiers come from two places:

* the **default verifier** — one per session, model from plugin settings;
* the **roster** — ``VERIFIER.yml`` entries with their own name, enable flag,
  model, tool grant (validated, not yet executable), and specialization
  prompt. Each entry runs as its own verifier with its own emission guard,
  stats, and JSONL log.

Hermes imports are lazy (inside functions) so this module imports cleanly in
tests and under the plugin loader.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import advisor_tools as tools_mod
from . import brief, delta as delta_mod, guard as guard_mod
from . import gate as gate_mod

logger = logging.getLogger("hermes-verifier")

PLUGIN_ID = "hermes-verifier"
DEFAULT_VERIFIER = "default"

# Claude Code found a turn-2 consultation reminder raised Haiku pass rates by
# ~7pp, had no measurable effect on Sonnet, and slightly lowered Opus rates.
# Ported with that caveat attached, and off by default.
NUDGE_TEXT = (
    "You have an independent advisor available. Before committing to an approach "
    "for this task, consider consulting it — it is most useful for non-obvious "
    "design decisions and failure modes you have not yet ruled out."
)

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
            "description": "Diagnostics, shown only in /verify dump raw. Keep to one or two sentences.",
        },
    },
    # severity + finding ARE the deliverable; reasoning is diagnostics. Requiring
    # the trailing diagnostics field meant a response truncated after `finding`
    # failed validation and lost the entire review.
    "required": ["severity", "finding"],
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
_BACKLOG_WAIT_CAP = 30.0
_BACKLOG_POLL = 0.1

# In-flight review threads, so a short-lived process can wait for them at exit.
# Review threads are daemon threads (they must never block a normal exit), which
# means `hermes chat -q` kills them at turn end and the review plus its
# telemetry are lost. drain_reviews() is the explicit, bounded join.
_THREADS_LOCK = threading.Lock()
_ACTIVE_REVIEWS: "set" = set()
_DRAIN_TIMEOUT = 20.0

# Event log for the dashboard websocket + UI: bounded ring with sequence ids.
EVENT_LOG: "List[Dict[str, Any]]" = []
_EVENT_SEQ = 0
_EVENT_LOCK = threading.Lock()

# Facade bound at register() time: PluginContext (get_config / llm / state).
_CTX: Any = None
# Injectable overrides for tests.
_DATA_DIR_OVERRIDE: Optional[Path] = None
_REVIEW_FN_OVERRIDE: Optional[Callable] = None

# Test seam: replaces the engagement decision so tests never spend a model call.
_GATE_FN_OVERRIDE: Optional[Callable] = None
_ROSTER_OVERRIDE: Optional[List[Dict[str, Any]]] = None
_RUNTIME_OVERRIDES: Dict[str, Any] = {}  # /verify on|off session-scoped toggles
_RECENT_SESSION: Optional[str] = None


def publish_event(kind: str, data: Dict[str, Any]) -> None:
    global _EVENT_SEQ
    with _EVENT_LOCK:
        _EVENT_SEQ += 1
        EVENT_LOG.append({"seq": _EVENT_SEQ, "kind": kind, "ts": time.time(), **data})
        while len(EVENT_LOG) > 200:
            EVENT_LOG.pop(0)


def events_after(seq: int) -> List[Dict[str, Any]]:
    with _EVENT_LOCK:
        return [e for e in EVENT_LOG if e["seq"] > seq]


def last_event_seq() -> int:
    with _EVENT_LOCK:
        return _EVENT_SEQ


@dataclass
class Verifier:
    """One reviewer attached to a session (default or roster entry)."""

    slug: str
    name: str
    model: str = ""          # "provider/model" override; "" = inherit default
    tools: List[str] = field(default_factory=list)
    tool_grant: List[str] = field(default_factory=list)
    instructions: str = ""
    shared_instructions: str = ""
    guard: guard_mod.EmissionGuard = field(default_factory=guard_mod.EmissionGuard)
    pending: Optional[Dict[str, Any]] = None
    stats: Dict[str, Any] = field(
        default_factory=lambda: {
            "reviews": 0, "findings": 0, "blockers": 0, "cautions": 0,
            "notes": 0, "input_tokens": 0, "output_tokens": 0,
            "cost_usd": 0.0, "model": "",
        }
    )

    def reset(self) -> None:
        self.guard.reset()
        self.pending = None


@dataclass
class SessionVerifier:
    session_id: str
    cursor_len: Optional[int] = None
    verifiers: Dict[str, Verifier] = field(default_factory=dict)
    immunity_left: int = 0
    backlog: int = 0
    reviewing: bool = False
    consecutive_failures: int = 0
    empty_responses: int = 0
    last_error: str = ""
    roster_signature: Any = None
    reviews_used: int = 0
    turns_seen: int = 0
    last_gate: Any = None
    gate_checked: int = 0
    gate_skipped: int = 0

    def ensure_verifiers(self, roster: List[Dict[str, Any]]) -> None:
        """Rebuild the verifier list when the roster changes."""
        entries = {e.get("slug") or e.get("name") for e in roster if e.get("enabled", True)}
        entries.add(DEFAULT_VERIFIER)
        for slug in list(self.verifiers):
            if slug not in entries:
                del self.verifiers[slug]
        if DEFAULT_VERIFIER not in self.verifiers:
            self.verifiers[DEFAULT_VERIFIER] = Verifier(
                slug=DEFAULT_VERIFIER, name="Default",
            )
        # The default verifier takes its grant from the default_tools setting.
        default_grant = _config(_CTX, "default_tools", []) or []
        self.verifiers[DEFAULT_VERIFIER].tool_grant = tools_mod.resolve_grant(default_grant)
        for entry in roster:
            slug = str(entry.get("slug") or entry.get("name"))
            if not entry.get("enabled", True):
                self.verifiers.pop(slug, None)
                continue
            verifier = self.verifiers.get(slug)
            if verifier is None:
                verifier = Verifier(slug=slug, name=str(entry.get("name", slug)))
                self.verifiers[slug] = verifier
            verifier.name = str(entry.get("name", slug))
            verifier.model = str(entry.get("model") or "")
            verifier.tools = list(entry.get("tools") or [])
            verifier.instructions = str(entry.get("instructions") or "")
            verifier.shared_instructions = str(entry.get("shared_instructions") or "")
            verifier.tool_grant = tools_mod.resolve_grant(verifier.tools)

    def enabled_verifiers(self) -> List[Verifier]:
        return [v for v in self.verifiers.values()]

    def reset(self) -> None:
        """Re-prime after compaction / session reset / /new."""
        self.cursor_len = None
        self.immunity_left = 0
        self.backlog = 0
        self.reviewing = False
        self.reviews_used = 0
        self.turns_seen = 0
        self.roster_signature = None
        self.empty_responses = 0
        for verifier in self.verifiers.values():
            verifier.reset()


def _session_key(session_id: Any) -> str:
    return str(session_id or "default")


def _get_state(session_id: Any, create: bool = True) -> Optional[SessionVerifier]:
    key = _session_key(session_id)
    with _LOCK:
        state = _SESSIONS.get(key)
        if state is None and create:
            if len(_SESSIONS) >= _MAX_SESSIONS:
                oldest = next(iter(_SESSIONS))
                _SESSIONS.pop(oldest, None)
            state = SessionVerifier(session_id=key)
            state.ensure_verifiers(discover_roster())
            _SESSIONS[key] = state
        return state


def _sync_roster(state: SessionVerifier) -> None:
    """Re-resolve the roster and tool grants for a live session.

    ``ensure_verifiers`` only ran at session creation, so a ``VERIFIER.yml``
    edited mid-session — or a ``default_tools`` change made from Settings —
    was silently ignored until the next session. Cheap to re-run (a dict
    comparison of slugs plus one setting read), so it runs at every turn.
    """
    roster = discover_roster()
    signature = [(str(e.get("slug") or e.get("name")), bool(e.get("enabled", True)),
                  str(e.get("model") or ""), tuple(e.get("tools") or []),
                  str(e.get("instructions") or "")) for e in roster]
    if signature != state.roster_signature:
        state.roster_signature = signature
        state.ensure_verifiers(roster)
        return
    # Same roster shape, but default_tools / advisor_tools may still have moved.
    default_grant = _config(_CTX, "default_tools", []) or []
    resolved = tools_mod.resolve_grant(default_grant)
    if resolved != state.verifiers[DEFAULT_VERIFIER].tool_grant:
        state.verifiers[DEFAULT_VERIFIER].tool_grant = resolved


def bind_ctx(ctx: Any) -> None:
    global _CTX
    _CTX = ctx


def discover_roster() -> List[Dict[str, Any]]:
    if _ROSTER_OVERRIDE is not None:
        return list(_ROSTER_OVERRIDE)
    try:
        return brief.load_roster()
    except Exception as exc:  # pragma: no cover - never break the loop
        logger.warning("roster discovery failed: %s", exc)
        return []


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


def safe_int(value: Any, default: int = 0) -> int:
    """Coerce a config or telemetry value to int without ever raising.

    Settings are hand-edited YAML. ``nudge_first_turns: abc`` used to raise
    ValueError straight out of ``/verify status`` and 500 the dashboard
    ``/state`` endpoint -- i.e. a bad setting crashed the very surface that
    would have shown the operator what was wrong. Degrade to the default
    instead.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _log_path(session_id: str, slug: str) -> Path:
    return data_dir() / f"verify.{_slug_session(session_id)}.{slug}.jsonl"


def _gate_log_path(session_id: str) -> Path:
    return data_dir() / f"gate.{_slug_session(session_id)}.jsonl"


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("failed to append %s: %s", path, exc)


def read_jsonl_tail(path: Path, limit: int = 100) -> List[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return []
    records: List[Dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


# ---------------------------------------------------------------------------
# Review loop
# ---------------------------------------------------------------------------

def _render_finding(slug: str, finding: Dict[str, Any]) -> str:
    severity = finding.get("severity", "note")
    text = str(finding.get("finding", "")).strip()
    attrs = [f'verifier="{slug}"', f'severity="{severity}"']
    # Name the turn this finding reviewed whenever it is not the turn now
    # running. Without it the finding reads as a judgement on the CURRENT turn's
    # work, which is how a review of turn 4 gets answered by turn 5.
    origin = int(finding.get("origin_turn") or 0)
    if origin:
        attrs.append(f'reviewed_turn="{origin}"')
    if slug == DEFAULT_VERIFIER:
        attrs = [a for a in attrs if not a.startswith("verifier=")]
    return (
        f"<finding {' '.join(attrs)} guidance=\"weigh, don't blindly obey\">\n"
        f"{text}\n</finding>"
    )


def _split_model(raw: str) -> tuple:
    """Return (provider, model) from a 'provider/model' selector."""
    raw = (raw or "").strip()
    if not raw:
        return None, None
    if "/" in raw:
        provider, _, model = raw.partition("/")
        return provider.strip(), model.strip()
    return None, raw


def _resolve_model_for(ctx: Any, verifier: Verifier) -> tuple:
    """Per-verifier model: roster entry wins, else the plugin setting.

    The setting key is ``verifier_model`` — a bare ``model`` key is a reserved
    root in the plugin settings namespace and is rejected on read and write.
    """
    selector = verifier.model or str(_config(ctx, "verifier_model", "") or "")
    return _split_model(selector)


def _call_structured(ctx: Any, *, system_prompt: str, instructions: str,
                     input_text: str, provider: Optional[str],
                     model: Optional[str],
                     max_tokens: Optional[int] = None) -> Any:
    from agent.plugin_llm import PluginLlmTrustError  # noqa: F401

    kwargs: Dict[str, Any] = dict(
        instructions=instructions,
        input=[{"type": "text", "text": input_text}],
        json_schema=REVIEW_SCHEMA,
        json_mode=True,
        schema_name="verifier_review",
        system_prompt=system_prompt,
        purpose="hermes-verifier review",
        timeout=120.0,
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if provider:
        kwargs["provider"] = provider
    if model:
        kwargs["model"] = model
    return ctx.llm.complete_structured(**kwargs)


def _record_and_deliver(state: SessionVerifier, verifier: Verifier,
                        parsed: Dict[str, Any], accepted: bool,
                        usage: Any, delta_excerpt: str = "",
                        prompt_parts: Optional[Dict[str, Any]] = None,
                        origin_turn: int = 0) -> None:
    severity = str(parsed.get("severity") or "none").strip().lower()
    finding_text = str(parsed.get("finding") or "").strip()
    reasoning = str(parsed.get("reasoning") or "").strip()
    record = {
        "ts": time.time(),
        "session_id": state.session_id,
        "verifier": verifier.slug,
        "severity": severity,
        "finding": finding_text,
        "reasoning": reasoning,
        "accepted": accepted,
        # The primary turn whose delta was reviewed. A finding is accounted to
        # this turn, never to the turn that later delivers it.
        "origin_turn": origin_turn,
        "model": verifier.stats["model"],
        "delta_excerpt": delta_excerpt[:2000],
        "prompt_parts": prompt_parts or {},
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cost_usd": getattr(usage, "cost_usd", None),
        } if usage is not None else {},
    }
    _append_jsonl(_log_path(state.session_id, verifier.slug), record)
    publish_event("review", {"session_id": state.session_id,
                             "verifier": verifier.slug, "severity": severity,
                             "finding": finding_text[:240], "accepted": accepted,
                             "origin_turn": origin_turn})
    if not accepted:
        return
    verifier.stats["findings"] += 1
    verifier.stats[severity + "s"] = verifier.stats.get(severity + "s", 0) + 1
    verifier.pending = {
        "severity": severity,
        "finding": finding_text,
        "reasoning": reasoning,
        "origin_turn": origin_turn,
    }
    publish_event("finding", {"session_id": state.session_id,
                              "verifier": verifier.slug, "severity": severity,
                              "finding": finding_text[:240],
                              "origin_turn": origin_turn})
    logger.info(
        "verifier %s raised %s finding for %s: %s",
        verifier.slug, severity, state.session_id, finding_text[:120],
    )


def _plugin_llm_trust_error_type() -> Any:
    try:
        from agent.plugin_llm import PluginLlmTrustError

        return PluginLlmTrustError
    except Exception:
        return None


def _review_one(state: SessionVerifier, verifier: Verifier, delta_text: str,
                effort: str = "deep", origin_turn: int = 0) -> None:
    ctx = _CTX
    if ctx is None:
        return
    trust_error_type = _plugin_llm_trust_error_type()
    verifier.guard.begin_cycle()
    provider, model = _resolve_model_for(ctx, verifier)
    try:
        system_prompt = brief.build_system_prompt(
            extra_instructions=verifier.shared_instructions + "\n\n" + verifier.instructions
        )
        # Tools only engage when the operator opts in AND the roster granted some.
        grant = verifier.tool_grant if _config(ctx, "advisor_tools", False) else []
        if grant:
            block = tools_mod.render_available(grant)
            if block:
                system_prompt = f"{system_prompt}\n\n{block}"
        cap = int(_config(ctx, "advisor_max_tokens", 0) or 0)
        if effort == "routine":
            # The gate judged this turn trivial. Hold the review to a smaller
            # output budget -- but never below 1024, the measured floor under which
            # a response is truncated mid-JSON and the whole review is lost.
            routine_cap = int(_config(ctx, "advisor_routine_tokens", 1024) or 1024)
            cap = min(cap, routine_cap) if cap else routine_cap
        cap_kwargs = {"max_tokens": cap} if cap >= 1024 else {}
        transcript = delta_text
        result = None
        empty_retried = False
        # Bounded: the advisor may verify a claim, then must produce its envelope.
        # A verifier with no grant is never told about tools, so it makes exactly
        # one call — parse_tool_call would refuse anyway, but skipping the loop
        # keeps its transcript and prompt byte-identical to the pre-tools build.
        attempts = tools_mod.MAX_TOOL_CALLS + 1 if grant else 1
        for _attempt in range(attempts):
            result = _call_structured(
                ctx, system_prompt=system_prompt, instructions=brief.REVIEW_TASK,
                input_text=transcript, provider=provider, model=model, **cap_kwargs,
            )
            # An empty completion surfaces as content_type="text" with no payload:
            # _parse_structured_text returns (None, "text") on an empty body.
            # Retry once, inside this try so the failure bookkeeping still applies.
            if (not empty_retried and getattr(result, "parsed", None) is None
                    and not getattr(result, "text", "")):
                empty_retried = True
                with _LOCK:
                    state.empty_responses += 1
                logger.warning(
                    "verifier %s got an empty completion for %s (empty #%d); retrying",
                    verifier.slug, state.session_id, state.empty_responses,
                )
                result = _call_structured(
                    ctx, system_prompt=system_prompt, instructions=brief.REVIEW_TASK,
                    input_text=transcript, provider=provider, model=model, **cap_kwargs,
                )
            call = tools_mod.parse_tool_call(getattr(result, "text", "") or "", grant)
            if call is None:
                break
            output = tools_mod.dispatch(ctx, call)
            transcript = (
                f"{delta_text}\n\n---\nTool result from {call['tool']}:\n{output}\n"
                "Now give your final review envelope (severity/finding/reasoning)."
            )
    except Exception as exc:
        cls = type(exc).__name__
        if trust_error_type is not None and isinstance(exc, trust_error_type):
            hint = (
                "model override denied — set plugins.entries.hermes-verifier."
                "llm.allow_model_override: true (or grant the llm.model_override "
                "capability) to use the configured verifier model"
            )
        else:
            hint = str(exc)
        state.last_error = f"{cls}: {hint}"[:300]
        state.consecutive_failures += 1
        publish_event("error", {"session_id": state.session_id,
                                "verifier": verifier.slug, "error": state.last_error})
        logger.warning("verifier %s review failed for %s: %s",
                       verifier.slug, state.session_id, hint)
        return

    state.consecutive_failures = 0
    state.last_error = ""
    usage = getattr(result, "usage", None)
    if usage is not None:
        verifier.stats["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
        verifier.stats["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
        cost = getattr(usage, "cost_usd", None)
        if cost:
            verifier.stats["cost_usd"] += float(cost)
    verifier.stats["model"] = getattr(result, "model", "") or ""
    verifier.stats["reviews"] += 1

    parsed = getattr(result, "parsed", None)
    if not isinstance(parsed, dict):
        # Non-empty but unparseable, or still empty after the in-loop retry.
        # Both are surfaced rather than dropped silently.
        text_len = len(getattr(result, "text", "") or "") if result is not None else 0
        if text_len:
            # A cut-off JSON response (e.g. advisor_max_tokens truncation) lands
            # here. It must be visible: silent loss looks identical to "the
            # advisor had nothing to say".
            with _LOCK:
                state.last_error = (
                    f"unparseable review response ({text_len} chars, not JSON) — "
                    "likely truncated by advisor_max_tokens or the provider"
                )[:300]
            publish_event("error", {"session_id": state.session_id,
                                    "verifier": verifier.slug,
                                    "error": state.last_error})
            logger.warning("verifier %s returned unparseable output for %s (%d chars)",
                           verifier.slug, state.session_id, text_len)
        else:
            with _LOCK:
                state.last_error = (
                    f"empty completion persisted after retry (empty #"
                    f"{state.empty_responses})"
                )[:300]
            publish_event("error", {"session_id": state.session_id,
                                    "verifier": verifier.slug,
                                    "error": state.last_error})
            logger.warning("verifier %s still empty after retry for %s",
                           verifier.slug, state.session_id)
        return
    severity = str(parsed.get("severity") or "none").strip().lower()
    finding_text = str(parsed.get("finding") or "").strip()
    parts = {
        "system_prompt": len(system_prompt),
        "instructions": len(brief.REVIEW_TASK),
        "transcript": len(transcript),
        # Token estimate for budget tuning. Measured, not modelled: across three
        # live reviews the provider billed ~2.48x the tiktoken/cl100k_base count
        # (ratios 2.46 / 2.39 / 2.59), i.e. ~1.60 chars per billed token on this
        # content. The authoritative number is usage.input_tokens, in this row.
        "est_tokens": int((len(system_prompt) + len(transcript)) / 1.60),
    }
    # Latest values for /verify status. Updated every review (not setdefault).
    verifier.stats["est_tokens"] = parts["est_tokens"]
    verifier.stats["last_input_tokens"] = int(getattr(usage, "input_tokens", 0) or 0)
    if severity == "none" or not finding_text:
        # Still record: cost must be measurable for reviews that find nothing,
        # otherwise the majority of spend is invisible.
        _record_and_deliver(state, verifier, parsed, False, usage,
                            delta_excerpt=delta_text, prompt_parts=parts,
                            origin_turn=origin_turn)
        return
    # Restatement suppression lives in guard.py as exact-text dedupe. A
    # semantic pass was built here and removed: measured against realistic
    # same-session negatives it could not separate a paraphrase from a distinct
    # finding (0.29-0.81 overlapping 0.59-0.78), so enabling it suppressed
    # genuine findings. See sidecar/NEGATIVE-RESULTS.md.
    accepted = verifier.guard.accept(severity, finding_text)
    _record_and_deliver(state, verifier, parsed, accepted, usage,
                        delta_excerpt=delta_text, prompt_parts=parts,
                        origin_turn=origin_turn)


def _run_review(state: SessionVerifier, delta_text: str,
                origin_turn: int = 0) -> None:
    """One review cycle for every enabled verifier. Background thread.

    ``origin_turn`` is the primary turn that produced ``delta_text``. It is
    carried through to the finding so the review is accounted to the turn that
    produced it, not the turn that receives it.
    """
    # _schedule_review counts this delta into backlog BEFORE starting the
    # thread, so this thread owns that slot and must consume it on EVERY exit
    # path, including the two early returns below. Missing them leaked backlog
    # upward permanently, and because _backlog_wait waits while backlog >=
    # threshold, a leaked backlog of 1 stalls every later turn for the full
    # _BACKLOG_WAIT_CAP. The outer try makes that structural rather than a rule
    # each new early return has to remember.
    try:
        if _CTX is None:
            return
        if not _INFLIGHT.acquire(blocking=False):
            logger.debug("verifier busy; skipping delta for %s", state.session_id)
            return
        try:
            with _LOCK:
                state.reviewing = True
            publish_event("review_started", {"session_id": state.session_id})
            try:
                decision = _gate_decision(state, delta_text)
                with _LOCK:
                    state.last_gate = decision
                    state.gate_checked += 1
                    if not decision.engage:
                        state.gate_skipped += 1
                publish_event("gate_decided", {
                    "session_id": state.session_id,
                    "engage": decision.engage,
                    "effort": decision.tier,
                    "source": decision.source,
                })
                # Persist the decision, INCLUDING a skip. A skipped turn runs no
                # verifier and so writes no review row, which made "why was this
                # turn not reviewed?" unanswerable after the fact. The digest
                # excerpt is the audit trail: it is what the decision was
                # actually made on.
                _append_jsonl(_gate_log_path(state.session_id), {
                    "ts": time.time(),
                    "session_id": state.session_id,
                    "origin_turn": origin_turn,
                    "engage": decision.engage,
                    "effort": decision.tier,
                    "source": decision.source,
                    "digest_excerpt": gate_mod.digest_from_delta(delta_text)[:400],
                })
                # A skip ends the cycle here, before any verifier call is paid
                # for. The finally below still releases the slot, because the
                # delta has been consumed either way.
                if not decision.engage:
                    logger.debug("gate skipped %s (%s)", state.session_id,
                                 decision.source)
                    return
                for verifier in state.enabled_verifiers():
                    _review_one(state, verifier, delta_text,
                                effort=decision.tier, origin_turn=origin_turn)
            finally:
                with _LOCK:
                    state.reviewing = False
                publish_event("review_finished", {"session_id": state.session_id})
        finally:
            _INFLIGHT.release()
    finally:
        # Both things _schedule_review handed this thread -- the backlog slot and
        # the registry entry -- are released on EVERY path, so a review that
        # cannot start leaks neither.
        with _LOCK:
            state.backlog = max(0, state.backlog - 1)
        with _THREADS_LOCK:
            _ACTIVE_REVIEWS.discard(threading.current_thread())


def _gate_decision(state: SessionVerifier, delta_text: str) -> "gate_mod.GateDecision":
    """Run the engagement gate, or the test override.

    Costs one cheap call (~180 input tokens measured, against ~9,500 for the
    review it can avoid). Off by default so an existing install is unchanged;
    when off this returns an unconditional engage with no model call at all.
    """
    if _GATE_FN_OVERRIDE is not None:
        return _GATE_FN_OVERRIDE(state, delta_text)
    if not _config(_CTX, "gate_enabled", False):
        return gate_mod.GateDecision(engage=True, effort="deep", source="disabled")
    provider, model = _resolve_model_for(_CTX, state.enabled_verifiers()[0]) \
        if state.enabled_verifiers() else (None, None)
    try:
        return gate_mod.decide(
            _CTX,
            digest=gate_mod.digest_from_delta(delta_text),
            provider=provider,
            model=model,
        )
    except Exception as exc:  # decide() does not raise, but never risk a review on this
        logger.warning("gate error for %s: %s", state.session_id, exc)
        return gate_mod.FAIL_SAFE


def _schedule_review(state: SessionVerifier, delta_text: str,
                     origin_turn: int = 0) -> None:
    # Claude Code caps advisor calls per request (max_uses); OMP has no cap at
    # all, so a noisy roster silently runs up bill. This is the session-scoped
    # version. The counter increments before dispatch, so a review that fails
    # still consumes budget — that is what makes it a cost control.
    cap = int(_config(_CTX, "max_reviews_per_session", 0) or 0)
    with _LOCK:
        if cap and state.reviews_used >= cap:
            logger.debug("review budget exhausted for %s (%d)", state.session_id, cap)
            return
        state.reviews_used += 1
    if _REVIEW_FN_OVERRIDE is not None:
        _REVIEW_FN_OVERRIDE(state, delta_text, origin_turn)
        return
    with _LOCK:
        state.backlog += 1
    thread = threading.Thread(
        target=_run_review, args=(state, delta_text, origin_turn),
        name=f"hermes-verifier:{state.session_id[:24]}", daemon=True,
    )
    with _THREADS_LOCK:
        _ACTIVE_REVIEWS.add(thread)
    thread.start()


def drain_reviews(timeout: float = _DRAIN_TIMEOUT) -> bool:
    """Wait briefly for in-flight reviews so a short-lived process does not
    kill them. Returns True when everything finished within the timeout.

    Review threads are daemons so they never block a normal exit; this is called
    explicitly at process teardown and is always bounded.
    """
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        with _THREADS_LOCK:
            pending = [t for t in _ACTIVE_REVIEWS if t.is_alive()]
        if not pending:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("drain_reviews timed out with %d review(s) in flight",
                           len(pending))
            return False
        pending[0].join(timeout=min(0.1, remaining))


# ---------------------------------------------------------------------------
# Catch-up window (sync_backlog)
# ---------------------------------------------------------------------------

def _parse_sync_backlog(raw: Any) -> int:
    """Normalize the sync_backlog setting to a turn threshold (0 = off).

    OMP's advisor.syncBacklog takes the enum strings "1" | "3" | "5" or off.
    Unquoted YAML gives us an int for ``sync_backlog: 1``, so accept both
    shapes and treat anything unrecognised as off rather than raising — the
    caller runs inside a broad except that would silently drop the finding.
    """
    if raw is None:
        return 0
    text = str(raw).strip().lower()
    if text in {"off", "none", "", "0"}:
        return 0
    return int(text) if text.isdigit() and 1 <= int(text) <= 5 else 0


def _backlog_wait(state: SessionVerifier, threshold: Optional[int]) -> None:
    """Optionally pace the primary turn while the verifier catches up.

    Only the primary's own turn boundary calls this; the wait is capped so a
    failing verifier can never stall the session. A verifier in a failure
    streak releases waiters immediately.
    """
    if not threshold or threshold < 1:
        return
    deadline = time.monotonic() + _BACKLOG_WAIT_CAP
    while time.monotonic() < deadline:
        with _LOCK:
            backlog = state.backlog
            failing = state.consecutive_failures >= 3
        if backlog < threshold or failing:
            return
        time.sleep(_BACKLOG_POLL)


# ---------------------------------------------------------------------------
# Ask channel (desktop composer → verifier)
# ---------------------------------------------------------------------------

def ask_verifier(session_id: Any, question: str) -> Dict[str, Any]:
    """Send the verifier a direct question. Returns a structured answer."""
    ctx = _CTX
    if ctx is None:
        return {"ok": False, "error": "verifier backend not loaded"}
    state = _get_state(session_id, create=False)
    if state is None:
        return {"ok": False, "error": "no active verifier session (run a turn first)"}
    verifier = state.verifiers.get(DEFAULT_VERIFIER)
    if verifier is None:
        return {"ok": False, "error": "default verifier missing"}
    if not _INFLIGHT.acquire(blocking=False):
        return {"ok": False, "error": "verifier busy; try again shortly"}
    try:
        provider, model = _resolve_model_for(ctx, verifier)
        system_prompt = brief.build_system_prompt(
            extra_instructions=verifier.shared_instructions + "\n\n" + verifier.instructions
        )
        try:
            result = _call_structured(
                ctx, system_prompt=system_prompt,
                instructions=f"Answer this question about the session:\n\n{question}",
                input_text="(direct question; no transcript delta provided)",
                provider=provider, model=model,
            )
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        usage = getattr(result, "usage", None)
        record = {
            "ts": time.time(), "session_id": state.session_id,
            "verifier": verifier.slug, "kind": "ask",
            "question": question[:1000],
            "answer": getattr(result, "text", "")[:4000],
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
            } if usage is not None else {},
        }
        _append_jsonl(_log_path(state.session_id, verifier.slug), record)
        return {
            "ok": True,
            "answer": getattr(result, "text", ""),
            "model": getattr(result, "model", "") or "",
        }
    finally:
        _INFLIGHT.release()


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
        _sync_roster(state)
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
        # The turn that PRODUCED this delta. pre_llm_call already incremented
        # turns_seen for the turn that is now ending, so this is that turn's
        # number. A finding from this delta belongs to this turn even though the
        # review finishes after it, and it must never be accounted to the turn
        # that happens to deliver it.
        with _LOCK:
            origin_turn = state.turns_seen
        _schedule_review(state, delta_text, origin_turn)
        backlog_threshold = _parse_sync_backlog(_config(_CTX, "sync_backlog", "off"))
        if backlog_threshold:
            _backlog_wait(state, backlog_threshold)
    except Exception as exc:  # pragma: no cover - defense in depth
        logger.warning("post_llm_call handler failed: %s", exc)


def handle_pre_llm_call(session_id: Any = "", **kwargs: Any) -> Optional[Dict[str, str]]:
    """Deliver pending findings into the next primary turn's context.

    Also drives the optional first-turn nudge. The nudge goes through
    ``pre_llm_call`` rather than the system prompt so the cached prefix is
    preserved — injecting into the user message is the documented cache-safe
    path.
    """
    # Only materialize a session when the subsystem is actually enabled for it:
    # a hook firing for an unrelated or disabled session must not create state
    # that lingers in _SESSIONS. The nudge window needs a session to count in,
    # so it is checked after enablement but before any early return.
    if not subsystem_enabled(session_id):
        return None
    state = _get_state(session_id)
    if state is None:  # pragma: no cover - create=True above
        return None
    nudge_turns = int(_config(_CTX, "nudge_first_turns", 0) or 0)
    nudge = None
    with _LOCK:
        if nudge_turns and state.turns_seen < nudge_turns:
            nudge = NUDGE_TEXT
        state.turns_seen += 1
    with _LOCK:
        pending = {slug: v.pending for slug, v in state.verifiers.items()
                   if v.pending is not None}
        for verifier in state.verifiers.values():
            verifier.pending = None
        # The turn now starting. A finding is accounted to its origin_turn, so
        # this is recorded only to make the gap visible -- a finding that lands
        # many turns late is a pacing symptom, not a finding about this turn.
        delivered_turn = state.turns_seen
    for slug, finding in pending.items():
        publish_event("finding_delivered", {
            "session_id": state.session_id,
            "verifier": slug,
            "severity": finding.get("severity", "note"),
            "origin_turn": int(finding.get("origin_turn") or 0),
            "delivered_turn": delivered_turn,
        })
    if not pending and not nudge:
        # A quiet turn still ticks the immunity clock down. Without this the
        # cooldown could never drain, so one early blocker would mute the
        # advisor for the rest of the session.
        state.immunity_left = max(0, state.immunity_left - 1)
        return None
    severities = {f.get("severity", "note") for f in pending.values()}
    if severities & {"caution", "blocker"}:
        # Only arm the cooldown when it is not already running. Re-arming on
        # every delivery would make a recurring blocker permanently immune,
        # because each delivery reset the counter before it could count down.
        state.immunity_left = max(state.immunity_left,
                                  int(_config(_CTX, "immunity_turns", 3) or 3))
    else:
        state.immunity_left = max(0, state.immunity_left - 1)
    blocks = [_render_finding(slug, finding) for slug, finding in pending.items()]
    if nudge:
        blocks.insert(0, nudge)
    return {"context": "\n\n".join(blocks)}


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
        blocker: Optional[Dict[str, Any]] = None
        blocker_verifier: Optional[Verifier] = None
        for verifier in state.enabled_verifiers():
            if verifier.pending and verifier.pending.get("severity") == "blocker":
                blocker = verifier.pending
                verifier.pending = None
                blocker_verifier = verifier
                break
        if blocker is None:
            return None
        if state.immunity_left > 0:
            # Not allowed to interrupt; the finding stays pending and rides the
            # next pre_llm_call injection instead.
            if blocker_verifier is not None:
                blocker_verifier.pending = blocker
            return None
        state.immunity_left = int(_config(_CTX, "immunity_turns", 3) or 3)
    text = str(blocker.get("finding", "")).strip()
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
# Settings writer (desktop model dropdown)
# ---------------------------------------------------------------------------

def set_model(value: str) -> bool:
    """Persist the default verifier model through the plugin settings namespace.

    Writes ``settings.verifier_model`` (``model`` is a reserved settings root).
    """
    cleaned = (value or "").strip()
    if _CTX is not None:
        try:
            _CTX.set_config("verifier_model", cleaned)
            return True
        except Exception as exc:
            logger.warning("ctx.set_config('verifier_model') failed: %s", exc)
    try:  # fallback: write config directly (gateway dashboard context)
        from hermes_cli import config as config_mod

        partial = {
            "plugins": {"entries": {PLUGIN_ID: {"settings": {"verifier_model": cleaned}}}}
        }
        full_path = ("plugins", "entries", PLUGIN_ID, "settings", "verifier_model")
        config_mod.save_config(partial, preserve_keys={full_path}, merge_existing=True)
        return True
    except Exception as exc:
        logger.warning("direct config write for verifier_model failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Designated verifier session (the Verifier Mode screen's real session)
# ---------------------------------------------------------------------------

_VERIFIER_SESSION: Optional[str] = None


def mark_verifier_session(session_id: Any) -> str:
    """Designate ``session_id`` as the verifier session and attach the
    verifier to it. Persists the designation in plugin settings so the
    desktop UI can re-open the same session after a restart.
    """
    global _VERIFIER_SESSION, _RECENT_SESSION
    key = _session_key(session_id)
    with _LOCK:
        _VERIFIER_SESSION = key
        _RECENT_SESSION = key
        _RUNTIME_OVERRIDES[key] = True
    if _CTX is not None:
        try:
            _CTX.set_config("verifier_session", key)
        except Exception as exc:
            logger.warning("ctx.set_config('verifier_session') failed: %s", exc)
    publish_event("status", {"session_id": key, "enabled": True, "designated": True})
    return key


def designated_session() -> Optional[str]:
    global _VERIFIER_SESSION
    with _LOCK:
        key = _VERIFIER_SESSION
    if key:
        return key
    # Fall back to the persisted designation (fresh process, no marks yet).
    persisted = _config(_CTX, "verifier_session", "") if _CTX is not None else ""
    if persisted:
        with _LOCK:
            _VERIFIER_SESSION = str(persisted)
            _RUNTIME_OVERRIDES.setdefault(str(persisted), True)
        return str(persisted)
    return None


# ---------------------------------------------------------------------------
# Status / dump helpers (shared with the slash command + dashboard)
# ---------------------------------------------------------------------------

def snapshot() -> Dict[str, Any]:
    with _LOCK:
        sessions = {}
        for key, st in _SESSIONS.items():
            verifiers = {}
            for slug, verifier in st.verifiers.items():
                verifiers[slug] = {
                    "name": verifier.name,
                    "model": verifier.model,
                    "tools": list(verifier.tools),
                    "stats": dict(verifier.stats),
                    "pending": verifier.pending,
                }
            sessions[key] = {
                "session_id": st.session_id,
                "active": subsystem_enabled(st.session_id),
                "reviewing": st.reviewing,
                "backlog": st.backlog,
                "reviews_used": st.reviews_used,
                # Counted in _review_one but previously never exposed, so both
                # consumers (commands._status_lines, plugin_api state) read it
                # via .get(..., 0) and reported 0 forever. That silently disabled
                # the one documented signal that the reviewer is DEAD rather
                # than finding nothing: "a rising empty count with a flat review
                # count means the advisor model is not answering".
                "empty_responses": st.empty_responses,
                "gate_checked": st.gate_checked,
                "gate_skipped": st.gate_skipped,
                "last_gate": ({"engage": st.last_gate.engage,
                               "effort": st.last_gate.tier,
                               "source": st.last_gate.source}
                              if st.last_gate is not None else None),
                "immunity_left": st.immunity_left,
                "last_error": st.last_error,
                "verifiers": verifiers,
            }
        recent = _RECENT_SESSION
        overrides = dict(_RUNTIME_OVERRIDES)
    return {"recent": recent, "sessions": sessions, "overrides": overrides,
            "events": list(EVENT_LOG[-100:]), "last_event_seq": last_event_seq()}


def set_session_override(session_id: Any, enabled: bool) -> None:
    with _LOCK:
        _RUNTIME_OVERRIDES[_session_key(session_id)] = enabled


def clear_session_override(session_id: Any) -> None:
    with _LOCK:
        _RUNTIME_OVERRIDES.pop(_session_key(session_id), None)


def recent_session() -> Optional[str]:
    with _LOCK:
        return _RECENT_SESSION
