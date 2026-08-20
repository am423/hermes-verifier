"""/verify slash command.

Session-scoped control surface while the desktop UI does not exist yet:

    /verify                 show status (same as /verify status)
    /verify on|off          toggle the verifier for the current session
    /verify reset           re-prime the current session's verifier state
    /verify dump [raw]      show recent accepted findings (raw adds reasoning)

Handlers receive only the raw argument string, so "the current session" is
the most recently active primary session the hooks have seen.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from . import runtime

_HELP = """\
/verify — session verifier control

Subcommands:
  status         Runtime state, model, findings, token usage, cost.
  on | off       Enable / disable the verifier for this session (session-scoped).
  reset          Re-prime this session's verifier (clear dedupe + cursor).
  dump [raw]     Recent accepted findings. 'raw' includes private reasoning.

Config (plugins.entries.hermes-verifier.settings):
  enabled, model, sync_backlog, immunity_turns, max_delta_chars, blocker_steer

Model override requires plugins.entries.hermes-verifier.llm.allow_model_override:
true (or the llm.model_override capability grant).
"""


def _fmt_cost(value: float) -> str:
    if not value:
        return "—"
    return f"${value:.4f}"


def _status_lines() -> List[str]:
    snap = runtime.snapshot()
    cfg = runtime._CTX
    lines: List[str] = []
    default_enabled = bool(runtime._config(cfg, "enabled", False))
    model_cfg = str(runtime._config(cfg, "model", "") or "").strip() or "(primary model)"
    lines.append(f"[hermes-verifier] config default: {'on' if default_enabled else 'off'} | model: {model_cfg}")

    sessions = snap["sessions"]
    if not sessions:
        lines.append("  No sessions seen yet. Review runs after each primary turn.")
        return lines

    for key in sorted(sessions):
        sess = sessions[key]
        active = "on " if sess["active"] else "off"
        reviewing = " [reviewing]" if sess.get("reviewing") else ""
        verifiers = sess.get("verifiers") or {}
        agg = {"reviews": 0, "findings": 0, "notes": 0, "cautions": 0,
               "blockers": 0, "input_tokens": 0, "output_tokens": 0,
               "cost_usd": 0.0}
        models = []
        for slug, verifier in sorted(verifiers.items()):
            stats = verifier.get("stats") or {}
            for field in agg:
                agg[field] += stats.get(field, 0) or 0
            if stats.get("model"):
                models.append(stats["model"])
        pending = ""
        for slug, verifier in sorted(verifiers.items()):
            if verifier.get("pending"):
                pending = f" | pending: {verifier['pending']['severity']} ({slug})"
                break
        last_err = sess.get("last_error") or ""
        if last_err:
            last_err = f" | error: {last_err[:80]}"
        line = (
            f"  {sess['session_id'][:28]:<28} {active} | reviews: {agg['reviews']} "
            f"| findings: {agg['findings']} (n:{agg['notes']} "
            f"c:{agg['cautions']} b:{agg['blockers']}) "
            f"| tokens: {agg['input_tokens'] + agg['output_tokens']} "
            f"| cost: {_fmt_cost(agg['cost_usd'])}"
        )
        lines.append(line + reviewing + pending + last_err)
        if models:
            lines.append(f"      verifier models: {', '.join(models[:3])}")
    return lines


def _dump_lines(raw: bool, limit: int = 12) -> List[str]:
    data_dir = runtime.data_dir()
    if not data_dir.is_dir():
        return ["  No verifier transcripts yet."]
    recent = runtime.recent_session()
    pattern = re.compile(r"^verify\..*\.jsonl$")
    files = sorted(data_dir.glob("verify.*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return ["  No verifier transcripts yet."]

    lines: List[str] = []
    shown = 0
    for path in files[:4]:
        session = path.stem[len("verify."):]
        marker = " ←" if session == runtime._slug_session(recent or "") else ""
        lines.append(f"  {session}{marker}")
        try:
            tail = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-limit:]
        except OSError:
            continue
        for line in tail:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            severity = record.get("severity", "?")
            accepted = "" if record.get("accepted", True) else " [suppressed]"
            finding = str(record.get("finding", ""))[:240].replace("\n", " ")
            lines.append(f"    [{severity}]{accepted} {finding}")
            if raw and record.get("reasoning"):
                lines.append(f"        reasoning: {str(record['reasoning'])[:200]}")
            shown += 1
    if not shown:
        lines.append("  (no findings recorded)")
    return lines


def make_handler(ctx: Any) -> Callable[[str], Optional[str]]:
    runtime.bind_ctx(ctx)

    def handler(raw_args: str) -> Optional[str]:
        argv = (raw_args or "").strip().split()
        if not argv or argv[0] in {"status", "help", "-h", "--help"}:
            return "\n".join(_status_lines() + ["", _HELP])

        sub = argv[0]
        recent = runtime.recent_session()

        if sub in {"on", "off"}:
            if recent is None:
                return (
                    "[hermes-verifier] No active session seen yet. Enable globally via "
                    "plugins.entries.hermes-verifier.settings.enabled: true in config.yaml."
                )
            runtime.set_session_override(recent, sub == "on")
            return f"[hermes-verifier] verifier {sub} for session {recent[:28]} (session-scoped)."

        if sub == "reset":
            if recent is None:
                return "[hermes-verifier] No active session seen yet."
            state = runtime._get_state(recent, create=False)
            if state is not None:
                state.reset()
            return f"[hermes-verifier] re-primed verifier state for {recent[:28]}."

        if sub == "dump":
            raw = len(argv) > 1 and argv[1] == "raw"
            return "\n".join(_dump_lines(raw))

        return f"Unknown subcommand: {sub}\n\n{_HELP}"

    return handler
