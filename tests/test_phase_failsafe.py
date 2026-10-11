"""The gate must never make the host BLOCK a call the user asked for.

The host bounds every `pre_tool_call` callback at `plugins.hook_callback_timeout`
(30 s by default) and converts a timeout -- or ANY exception escaping the
callback -- into a block directive
(hermes_cli/plugins_dispatch.py: "pre_tool_call plugin callback timed out or is
still running"). After a timeout it also suppresses the callback for ~60 s, so
the next high-risk call is blocked again without any gate call at all.

The plugin believed it failed OPEN. It did, for errors inside its own try blocks
-- but the gate's model call was allowed 120 s, four times the host budget. A
slow provider therefore produced exactly the failure the design says must never
happen: a blocked `git push`, with `phase.<session>.jsonl` recording `allow` for
an action that was actually blocked, poisoning the analysis it feeds.

Two invariants pinned here:
  1. the gate's own call is capped well under the host's 30 s budget
  2. no exception can escape the handler, because the host turns one into a veto
"""

from __future__ import annotations

import json

from verifier_pkg import runtime

from .test_runtime import FakeResult, ctx  # noqa: F401


def test_the_gate_call_is_capped_under_the_host_budget(ctx, monkeypatch):  # noqa: F811
    """120 s against a 30 s host budget meant a slow provider blocked the user."""
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "allow", "finding": ""})
    runtime.handle_pre_tool_call(tool_name="terminal",
                                 args={"command": "git push origin main"},
                                 session_id="s1")
    used = ctx.llm.calls[-1].get("timeout")
    assert used is not None, "the gate call relies on a default timeout"
    assert used < 30.0, (
        f"the gate allows {used}s but the host vetoes the callback at 30s; "
        "a slow provider then BLOCKS the user's action"
    )
    assert used == runtime._PHASE_TIMEOUT


def test_no_exception_can_escape_the_handler(ctx, monkeypatch):  # noqa: F811
    """The host converts an escaping exception into a block, so the handler must
    swallow everything and allow."""
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.raise_error = RuntimeError("provider exploded")
    out = runtime.handle_pre_tool_call(tool_name="terminal",
                                       args={"command": "git push origin main"},
                                       session_id="s1")
    assert out is None, "an error path returned a directive instead of allowing"


def test_a_failure_inside_the_pattern_match_still_allows(ctx, monkeypatch):  # noqa: F811
    """Even a bug in the detector must not block the user."""
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")

    def boom(*a, **k):
        raise ValueError("detector is broken")

    monkeypatch.setattr(runtime, "_action_is_high_risk", boom)
    out = runtime.handle_pre_tool_call(tool_name="terminal",
                                       args={"command": "git push origin main"},
                                       session_id="s1")
    assert out is None


def test_a_failed_gate_call_is_recorded_as_an_error(ctx, monkeypatch):  # noqa: F811
    """A gate that could not answer must leave a trace.

    It is exactly when the gate silently stopped protecting that the autoresearch
    question -- how often did it block, and was it right -- needs an answer.
    """
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.raise_error = RuntimeError("provider down")
    runtime.handle_pre_tool_call(tool_name="terminal",
                                 args={"command": "git push origin main"},
                                 session_id="s1")
    path = runtime.data_dir() / f"phase.{runtime._slug_session('s1')}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert rows, "a gate that failed wrote nothing at all"
    assert rows[-1]["verdict"] == "error"
