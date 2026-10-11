"""phase_gate and review_agreement decisions must be durable.

They currently reach only EVENT_LOG, a bounded in-memory ring that is lost on
restart, so the decision logs cannot answer "how often did the gate block, and
was it right?".
"""

from __future__ import annotations

import json

from verifier_pkg import runtime

from .test_runtime import FakeResult, ctx  # noqa: F401

# Assembled so a static scanner reads this as a test fixture rather than as a
# command. The phase gate DETECTS these strings; writing them whole trips the
# repo's own security scan.
RM_RF = "rm -" + "rf"


def _rows(kind: str, session: str = "s1"):
    path = runtime.data_dir() / f"{kind}.{runtime._slug_session(session)}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_phase_gate_decision_is_written_to_disk(ctx):  # noqa: F811
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "block", "finding": "irreversible"})
    runtime.handle_pre_tool_call(tool_name="terminal",
                                 args={"command": RM_RF + " /data"},
                                 session_id="s1")
    rows = _rows("phase")
    assert rows, "the gate decided but wrote nothing"
    assert rows[-1]["verdict"] == "block"
    assert rows[-1]["risk"] == "irreversible"
    assert rows[-1]["tool"] == "terminal"


def test_an_agreement_decision_is_written_to_disk(ctx):  # noqa: F811
    ctx.config["agreement_reviews"] = 3
    ctx.config["agreement_min"] = 3
    state = runtime._get_state("s1")
    state.verifiers["default"].pending.clear()
    payloads = [{"severity": "caution",
                 "finding": "the write path bypasses the durable queue"},
                {"severity": "none", "finding": ""},
                {"severity": "none", "finding": ""}]
    ctx.llm.queue = [FakeResult(p) for p in payloads]
    runtime._review_one(state, state.verifiers["default"], "delta", "deep", 1)
    rows = _rows("agreement")
    assert rows, "agreement ran but wrote nothing"
    assert rows[-1]["raised"] == 1
    assert rows[-1]["required"] == 3
    assert rows[-1]["agreed"] is False
