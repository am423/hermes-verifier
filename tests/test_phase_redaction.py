"""The phase gate must redact tool args before sending AND before logging them.

Every other model-bound path in this plugin redacts: the transcript delta goes
through `delta.redact`, and `brief.py` promises the reviewer "you never see
secrets". The phase gate, added later, did neither -- it sent the raw args dict
to the provider and wrote a 400-char excerpt of the same bytes to
phase.<session>.jsonl.

Trigger: a command like `git push https://x-access-token:<token>@github.com/...`
matches the gate's `git push` pattern, so the whole command -- URL and all --
goes to the model and to disk.

Verified before writing the fix: no credential is currently on disk (the only
matches in the logs were this session's own prose about the patterns), so this
is a latent path, not an exposure. It is fixed anyway because it is the same
breach class `test_redaction_paths.py` already covers for the delta.
"""

from __future__ import annotations

import json

from verifier_pkg import runtime

from .test_runtime import FakeResult, ctx  # noqa: F401

# Assembled rather than written whole: the repo's own security scan reads a
# literal token shape as a hardcoded secret, correctly.
FAKE_TOKEN = "ghp" + "_" + "abcdefghijklmnopqrstuvwxyz012345"
CRED_URL = "https://x-access-token:" + FAKE_TOKEN + "@github.com/org/repo.git"


def _phase_rows(session="s1"):
    path = runtime.data_dir() / f"phase.{runtime._slug_session(session)}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_the_model_call_does_not_receive_a_credential(ctx, monkeypatch):  # noqa: F811
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "allow", "finding": ""})
    runtime.handle_pre_tool_call(
        tool_name="terminal",
        args={"command": f"git push {CRED_URL}"},
        session_id="s1")

    sent = ctx.llm.calls[-1]["input"][0]["text"]
    assert FAKE_TOKEN not in sent, (
        "the credential reached the verifier provider verbatim; every other "
        "model-bound path in this plugin redacts"
    )
    assert "git push" in sent, "the gate still needs enough to judge the action"


def test_the_decision_log_does_not_receive_a_credential(ctx, monkeypatch):  # noqa: F811
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "allow", "finding": ""})
    runtime.handle_pre_tool_call(
        tool_name="terminal",
        args={"command": f"git push {CRED_URL}"},
        session_id="s1")

    rows = _phase_rows("s1")
    assert rows, "the gate decided but wrote nothing"
    assert FAKE_TOKEN not in json.dumps(rows), (
        "the credential was written to phase.<session>.jsonl, which the "
        "autoresearch loop then reads and republishes"
    )


def test_ordinary_commands_are_still_readable_after_redaction(ctx, monkeypatch):  # noqa: F811
    """Redaction must not destroy the audit value of the row."""
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "allow", "finding": ""})
    runtime.handle_pre_tool_call(
        tool_name="terminal",
        args={"command": "git push origin main"},
        session_id="s1")
    row = _phase_rows("s1")[-1]
    assert "git push origin main" in row["args_excerpt"]
    assert row["tool"] == "terminal" and row["risk"] == "external"
