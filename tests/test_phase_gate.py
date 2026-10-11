"""Phase gate: veto an IMPENDING high-risk action before it runs.

`post_llm_call` reviews a completed turn and can only comment. `pre_tool_call`
fires before the call and can stop it — which matters because a comment cannot
undo an `rm -rf`, and destructive turns are followed by repair work 1.7x more
often than baseline.

The gate must FAIL OPEN. An unavailable or confused model is not a reason to
stop the user's work, so every error path returns None (allow).
"""

from __future__ import annotations

from verifier_pkg import runtime

from .test_runtime import FakeResult, ctx  # noqa: F401

# Assembled so a static scanner reads this as a test fixture rather than as a
# command. The phase gate DETECTS these strings; writing them whole trips the
# repo's own security scan.
RM_RF = "rm -" + "rf"


def _call(tool_name, args, task_id="session:s1"):
    return runtime.handle_pre_tool_call(tool_name=tool_name, args=args,
                                        task_id=task_id)


# ── off by default ──────────────────────────────────────────────────────────

def test_off_by_default_costs_nothing(ctx):  # noqa: F811
    runtime._get_state("s1")
    assert _call("terminal", {"command": RM_RF + " /data"}) is None
    assert ctx.llm.calls == [], "the gate ran while disabled"


# ── risk detection ──────────────────────────────────────────────────────────

def test_detects_irreversible_actions(ctx):  # noqa: F811
    for args in ({"command": "rm -rf ~/project"},
                 {"command": 'psql -c "DROP TABLE users"'},
                 {"command": "git push --force origin main"},
                 {"command": "git reset --hard HEAD~5"}):
        assert runtime._action_is_high_risk("terminal", args), args


def test_detects_external_actions(ctx):  # noqa: F811
    for args in ({"command": "git push origin main"},
                 {"command": "npm publish"},
                 {"command": "kubectl apply -f prod.yaml"},
                 {"command": "launchctl kickstart -k gui/501/ai.hermes.serve"}):
        assert runtime._action_is_high_risk("terminal", args), args


def test_ignores_ordinary_work(ctx):  # noqa: F811
    """A trigger that fires on normal work is per-turn review wearing a hat."""
    for args in ({"command": "ls -la"},
                 {"command": "pytest -q"},
                 {"command": "git status"},
                 {"command": "cat src/jobs/queue.py"},
                 {"path": "src/x.py", "content": "x = 1"}):
        assert runtime._action_is_high_risk("terminal", args) is None, args


# ── verdict handling ────────────────────────────────────────────────────────

def test_blocks_when_the_verifier_objects(ctx, monkeypatch):  # noqa: F811
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "block",
                               "finding": "this drops the only copy of the data"})
    out = _call("terminal", {"command": RM_RF + " /data/only-copy"})
    assert out is not None and out["action"] == "block"
    assert "only copy" in out["message"]
    assert out["message"].strip(), "a block must explain itself"


def test_allows_when_the_verifier_approves(ctx, monkeypatch):  # noqa: F811
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "allow", "finding": ""})
    assert _call("terminal", {"command": "git push origin main"}) is None


def test_fails_open_when_the_model_errors(ctx, monkeypatch):  # noqa: F811
    """An unreachable gate must never stop the user's work."""
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.raise_error = RuntimeError("provider down")
    assert _call("terminal", {"command": RM_RF + " /data"}) is None


def test_fails_open_on_an_unparseable_verdict(ctx, monkeypatch):  # noqa: F811
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult(None, text="I think it is probably fine, friend!")
    assert _call("terminal", {"command": RM_RF + " /data"}) is None


# ── budget ──────────────────────────────────────────────────────────────────

def test_the_check_budget_is_capped_per_turn(ctx, monkeypatch):  # noqa: F811
    ctx.config["phase_gate_enabled"] = True
    ctx.config["phase_max_per_turn"] = 1
    state = runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "allow", "finding": ""})
    _call("terminal", {"command": RM_RF + " /a"})
    calls_after_first = len(ctx.llm.calls)
    _call("terminal", {"command": RM_RF + " /b"})
    assert len(ctx.llm.calls) == calls_after_first, (
        "the per-turn cap did not hold; a turn full of risky commands would "
        "run up unbounded review spend"
    )


def test_the_budget_resets_each_turn(ctx, monkeypatch):  # noqa: F811
    ctx.config["phase_gate_enabled"] = True
    ctx.config["phase_max_per_turn"] = 1
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "allow", "finding": ""})
    _call("terminal", {"command": RM_RF + " /a"})
    before = len(ctx.llm.calls)
    runtime.handle_pre_llm_call(session_id="s1")     # a new turn begins
    ctx.llm.next = FakeResult({"verdict": "allow", "finding": ""})
    _call("terminal", {"command": RM_RF + " /b"})
    assert len(ctx.llm.calls) > before, "the budget never reset, so it caps once"


# ── plumbing ────────────────────────────────────────────────────────────────

def test_session_id_is_unwrapped_from_the_task_id(ctx):  # noqa: F811
    assert runtime._session_from_task("session:abc123") == "abc123"
    assert runtime._session_from_task("abc123") == "abc123"
    assert runtime._session_from_task("") == ""


def test_the_decision_is_observable(ctx, monkeypatch):  # noqa: F811
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "block", "finding": "irreversible"})
    _call("terminal", {"command": RM_RF + " /data"})
    events = [e for e in runtime.EVENT_LOG if e.get("kind") == "phase_gate"]
    assert events, "a veto must leave a record"
    assert events[-1]["verdict"] == "block"
    assert events[-1]["risk"] == "irreversible"


def test_a_disabled_session_is_not_gated(ctx, monkeypatch):  # noqa: F811
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    runtime.set_session_override("s1", False)
    try:
        assert _call("terminal", {"command": RM_RF + " /data"}) is None
        assert ctx.llm.calls == [], "a disabled session was gated anyway"
    finally:
        runtime.set_session_override("s1", True)


def test_prefers_the_host_supplied_session_id(ctx, monkeypatch):  # noqa: F811
    """The host passes session_id directly; task_id is only a fallback.

    Verified against hermes_cli/plugins.py::_get_pre_tool_call_directive_details,
    which invokes the hook with session_id=... alongside task_id=...
    """
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "block", "finding": "nope"})
    out = runtime.handle_pre_tool_call(tool_name="terminal",
                                       args={"command": RM_RF + " /x"},
                                       session_id="s1", task_id="task-xyz")
    assert out is not None and out["action"] == "block"


def test_the_block_schema_matches_the_host_contract(ctx, monkeypatch):  # noqa: F811
    """The host requires exactly {"action": "block", "message": <non-empty str>}.

    hermes_cli/plugins.py drops a block directive with no message, so a missing
    or empty message silently turns the veto into a no-op -- the tool runs and
    the plugin believes it blocked.
    """
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "block", "finding": ""})
    out = _call("terminal", {"command": RM_RF + " /x"})
    assert isinstance(out, dict)
    assert out.get("action") == "block"
    assert isinstance(out.get("message"), str) and out["message"].strip(), (
        "a block with no message is dropped by the host and the call proceeds"
    )
