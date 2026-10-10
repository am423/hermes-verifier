"""Runtime wiring tests against a fake PluginContext.

Covers: enable state, the review loop (synchronous via the override hook),
JSONL persistence, finding delivery, immunity, the steer gate, roster
verifiers, the catch-up window, the ask channel, and model persistence.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from verifier_pkg import runtime


class FakeUsage:
    def __init__(self, i: int = 12, o: int = 7, cost: float = 0.001):
        self.input_tokens = i
        self.output_tokens = o
        self.total_tokens = i + o
        self.cost_usd = cost


class FakeResult:
    def __init__(self, parsed: Any, model: str = "fake/model",
                 usage: FakeUsage | None = None, text: str = ""):
        self.parsed = parsed
        self.model = model
        self.usage = usage or FakeUsage()
        self.text = text


class FakeLlm:
    def __init__(self) -> None:
        self.calls: list = []
        self.next: Any = None
        self.queue: list = []
        self.raise_error: Exception | None = None
        # Overridden per-test when a test asserts on tool dispatch.
        self.dispatch_tool = lambda name, args: "[fake tool result]"

    def complete_structured(self, **kwargs) -> FakeResult:
        self.calls.append(kwargs)
        if self.raise_error is not None:
            raise self.raise_error
        if self.queue:
            return self.queue.pop(0)
        result = self.next if self.next is not None else FakeResult(None)
        self.next = None
        return result


class FakeCtx:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.llm = FakeLlm()
        self.set_config_calls: list = []

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        self.set_config_calls.append((key, value))
        self.config[key] = value


@pytest.fixture()
def ctx(tmp_path):
    runtime._DATA_DIR_OVERRIDE = tmp_path / "data"
    runtime._RUNTIME_OVERRIDES.clear()
    runtime._SESSIONS.clear()
    runtime._RECENT_SESSION = None
    runtime._ROSTER_OVERRIDE = None
    runtime._VERIFIER_SESSION = None
    runtime.EVENT_LOG.clear()
    runtime._EVENT_SEQ = 0
    fake = FakeCtx({"enabled": True, "verifier_model": "deepseek/deepseek-v4",
                    "immunity_turns": 3, "max_delta_chars": 20000})
    runtime.bind_ctx(fake)
    yield fake
    runtime._DATA_DIR_OVERRIDE = None
    runtime._CTX = None
    runtime._ROSTER_OVERRIDE = None


def _review_sync(state, delta_text):
    runtime._run_review(state, delta_text)


def _turn(ctx, session="s1", history=None, response="final answer"):
    runtime.handle_post_llm_call(
        session_id=session,
        user_message="do the thing",
        conversation_history=history if history is not None else [],
        assistant_response=response,
    )


def _default(state):
    return state.verifiers["default"]


def test_enable_state(ctx):
    assert runtime.subsystem_enabled("any") is True
    ctx.config["enabled"] = False
    assert runtime.subsystem_enabled("any") is False
    runtime.set_session_override("any", True)
    assert runtime.subsystem_enabled("any") is True


def test_model_resolution(ctx):
    ctx.config["verifier_model"] = "deepseek/deepseek-v4"
    assert runtime._resolve_model_for(ctx, runtime.Verifier(slug="d", name="d")) == ("deepseek", "deepseek-v4")
    ctx.config["verifier_model"] = "gpt-something"
    assert runtime._resolve_model_for(ctx, runtime.Verifier(slug="d", name="d")) == (None, "gpt-something")
    ctx.config["verifier_model"] = ""
    assert runtime._resolve_model_for(ctx, runtime.Verifier(slug="d", name="d")) == (None, None)
    # Roster entry model wins over the setting.
    v = runtime.Verifier(slug="arch", name="Architecture", model="anthropic/claude-x")
    assert runtime._resolve_model_for(ctx, v) == ("anthropic", "claude-x")


def test_full_cycle_raise_and_deliver(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({
        "severity": "caution",
        "finding": "Bypasses the durable queue in src/jobs",
        "reasoning": "the write path skips the queue",
    })

    _turn(ctx)

    state = runtime._get_state("s1", create=False)
    assert state is not None
    verifier = _default(state)
    assert verifier.pending["severity"] == "caution"
    assert verifier.stats["reviews"] == 1
    assert verifier.stats["cautions"] == 1

    log = runtime._log_path("s1", "default")
    records = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["accepted"] is True
    assert records[0]["severity"] == "caution"
    assert records[0]["usage"]["cost_usd"] == 0.001
    assert "final answer" in records[0]["delta_excerpt"]

    call = ctx.llm.calls[-1]
    assert call["provider"] == "deepseek"
    assert call["model"] == "deepseek-v4"
    assert call["json_schema"] == runtime.REVIEW_SCHEMA
    assert call["system_prompt"].startswith("You are the Verifier")
    assert call["input"][0]["type"] == "text"
    assert "final answer" in call["input"][0]["text"]

    injected = runtime.handle_pre_llm_call(session_id="s1")
    assert injected is not None
    assert '<finding severity="caution"' in injected["context"]
    assert "Bypasses the durable queue" in injected["context"]
    assert runtime.handle_pre_llm_call(session_id="s1") is None
    assert state.immunity_left == 3


def test_duplicate_suppressed(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "note", "finding": "Tidy the imports",
                               "reasoning": "unused"})
    _turn(ctx)
    assert _default(runtime._get_state("s1")).pending is not None
    assert runtime.handle_pre_llm_call(session_id="s1") is not None
    assert _default(runtime._get_state("s1")).pending is None

    ctx.llm.next = FakeResult({"severity": "note", "finding": "Tidy the imports",
                               "reasoning": "again"})
    _turn(ctx)
    assert _default(runtime._get_state("s1")).pending is None
    assert _default(runtime._get_state("s1")).guard.stats["deduped"] == 1

    log = runtime._log_path("s1", "default")
    records = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(records) == 2
    assert records[1]["accepted"] is False


def test_none_severity_no_finding(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "none", "finding": "",
                               "reasoning": "clean turn"})
    _turn(ctx)
    assert _default(runtime._get_state("s1")).pending is None
    assert _default(runtime._get_state("s1")).stats["findings"] == 0


def test_llm_failure_records_error_no_crash(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.raise_error = RuntimeError("provider 500")
    _turn(ctx)
    state = runtime._get_state("s1", create=False)
    assert "RuntimeError" in state.last_error
    assert _default(state).pending is None


def test_non_json_result_ignored(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult(None)
    _turn(ctx)
    state = runtime._get_state("s1", create=False)
    assert _default(state).pending is None
    assert _default(state).stats["reviews"] == 1


def test_blocker_steers_pre_verify(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({
        "severity": "blocker",
        "finding": "The change was never exercised",
        "reasoning": "test run is missing",
    })
    _turn(ctx)
    state = runtime._get_state("s1", create=False)

    steer = runtime.handle_pre_verify(session_id="s1")
    assert steer is not None
    assert steer["action"] == "continue"
    assert "never exercised" in steer["message"]
    assert _default(state).pending is None
    assert state.immunity_left == 3

    ctx.llm.next = FakeResult({"severity": "blocker", "finding": "Second blocker",
                               "reasoning": "x"})
    _turn(ctx)
    assert runtime.handle_pre_verify(session_id="s1") is None
    assert _default(state).pending is not None


def test_caution_does_not_steer_pre_verify(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "caution", "finding": "Risky path",
                               "reasoning": "x"})
    _turn(ctx)
    assert runtime.handle_pre_verify(session_id="s1") is None
    assert runtime.handle_pre_llm_call(session_id="s1") is not None


def test_session_reset_reprimes(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "note", "finding": "cleanup",
                               "reasoning": "x"})
    _turn(ctx, history=[{"role": "user", "content": "a"}])
    state = runtime._get_state("s1", create=False)
    assert state.cursor_len == 1
    assert _default(state).pending is not None

    runtime.handle_session_reset(session_id="s1")
    assert state.cursor_len is None
    assert _default(state).pending is None
    assert len(_default(state).guard) == 0


def test_disabled_session_skips_everything(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["enabled"] = False
    runtime.clear_session_override("s1")
    _turn(ctx)
    assert runtime._get_state("s1", create=False) is None
    assert ctx.llm.calls == []


# ── Roster ──────────────────────────────────────────────────────────────────


def test_roster_multi_verifier_review(ctx, monkeypatch):
    runtime._ROSTER_OVERRIDE = [
        {"slug": "architecture", "name": "Architecture",
         "enabled": True, "model": "anthropic/claude-x", "tools": ["read_file"],
         "instructions": "Watch coupling."},
        {"slug": "fixer", "name": "Fixer", "enabled": False, "model": "",
         "tools": [], "instructions": ""},
    ]
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.queue = [
        FakeResult({"severity": "note", "finding": "Default sees this",
                    "reasoning": "x"}),
        FakeResult({"severity": "note", "finding": "Coupling grew",
                    "reasoning": "x"}),
    ]

    _turn(ctx)
    state = runtime._get_state("s1", create=False)
    assert set(state.verifiers) == {"default", "architecture"}  # fixer disabled
    arch = state.verifiers["architecture"]
    assert arch.stats["reviews"] == 1
    assert _default(state).stats["reviews"] == 1
    assert arch.pending is not None and _default(state).pending is not None

    # Both pendings deliver together on the next turn.
    injected = runtime.handle_pre_llm_call(session_id="s1")
    assert injected is not None
    assert 'verifier="architecture"' in injected["context"]
    assert "Coupling grew" in injected["context"]
    assert "Default sees this" in injected["context"]

    # Per-verifier JSONL + per-verifier model override.
    assert runtime._log_path("s1", "architecture").exists()
    assert runtime._log_path("s1", "default").exists()
    arch_call = [c for c in ctx.llm.calls if c.get("model") == "claude-x"]
    assert arch_call
    # Roster instructions land in the system prompt.
    assert "Watch coupling." in arch_call[0]["system_prompt"]


def test_roster_discovery_from_files(tmp_path):
    from verifier_pkg import brief

    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    (proj / "VERIFIER.yml").write_text(
        "instructions: |\n  Shared line.\nverifiers:\n"
        "  - name: Architecture\n    model: anthropic/claude-x\n"
        "  - name: Bad\n    enabled: false\n"
    )
    roster = brief.load_roster(start_dir=proj, user_home=home)
    assert len(roster) == 2
    arch = next(e for e in roster if e["slug"] == "architecture")
    assert arch["model"] == "anthropic/claude-x"
    assert arch["shared_instructions"] == "Shared line."


def test_roster_specificity_override(tmp_path):
    from verifier_pkg import brief

    home = tmp_path / "home"
    proj = tmp_path / "proj"
    (home / ".git").mkdir(parents=True)
    (proj / ".git").mkdir(parents=True)
    (home / "VERIFIER.yml").write_text("verifiers:\n  - name: Architecture\n    model: user-model\n")
    (proj / "VERIFIER.yml").write_text("verifiers:\n  - name: Architecture\n    model: project-model\n")
    roster = brief.load_roster(start_dir=proj, user_home=home)
    arch = next(e for e in roster if e["slug"] == "architecture")
    assert arch["model"] == "project-model"


def test_brief_import_expansion(tmp_path):
    from verifier_pkg import brief

    proj = tmp_path / "proj"
    proj.mkdir()
    home = tmp_path / "none"
    (proj / "shared.md").write_text("shared priority line")
    (proj / "VERIFIER.md").write_text(
        "# brief\n@shared.md\n```\n@literal.md\n```\n@missing.md\n"
    )
    blocks = brief.load_brief_blocks(start_dir=proj, user_home=home)
    joined = "\n".join(blocks)
    assert "shared priority line" in joined
    assert "@literal.md" in joined          # fenced code stays literal
    assert "@missing.md" in joined          # missing import stays literal


# ── Catch-up window ─────────────────────────────────────────────────────────


def test_backlog_wait_noop_when_off(ctx):
    state = runtime._get_state("s1")
    start = time.monotonic()
    runtime._backlog_wait(state, None)
    assert time.monotonic() - start < 0.05


def test_backlog_wait_drains_when_caught_up(ctx):
    state = runtime._get_state("s1")
    state.backlog = 0
    start = time.monotonic()
    runtime._backlog_wait(state, 1)
    assert time.monotonic() - start < 0.2


def test_backlog_wait_capped(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_BACKLOG_WAIT_CAP", 0.25)
    state = runtime._get_state("s1")
    state.backlog = 3  # never drains
    start = time.monotonic()
    runtime._backlog_wait(state, 1)
    elapsed = time.monotonic() - start
    assert 0.2 <= elapsed < 2.0


def test_backlog_wait_released_when_failing(ctx):
    state = runtime._get_state("s1")
    state.backlog = 5
    state.consecutive_failures = 3
    start = time.monotonic()
    runtime._backlog_wait(state, 1)
    assert time.monotonic() - start < 0.2


# ── Ask channel + model persistence ─────────────────────────────────────────


def test_ask_verifier(ctx):
    runtime._get_state("s1")  # create session state without a review thread
    ctx.llm.next = FakeResult(None, model="fake/model", text="It looks fine to me.")
    result = runtime.ask_verifier("s1", "Is this safe?")
    assert result["ok"] is True
    assert "looks fine" in result["answer"]

    records = [json.loads(l) for l in
               runtime._log_path("s1", "default").read_text().splitlines()]
    asks = [r for r in records if r.get("kind") == "ask"]
    assert len(asks) == 1
    assert asks[0]["question"] == "Is this safe?"


def test_ask_verifier_no_session(ctx):
    result = runtime.ask_verifier("never-seen", "question?")
    assert result["ok"] is False


def test_set_model_via_ctx(ctx):
    assert runtime.set_model("openai/gpt-4o") is True
    assert ctx.set_config_calls == [("verifier_model", "openai/gpt-4o")]
    assert ctx.config["verifier_model"] == "openai/gpt-4o"


def test_mark_verifier_session(ctx):
    runtime.mark_verifier_session("vs1")
    assert runtime.designated_session() == "vs1"
    assert runtime.subsystem_enabled("vs1") is True
    assert ctx.set_config_calls == [("verifier_session", "vs1")]


def test_designated_session_persisted(ctx):
    ctx.config["verifier_session"] = "stored-session"
    assert runtime.designated_session() == "stored-session"
    # The persisted designation also arms the session override.
    assert runtime.subsystem_enabled("stored-session") is True


# ── Events ──────────────────────────────────────────────────────────────────


def test_event_log(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "note", "finding": "x is wrong",
                               "reasoning": "y"})
    _turn(ctx)
    kinds = {e["kind"] for e in runtime.EVENT_LOG}
    assert {"review_started", "review", "finding", "review_finished"} <= kinds
    seq = runtime.last_event_seq()
    assert seq == len(runtime.EVENT_LOG)
    assert runtime.events_after(0) == runtime.EVENT_LOG
    assert runtime.events_after(seq) == []


# ── sync_backlog normalization + catch-up window ─────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("off", 0), ("1", 1), ("3", 3), ("5", 5),
    (1, 1), (3, 3),          # unquoted YAML int — the regression
    (0, 0), ("0", 0), ("none", 0), ("", 0), (None, 0),
    ("bogus", 0), (7, 0),    # out of OMP's enum range
])
def test_parse_sync_backlog(raw, expected):
    assert runtime._parse_sync_backlog(raw) == expected


def test_sync_backlog_int_setting_does_not_drop_finding(ctx, monkeypatch):
    """Regression: an int setting crashed .isdigit() and the finding was lost."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["sync_backlog"] = 1          # what YAML actually gives us
    ctx.llm.next = FakeResult({"severity": "caution",
                               "finding": "Bypasses the durable queue in src/jobs",
                               "reasoning": "skips the queue"})

    _turn(ctx)

    state = runtime._get_state("s1", create=False)
    assert state.last_error == "", "handler swallowed an exception"
    assert _default(state).pending is not None, "finding was dropped"


def test_backlog_wait_waits_then_releases(monkeypatch):
    """Waits while backlog >= threshold; releases on a failure streak."""
    clock = {"t": 0.0}
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(runtime.time, "sleep",
                        lambda s: (clock.__setitem__("t", clock["t"] + s)))

    state = runtime.SessionVerifier(session_id="s")
    state.backlog = 2
    runtime._backlog_wait(state, 1)
    assert clock["t"] > 0, "should have waited while backlog was above threshold"

    # Failure streak releases immediately even when still behind.
    clock["t"] = 0.0
    state.consecutive_failures = 3
    runtime._backlog_wait(state, 1)
    assert clock["t"] == 0, "a failing verifier must never stall the session"


def test_backlog_wait_is_capped(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(runtime.time, "sleep",
                        lambda s: clock.__setitem__("t", clock["t"] + s))
    state = runtime.SessionVerifier(session_id="s")
    state.backlog = 99                       # never clears
    runtime._backlog_wait(state, 1)
    # Bounded by the cap plus at most one poll interval of overshoot.
    assert clock["t"] <= runtime._BACKLOG_WAIT_CAP + runtime._BACKLOG_POLL
    assert clock["t"] >= runtime._BACKLOG_WAIT_CAP, "should have waited the full cap"


# ── Advisor tool verification ────────────────────────────────────────────────


def test_advisor_tools_verify_a_claim(ctx, monkeypatch):
    """With the grant on, the advisor's tool result reaches the review prompt.

    Tools are per-verifier: a roster entry (or default_tools) must request them
    AND advisor_tools must be on. Only the 'claims' verifier has a grant, so
    only it may make tool calls — the default verifier reviews prose alone.
    """
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["advisor_tools"] = True
    runtime._ROSTER_OVERRIDE = [{
        "name": "Claims", "slug": "claims", "enabled": True,
        "model": "", "tools": ["read_file"], "instructions": "",
    }]
    # The default verifier runs first and has no grant: it must not consume the
    # tool call. Route responses per verifier so each sees its own script.
    responses = {
        "claims": [
            FakeResult({"severity": "none", "finding": "", "reasoning": ""},
                       text='TOOL_CALL: read_file {"path": "src/jobs/queue.py"}'),
            FakeResult({"severity": "caution",
                        "finding": "Confirmed: src/jobs/write.py bypasses the durable queue",
                        "reasoning": "read both files"}),
        ],
    }

    def _fake_complete(**kwargs):
        ctx.llm.calls.append(kwargs)
        has_tools = "TOOL_CALL" in kwargs["system_prompt"]
        if not has_tools:
            # Default verifier: no grant, prose review only.
            return FakeResult({"severity": "none", "finding": "", "reasoning": ""})
        return responses["claims"].pop(0)

    ctx.llm.complete_structured = _fake_complete
    dispatched = []
    ctx.dispatch_tool = lambda name, args: dispatched.append(name) or "def enqueue(job): ..."

    _turn(ctx)

    calls = ctx.llm.calls
    assert len(calls) >= 2, "advisor should have made a verification call"
    # The tool result must appear in the review that follows the tool call.
    verifying = [c for c in calls if "def enqueue" in c["input"][0]["text"]]
    assert verifying, "tool output must feed the review"
    assert dispatched == ["read_file"]
    state = runtime._get_state("s1", create=False)
    assert state.verifiers["claims"].pending is not None
    assert state.verifiers["claims"].pending["severity"] == "caution"


def test_default_tools_grant_applies_without_a_roster(ctx, monkeypatch):
    """default_tools grants the default verifier tools when advisor_tools is on."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["advisor_tools"] = True
    ctx.config["default_tools"] = ["read_file", "terminal"]   # terminal must be refused
    ctx.llm.queue = [
        FakeResult({"severity": "none", "finding": "", "reasoning": ""},
                   text='TOOL_CALL: read_file {"path": "a.py"}'),
        FakeResult({"severity": "note", "finding": "verified", "reasoning": ""}),
    ]
    dispatched = []
    ctx.dispatch_tool = lambda name, args: dispatched.append(name) or "contents"

    _turn(ctx)

    assert dispatched == ["read_file"], "only the allowlisted tool may run"
    assert _default(runtime._get_state("s1", create=False)).pending is not None


def test_advisor_tools_off_by_default(ctx, monkeypatch):
    """Zero-config behavior is unchanged: one call, no tool dispatch."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "note", "finding": "nit: rename x", "reasoning": ""})
    dispatched = []
    ctx.dispatch_tool = lambda name, args: dispatched.append(n) or "should not happen"

    _turn(ctx)

    assert len(ctx.llm.calls) == 1
    assert dispatched == []
    assert "TOOL_CALL" not in ctx.llm.calls[0]["system_prompt"]
