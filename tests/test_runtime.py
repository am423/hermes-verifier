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
    # A quiet turn ticks the cooldown down, so the caution armed 3 and the
    # call above drained one. This is deliberate: without the drain a single
    # early finding could mute the advisor for the rest of the session.
    assert state.immunity_left == 2


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


# ── Review budget (max_uses, session-scoped) ─────────────────────────────────


def test_review_budget_blocks_after_cap(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["max_reviews_per_session"] = 2
    ctx.llm.next = FakeResult({"severity": "note", "finding": "nit: rename x",
                               "reasoning": ""})

    _turn(ctx)
    _turn(ctx, history=[{"role": "user", "content": "again"}], response="second")
    state = runtime._get_state("s1", create=False)
    assert state.reviews_used == 2

    _turn(ctx, history=[{"role": "user", "content": "third"}], response="third")
    assert state.reviews_used == 2, "budget exhausted — no further reviews"
    before = len(ctx.llm.calls)
    _turn(ctx, history=[{"role": "user", "content": "fourth"}], response="fourth")
    assert len(ctx.llm.calls) == before, "no provider call may be made past the cap"
    assert state.reviews_used == 2


def test_review_budget_unlimited_by_default(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "note", "finding": "nit: rename x",
                               "reasoning": ""})
    for i in range(4):
        _turn(ctx, history=[{"role": "user", "content": f"t{i}"}], response=f"a{i}")
    assert runtime._get_state("s1", create=False).reviews_used == 4


def test_budget_resets_on_session_reset(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["max_reviews_per_session"] = 1
    ctx.llm.next = FakeResult({"severity": "note", "finding": "nit: rename x",
                               "reasoning": ""})
    _turn(ctx)
    runtime.handle_session_reset(session_id="s1")
    _turn(ctx, history=[{"role": "user", "content": "again"}], response="again")
    assert runtime._get_state("s1", create=False).reviews_used == 1


def test_budget_counts_attempts_not_successes(ctx, monkeypatch):
    """A failing review still consumes budget — it is a cost control."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["max_reviews_per_session"] = 1
    ctx.llm.raise_error = RuntimeError("provider down")

    _turn(ctx)

    state = runtime._get_state("s1", create=False)
    assert state.reviews_used == 1
    assert state.consecutive_failures == 1


# ── First-turn advisor nudge ─────────────────────────────────────────────────


def test_nudge_injected_on_early_turns(ctx):
    ctx.config["nudge_first_turns"] = 2
    injected = runtime.handle_pre_llm_call(
        session_id="s1", user_message="build the thing",
        conversation_history=[], is_first_turn=True)
    assert injected is not None
    assert "advisor" in injected["context"].lower()


def test_nudge_not_injected_when_disabled(ctx):
    ctx.config["nudge_first_turns"] = 0
    assert runtime.handle_pre_llm_call(
        session_id="s1", user_message="build the thing",
        conversation_history=[], is_first_turn=True) is None


def test_nudge_stops_after_n_turns(ctx):
    ctx.config["nudge_first_turns"] = 1
    for _ in range(3):
        runtime.handle_pre_llm_call(session_id="s1", user_message="m",
                                    conversation_history=[])
    assert runtime.handle_pre_llm_call(
        session_id="s1", user_message="m", conversation_history=[]) is None


def test_nudge_counts_turns_while_enabled(ctx):
    """The nudge window advances per turn while the subsystem is enabled.

    While disabled the handler returns before materializing a session (see
    test_pre_llm_call_does_not_leak_sessions), so the window starts counting
    from the first enabled turn rather than from an earlier disabled one.
    """
    ctx.config["nudge_first_turns"] = 2
    ctx.config["enabled"] = True
    assert runtime.handle_pre_llm_call(
        session_id="s1", user_message="m", conversation_history=[]) is not None
    assert runtime.handle_pre_llm_call(
        session_id="s1", user_message="m", conversation_history=[]) is not None
    state = runtime._get_state("s1", create=False)
    assert state.turns_seen == 2
    # Window closed after 2 turns.
    assert runtime.handle_pre_llm_call(
        session_id="s1", user_message="m", conversation_history=[]) is None


# ── Status surfacing ─────────────────────────────────────────────────────────


def test_status_lines_surface_new_controls(ctx):
    from verifier_pkg import commands
    ctx.config.update({"max_reviews_per_session": 5, "advisor_tools": True,
                       "nudge_first_turns": 2})
    text = "\n".join(commands._status_lines())
    assert "advisor tools: on" in text
    assert "nudge: first 2" in text
    assert "review budget: 5" in text


def test_status_budget_unlimited_by_default(ctx):
    from verifier_pkg import commands
    text = "\n".join(commands._status_lines())
    assert "review budget: unlimited" in text


# ── Gap fixes: live roster/default_tools, session hygiene, dashboard parity ──


def test_roster_grants_refresh_mid_session(ctx, monkeypatch):
    """GAP 1: a roster written after session creation must still take effect.

    ensure_verifiers only ran at creation, so a VERIFIER.yml edited (or a
    default_tools change made) while a session was live was ignored until the
    next session. The grant now re-resolves each turn.
    """
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    runtime._ROSTER_OVERRIDE = []                       # created with no roster
    ctx.llm.next = FakeResult({"severity": "note", "finding": "nit", "reasoning": ""})
    _turn(ctx)
    state = runtime._get_state("s1", create=False)
    assert state.verifiers["default"].tool_grant == []

    # Operator edits VERIFIER.yml mid-session.
    runtime._ROSTER_OVERRIDE = [{
        "name": "Claims", "slug": "claims", "enabled": True,
        "model": "", "tools": ["read_file"], "instructions": "",
    }]
    ctx.llm.next = FakeResult({"severity": "note", "finding": "nit2", "reasoning": ""})
    _turn(ctx, history=[{"role": "user", "content": "again"}], response="again")

    assert "claims" in state.verifiers, "roster change must be picked up mid-session"
    assert state.verifiers["claims"].tool_grant == ["read_file"]


def test_default_tools_refresh_mid_session(ctx, monkeypatch):
    """GAP 2: changing default_tools must re-resolve the live grant."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "note", "finding": "nit", "reasoning": ""})
    _turn(ctx)
    state = runtime._get_state("s1", create=False)
    assert state.verifiers["default"].tool_grant == []

    ctx.config["default_tools"] = ["read_file", "terminal"]   # terminal refused
    _turn(ctx, history=[{"role": "user", "content": "again"}], response="again")

    assert state.verifiers["default"].tool_grant == ["read_file"]


def test_pre_llm_call_does_not_leak_sessions(ctx):
    """GAP 3: pre_llm_call must not materialize a session for a turn that
    never produced a review — otherwise disabled/short sessions accumulate
    state that never gets used."""
    ctx.config["enabled"] = False                       # subsystem off
    runtime.handle_pre_llm_call(session_id="ghost", user_message="hi",
                                conversation_history=[])
    assert runtime._get_state("ghost", create=False) is None, \
        "a disabled session must not be materialized by pre_llm_call"

    # With the subsystem on, the session is created (it must be, to hold state).
    ctx.config["enabled"] = True
    ctx.config["nudge_first_turns"] = 1
    runtime.handle_pre_llm_call(session_id="live", user_message="hi",
                                conversation_history=[])
    assert runtime._get_state("live", create=False) is not None


def test_dashboard_state_exposes_new_settings():
    """GAP 4: the dashboard /state payload must expose the new controls so the
    UI does not silently disagree with /verify status."""
    import sys
    import types
    mod = types.ModuleType("fake_api_mod")

    api_path = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "dashboard" / "plugin_api.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("hv_plugin_api", api_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["hv_plugin_api"] = m
    spec.loader.exec_module(m)

    m._plugin_settings = lambda: {
        "enabled": True, "verifier_model": "a/b", "sync_backlog": "off",
        "immunity_turns": 3, "max_delta_chars": 20000, "blocker_steer": True,
        "advisor_tools": True, "default_tools": ["read_file"],
        "max_reviews_per_session": 7, "advisor_max_tokens": 1024,
        "nudge_first_turns": 2,
    }
    m._primary_model = lambda: None
    m._transcript_from_disk = lambda *a, **k: []
    m.runtime = types.SimpleNamespace(last_event_seq=lambda: 0,
                                      designated_session=lambda: None)
    payload = m._state_payload()
    s = payload["settings"]
    assert s["advisor_tools"] is True
    assert s["default_tools"] == ["read_file"]
    assert s["max_reviews_per_session"] == 7
    assert s["advisor_max_tokens"] == 1024
    assert s["nudge_first_turns"] == 2


# ── Empty advisor response: retry + observability ────────────────────────────


def test_empty_response_retries_then_records(ctx, monkeypatch):
    """An empty provider completion (content_type=text, tokens=0) is retried.

    Root cause: _parse_structured_text returns (None, "text") on an empty body,
    so a silent empty completion looked like "no finding" and was dropped with
    only a debug line. The review now retries and records the outcome.
    """
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    attempts = {"n": 0}

    class FlakyLlm:
        def complete_structured(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                # Empty completion: no parsed payload, no text.
                return FakeResult(None, text="")
            return FakeResult({"severity": "caution",
                               "finding": "Retry recovered the review",
                               "reasoning": "second attempt"})

    ctx.llm.complete_structured = FlakyLlm().complete_structured
    _turn(ctx)

    assert attempts["n"] == 2, "an empty response must be retried"
    state = runtime._get_state("s1", create=False)
    assert _default(state).pending is not None
    assert _default(state).pending["finding"] == "Retry recovered the review"


def test_persistent_empty_response_is_logged_and_counted(ctx, monkeypatch):
    """A review that stays empty is recorded as an outcome, not silently lost."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.complete_structured = lambda **kw: FakeResult(None, text="")

    _turn(ctx)

    state = runtime._get_state("s1", create=False)
    v = _default(state)
    assert state.empty_responses >= 1, "empty completions must be counted"
    assert state.last_error, "an unrecovered empty response must be surfaced"
    assert v.stats["reviews"] >= 1, "the attempts must still count as reviews"


def test_no_retry_when_a_real_finding_returns(ctx, monkeypatch):
    """A good first response must not trigger a retry."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    calls = {"n": 0}

    def _ok(**kw):
        calls["n"] += 1
        return FakeResult({"severity": "note", "finding": "fine", "reasoning": ""})

    ctx.llm.complete_structured = _ok
    _turn(ctx)
    assert calls["n"] == 1


# ── Production readiness: engagement conditions ──────────────────────────────


def test_empty_delta_does_not_review(ctx, monkeypatch):
    """No new transcript content means no paid provider call."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    _turn(ctx, history=[], response="")
    assert ctx.llm.calls == [], "an empty delta must not spend a provider call"


def test_identical_consecutive_turns_do_not_review(ctx, monkeypatch):
    """A turn that adds no history rows must not re-review old content.

    The cursor tracks row count, so a replayed transcript with no new rows
    yields an empty delta and therefore no paid provider call. (A turn that
    genuinely repeats content still warrants review — that is the guard's job,
    not the cursor's.)
    """
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    hist = [{"role": "assistant", "content": "same"}]
    _turn(ctx, history=hist, response="")
    first = len(ctx.llm.calls)
    # Replay the identical history with no assistant response: no new rows.
    _turn(ctx, history=hist, response="")
    assert len(ctx.llm.calls) == first, "no new delta → no second review"


def test_review_exception_never_propagates(ctx, monkeypatch):
    """A failing review must be contained: the primary agent is unaffected."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.raise_error = RuntimeError("provider exploded")

    # Must not raise.
    _turn(ctx)

    state = runtime._get_state("s1", create=False)
    assert state.consecutive_failures >= 1, "the failure must be recorded"
    assert state.last_error, "the failure must be surfaced, not swallowed silently"


def test_failure_streak_releases_backlog_wait(ctx, monkeypatch):
    """A broken advisor must never stall the primary turn."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.raise_error = RuntimeError("down")
    for i in range(4):
        _turn(ctx, history=[{"role": "user", "content": f"t{i}"}], response=f"a{i}")
    state = runtime._get_state("s1", create=False)
    assert state.consecutive_failures >= 3

    clock = {"t": 0.0}
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(runtime.time, "sleep",
                        lambda s: clock.__setitem__("t", clock["t"] + s))
    runtime._backlog_wait(state, 1)
    assert clock["t"] == 0, "a failing verifier must never stall the session"


def test_identical_finding_not_raised_on_consecutive_turns(ctx, monkeypatch):
    """The same finding must not re-fire every turn — that is nagging."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    for i in range(3):
        ctx.llm.next = FakeResult({"severity": "caution",
                                   "finding": "Bypasses the durable queue in src/jobs",
                                   "reasoning": "same issue"})
        _turn(ctx, history=[{"role": "user", "content": f"t{i}"}], response=f"a{i}")
        runtime.handle_pre_llm_call(session_id="s1", user_message="m",
                                    conversation_history=[])
    stats = _default(runtime._get_state("s1", create=False)).guard.stats
    assert stats["deduped"] >= 2, (
        f"identical findings must be deduped across turns, got {stats}"
    )


def test_escalating_severity_still_delivers(ctx, monkeypatch):
    """A note escalating to a blocker is new information, not a duplicate."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "note",
                               "finding": "queue bypass", "reasoning": ""})
    _turn(ctx)
    runtime.handle_pre_llm_call(session_id="s1", user_message="m",
                                conversation_history=[])
    ctx.llm.next = FakeResult({"severity": "blocker",
                               "finding": "queue bypass", "reasoning": ""})
    _turn(ctx, history=[{"role": "user", "content": "t2"}], response="a2")
    assert _default(runtime._get_state("s1", create=False)).pending is not None, \
        "an escalation must survive dedupe"


def test_nudge_and_pending_finding_both_delivered(ctx, monkeypatch):
    """A nudge and a real finding in the same turn must both arrive."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["nudge_first_turns"] = 5
    ctx.llm.next = FakeResult({"severity": "caution",
                               "finding": "queue bypass", "reasoning": ""})
    _turn(ctx)
    runtime.handle_pre_llm_call(session_id="s1", user_message="m",
                                conversation_history=[])
    ctx.llm.next = FakeResult({"severity": "blocker",
                               "finding": "credential leak", "reasoning": ""})
    _turn(ctx, history=[{"role": "user", "content": "t2"}], response="a2")

    injected = runtime.handle_pre_llm_call(session_id="s1", user_message="m",
                                           conversation_history=[])
    assert injected is not None
    assert "advisor" in injected["context"].lower(), "the nudge must be present"
    assert "credential leak" in injected["context"], "the finding must be present"


def test_immunity_cooldown_expires(ctx, monkeypatch):
    """The cooldown counts down on quiet turns instead of muting forever.

    While blockers keep arriving the cooldown legitimately stays armed (that is
    the rate limit doing its job). It must decrement once findings stop, so a
    later blocker can steer again.
    """
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["immunity_turns"] = 2

    # Turn 1: a blocker interrupts and arms the cooldown.
    ctx.llm.next = FakeResult({"severity": "blocker",
                               "finding": "first blocker", "reasoning": ""})
    _turn(ctx)
    assert runtime.handle_pre_verify(session_id="s1") is not None, \
        "the first blocker must steer"
    state = runtime._get_state("s1", create=False)
    assert state.immunity_left == 2

    # Turns 2-3: quiet turns (severity none) must decrement the cooldown.
    for i in (2, 3):
        ctx.llm.next = FakeResult({"severity": "none", "finding": "",
                                   "reasoning": ""})
        _turn(ctx, history=[{"role": "user", "content": f"t{i}"}], response=f"a{i}")
        runtime.handle_pre_llm_call(session_id="s1", user_message="m",
                                    conversation_history=[])
    assert state.immunity_left == 0, (
        f"quiet turns must drain the cooldown, got {state.immunity_left}"
    )

    # Turn 4: a real blocker must steer again now that immunity expired.
    # pre_verify runs before pre_llm_call in the real turn lifecycle, so the
    # blocker must still be pending when the gate looks for it.
    ctx.llm.next = FakeResult({"severity": "blocker",
                               "finding": "later blocker", "reasoning": ""})
    _turn(ctx, history=[{"role": "user", "content": "t4"}], response="a4")
    assert runtime.handle_pre_verify(session_id="s1") is not None, \
        "immunity must expire so a real blocker can steer"


def test_budget_counts_retry_as_one_review(ctx, monkeypatch):
    """The empty-response retry must not double-spend the budget."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["max_reviews_per_session"] = 2
    calls = {"n": 0}

    def _flaky(**kw):
        calls["n"] += 1
        # First call of each review is empty; the retry returns a finding.
        if calls["n"] % 2 == 1:
            return FakeResult(None, text="")
        return FakeResult({"severity": "note", "finding": "queue nit",
                           "reasoning": ""})

    ctx.llm.complete_structured = _flaky
    _turn(ctx)
    _turn(ctx, history=[{"role": "user", "content": "t2"}], response="a2")

    state = runtime._get_state("s1", create=False)
    assert state.reviews_used == 2, "two reviews consumed, retries included"
    assert calls["n"] == 4, "2 reviews x (1 empty + 1 retry) = 4 provider calls"


def test_budget_blocks_after_cap_with_retries(ctx, monkeypatch):
    """No provider call at all once the cap is reached."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["max_reviews_per_session"] = 1
    ctx.llm.complete_structured = lambda **kw: FakeResult(None, text="")
    _turn(ctx)
    before = len(ctx.llm.calls)
    _turn(ctx, history=[{"role": "user", "content": "t2"}], response="a2")
    assert len(ctx.llm.calls) == before, "the cap must hold even when retrying"


def test_session_eviction_is_bounded():
    """_SESSIONS must not grow without bound across many sessions."""
    original = dict(runtime._SESSIONS)
    try:
        runtime._SESSIONS.clear()
        for i in range(runtime._MAX_SESSIONS + 10):
            runtime._get_state(f"session-{i}")
        assert len(runtime._SESSIONS) <= runtime._MAX_SESSIONS, (
            f"_SESSIONS grew to {len(runtime._SESSIONS)} "
            f"(cap {runtime._MAX_SESSIONS})"
        )
    finally:
        runtime._SESSIONS.clear()
        runtime._SESSIONS.update(original)
