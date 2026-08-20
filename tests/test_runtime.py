"""Runtime wiring tests against a fake PluginContext.

Covers: enable state, the review loop (synchronous via the override hook),
JSONL persistence, finding delivery, immunity, and the steer gate.
"""

from __future__ import annotations

import json
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
                 usage: FakeUsage | None = None):
        self.parsed = parsed
        self.model = model
        self.usage = usage or FakeUsage()


class FakeLlm:
    def __init__(self) -> None:
        self.calls: list = []
        self.next: Any = None
        self.raise_error: Exception | None = None

    def complete_structured(self, **kwargs) -> FakeResult:
        self.calls.append(kwargs)
        if self.raise_error is not None:
            raise self.raise_error
        result = self.next if self.next is not None else FakeResult(None)
        self.next = None
        return result


class FakeCtx:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.llm = FakeLlm()

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


@pytest.fixture()
def ctx(tmp_path):
    runtime._DATA_DIR_OVERRIDE = tmp_path / "data"
    runtime._RUNTIME_OVERRIDES.clear()
    runtime._SESSIONS.clear()
    runtime._RECENT_SESSION = None
    fake = FakeCtx({"enabled": True, "model": "deepseek/deepseek-v4",
                    "immunity_turns": 3, "max_delta_chars": 20000})
    runtime.bind_ctx(fake)
    yield fake
    runtime._DATA_DIR_OVERRIDE = None
    runtime._CTX = None


def _review_sync(state, delta_text):
    # Run the real review inline instead of on a thread.
    runtime._run_review(state, delta_text)


def _turn(ctx, session="s1", history=None, response="final answer"):
    runtime.handle_post_llm_call(
        session_id=session,
        user_message="do the thing",
        conversation_history=history if history is not None else [],
        assistant_response=response,
    )


def test_enable_state(ctx):
    assert runtime.subsystem_enabled("any") is True
    ctx.config["enabled"] = False
    assert runtime.subsystem_enabled("any") is False
    runtime.set_session_override("any", True)
    assert runtime.subsystem_enabled("any") is True


def test_model_resolution(ctx):
    ctx.config["model"] = "deepseek/deepseek-v4"
    assert runtime._resolve_model(ctx) == ("deepseek", "deepseek-v4")
    ctx.config["model"] = "gpt-something"
    assert runtime._resolve_model(ctx) == (None, "gpt-something")
    ctx.config["model"] = ""
    assert runtime._resolve_model(ctx) == (None, None)


def test_full_cycle_raise_and_deliver(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({
        "severity": "caution",
        "finding": "Bypasses the durable queue in src/jobs",
        "reasoning": "the write path skips the queue",
    })

    _turn(ctx)

    # The review ran synchronously: pending finding + JSONL record.
    state = runtime._get_state("s1", create=False)
    assert state is not None
    assert state.pending["severity"] == "caution"
    assert state.stats["reviews"] == 1
    assert state.stats["cautions"] == 1

    log = runtime._log_path("s1")
    records = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["accepted"] is True
    assert records[0]["severity"] == "caution"
    assert records[0]["usage"]["cost_usd"] == 0.001

    # Structured call carried the model override + schema + persona.
    call = ctx.llm.calls[-1]
    assert call["provider"] == "deepseek"
    assert call["model"] == "deepseek-v4"
    assert call["json_schema"] == runtime.REVIEW_SCHEMA
    assert call["system_prompt"].startswith("You are the Verifier")
    assert call["input"][0]["type"] == "text"
    assert "final answer" in call["input"][0]["text"]

    # Delivery: next primary turn's pre_llm_call injects the finding.
    injected = runtime.handle_pre_llm_call(session_id="s1")
    assert injected is not None
    assert '<finding verifier="default" severity="caution"' in injected["context"]
    assert "Bypasses the durable queue" in injected["context"]
    # Consumed on delivery.
    assert runtime.handle_pre_llm_call(session_id="s1") is None
    # Interrupting delivery armed the immunity cooldown.
    assert state.immunity_left == 3


def test_duplicate_suppressed(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "note", "finding": "Tidy the imports",
                               "reasoning": "unused"})
    _turn(ctx)
    assert runtime._get_state("s1").pending is not None
    # The primary turn picks the finding up; pending is consumed.
    assert runtime.handle_pre_llm_call(session_id="s1") is not None
    assert runtime._get_state("s1").pending is None

    # Same finding again (a later turn): emission guard drops it silently.
    ctx.llm.next = FakeResult({"severity": "note", "finding": "Tidy the imports",
                               "reasoning": "again"})
    _turn(ctx)
    assert runtime._get_state("s1").pending is None
    assert runtime._get_state("s1").guard.stats["deduped"] == 1

    log = runtime._log_path("s1")
    records = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(records) == 2
    assert records[1]["accepted"] is False


def test_none_severity_no_finding(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "none", "finding": "",
                               "reasoning": "clean turn"})
    _turn(ctx)
    state = runtime._get_state("s1", create=False)
    assert state.pending is None
    assert state.stats["findings"] == 0


def test_llm_failure_records_error_no_crash(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.raise_error = RuntimeError("provider 500")
    _turn(ctx)
    state = runtime._get_state("s1", create=False)
    assert "RuntimeError" in state.stats["last_error"]
    assert state.pending is None


def test_non_json_result_ignored(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult(None)  # parsed is None: refused / non-JSON
    _turn(ctx)
    state = runtime._get_state("s1", create=False)
    assert state.pending is None
    assert state.stats["reviews"] == 1


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
    assert state.pending is None
    assert state.immunity_left == 3

    # During immunity, a blocker no longer steers (it rides next-turn injection).
    ctx.llm.next = FakeResult({"severity": "blocker", "finding": "Second blocker",
                               "reasoning": "x"})
    _turn(ctx)
    assert runtime.handle_pre_verify(session_id="s1") is None
    assert state.pending is not None


def test_caution_does_not_steer_pre_verify(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "caution", "finding": "Risky path",
                               "reasoning": "x"})
    _turn(ctx)
    assert runtime.handle_pre_verify(session_id="s1") is None
    # But the caution is still delivered on the next turn.
    assert runtime.handle_pre_llm_call(session_id="s1") is not None


def test_session_reset_reprimes(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "note", "finding": "cleanup",
                               "reasoning": "x"})
    _turn(ctx, history=[{"role": "user", "content": "a"}])
    state = runtime._get_state("s1", create=False)
    assert state.cursor_len == 1
    assert state.pending is not None

    runtime.handle_session_reset(session_id="s1")
    assert state.cursor_len is None
    assert state.pending is None
    assert len(state.guard) == 0


def test_disabled_session_skips_everything(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["enabled"] = False
    runtime.clear_session_override("s1")
    _turn(ctx)
    # Disabled: no state created, no reviews, no LLM calls.
    assert runtime._get_state("s1", create=False) is None
    assert ctx.llm.calls == []
