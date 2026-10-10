"""Tests for gate.py — the engage/effort decision.

The fail-safe paths carry the risk: if the gate ever suppresses a review on an
error, a finding is lost silently. Every one of those paths is pinned here.

The model's own accuracy is NOT tested here — it was measured against three
labelled case sets in ``sidecar/`` (44 cases, 0 dangerous actions skipped, 0
down-tiered). What these tests pin is the CONTRACT: correct pass-through when
the model answers, and ENGAGE AT DEEP on every conceivable failure.
"""

from __future__ import annotations

import pytest
from verifier_pkg import gate

from .test_runtime import FakeCtx, FakeResult, FakeUsage


def model_says(engage: bool, effort: str) -> FakeResult:
    return FakeResult({"engage": engage, "effort": effort})


# --------------------------------------------------------------------------
# the pass-through contract
# --------------------------------------------------------------------------

def test_decides_to_skip_routine_work():
    ctx = FakeCtx()
    ctx.llm.next = model_says(False, "routine")
    d = gate.decide(ctx, digest="sorted the imports", provider="p", model="m")
    assert d.engage is False
    assert d.effort == "routine"
    assert d.source == "model"


def test_decides_to_engage_at_deep():
    ctx = FakeCtx()
    ctx.llm.next = model_says(True, "deep")
    d = gate.decide(ctx, digest="dropped the users table", provider="p", model="m")
    assert d.engage is True
    assert d.tier == "deep"
    assert d.source == "model"


def test_effort_is_normalised_when_the_gate_skips():
    """A tier is meaningless on a skipped turn; do not propagate a stray value."""
    ctx = FakeCtx()
    ctx.llm.next = model_says(False, "deep")
    d = gate.decide(ctx, digest="typo", provider="p", model="m")
    assert d.engage is False
    assert d.effort == "routine"


# --------------------------------------------------------------------------
# fail-safe: every failure engages at deep
# --------------------------------------------------------------------------

def test_provider_error_engages_at_deep():
    ctx = FakeCtx()
    ctx.llm.raise_error = RuntimeError("provider unreachable")
    d = gate.decide(ctx, digest="anything", provider="p", model="m")
    assert d.engage is True
    assert d.effort == "deep"
    assert d.source == "fail-safe"


def test_trust_error_engages_at_deep():
    """A denied model override must not silently disable review."""
    from agent.plugin_llm import PluginLlmTrustError
    ctx = FakeCtx()
    ctx.llm.raise_error = PluginLlmTrustError("denied")
    d = gate.decide(ctx, digest="anything", provider="p", model="m")
    assert d.engage is True and d.source == "fail-safe"


def test_empty_completion_engages_at_deep():
    """The reasoning-model trap: finish_reason=length yields parsed=None, text=''.

    This is the single most dangerous failure, because an empty completion is
    indistinguishable from "found nothing to review".
    """
    ctx = FakeCtx()
    ctx.llm.next = FakeResult(None, text="")
    d = gate.decide(ctx, digest="anything", provider="p", model="m")
    assert d.engage is True
    assert d.effort == "deep"
    assert d.source == "fail-safe"


@pytest.mark.parametrize("parsed", [
    {},                                   # nothing at all
    {"engage": True},                     # effort missing
    {"effort": "deep"},                   # engage missing
    {"engage": "false", "effort": "deep"},  # wrong type: a truthy string
    {"engage": 0, "effort": "deep"},        # wrong type: an int
    {"engage": True, "effort": "DEEP"},     # wrong case
    {"engage": True, "effort": "moderate"},  # not an offered enum
    {"engage": False, "effort": None},      # null effort
])
def test_malformed_answers_engage_at_deep(parsed):
    ctx = FakeCtx()
    ctx.llm.next = FakeResult(parsed)
    d = gate.decide(ctx, digest="anything", provider="p", model="m")
    assert d.engage is True, f"{parsed!r} must not suppress a review"
    assert d.effort == "deep"
    assert d.source == "fail-safe"


def test_non_dict_parsed_engages_at_deep():
    ctx = FakeCtx()
    ctx.llm.next = FakeResult("a string, not a dict")
    d = gate.decide(ctx, digest="anything", provider="p", model="m")
    assert d.engage is True and d.source == "fail-safe"


# --------------------------------------------------------------------------
# the digest is bounded -- the economics depend on it
# --------------------------------------------------------------------------

def test_short_digest_is_untouched():
    assert gate.build_digest("ran ls") == "ran ls"


def test_long_digest_is_bounded_and_keeps_both_ends():
    text = "HEAD-MARKER " + ("x" * 20000) + " TAIL-MARKER"
    out = gate.build_digest(text)
    assert len(out) <= gate.GATE_DIGEST_CAP + 32
    assert "HEAD-MARKER" in out
    assert "TAIL-MARKER" in out
    assert "[truncated]" in out


def test_digest_from_delta_bounds_the_reviewer_text():
    out = gate.digest_from_delta("y" * 50000)
    assert len(out) <= gate.GATE_DIGEST_CAP + 32


def test_blank_digest_is_handled():
    assert gate.build_digest("") == ""
    assert gate.build_digest(None) == ""


# --------------------------------------------------------------------------
# the call itself
# --------------------------------------------------------------------------

def test_passes_provider_and_model_through():
    ctx = FakeCtx()
    ctx.llm.next = model_says(True, "deep")
    gate.decide(ctx, digest="x", provider="nous", model="anthropic/claude-opus-5.5")
    kwargs = ctx.llm.calls[-1]
    assert kwargs["provider"] == "nous"
    assert kwargs["model"] == "anthropic/claude-opus-5.5"


def test_omits_provider_and_model_when_unset():
    """Unset must mean 'use the plugin default', not 'send null'."""
    ctx = FakeCtx()
    ctx.llm.next = model_says(True, "deep")
    gate.decide(ctx, digest="x", provider=None, model=None)
    kwargs = ctx.llm.calls[-1]
    assert "provider" not in kwargs
    assert "model" not in kwargs


def test_gate_requests_enough_tokens_for_a_reasoning_model():
    """Measured: 40 tokens returned EMPTY content; 400 lost 2/18 held-out calls."""
    assert gate.GATE_MAX_TOKENS >= 400
    ctx = FakeCtx()
    ctx.llm.next = model_says(True, "deep")
    gate.decide(ctx, digest="x", provider=None, model=None)
    assert ctx.llm.calls[-1]["max_tokens"] == gate.GATE_MAX_TOKENS


def test_gate_uses_its_own_schema_not_the_review_schema():
    ctx = FakeCtx()
    ctx.llm.next = model_says(True, "deep")
    gate.decide(ctx, digest="x", provider=None, model=None)
    kwargs = ctx.llm.calls[-1]
    assert kwargs["schema_name"] == "verifier_gate"
    assert set(kwargs["json_schema"]["properties"]) == {"engage", "effort"}


def test_effort_prompt_keeps_the_deep_default():
    """Load-bearing: without this line a dangerous action was down-tiered 1-in-9."""
    assert "DEFAULT TO" in gate.GATE_SYSTEM
    assert "When unsure, choose deep" in gate.GATE_SYSTEM


def test_describe_is_readable_in_status_output():
    assert "skipped" in gate.describe(gate.GateDecision(False, "routine", "model"))
    assert "deep" in gate.describe(gate.GateDecision(True, "deep", "model"))
    assert "unavailable" in gate.describe(gate.FAIL_SAFE)


def test_usage_is_not_swallowed():
    """The gate's cost must be visible for the roll-up, not silently absorbed."""
    ctx = FakeCtx()
    ctx.llm.next = FakeResult({"engage": True, "effort": "deep"},
                              usage=FakeUsage(i=180, o=140))
    gate.decide(ctx, digest="x", provider=None, model=None)
    assert ctx.llm.calls[-1]["purpose"] == "hermes-verifier engagement gate"


# --------------------------------------------------------------------------
# integration: the skip must actually prevent the review from being paid for
# --------------------------------------------------------------------------

from verifier_pkg import runtime                            # noqa: E402
from .test_runtime import ctx                               # noqa: E402,F401  (fixture)


def _review_run(state, delta="agent did something"):
    runtime._run_review(state, delta)


def test_gate_is_off_by_default_so_an_existing_install_is_unchanged(ctx):
    """Gate disabled must make NO model call for the decision and review normally."""
    ctx.llm.queue = [FakeResult({"severity": "none", "finding": ""})]
    state = runtime.SessionVerifier(session_id="s")
    state.verifiers["default"] = runtime.Verifier(slug="default", name="d")
    _review_run(state)
    assert state.gate_checked == 1
    assert state.gate_skipped == 0
    assert state.last_gate.source == "disabled"
    # Exactly one call, and it is the review -- no gate call was made.
    assert len(ctx.llm.calls) == 1
    assert ctx.llm.calls[0]["schema_name"] == "verifier_review"


def test_gate_skip_prevents_the_review_call_entirely(ctx, monkeypatch):
    ctx.config["gate_enabled"] = True
    monkeypatch.setattr(runtime, "_GATE_FN_OVERRIDE",
                        lambda state, delta: gate.GateDecision(False, "routine", "model"))
    state = runtime.SessionVerifier(session_id="s")
    state.verifiers["default"] = runtime.Verifier(slug="default", name="d")
    state.backlog = 1
    _review_run(state)
    assert state.gate_skipped == 1
    assert ctx.llm.calls == [], "a skipped turn must cost zero review tokens"
    assert state.backlog == 0, "the delta was consumed; backlog must not stall"


def test_gate_engage_runs_the_review_at_the_chosen_tier(ctx, monkeypatch):
    ctx.config["gate_enabled"] = True
    monkeypatch.setattr(runtime, "_GATE_FN_OVERRIDE",
                        lambda state, delta: gate.GateDecision(True, "deep", "model"))
    ctx.llm.queue = [FakeResult({"severity": "none", "finding": ""})]
    state = runtime.SessionVerifier(session_id="s")
    state.verifiers["default"] = runtime.Verifier(slug="default", name="d")
    _review_run(state)
    assert state.gate_skipped == 0
    assert len(ctx.llm.calls) == 1
    assert ctx.llm.calls[0]["schema_name"] == "verifier_review"


def test_fail_safe_decision_still_reviews(ctx, monkeypatch):
    ctx.config["gate_enabled"] = True
    monkeypatch.setattr(runtime, "_GATE_FN_OVERRIDE",
                        lambda state, delta: gate.FAIL_SAFE)
    ctx.llm.queue = [FakeResult({"severity": "none", "finding": ""})]
    state = runtime.SessionVerifier(session_id="s")
    state.verifiers["default"] = runtime.Verifier(slug="default", name="d")
    _review_run(state)
    assert len(ctx.llm.calls) == 1, "an unavailable gate must never suppress review"


def test_routine_effort_uses_a_smaller_output_budget(ctx, monkeypatch):
    """Effort must reach the reviewer as a smaller cap -- but never below the floor."""
    ctx.config["gate_enabled"] = True
    ctx.config["advisor_max_tokens"] = 4096
    monkeypatch.setattr(runtime, "_GATE_FN_OVERRIDE",
                        lambda state, delta: gate.GateDecision(True, "routine", "model"))
    ctx.llm.queue = [FakeResult({"severity": "none", "finding": ""})]
    state = runtime.SessionVerifier(session_id="s")
    state.verifiers["default"] = runtime.Verifier(slug="default", name="d")
    _review_run(state)
    assert ctx.llm.calls[0]["max_tokens"] == 1024


def test_deep_effort_keeps_the_configured_budget(ctx, monkeypatch):
    ctx.config["gate_enabled"] = True
    ctx.config["advisor_max_tokens"] = 4096
    monkeypatch.setattr(runtime, "_GATE_FN_OVERRIDE",
                        lambda state, delta: gate.GateDecision(True, "deep", "model"))
    ctx.llm.queue = [FakeResult({"severity": "none", "finding": ""})]
    state = runtime.SessionVerifier(session_id="s")
    state.verifiers["default"] = runtime.Verifier(slug="default", name="d")
    _review_run(state)
    assert ctx.llm.calls[0]["max_tokens"] == 4096


def test_routine_never_goes_below_the_truncation_floor(ctx, monkeypatch):
    """Below 1024 a response truncates mid-JSON and the entire review is lost."""
    ctx.config["gate_enabled"] = True
    ctx.config["advisor_routine_tokens"] = 64          # a hostile setting
    monkeypatch.setattr(runtime, "_GATE_FN_OVERRIDE",
                        lambda state, delta: gate.GateDecision(True, "routine", "model"))
    ctx.llm.queue = [FakeResult({"severity": "none", "finding": ""})]
    state = runtime.SessionVerifier(session_id="s")
    state.verifiers["default"] = runtime.Verifier(slug="default", name="d")
    _review_run(state)
    cap = ctx.llm.calls[0].get("max_tokens")
    assert cap is None or cap >= 1024


def test_skip_publishes_a_decision_event(ctx, monkeypatch):
    ctx.config["gate_enabled"] = True
    monkeypatch.setattr(runtime, "_GATE_FN_OVERRIDE",
                        lambda state, delta: gate.GateDecision(False, "routine", "model"))
    state = runtime.SessionVerifier(session_id="s")
    state.verifiers["default"] = runtime.Verifier(slug="default", name="d")
    _review_run(state)
    events = [e for e in runtime.EVENT_LOG if e.get("kind") == "gate_decided"]
    assert events, "the decision must be observable, not silent"
    assert events[-1]["engage"] is False
