"""The review budget counts REVIEWS, failures are tracked per verifier, and a
designation reports whether it actually persisted.
"""

from __future__ import annotations

from verifier_pkg import runtime

from .test_runtime import FakeResult, _review_sync, _turn, ctx  # noqa: F401


# ── the cap counts reviews, not deltas ──────────────────────────────────────

def test_one_delta_with_two_verifiers_consumes_two_reviews(ctx, monkeypatch):  # noqa: F811
    """A delta runs EVERY enabled verifier, so counting deltas made a cap of N
    really N x verifiers -- not what 'hard cap on advisor reviews' says."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    runtime._ROSTER_OVERRIDE = [{"name": "Security", "slug": "security",
                                 "enabled": True}]
    ctx.config["max_reviews_per_session"] = 10
    ctx.llm.next = FakeResult({"severity": "none", "finding": "", "reasoning": ""})
    _turn(ctx, history=[{"role": "user", "content": "one"}])

    state = runtime._get_state("s1")
    assert len(state.enabled_verifiers()) == 2, "precondition: two verifiers"
    assert state.reviews_used == 2, (
        f"one delta ran 2 verifiers but the budget counted "
        f"{state.reviews_used}"
    )


def test_the_cap_still_blocks_once_reached(ctx, monkeypatch):  # noqa: F811
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["max_reviews_per_session"] = 1
    ctx.llm.next = FakeResult({"severity": "none", "finding": "", "reasoning": ""})
    _turn(ctx, history=[{"role": "user", "content": "one"}])
    state = runtime._get_state("s1")
    used = state.reviews_used
    assert used >= 1

    ctx.llm.next = FakeResult({"severity": "none", "finding": "", "reasoning": ""})
    _turn(ctx, session="s1",
          history=[{"role": "user", "content": "one"},
                   {"role": "assistant", "content": "final answer"},
                   {"role": "user", "content": "two"}],
          response="second answer")
    assert state.reviews_used == used, "the cap stopped counting reviews"


def test_a_gate_skip_does_not_consume_the_review_budget(ctx, monkeypatch):  # noqa: F811
    """A skipped turn costs only the gate call; it is not a review."""
    from verifier_pkg import gate
    ctx.config["gate_enabled"] = True
    ctx.config["max_reviews_per_session"] = 10
    monkeypatch.setattr(runtime, "_GATE_FN_OVERRIDE",
                        lambda state, delta: gate.GateDecision(False, "routine", "model"))
    _turn(ctx, history=[{"role": "user", "content": "ran the linter"}])
    state = runtime._get_state("s1")
    assert state.reviews_used == 0, (
        "a turn that was never reviewed consumed review budget"
    )


# ── failures are per verifier ───────────────────────────────────────────────

def test_a_broken_verifier_trips_the_backlog_escape_in_a_mixed_roster(ctx, monkeypatch):  # noqa: F811
    """The session counter was reset by ANY verifier's success, so a permanently
    broken entry in a mixed roster never reached the failure streak that
    releases _backlog_wait's waiters -- it kept costing the full cap."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    runtime._ROSTER_OVERRIDE = [{"name": "Security", "slug": "security",
                                 "enabled": True}]
    state = runtime._get_state("s1")
    state.ensure_verifiers(runtime.discover_roster())

    # The default verifier is broken; security keeps succeeding.
    state.verifiers["default"].consecutive_failures = 3
    state.consecutive_failures = 0          # security just succeeded

    clock = {"t": 0.0}
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock["t"])
    state.backlog = 99
    runtime._backlog_wait(state, 1)
    assert clock["t"] == 0, (
        "a broken verifier did not release the waiter; it will cost the full cap "
        "on every turn"
    )


def test_a_verifier_failure_streak_is_reset_only_by_its_own_success(ctx, monkeypatch):  # noqa: F811
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    state = runtime._get_state("s1")
    verifier = state.verifiers["default"]
    verifier.consecutive_failures = 2

    ctx.llm.next = FakeResult({"severity": "none", "finding": "", "reasoning": ""})
    _turn(ctx, history=[{"role": "user", "content": "x"}])
    assert verifier.consecutive_failures == 0, "a good review must clear the streak"


def test_garbage_output_does_not_clear_the_streak(ctx, monkeypatch):  # noqa: F811
    """A malformed response is a FAILURE. The reset used to run before
    validation, so a verifier that persistently returned garbage looked healthy
    and the failure escape could never trip."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    state = runtime._get_state("s1")
    verifier = state.verifiers["default"]
    verifier.consecutive_failures = 1
    state.consecutive_failures = 1

    ctx.llm.next = FakeResult(None, text="not json at all, just prose")
    _turn(ctx, history=[{"role": "user", "content": "x"}])

    assert verifier.consecutive_failures == 2, (
        "an unparseable response cleared the failure streak"
    )
    assert state.last_error, "the failure must be recorded"


# ── designation honesty ─────────────────────────────────────────────────────

def test_designation_reports_when_it_could_not_persist(ctx):  # noqa: F811
    """The docstring promises the designation survives a restart, and the API
    answered ok:true regardless -- so the UI showed a durable setting that would
    vanish on the next launch."""
    def _fail(key, value):
        raise RuntimeError("settings are read-only")

    ctx.set_config = _fail
    out = runtime.mark_verifier_session("s1")
    assert isinstance(out, dict), "callers need the persistence outcome"
    assert out["session_id"]
    assert out["persisted"] is False


def test_designation_reports_success(ctx):  # noqa: F811
    out = runtime.mark_verifier_session("s2")
    assert out["persisted"] is True
