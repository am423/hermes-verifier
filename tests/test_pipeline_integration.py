"""The production pipeline, exercised with NO test seam.

Every other integration test replaces `_REVIEW_FN_OVERRIDE` or `_GATE_FN_OVERRIDE`,
which skips the real thread, the real budget accounting, and the real gate call.
That means a regression in the wiring itself -- the gate sending the whole
unbounded delta as its digest, the review running at the wrong tier, concurrency
exceeding _MAX_INFLIGHT -- would ship invisibly.
"""

from __future__ import annotations

import threading
import time

from verifier_pkg import runtime

from .test_runtime import FakeResult, ctx  # noqa: F401


def _calls_by_schema(ctx):
    out = {}
    for call in ctx.llm.calls:
        out.setdefault(call.get("schema_name"), []).append(call)
    return out


# ── the gate is wired to the real call path ─────────────────────────────────

def test_the_gate_runs_live_and_its_digest_is_bounded(ctx):  # noqa: F811
    """No override: the gate must make its own call, and a cheap one.

    The entire economic case is that the gate costs ~180 tokens against a
    ~9,500-token review. If the wiring handed it the raw delta, the gate would
    cost as much as the review it exists to avoid -- and every unit test would
    still pass, because they call gate.decide() directly.
    """
    ctx.config["gate_enabled"] = True
    long_delta = "\n".join(f"2026-10-10 10:00:{i % 60:02d} INFO step {i} completed"
                           for i in range(500))
    ctx.llm.queue = [
        FakeResult({"engage": True, "effort": "deep"}),
        FakeResult({"severity": "none", "finding": "", "reasoning": ""}),
    ]
    runtime._run_review(runtime._get_state("s1"), long_delta, 1)

    by_schema = _calls_by_schema(ctx)
    gate_calls = by_schema.get("verifier_gate") or []
    assert gate_calls, f"the gate never ran; schemas seen: {list(by_schema)}"
    digest = gate_calls[0]["input"][0]["text"]
    assert len(digest) <= runtime.gate_mod.GATE_DIGEST_CAP + 40, (
        f"the gate was handed {len(digest)} chars; it is meant to see a bounded "
        "digest, not the whole delta"
    )
    assert by_schema.get("verifier_review"), "the review never ran after an engage"


def test_the_gate_skip_costs_no_review_call(ctx):  # noqa: F811
    ctx.config["gate_enabled"] = True
    ctx.llm.queue = [FakeResult({"engage": False, "effort": "routine"})]
    runtime._run_review(runtime._get_state("s1"), "ran the linter and formatted", 1)

    by_schema = _calls_by_schema(ctx)
    assert by_schema.get("verifier_gate"), "the gate did not run"
    assert not by_schema.get("verifier_review"), (
        "a skipped turn still paid for a review"
    )


def test_a_routine_tier_reaches_the_reviewer_as_a_smaller_cap(ctx):  # noqa: F811
    ctx.config["gate_enabled"] = True
    ctx.config["advisor_max_tokens"] = 4096
    ctx.llm.queue = [
        FakeResult({"engage": True, "effort": "routine"}),
        FakeResult({"severity": "none", "finding": "", "reasoning": ""}),
    ]
    runtime._run_review(runtime._get_state("s1"), "renamed a local variable", 1)

    review = _calls_by_schema(ctx)["verifier_review"][0]
    assert review["max_tokens"] == 1024, (
        "the gate's tier did not reach the reviewer"
    )


def test_a_deep_tier_keeps_the_configured_cap(ctx):  # noqa: F811
    ctx.config["gate_enabled"] = True
    ctx.config["advisor_max_tokens"] = 4096
    ctx.llm.queue = [
        FakeResult({"engage": True, "effort": "deep"}),
        FakeResult({"severity": "none", "finding": "", "reasoning": ""}),
    ]
    runtime._run_review(runtime._get_state("s1"), "dropped the users table", 1)
    assert _calls_by_schema(ctx)["verifier_review"][0]["max_tokens"] == 4096


def test_a_failed_gate_still_reviews(ctx):  # noqa: F811
    """Fail-safe on the real path, not just in the unit test."""
    ctx.config["gate_enabled"] = True
    ctx.llm.queue = [
        FakeResult(None, text=""),        # empty gate completion -> FAIL_SAFE
        FakeResult({"severity": "none", "finding": "", "reasoning": ""}),
    ]
    runtime._run_review(runtime._get_state("s1"), "something happened", 1)
    assert _calls_by_schema(ctx).get("verifier_review"), (
        "an unusable gate decision suppressed the review"
    )


# ── concurrency is capped for real ──────────────────────────────────────────

def test_concurrent_reviews_never_exceed_the_inflight_cap(ctx):  # noqa: F811
    """_MAX_INFLIGHT is the only thing bounding parallel model spend."""
    inside = []
    peak = []
    lock = threading.Lock()

    def _complete(**kwargs):
        ctx.llm.calls.append(kwargs)
        with lock:
            inside.append(1)
            peak.append(len(inside))
        time.sleep(0.05)
        with lock:
            inside.pop()
        return FakeResult({"severity": "none", "finding": "", "reasoning": ""})

    ctx.llm.complete_structured = _complete
    state = runtime._get_state("s1")
    threads = [threading.Thread(target=runtime._run_review,
                                args=(state, f"delta {i}", i), daemon=True)
               for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert peak and max(peak) <= runtime._MAX_INFLIGHT, (
        f"observed {max(peak)} concurrent reviews; the cap is {runtime._MAX_INFLIGHT}"
    )


def test_a_delta_dropped_by_a_full_pipeline_is_recorded(ctx):  # noqa: F811
    """The no-override path: a skip must be visible, not a debug line."""
    state = runtime._get_state("s1")
    before = state.reviews_dropped
    held = 0
    while runtime._INFLIGHT.acquire(blocking=False):
        held += 1
    try:
        runtime._run_review(state, "a delta nobody will review", 3)
    finally:
        for _ in range(held):
            runtime._INFLIGHT.release()

    assert state.reviews_dropped == before + 1
    kinds = [e.get("kind") for e in runtime.EVENT_LOG]
    assert "review_skipped" in kinds


# ── the slash command actually toggles ──────────────────────────────────────

def test_verify_on_and_off_toggle_the_session(ctx):  # noqa: F811
    from verifier_pkg import commands
    runtime._get_state("s1")
    runtime.mark_verifier_session("s1")

    # Drive the module's own accessors rather than guessing at internals.
    runtime.set_session_override("s1", False)
    assert runtime.subsystem_enabled("s1") is False
    runtime.set_session_override("s1", True)
    assert runtime.subsystem_enabled("s1") is True

    lines = commands._status_lines()
    assert any("hermes-verifier" in line for line in lines), lines
    assert any("gate:" in line for line in lines), (
        "status must report the gate now that it exists"
    )
