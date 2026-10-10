"""Lifecycle robustness: bootstrapping, resource cleanup, and bounded state.

The plugin lives inside a long-running gateway process, so anything that grows
without a bound or fails to release on an error path eventually degrades the
whole host -- not just the verifier. These pin that.
"""

from __future__ import annotations

from verifier_pkg import runtime

from .test_runtime import FakeResult, ctx  # noqa: F401


# ── bootstrapping ───────────────────────────────────────────────────────────

def test_a_session_is_recorded_even_when_the_subsystem_is_disabled(ctx):  # noqa: F811
    """/verify on bootstraps from _RECENT_SESSION, and the shipped default is
    enabled: false -- so if the session is only recorded when enabled, the
    documented toggle can never turn the verifier on for the first time."""
    ctx.config["enabled"] = False
    runtime._RECENT_SESSION = None
    runtime.handle_post_llm_call(
        session_id="s1", user_message="hi",
        conversation_history=[{"role": "user", "content": "hi"}],
        assistant_response="hello")
    assert runtime.recent_session() is not None, (
        "no session recorded while disabled; /verify on cannot bootstrap"
    )


def test_ask_survives_a_completion_with_no_text(ctx):  # noqa: F811
    """A None completion is reachable (same shape the review path retries on).

    Slicing it raised TypeError straight out of the API handler: a 500 with no
    answer and no record written.
    """
    runtime._get_state("s1")            # ask_verifier needs an existing session
    ctx.llm.next = FakeResult(None, text=None)
    try:
        out = runtime.ask_verifier(session_id="s1", question="what changed?")
    except TypeError as exc:  # pragma: no cover - that is the bug
        raise AssertionError(f"ask_verifier crashed on an empty completion: {exc}")
    assert isinstance(out, dict)
    assert out.get("ok") is True


# ── resource release ────────────────────────────────────────────────────────

def test_a_thread_that_cannot_start_releases_what_it_was_handed(ctx, monkeypatch):  # noqa: F811
    """thread.start() can fail (thread limits, EAGAIN). When it does, no thread
    body runs, so nothing releases the backlog slot or the registry entry that
    _schedule_review had already allocated."""
    state = runtime._get_state("s1")
    state.backlog = 0
    before = len(runtime._ACTIVE_REVIEWS)

    def boom(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(runtime.threading.Thread, "start", boom)
    runtime._schedule_review(state, "some delta", 1)
    monkeypatch.undo()

    assert state.backlog == 0, "a thread that never started leaked its backlog slot"
    assert len(runtime._ACTIVE_REVIEWS) == before, (
        "a thread that never started stayed in the registry forever"
    )


# ── bounded state ───────────────────────────────────────────────────────────

def test_session_overrides_do_not_grow_without_bound(ctx):  # noqa: F811
    """Sessions are capped at _MAX_SESSIONS; the override map was not, so every
    /verify on|off added an entry that was never reclaimed."""
    for i in range(runtime._MAX_SESSIONS * 6):
        runtime.set_session_override(f"ghost-{i}", True)
    assert len(runtime._RUNTIME_OVERRIDES) <= runtime._MAX_SESSIONS * 4, (
        f"override map grew to {len(runtime._RUNTIME_OVERRIDES)} entries"
    )


def test_eviction_prefers_sessions_with_nothing_in_flight(ctx):  # noqa: F811
    """FIFO eviction removed the oldest-CREATED session -- typically the
    long-lived main one -- discarding pending findings and resetting
    reviews_used, which is the cost cap."""
    runtime._SESSIONS.clear()
    busy = runtime._get_state("busy")
    busy.reviews_used = 9
    runtime._record_and_deliver(
        busy, busy.verifiers["default"],
        {"severity": "blocker", "finding": "hold on", "reasoning": ""},
        True, None, origin_turn=1)

    for i in range(runtime._MAX_SESSIONS + 3):
        runtime._get_state(f"churn-{i}")

    assert "busy" in runtime._SESSIONS, (
        "a session with an undelivered finding was evicted while idle ones lived"
    )


# ── the gate's cost, and silent skips ───────────────────────────────────────

def test_the_gate_call_is_billed_to_the_session(ctx, monkeypatch):  # noqa: F811
    """The gate is the cheap half of the design but it is not free. Cost
    accounting that ignored it under-reports every turn it examined."""
    from verifier_pkg import gate
    ctx.config["gate_enabled"] = True

    class _Usage:
        input_tokens, output_tokens = 180, 40

    monkeypatch.setattr(runtime, "_GATE_FN_OVERRIDE",
                        lambda state, delta: gate.GateDecision(
                            True, "deep", "model", usage=_Usage()))
    state = runtime._get_state("s1")
    ctx.llm.queue = [FakeResult({"severity": "none", "finding": "", "reasoning": ""})]
    runtime._run_review(state, "some delta", 1)

    snap = runtime.snapshot()["sessions"][state.session_id]
    assert snap["gate_input_tokens"] == 180, "the decision's input tokens went unbilled"
    assert snap["gate_output_tokens"] == 40


def test_a_delta_that_could_not_be_reviewed_is_visible(ctx, full_inflight=None):  # noqa: F811
    """When every slot is busy the delta is skipped forever -- the cursor already
    advanced. That was a debug line only: no event, no counter, nothing in
    status, so a busy reviewer was indistinguishable from a quiet one."""
    state = runtime._get_state("s1")
    held = 0
    while runtime._INFLIGHT.acquire(blocking=False):
        held += 1
    try:
        runtime._run_review(state, "a delta nobody will review", 7)
    finally:
        for _ in range(held):
            runtime._INFLIGHT.release()

    assert state.reviews_dropped == 1
    events = [e for e in runtime.EVENT_LOG if e.get("kind") == "review_skipped"]
    assert events, "the skip left no trace at all"
    assert events[-1]["origin_turn"] == 7
    assert runtime.snapshot()["sessions"][state.session_id]["reviews_dropped"] == 1


# ── concurrent cycles must not interleave on one verifier's guard ───────────

def test_reviews_for_one_verifier_do_not_overlap(ctx):  # noqa: F811
    """Two cycles can be in flight at once and they share the verifier's
    EmissionGuard, which holds per-CYCLE state: begin_cycle resets the budget,
    so the first finisher consumed it and the second review's distinct finding
    was dropped as rate_limited."""
    import threading
    import time

    state = runtime._get_state("s1")
    verifier = state.verifiers["default"]

    inside = []
    overlaps = []

    def slow(**kwargs):
        inside.append(1)
        if len(inside) > 1:
            overlaps.append(1)
        time.sleep(0.06)
        inside.pop()
        return FakeResult({"severity": "none", "finding": "", "reasoning": ""})

    ctx.llm.complete_structured = slow
    threads = [threading.Thread(target=runtime._review_one,
                                args=(state, verifier, f"delta {i}", "deep", i),
                                daemon=True) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not overlaps, (
        "two review cycles ran concurrently on one verifier, interleaving on its "
        "per-cycle guard state"
    )
