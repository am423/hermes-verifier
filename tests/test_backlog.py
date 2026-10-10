"""The delta's backlog slot must be consumed on every exit path.

`_schedule_review` increments ``state.backlog`` and THEN starts the review
thread. So the thread owns that slot and must release it however it exits --
including when it cannot start at all because ``_INFLIGHT`` is already at
``_MAX_INFLIGHT``.

If it does not, backlog leaks upward permanently. That matters because
``_backlog_wait`` (engaged when ``sync_backlog`` is on) waits while
``backlog >= threshold``, so a permanently leaked backlog of 1 makes EVERY
later turn stall for the full ``_BACKLOG_WAIT_CAP``.
"""

from __future__ import annotations

import pytest

from verifier_pkg import runtime

from .test_runtime import FakeResult, ctx  # noqa: F401


@pytest.fixture()
def full_inflight():
    """Exhaust _INFLIGHT, then restore it exactly. A test that leaks semaphore
    slots would be far worse than the bug it is testing for."""
    runtime._INFLIGHT._value  # noqa: B018 - touch to confirm the attribute exists
    held = 0
    while runtime._INFLIGHT.acquire(blocking=False):
        held += 1
    assert held > 0
    try:
        yield held
    finally:
        for _ in range(held):
            runtime._INFLIGHT.release()


def _state(backlog: int = 1):
    state = runtime.SessionVerifier(session_id="s")
    state.verifiers["default"] = runtime.Verifier(slug="default", name="d")
    state.backlog = backlog
    return state


def test_backlog_is_consumed_when_the_review_cannot_start(ctx, full_inflight):  # noqa: F811
    """_INFLIGHT full -> the thread returns early. It must still release the slot."""
    state = _state(backlog=1)
    runtime._run_review(state, "the agent did something")
    assert state.backlog == 0, (
        "a review that could not start still owns its backlog slot; leaking it "
        "makes every subsequent turn wait the full backlog cap"
    )


def test_backlog_is_consumed_when_the_gate_skips(ctx, monkeypatch):
    """The gate-skip path returns from inside the cycle; the slot still frees."""
    from verifier_pkg import gate
    ctx.config["gate_enabled"] = True
    monkeypatch.setattr(runtime, "_GATE_FN_OVERRIDE",
                        lambda state, delta: gate.GateDecision(False, "routine", "model"))
    state = _state(backlog=1)
    runtime._run_review(state, "ran the linter")
    assert state.backlog == 0
    assert ctx.llm.calls == []


def test_backlog_does_not_go_negative(ctx):
    """max(0, ...) guards against a double-release; pin it."""
    state = _state(backlog=0)
    ctx.llm.queue = [FakeResult({"severity": "none", "finding": ""})]
    runtime._run_review(state, "delta")
    assert state.backlog == 0


def test_thread_registry_is_cleaned_when_the_review_cannot_start(ctx, full_inflight):  # noqa: F811
    """_schedule_review registers the thread BEFORE starting it.

    If the discard lives only on the post-acquire path, every overflow event
    leaves a dead Thread in _ACTIVE_REVIEWS permanently. drain_reviews then
    scans them on every shutdown, and the registry grows without bound.
    """
    import threading
    me = threading.current_thread()
    with runtime._THREADS_LOCK:
        runtime._ACTIVE_REVIEWS.add(me)
    try:
        state = _state(backlog=1)
        runtime._run_review(state, "delta")
        assert me not in runtime._ACTIVE_REVIEWS, (
            "_ACTIVE_REVIEWS accumulated a dead thread that drain_reviews will "
            "scan forever"
        )
    finally:
        runtime._ACTIVE_REVIEWS.discard(me)
