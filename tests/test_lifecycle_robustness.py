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
