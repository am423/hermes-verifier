"""Undelivered findings are a QUEUE, not a slot.

`pending` used to be a single `Optional[Dict]`. Two review cycles can complete
between two primary turns (``_MAX_INFLIGHT`` is 2), and `_record_and_deliver`
assigned the slot unconditionally, so the second accepted finding OVERWROTE the
first: it was counted in `stats["findings"]`, written to the JSONL audit row as
``accepted: true``, and published as a `finding` event -- then never delivered.
The audit trail claimed a delivery that never happened.

These tests pin at-least-once delivery, the bound, and the fact that a drop is
never silent.
"""

from __future__ import annotations

from verifier_pkg import runtime

from .test_runtime import ctx  # noqa: F401


def _verifier():
    state = runtime._get_state("s1")
    verifier = state.verifiers["default"]
    verifier.pending.clear()
    return state, verifier


def _accept(state, verifier, text, severity="caution", origin_turn=0):
    runtime._record_and_deliver(
        state, verifier,
        {"severity": severity, "finding": text, "reasoning": ""},
        True, None, origin_turn=origin_turn,
    )


def _events(kind):
    return [e for e in runtime.EVENT_LOG if e.get("kind") == kind]


def test_two_accepted_findings_are_both_delivered(ctx):  # noqa: F811
    """The regression: with one slot the first finding was silently lost."""
    state, verifier = _verifier()
    _accept(state, verifier, "ALPHA the queue writes without a lock", origin_turn=1)
    _accept(state, verifier, "BETA the retry loop swallows errors", origin_turn=2)

    out = runtime.handle_pre_llm_call(session_id="s1")
    assert out is not None, "both findings were queued; something must be delivered"
    assert "ALPHA" in out["context"], (
        "the first accepted finding was overwritten by the second -- counted and "
        "audited as accepted, never delivered"
    )
    assert "BETA" in out["context"]


def test_delivery_is_oldest_first(ctx):  # noqa: F811
    state, verifier = _verifier()
    _accept(state, verifier, "FIRST finding", origin_turn=1)
    _accept(state, verifier, "SECOND finding", origin_turn=2)
    out = runtime.handle_pre_llm_call(session_id="s1")
    assert out["context"].index("FIRST") < out["context"].index("SECOND")


def test_the_queue_drains_so_a_finding_is_not_repeated(ctx):  # noqa: F811
    state, verifier = _verifier()
    _accept(state, verifier, "ONCE only")
    first = runtime.handle_pre_llm_call(session_id="s1")
    assert first and "ONCE" in first["context"]
    second = runtime.handle_pre_llm_call(session_id="s1")
    assert second is None or "ONCE" not in second["context"]


def test_every_delivered_finding_publishes_an_event(ctx):  # noqa: F811
    """Including the ones that used to vanish, and naming their origin turn."""
    state, verifier = _verifier()
    _accept(state, verifier, "one", origin_turn=3)
    _accept(state, verifier, "two", origin_turn=4)
    runtime.handle_pre_llm_call(session_id="s1")

    delivered = _events("finding_delivered")
    assert len(delivered) == 2, "each delivered finding must be observable"
    assert {e["origin_turn"] for e in delivered} == {3, 4}


def test_the_queue_is_bounded_and_a_drop_is_never_silent(ctx):  # noqa: F811
    """A reviewer that stays slow must not queue without limit -- and losing a
    finding must be published, not swallowed."""
    state, verifier = _verifier()
    for i in range(runtime._MAX_PENDING + 3):
        _accept(state, verifier, f"finding {i}")

    assert len(verifier.pending) == runtime._MAX_PENDING, "queue must be bounded"
    drops = _events("finding_dropped")
    assert len(drops) == 3, "each overflow must be reported"
    assert drops[0]["reason"] == "pending queue overflow"


def test_overflow_drops_the_oldest_not_the_newest(ctx):  # noqa: F811
    """The newest finding is about the current state of the work."""
    state, verifier = _verifier()
    for i in range(runtime._MAX_PENDING + 1):
        _accept(state, verifier, f"finding {i}")
    texts = [f["finding"] for f in verifier.pending]
    assert "finding 0" not in texts, "the oldest should be dropped"
    assert f"finding {runtime._MAX_PENDING}" in texts, "the newest must survive"


def test_a_steered_blocker_leaves_the_rest_queued(ctx):  # noqa: F811
    """pre_verify takes only the blocker; other findings still ride to delivery."""
    state, verifier = _verifier()
    _accept(state, verifier, "NOTE keep the queue", severity="note")
    _accept(state, verifier, "BLOCKER stop here", severity="blocker")

    steered = runtime.handle_pre_verify(session_id="s1")
    assert steered and steered.get("action") == "continue"
    assert [f["severity"] for f in verifier.pending] == ["note"], (
        "steering a blocker must not discard unrelated queued findings"
    )
    out = runtime.handle_pre_llm_call(session_id="s1")
    assert out and "NOTE keep the queue" in out["context"]


def test_reset_empties_the_queue(ctx):  # noqa: F811
    state, verifier = _verifier()
    _accept(state, verifier, "stale")
    verifier.reset()
    assert not verifier.pending
