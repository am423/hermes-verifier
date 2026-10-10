"""A verifier finding is accounted to the turn that PRODUCED it.

The review runs on a background thread, so it finishes after the turn that
triggered it has ended and its finding is delivered at the start of the next
turn. That delivery timing is unavoidable; the ACCOUNTING must not follow it.

Every finding therefore carries ``origin_turn`` -- the primary turn whose delta
was reviewed -- and that is the turn it is attributed to everywhere: the metrics
row, the review and finding events, and the rendered block handed to the agent.
Without this a review of turn 4 is delivered during turn 5 and reads as a
judgement on turn 5's work, which is both wrong and confusing to answer.
"""

from __future__ import annotations

import json

from verifier_pkg import runtime

from .test_runtime import FakeResult, _review_sync, _turn, ctx  # noqa: F401


def _records(ctx):
    path = runtime._log_path("s1", "default")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _finding_event():
    return [e for e in runtime.EVENT_LOG if e.get("kind") == "finding"]


def test_finding_is_accounted_to_the_turn_that_produced_it(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    runtime.handle_pre_llm_call(session_id="s1")          # turn 1 begins
    ctx.llm.next = FakeResult({"severity": "caution", "finding": "something is wrong"})
    _turn(ctx, history=[{"role": "user", "content": "do the thing"}])

    rows = _records(ctx)
    assert rows, "the review should have written a metrics row"
    assert rows[-1]["origin_turn"] == 1, "the finding belongs to turn 1"


def test_each_turn_stamps_its_own_number(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    runtime.handle_pre_llm_call(session_id="s1")          # turn 1
    ctx.llm.next = FakeResult({"severity": "note", "finding": "first"})
    _turn(ctx, history=[{"role": "user", "content": "one"}])

    runtime.handle_pre_llm_call(session_id="s1")          # turn 2
    ctx.llm.next = FakeResult({"severity": "note", "finding": "second"})
    _turn(ctx, session="s1",
          history=[{"role": "user", "content": "one"},
                   {"role": "assistant", "content": "final answer"},
                   {"role": "user", "content": "two"}],
          response="second answer")

    rows = _records(ctx)
    assert [r["origin_turn"] for r in rows] == [1, 2]


def test_review_and_finding_events_carry_the_origin_turn(ctx, monkeypatch):
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    runtime.handle_pre_llm_call(session_id="s1")
    ctx.llm.next = FakeResult({"severity": "caution",
                               "finding": "the delta deletes rows without a backup"})
    _turn(ctx, history=[{"role": "user", "content": "risky"}])

    for pred in (lambda e: e.get("kind") == "review",
                 lambda e: e.get("kind") == "finding"):
        events = [e for e in runtime.EVENT_LOG if pred(e)]
        assert events, "event missing"
        assert events[-1]["origin_turn"] == 1


def test_delivered_finding_names_the_turn_it_reviewed(ctx, monkeypatch):
    """The block handed to the agent must name the reviewed turn, not the current one."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    runtime.handle_pre_llm_call(session_id="s1")          # turn 1 produces the review
    ctx.llm.next = FakeResult({"severity": "caution", "finding": "the delta is risky"})
    _turn(ctx, history=[{"role": "user", "content": "first"}])

    # Turn 2 begins and the finding is delivered here.
    out = runtime.handle_pre_llm_call(session_id="s1")
    assert out and "reviewed_turn=\"1\"" in out["context"], out


def test_delivery_event_records_the_gap(ctx, monkeypatch):
    """A late delivery is a pacing symptom and must be visible as one."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    runtime.handle_pre_llm_call(session_id="s1")
    ctx.llm.next = FakeResult({"severity": "caution", "finding": "risky"})
    _turn(ctx, history=[{"role": "user", "content": "first"}])
    runtime.handle_pre_llm_call(session_id="s1")          # delivers during turn 2

    delivered = [e for e in runtime.EVENT_LOG if e.get("kind") == "finding_delivered"]
    assert delivered, "delivery must be observable"
    assert delivered[-1]["origin_turn"] == 1
    assert delivered[-1]["delivered_turn"] >= 2


def test_unknown_origin_turn_is_not_rendered(ctx):
    """turns_seen can be 0 (a hook fired without pre_llm_call); omit rather than lie."""
    block = runtime._render_finding("default", {
        "severity": "note", "finding": "text", "origin_turn": 0,
    })
    assert "reviewed_turn" not in block
    assert "note" in block and "text" in block


def test_a_stamped_finding_renders_its_turn(ctx):
    block = runtime._render_finding("default", {
        "severity": "caution", "finding": "text", "origin_turn": 7,
    })
    assert 'reviewed_turn="7"' in block


def test_non_default_verifier_keeps_its_slug_and_its_turn(ctx):
    block = runtime._render_finding("security", {
        "severity": "blocker", "finding": "leak", "origin_turn": 3,
    })
    assert 'verifier="security"' in block
    assert 'reviewed_turn="3"' in block
    assert "blocker" in block


def test_delivery_clears_pending_so_a_finding_is_not_repeated(ctx, monkeypatch):
    """Attribution must not become re-delivery: the finding is handed over once."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    runtime.handle_pre_llm_call(session_id="s1")
    ctx.llm.next = FakeResult({"severity": "caution", "finding": "risky"})
    _turn(ctx, history=[{"role": "user", "content": "first"}])

    first = runtime.handle_pre_llm_call(session_id="s1")
    assert first and "risky" in first["context"]
    second = runtime.handle_pre_llm_call(session_id="s1")
    assert second is None or "risky" not in second["context"]
