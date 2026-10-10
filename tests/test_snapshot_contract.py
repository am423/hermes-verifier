"""Contract: what consumers read from `snapshot()` must actually be in it.

`empty_responses` was counted in `_review_one` but never added to the session
dict in `snapshot()`, while BOTH consumers read it as
`s.get("empty_responses", 0)`. The counter was therefore pinned at 0 forever,
which silently disabled the one documented signal that the reviewer is DEAD
rather than finding nothing:

    "a rising empty count with a flat review count means the advisor model is
     not answering, not that it found nothing."     -- README

The suite missed it because the dashboard test hand-built a snapshot dict that
already contained the field, so it could never disagree with the producer. These
tests exercise the real producer instead.
"""

from __future__ import annotations

from verifier_pkg import commands, runtime

from .test_runtime import FakeResult, ctx  # noqa: F401


def _session_key():
    state = runtime._get_state("s1")
    return state


def test_snapshot_exposes_the_empty_response_counter(ctx):  # noqa: F811
    """The producer must publish it -- the consumers already read it."""
    state = _session_key()
    with runtime._LOCK:
        state.empty_responses = 7
    snap = runtime.snapshot()
    entry = snap["sessions"][state.session_id]
    assert "empty_responses" in entry, (
        "consumers read this key with .get(..., 0), so omitting it silently "
        "reports 0 forever instead of raising"
    )
    assert entry["empty_responses"] == 7


def test_status_line_reports_a_rising_empty_count(ctx):  # noqa: F811
    """End-to-end: a dead advisor must be visible in /verify status."""
    state = _session_key()
    with runtime._LOCK:
        state.empty_responses = 3
    text = "\n".join(commands._status_lines())
    assert "3 empty response(s)" in text, text


def test_every_key_the_consumers_read_exists_in_snapshot(ctx):  # noqa: F811
    """Pin the contract generally, so the next added counter cannot be orphaned."""
    state = _session_key()
    snap = runtime.snapshot()
    entry = snap["sessions"][state.session_id]
    # Keys commands._status_lines() and dashboard/plugin_api.py read off a session.
    required = {
        "session_id", "active", "reviewing", "backlog", "reviews_used",
        "empty_responses", "immunity_left", "last_error", "verifiers",
        "gate_checked", "gate_skipped",
    }
    assert required <= set(entry), f"missing from snapshot(): {required - set(entry)}"


def test_status_survives_a_hostile_non_numeric_setting(ctx):  # noqa: F811
    """int(cfg[...]) on a hand-edited config must not crash the surface."""
    ctx.config["nudge_first_turns"] = "abc"
    try:
        lines = commands._status_lines()
    except ValueError as exc:  # pragma: no cover - that is the bug
        raise AssertionError(f"/verify status crashed on a bad setting: {exc}")
    assert lines, "status must still produce output"
