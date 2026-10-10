"""Roster edge cases that silently disable review entirely.

`ensure_verifiers` treats "default" as the RESERVED slug of the built-in
verifier and always re-creates it -- but it then walks the roster and pops any
DISABLED entry's slug. A roster entry whose name or slug maps onto "default"
(a VERIFIER.yml entry literally called `Default`, with `enabled: false`) could
therefore delete the built-in verifier, leaving the session with an empty
`verifiers` dict.

The damage was silent and total: the next turn's `_sync_roster` dereferenced
`state.verifiers[DEFAULT_VERIFIER]`, raised KeyError, and that landed in the
post-hook's broad `except Exception` -- so no verifier ran, no finding was
produced, no event was published, and `last_error` stayed empty.
"""

from __future__ import annotations

from verifier_pkg import runtime

from .test_runtime import FakeResult, _review_sync, _turn, ctx  # noqa: F401


def _roster(*entries):
    runtime._ROSTER_OVERRIDE = list(entries)


def test_a_disabled_entry_named_default_cannot_delete_the_builtin(ctx):  # noqa: F811
    """The reserved slug must survive whatever the roster says."""
    state = runtime._get_state("s1")
    _roster({"name": "Default", "slug": "default", "enabled": False})
    state.ensure_verifiers(runtime.discover_roster())

    assert "default" in state.verifiers, (
        "the built-in verifier was deleted by a roster entry; the session now "
        "has no verifier at all and every turn goes unreviewed"
    )
    assert state.enabled_verifiers(), "a session must always have a verifier"


def test_review_still_happens_after_a_disabled_default_entry(ctx, monkeypatch):  # noqa: F811
    """End to end: the turn must still be reviewed."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    _roster({"name": "Default", "slug": "default", "enabled": False})
    ctx.llm.next = FakeResult({"severity": "caution", "finding": "something is wrong"})

    _turn(ctx, history=[{"role": "user", "content": "do the thing"}])

    state = runtime._get_state("s1")
    assert not state.last_error, f"the hook failed silently: {state.last_error!r}"
    assert state.verifiers["default"].pending, (
        "no review ran at all -- the built-in verifier had been deleted"
    )


def test_a_normal_roster_entry_still_works(ctx):  # noqa: F811
    """The guard must not stop ordinary roster management."""
    state = runtime._get_state("s1")
    _roster({"name": "Security", "slug": "security", "enabled": True})
    state.ensure_verifiers(runtime.discover_roster())
    assert "security" in state.verifiers
    assert "default" in state.verifiers

    # And a disabled custom entry is still removed.
    _roster({"name": "Security", "slug": "security", "enabled": False})
    state.ensure_verifiers(runtime.discover_roster())
    assert "security" not in state.verifiers
    assert "default" in state.verifiers
