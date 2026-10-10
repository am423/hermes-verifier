"""Concurrent roster sync must not corrupt the verifier dict for readers.

`_sync_roster` runs on every turn and calls `ensure_verifiers`, which INSERTS
and DELETES keys in `state.verifiers`. Readers -- `snapshot()` (dashboard), the
review thread, and the gate -- iterate that same dict. Mutating it unlocked
while a reader iterates raises::

    RuntimeError: dictionary changed size during iteration

which killed reviews mid-flight and returned 500 from the dashboard's /state.
The writer now takes `_LOCK`, and the readers snapshot the list under it before
iterating, because `_review_one` does I/O per verifier and holding the lock
across that loop would serialise every session behind one review.
"""

from __future__ import annotations

import threading

from verifier_pkg import runtime

from .test_runtime import ctx  # noqa: F401

_ROSTER = [
    {"slug": "alpha", "name": "Alpha", "enabled": True},
    {"slug": "beta", "name": "Beta", "enabled": True},
    {"slug": "gamma", "name": "Gamma", "enabled": True},
]


def _hammer(seconds: float = 0.6):
    """Run the writer and both readers together; collect any explosion."""
    state = runtime._get_state("s1")
    errors: list = []
    stop = threading.Event()
    original = runtime.discover_roster

    def writer():
        i = 0
        while not stop.is_set():
            roster = _ROSTER[: (1 + i % len(_ROSTER))]
            runtime.discover_roster = lambda r=roster: r
            try:
                runtime._sync_roster(state)
            except Exception as exc:  # pragma: no cover - that is the bug
                errors.append(("writer", type(exc).__name__, str(exc)))
            i += 1

    def reader_snapshot():
        while not stop.is_set():
            try:
                runtime.snapshot()
            except Exception as exc:  # pragma: no cover
                errors.append(("snapshot", type(exc).__name__, str(exc)))

    def reader_iterate():
        while not stop.is_set():
            try:
                with runtime._LOCK:
                    list(state.enabled_verifiers())
            except Exception as exc:  # pragma: no cover
                errors.append(("iterate", type(exc).__name__, str(exc)))

    threads = [threading.Thread(target=f, daemon=True)
               for f in (writer, reader_snapshot, reader_iterate)]
    for t in threads:
        t.start()
    threading.Event().wait(seconds)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    runtime.discover_roster = original
    return errors


def test_roster_sync_and_readers_do_not_race(ctx):  # noqa: F811
    errors = _hammer()
    assert not errors, (
        f"roster sync raced its readers: {errors[:3]} "
        "(RuntimeError: dictionary changed size during iteration)"
    )
