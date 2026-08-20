"""Emission guard unit tests."""

from __future__ import annotations

from verifier_pkg.guard import EmissionGuard, is_content_free, normalize


def test_normalize_collapses_everything():
    assert normalize("Stop.") == "stop"
    assert normalize("*Stop*") == "stop"
    assert normalize("  stop  ") == "stop"
    assert normalize("Use   the      queue!!!") == "use the queue"
    # NFKC fullwidth folding
    assert normalize("\uff2c\uff27\uff34\uff2d") == "lgtm"


def test_content_free_phrases():
    for phrase in ("stop", "done", "lgtm", "nothing to add", "no issue continue",
                   "complete", "no findings", "all good"):
        assert is_content_free(normalize(phrase)), phrase
    assert not is_content_free(normalize("the queue is being bypassed in src/jobs"))


def test_dedupe_same_severity():
    guard = EmissionGuard()
    guard.begin_cycle()
    assert guard.accept("caution", "Bypasses the durable queue") is True
    guard.begin_cycle()
    assert guard.accept("caution", "Bypasses the durable queue") is False
    assert guard.stats["deduped"] == 1


def test_escalation_allowed_but_not_downgrade():
    guard = EmissionGuard()
    guard.begin_cycle()
    assert guard.accept("note", "Check the retry backoff") is True
    # Escalation to a higher severity passes the dedupe.
    guard.begin_cycle()
    assert guard.accept("blocker", "Check the retry backoff") is True
    # Downgrade re-raise is dropped.
    guard.begin_cycle()
    assert guard.accept("note", "Check the retry backoff") is False


def test_phrase_filter_silent():
    guard = EmissionGuard()
    guard.begin_cycle()
    assert guard.accept("note", "lgtm") is False
    assert guard.stats["phrase_filtered"] == 1
    assert len(guard) == 0


def test_one_finding_per_cycle():
    guard = EmissionGuard()
    guard.begin_cycle()
    assert guard.accept("note", "First finding") is True
    assert guard.accept("note", "Second different finding") is False
    assert guard.stats["rate_limited"] == 1
    guard.begin_cycle()
    assert guard.accept("note", "Second different finding") is True


def test_normalization_makes_dupes_identical():
    guard = EmissionGuard()
    guard.begin_cycle()
    assert guard.accept("caution", "Stop. Right now") is True
    guard.begin_cycle()
    assert guard.accept("caution", "***stop right now***") is False


def test_fifo_history_bounded():
    guard = EmissionGuard(max_history=5)
    for i in range(10):
        guard.begin_cycle()
        assert guard.accept("note", f"finding number {i}") is True
    assert len(guard) == 5
    # Oldest entries evicted: re-raising one of the first five is accepted.
    guard.begin_cycle()
    assert guard.accept("note", "finding number 1") is True


def test_reset_clears_everything():
    guard = EmissionGuard()
    guard.begin_cycle()
    assert guard.accept("caution", "Something real") is True
    guard.reset()
    guard.begin_cycle()
    assert guard.accept("caution", "Something real") is True
