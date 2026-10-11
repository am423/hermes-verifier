"""The rule predicates and labels live in one importable place."""

from __future__ import annotations

from autoresearch.rules import CANDIDATES, FAILURE, REPAIR, lift_on


def test_labels_detect_their_language():
    assert REPAIR.search("I need to revert that change")
    assert not REPAIR.search("the tests pass")
    assert FAILURE.search("Traceback (most recent call last)")


def test_candidates_include_the_shipped_rule():
    assert "destructive_or_external" in CANDIDATES


def test_lift_is_one_when_a_rule_fires_on_everything():
    turns = [{"repair_next": True, "calls": "", "says": "", "edits": 0}] * 4
    mask = [True] * 4
    assert lift_on(turns, mask) == 1.0


# --- the outcome label must mean what it says --------------------------------
#
# Measured: with no word boundaries and a search across prose + tool args +
# tool results, only 87 of 346 positives (25%) came from the agent's own
# words. `regress` matched "regression" (running a regression suite);
# `recover` matched git-log output. The label was mostly measuring "the next
# turn ran tests".


def test_boundaries_prevent_midword_matches():
    assert not REPAIR.search("running the regression suite")   # 'regress' inside
    assert not REPAIR.search("point-in-time recovery is enabled")
    assert REPAIR.search("I need to revert that")


def test_the_real_phrases_still_match():
    for phrase in ("let me revert this", "that was wrong", "broke the build",
                   "roll back the change", "my mistake"):
        assert REPAIR.search(phrase), phrase
    assert FAILURE.search("Traceback (most recent call last)")


def test_phrases_with_nonword_edges_still_match():
    """Word boundaries must not break entries that end (or start) with a
    non-word character: `error:`, `> /etc/`, `git clean -`. A naive
    ``\\b(?:...|error:|...)\\b`` compiles but silently stops matching them,
    because there is no word boundary between `:` and a space."""
    assert FAILURE.search("error: file not found")
    assert not FAILURE.search("terror: the movie")      # not mid-word 'error:'
    assert CANDIDATES["irreversible"]("echo pwned > /etc/hosts", "", 0)
