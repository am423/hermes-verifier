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
