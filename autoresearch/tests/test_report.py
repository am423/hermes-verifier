"""The report must state its own limits, every run, without being asked."""

from __future__ import annotations

from autoresearch.report import render


def test_the_caveat_is_always_present():
    text = render({"verdict": "no candidate cleared the bar",
                   "power": {"turns": 10, "sessions": 2, "positive_labels": 4}})
    assert "repair FOLLOWED" in text
    assert "not that a review would have prevented it" in text


def test_a_low_power_run_says_so():
    text = render({"verdict": "no candidate cleared the bar",
                   "power": {"turns": 40, "sessions": 3, "positive_labels": 5}})
    assert "UNDERPOWERED" in text, (
        "a null with few positive labels is not evidence, and the report must say so"
    )


def test_an_adequately_powered_run_does_not_cry_underpowered():
    text = render({"verdict": "no candidate cleared the bar",
                   "power": {"turns": 900, "sessions": 30, "positive_labels": 350}})
    assert "UNDERPOWERED" not in text
    assert "no candidate cleared the bar" in text


def test_the_verdict_is_stated_even_when_null():
    text = render({"verdict": "no candidate cleared the bar", "power": {}})
    assert "no candidate cleared the bar" in text
