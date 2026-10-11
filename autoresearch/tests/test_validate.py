"""Adoption requires BOTH the multiplicity bar and a held-out confirmation."""

from __future__ import annotations

from autoresearch.validate import split_by_session, validate


def _corpus():
    turns = []
    for s in range(20):
        for i in range(10):
            turns.append({
                "session_id": f"s{s}", "turn": i,
                "calls": "rm -rf x" if (s % 2 == 0 and i % 2 == 0) else "ls",
                "says": "", "edits": 1,
                "repair_next": bool(s % 2 == 0 and i % 2 == 0),
                "fail_next": False,
            })
    return turns


def test_the_split_is_by_session_not_by_turn():
    fit, val = split_by_session(_corpus(), seed=1)
    assert not ({t["session_id"] for t in fit} & {t["session_id"] for t in val}), (
        "turns from one session leaked across the split"
    )
    assert fit and val


def test_validate_reports_the_bar_and_the_held_out_gap():
    out = validate(_corpus(), {"always": lambda t: True,
                               "rm": lambda t: "rm -rf" in t["calls"]},
                   reference="always", bootstraps=200, seed=1)
    assert "verdict" in out
    assert "val_power" in out and "fit_power" in out
    assert isinstance(out["survivors"], list)


def test_a_null_verdict_says_so_plainly():
    out = validate(_corpus(), {"always": lambda t: True,
                               "never": lambda t: False},
                   reference="always", bootstraps=200, seed=1)
    assert out["verdict"] == "no candidate cleared the bar"
