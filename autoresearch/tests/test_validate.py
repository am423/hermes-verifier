"""Adoption requires BOTH the multiplicity bar and a held-out confirmation."""

from __future__ import annotations

from autoresearch.validate import split_by_session, validate


def _corpus():
    """24 sessions x 20 turns: above the per-half minimums so the validation
    tests exercise the real path rather than the insufficient-data refusal."""
    turns = []
    for s in range(24):
        for i in range(20):
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


def test_the_split_balances_turn_mass_not_session_count():
    """A count-balanced split let one 500-turn session sit opposite four
    100-turn ones: 600/300, not two comparable corpora."""
    turns = []
    for s, size in enumerate([500, 100, 100, 100, 100]):
        for i in range(size):
            turns.append({
                "session_id": f"s{s}", "turn": i + 1, "calls": "ls", "says": "",
                "edits": 1, "repair_next": (i % 2 == 0), "fail_next": False,
            })
    fit, val = split_by_session(turns, seed=1)
    assert abs(len(fit) - len(val)) <= 0.15 * len(turns), (
        f"turn mass unbalanced: fit {len(fit)} vs val {len(val)}"
    )


def test_thin_halves_are_insufficient_data_not_a_null():
    """A null on thin data is not evidence. Below the per-half minimums the
    verdict must say the data cannot decide."""
    turns = []
    for s in range(20):
        for i in range(10):
            turns.append({
                "session_id": f"s{s}", "turn": i + 1, "calls": "ls", "says": "",
                "edits": 1, "repair_next": bool(s % 2 == 0 and i % 2 == 0),
                "fail_next": False,
            })
    out = validate(turns, {"always": lambda t: True, "never": lambda t: False},
                   reference="always", bootstraps=100, seed=1)
    assert out["verdict"] == "insufficient data"
    assert out["survivors"] == []
    assert out["reason"], "the refusal must say what is missing"
