"""Every score must carry a CI, a power statement, and the multiplicity bar."""

from __future__ import annotations

from autoresearch.score import compare_rules, lift_ci


def _turns(n_repair, n_clean, fired_every=1):
    """fired_every=1 -> the rule fires on every turn; 2 -> every other."""
    out = []
    for i in range(n_repair):
        out.append({"session_id": "s1", "repair_next": True, "calls": "x",
                    "says": "", "edits": 0, "fire": (i % fired_every == 0)})
    for i in range(n_clean):
        out.append({"session_id": "s1", "repair_next": False, "calls": "x",
                    "says": "", "edits": 0, "fire": (i % fired_every == 0)})
    return out


def test_a_rule_firing_on_everything_has_lift_one():
    turns = _turns(40, 60)
    lo, hi = lift_ci(turns, lambda t: t["fire"], bootstraps=200, seed=1)
    assert lo <= 1.0 <= hi


def test_the_report_states_power():
    turns = _turns(40, 60)
    res = compare_rules(turns, {"always": lambda t: t["fire"]},
                        reference="always", bootstraps=200, seed=1)
    assert "power" in res
    assert res["power"]["turns"] == len(turns)
    assert res["power"]["positive_labels"] == 40
    assert res["power"]["sessions"] >= 1


def test_the_multiplicity_bar_widens_with_the_number_of_candidates():
    turns = _turns(40, 60)
    one = compare_rules(turns, {"always": lambda t: t["fire"],
                                "never": lambda t: False},
                        reference="always", bootstraps=100, seed=1)
    many = compare_rules(turns, {f"r{i}": (lambda t, i=i: t["fire"] and i % 2 == 0)
                                 for i in range(12)} | {"always": lambda t: t["fire"]},
                         reference="always", bootstraps=100, seed=1)
    assert many["multiplicity"]["alpha_each"] < one["multiplicity"]["alpha_each"], (
        "more candidates must mean a stricter bar, or the pass manufactures winners"
    )


def test_an_undefined_baseline_is_unscorable():
    """lift_on returned 0.0 for a rule that never fires, so a candidate was
    scored against a phantom zero and reported gap +4.0 with a CI of
    (3.42, 4.88). "Never fires" has no lift; it must not be a scoreable
    baseline."""
    turns = _turns(20, 20)
    res = compare_rules(turns, {"always": lambda t: False,
                                "never": lambda t: True},
                        reference="always", bootstraps=100, seed=1)
    assert res["results"]["never"]["clears_bar"] is False
    assert res["results"]["never"].get("unscorable") is True
