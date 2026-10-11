"""Every score must carry a CI, a power statement, and the multiplicity bar."""

from __future__ import annotations

import random

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


def test_the_bootstrap_resamples_sessions_not_turns():
    """Turns inside a session share an author and a project, so row-level
    resampling understates uncertainty. On a corpus where labels vary BY SESSION
    and the rule fires concentrated in a few sessions, the turn-level interval
    admitted a false survivor that the cluster interval rejected."""
    turns = []
    for s in range(10):
        rate = 0.8 if s < 3 else 0.1          # labels vary by session
        hot = s < 3                            # rule fires in a few sessions
        for i in range(40):
            turns.append({"session_id": f"s{s}", "turn": i, "calls": "", "says": "",
                          "edits": 0, "repair_next": (i % 10) < rate * 10,
                          "fire": hot})
    res = compare_rules(turns, {"ref": lambda t: t["fire"],
                                "cand": lambda t: t["fire"]},
                        reference="ref", bootstraps=200, seed=1)
    assert "cluster" in res, "the result must expose a cluster-level interval"


def _session_rate_corpus(seed, n_sessions=12, turns_per=30):
    """Labels that vary BY SESSION: each session draws its own label rate, so
    turns inside a session are correlated, and each rule fires on whole
    sessions -- the confound the cluster bootstrap exists for."""
    rng = random.Random(seed)
    rates = [rng.choice([0.05, 0.35, 0.85]) for _ in range(n_sessions)]
    fire_a = rng.sample(range(n_sessions), 5)
    fire_b = rng.sample(range(n_sessions), 5)
    turns = []
    for s in range(n_sessions):
        for i in range(turns_per):
            turns.append({
                "session_id": f"s{s}", "turn": i + 1, "calls": "", "says": "",
                "edits": 0,
                "repair_next": rng.random() < rates[s],
                "fire_a": s in fire_a, "fire_b": s in fire_b,
            })
    return turns


def test_the_turn_level_interval_admits_a_survivor_the_cluster_interval_rejects():
    """The demonstrated case: on a session-correlated corpus the turn-level
    interval's low end is above zero (+0.076) while the session-cluster
    interval -- the honest one, because turns share a session -- still spans
    zero (low -0.732). clears_bar must follow the cluster interval, and the
    rejection must come from the interval, not from a guard."""
    turns = _session_rate_corpus(seed=1)
    res = compare_rules(turns, {"ref": lambda t: t["fire_a"],
                                "cand": lambda t: t["fire_b"]},
                        reference="ref", bootstraps=200, seed=1)
    r = res["results"]["cand"]
    assert r["ci"][0] > 0, f"turn-level interval was {r['ci']}"
    assert r["cluster_ci"][0] < 0, f"cluster interval was {r['cluster_ci']}"
    assert r["clears_bar"] is False
    assert not r.get("unscorable"), (
        f"the rejection must be the interval's, not a guard's: {r.get('reason')}"
    )
