"""Split by SESSION, then require a candidate to survive the fit half AND hold
up on the validation half.

Splitting by turn leaks: turns inside one session share an author, a project and
a set of habits, so a turn-level split lets the fit half memorise the validation
half and every rule looks generalisable.

The split balances TURN MASS, not session count: balancing counts let one
500-turn session sit opposite four 100-turn ones, so the halves were not
comparable corpora. Below the minimums (8 sessions, 150 turns and 50 positive
labels per half) there is not enough data to decide anything, and the verdict
says so rather than dressing a confident null on thin data.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .score import compare_rules, power_of

# Per-half floors. Under any of these the run cannot distinguish a real rule
# from resampling luck, whatever the intervals say.
MIN_SESSIONS_PER_HALF = 8
MIN_TURNS_PER_HALF = 150
MIN_POSITIVES_PER_HALF = 50


def split_by_session(turns: List[Dict[str, Any]], seed: int = 20261010
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Balance TURN MASS across the halves, not session count.

    Balancing counts let one 500-turn session sit opposite four 100-turn ones,
    so the halves were not comparable corpora. Sessions go largest-first into
    whichever half currently holds less turn mass, ties broken by a seeded
    shuffle.
    """
    counts: Dict[str, int] = {}
    for t in turns:
        sid = str(t.get("session_id"))
        counts[sid] = counts.get(sid, 0) + 1
    ids = sorted(counts)
    random.Random(seed).shuffle(ids)
    ids.sort(key=lambda sid: -counts[sid])      # stable: ties keep shuffle order
    fit_ids: set = set()
    val_ids: set = set()
    fit_mass = val_mass = 0
    for sid in ids:
        if fit_mass <= val_mass:
            fit_ids.add(sid)
            fit_mass += counts[sid]
        else:
            val_ids.add(sid)
            val_mass += counts[sid]
    fit = [t for t in turns if str(t.get("session_id")) in fit_ids]
    val = [t for t in turns if str(t.get("session_id")) in val_ids]
    return fit, val


def _insufficiency(power: Dict[str, Any]) -> Optional[str]:
    """Why this half cannot decide anything, or None when it can."""
    if power["sessions"] < MIN_SESSIONS_PER_HALF:
        return f"{power['sessions']} sessions (need {MIN_SESSIONS_PER_HALF})"
    if power["turns"] < MIN_TURNS_PER_HALF:
        return f"{power['turns']} turns (need {MIN_TURNS_PER_HALF})"
    if power["positive_labels"] < MIN_POSITIVES_PER_HALF:
        return (f"{power['positive_labels']} positive labels "
                f"(need {MIN_POSITIVES_PER_HALF})")
    return None


def validate(turns: List[Dict[str, Any]], candidates: Dict[str, Callable], *,
             reference: str, bootstraps: int = 4000,
             seed: int = 20261010) -> Dict[str, Any]:
    fit, val = split_by_session(turns, seed=seed)
    fit_power, val_power = power_of(fit), power_of(val)
    thin = {"fit": _insufficiency(fit_power), "val": _insufficiency(val_power)}
    if any(thin.values()):
        return {
            "verdict": "insufficient data",
            "reason": "; ".join(f"{half} half: {why}"
                                for half, why in thin.items() if why),
            "reference": reference,
            "fit_power": fit_power,
            "val_power": val_power,
            "power": power_of(turns),
            "survivors": [],
            "held_out": {},
            "adoptable": [],
            "multiplicity": None,
        }

    fit_res = compare_rules(fit, candidates, reference=reference,
                            bootstraps=bootstraps, seed=seed)
    survivors = [k for k, r in fit_res["results"].items() if r["clears_bar"]]

    held_out: Dict[str, Any] = {}
    if survivors:
        val_res = compare_rules(val, candidates, reference=reference,
                                bootstraps=bootstraps, seed=seed)
        for k in survivors:
            vr = val_res["results"][k]
            held_out[k] = {
                "fit_gap": fit_res["results"][k]["gap"],
                "val_gap": vr["gap"],
                "val_ci": vr["ci"],
                "val_cluster_ci": vr["cluster_ci"],
                "confirmed": bool(vr["gap"] is not None and vr["gap"] > 0),
            }

    adoptable = [k for k, h in held_out.items() if h["confirmed"]]
    if not survivors:
        verdict = "no candidate cleared the bar"
    elif adoptable:
        verdict = "adopted: " + ", ".join(adoptable)
    else:
        verdict = "candidates cleared the fit bar but did not hold on validation"

    return {
        "verdict": verdict,
        "reference": reference,
        "fit_power": fit_power,
        "val_power": val_power,
        "power": power_of(turns),
        "survivors": survivors,
        "held_out": held_out,
        "adoptable": adoptable,
        "multiplicity": fit_res["multiplicity"],
    }
