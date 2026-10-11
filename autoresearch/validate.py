"""Split by SESSION, then require a candidate to survive the fit half AND hold
up on the validation half.

Splitting by turn leaks: turns inside one session share an author, a project and
a set of habits, so a turn-level split lets the fit half memorise the validation
half and every rule looks generalisable.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Tuple

from .score import compare_rules, power_of


def split_by_session(turns: List[Dict[str, Any]], seed: int = 20261010
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ids = sorted({t.get("session_id") for t in turns})
    random.Random(seed).shuffle(ids)
    half = len(ids) // 2
    fit_ids, val_ids = set(ids[:half]), set(ids[half:])
    fit = [t for t in turns if t.get("session_id") in fit_ids]
    val = [t for t in turns if t.get("session_id") in val_ids]
    return fit, val


def validate(turns: List[Dict[str, Any]], candidates: Dict[str, Callable], *,
             reference: str, bootstraps: int = 4000,
             seed: int = 20261010) -> Dict[str, Any]:
    fit, val = split_by_session(turns, seed=seed)
    fit_res = compare_rules(fit, candidates, reference=reference,
                            bootstraps=bootstraps, seed=seed)
    survivors = [k for k, r in fit_res["results"].items() if r["clears_bar"]]

    held_out: Dict[str, Any] = {}
    if survivors:
        val_res = compare_rules(val, candidates, reference=reference,
                                bootstraps=bootstraps, seed=seed)
        for k in survivors:
            held_out[k] = {
                "fit_gap": fit_res["results"][k]["gap"],
                "val_gap": val_res["results"][k]["gap"],
                "confirmed": val_res["results"][k]["gap"] > 0,
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
        "fit_power": power_of(fit),
        "val_power": power_of(val),
        "power": power_of(turns),
        "survivors": survivors,
        "held_out": held_out,
        "adoptable": adoptable,
        "multiplicity": fit_res["multiplicity"],
    }
