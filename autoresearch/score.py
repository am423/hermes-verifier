"""Score candidate rules with the guards a previous pass lacked.

Three rules, all mandatory:
  * every lift gets a bootstrap CI, so a point estimate cannot be read as signal
  * the multiplicity bar is Bonferroni over the number of candidates tested --
    testing N rules at 95% each gives 1 - 0.95^N odds of a spurious winner
  * the result carries a power statement (sessions, turns, positive labels),
    because a null under a strict bar is NOT confirmation when power is low
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Tuple

from .rules import lift_on

DEFAULT_BOOTSTRAPS = 4000
DEFAULT_SEED = 20261010
ALPHA = 0.05


def lift_ci(turns: List[Dict[str, Any]], predicate: Callable[[Dict[str, Any]], bool],
            bootstraps: int = DEFAULT_BOOTSTRAPS, seed: int = DEFAULT_SEED
            ) -> Tuple[float, float]:
    """95% bootstrap CI for one rule's lift."""
    rng = random.Random(seed)
    n = len(turns)
    mask = [predicate(t) for t in turns]
    vals = []
    for _ in range(bootstraps):
        pick = [rng.randrange(n) for _ in range(n)]
        vals.append(lift_on([turns[i] for i in pick], [mask[i] for i in pick]))
    vals.sort()
    return vals[int(len(vals) * 0.025)], vals[int(len(vals) * 0.975) - 1]


def power_of(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "turns": len(turns),
        "sessions": len({t.get("session_id") for t in turns}),
        "positive_labels": sum(1 for t in turns if t.get("repair_next")),
    }


def compare_rules(turns: List[Dict[str, Any]],
                  candidates: Dict[str, Callable],
                  *, reference: str,
                  bootstraps: int = DEFAULT_BOOTSTRAPS,
                  seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Paired bootstrap of every candidate against `reference`.

    Paired: both rules are scored on the SAME resampled turns, so shared noise
    cancels and the interval is on the actual difference rather than on two
    independently noisy estimates.
    """
    rng = random.Random(seed)
    n = len(turns)
    masks = {k: [f(t) for t in turns] for k, f in candidates.items()}
    point = {k: lift_on(turns, m) for k, m in masks.items()}

    draws: Dict[str, List[float]] = {k: [] for k in candidates}
    for _ in range(bootstraps):
        pick = [rng.randrange(n) for _ in range(n)]
        sub = [turns[i] for i in pick]
        for k, m in masks.items():
            draws[k].append(lift_on(sub, [m[i] for i in pick]))

    n_tests = max(1, len(candidates) - 1)
    lo_q = ALPHA / n_tests / 2
    hi_q = 1 - lo_q
    results = {}
    for k in candidates:
        if k == reference:
            continue
        d = sorted(a - b for a, b in zip(draws[k], draws[reference]))
        lo = d[int(len(d) * lo_q)]
        hi = d[int(len(d) * hi_q) - 1]
        results[k] = {"gap": point[k] - point[reference], "ci": (lo, hi),
                      "clears_bar": lo > 0}
    return {
        "reference": reference,
        "reference_lift": point[reference],
        "point": point,
        "results": results,
        "multiplicity": {"tests": n_tests, "alpha_each": ALPHA / n_tests},
        "power": power_of(turns),
    }
