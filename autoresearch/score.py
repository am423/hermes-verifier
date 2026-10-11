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
from typing import Any, Callable, Dict, List, Optional, Tuple

from .rules import lift_on

DEFAULT_BOOTSTRAPS = 4000
DEFAULT_SEED = 20261010
ALPHA = 0.05

# Above this fraction of cluster draws with an undefined side, the interval is
# a statement about the draws that happened to fire, not about the corpus.
MAX_UNDEFINED_DRAW_FRACTION = 0.05


def lift_ci(turns: List[Dict[str, Any]], predicate: Callable[[Dict[str, Any]], bool],
            bootstraps: int = DEFAULT_BOOTSTRAPS, seed: int = DEFAULT_SEED
            ) -> Tuple[float, float]:
    """95% bootstrap CI for one rule's lift (turn-level; diagnostic only).

    Draws where the lift is undefined (the rule never fires in the resample,
    or the resample has no positive label) are dropped; the decision path is
    `compare_rules`, which counts them instead of dropping them.
    """
    rng = random.Random(seed)
    n = len(turns)
    mask = [predicate(t) for t in turns]
    vals = []
    for _ in range(bootstraps):
        pick = [rng.randrange(n) for _ in range(n)] if n else []
        v = lift_on([turns[i] for i in pick], [mask[i] for i in pick])
        if v is not None:
            vals.append(v)
    if len(vals) < 2:
        return (None, None)  # type: ignore[return-value]
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

    Paired: both rules are scored on the SAME resampled corpus per draw, so
    shared noise cancels and the interval is on the actual difference rather
    than on two independently noisy estimates.

    Resampling is by SESSION (cluster), not by turn. Turns inside a session
    share an author, a project and a set of habits, so row-level resampling
    treats correlated turns as independent evidence and understates the
    interval -- on a corpus where labels vary by session it admitted a false
    survivor whose turn-level CI low was +0.086 while the cluster interval
    rejected it at -0.217. Both intervals are reported (`ci` turn-level,
    `cluster_ci`); `clears_bar` requires the CLUSTER interval to exclude zero.

    A comparison is `unscorable` when either rule's lift is undefined (the
    rule never fires, no positive labels, no turns), when there is no cluster
    variation to resample (fewer than 2 sessions), or when too many cluster
    draws come back undefined. A phantom 0.0 once made a rule that never fired
    look like a scoreable baseline and reported a candidate's gap as +4.0.
    """
    rng = random.Random(seed)
    n = len(turns)
    ids = sorted({str(t.get("session_id")) for t in turns})
    by_session: Dict[str, List[int]] = {}
    for i, t in enumerate(turns):
        by_session.setdefault(str(t.get("session_id")), []).append(i)
    masks = {k: [f(t) for t in turns] for k, f in candidates.items()}
    point = {k: lift_on(turns, m) for k, m in masks.items()}
    positives = sum(1 for t in turns if t.get("repair_next"))
    degenerate = n == 0 or positives == 0 or positives == n

    draws: Dict[str, List[Optional[float]]] = {k: [] for k in candidates}
    cluster_draws: Dict[str, List[Optional[float]]] = {k: [] for k in candidates}
    for _ in range(bootstraps):
        pick = [rng.randrange(n) for _ in range(n)] if n else []
        sub = [turns[i] for i in pick]
        for k, m in masks.items():
            draws[k].append(lift_on(sub, [m[i] for i in pick]))
        chosen = [ids[rng.randrange(len(ids))] for _ in ids] if ids else []
        idx = [i for sid in chosen for i in by_session[sid]]
        csub = [turns[i] for i in idx]
        for k, m in masks.items():
            cluster_draws[k].append(lift_on(csub, [m[i] for i in idx]))

    n_tests = max(1, len(candidates) - 1)
    lo_q = ALPHA / n_tests / 2
    hi_q = 1 - lo_q
    results = {}
    for k in candidates:
        if k == reference:
            continue
        pair: List[float] = []
        for a, b in zip(draws[k], draws[reference]):
            if a is not None and b is not None:
                pair.append(a - b)
        pair.sort()
        ci = ((pair[int(len(pair) * lo_q)], pair[int(len(pair) * hi_q) - 1])
              if len(pair) >= 2 else (None, None))
        cpair: List[float] = []
        for a, b in zip(cluster_draws[k], cluster_draws[reference]):
            if a is not None and b is not None:
                cpair.append(a - b)
        cpair.sort()
        cluster_ci = ((cpair[int(len(cpair) * lo_q)],
                       cpair[int(len(cpair) * hi_q) - 1])
                      if len(cpair) >= 2 else (None, None))
        undefined = bootstraps - len(cpair)
        gap = (point[k] - point[reference]
               if point[k] is not None and point[reference] is not None else None)
        reasons = []
        if degenerate:
            reasons.append("degenerate base rate")
        if point[reference] is None:
            reasons.append("reference lift undefined")
        if point[k] is None:
            reasons.append("candidate lift undefined")
        if len(ids) < 2:
            reasons.append("fewer than 2 sessions: no cluster variation")
        if bootstraps and undefined > MAX_UNDEFINED_DRAW_FRACTION * bootstraps:
            reasons.append(f"{undefined}/{bootstraps} cluster draws undefined")
        res: Dict[str, Any] = {
            "gap": gap, "ci": ci, "cluster_ci": cluster_ci,
            "clears_bar": bool(cluster_ci[0] is not None and cluster_ci[0] > 0
                               and not reasons),
        }
        if reasons:
            res["unscorable"] = True
            res["reason"] = "; ".join(reasons)
        results[k] = res

    multiplicity: Dict[str, Any] = {
        "tests": n_tests, "alpha_each": ALPHA / n_tests,
        "bootstraps": bootstraps, "quantile_draws": int(bootstraps * lo_q),
    }
    if bootstraps * lo_q < 10:
        # The lower bound is then within a few order statistics of the sample
        # minimum -- stable only by luck. 20 draws below the quantile is the
        # floor worth trusting; below that, raise bootstraps.
        multiplicity["warning"] = (
            f"only {bootstraps * lo_q:.1f} draws below the lower quantile; "
            "raise bootstraps for a stable bar"
        )
    return {
        "reference": reference,
        "reference_lift": point[reference],
        "point": point,
        "results": results,
        # The resampling unit and how many of them there are: a cluster
        # interval is only as trustworthy as the number of clusters.
        "cluster": {"unit": "session", "sessions": len(ids),
                    "bootstraps": bootstraps},
        "multiplicity": multiplicity,
        "power": power_of(turns),
        "seed": seed,
    }
