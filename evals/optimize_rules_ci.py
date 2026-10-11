"""Is any candidate rule actually BETTER, or is the ranking in-sample noise?

The previous pass scored 13 candidate triggers on 830 turns and reported lift.
That is selection on the evaluation set: with a proxy label and ~830 turns, a
1.48-vs-1.46 gap is not evidence of anything, and the "best" rule is whichever
one happened to win a noisy race.

This pass answers the only question that matters before changing a decision rule:

    is candidate X's lift reliably better than the CURRENT rule's?

via a paired bootstrap. Each resample scores BOTH rules on the SAME turns, so
the paired difference cancels the shared noise and gives a CI on the gap. A rule
is only worth adopting if that CI excludes 1 (i.e. no advantage).

Also reported: each rule's own lift CI, so a "winner" whose interval spans 1.0 is
visibly indistinguishable from no signal at all.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from optimize_rules import CANDIDATES, load_turns

CURRENT = "destructive_or_external"      # what the plugin ships today
BOOTSTRAPS = 2000
SEED = 20261010


def lift_on(turns: List[Dict[str, Any]], mask: List[bool]) -> float:
    """P(repair | fired) / P(repair) for one rule over these turns."""
    fired = [t for t, m in zip(turns, mask) if m]
    if not fired:
        return 0.0
    base = sum(1 for t in turns if t["repair_next"]) / max(1, len(turns))
    if base == 0:
        return 0.0
    return (sum(1 for t in fired if t["repair_next"]) / len(fired)) / base


def main() -> None:
    turns = load_turns()
    n = len(turns)
    rng = random.Random(SEED)

    masks = {name: [fn(t["calls"], t["says"], t["edits"]) for t in turns]
             for name, fn in CANDIDATES.items()}

    point = {name: lift_on(turns, m) for name, m in masks.items()}

    idx = list(range(n))
    samples: Dict[str, List[float]] = {name: [] for name in masks}
    for _ in range(BOOTSTRAPS):
        pick = [rng.randrange(n) for _ in range(n)]
        sub = [turns[i] for i in pick]
        for name, m in masks.items():
            samples[name].append(lift_on(sub, [m[i] for i in pick]))

    def ci(vals: List[float], lo: float = 2.5, hi: float = 97.5):
        s = sorted(vals)
        return s[int(len(s) * lo / 100)], s[int(len(s) * hi / 100)]

    print(f"turns: {n}   bootstraps: {BOOTSTRAPS}\n")
    print(f"  {'rule':30s} {'lift':>6s}  {'95% CI':>16s}   beats noise?")
    for name in sorted(masks, key=lambda k: -point[k]):
        lo, hi = ci(samples[name])
        verdict = "yes" if lo > 1.0 else "NO -- spans 1.0"
        print(f"  {name:30s} {point[name]:>6.2f}  [{lo:>5.2f}, {hi:>5.2f}]   {verdict}")

    print(f"\n  PAIRED vs the shipped rule ({CURRENT}, lift {point[CURRENT]:.2f}):")
    print(f"  {'rule':30s} {'lift gap':>9s}  {'95% CI of gap':>18s}   adopt?")
    for name in sorted(masks, key=lambda k: -(point[k] - point[CURRENT])):
        if name == CURRENT:
            continue
        diffs = [a - b for a, b in zip(samples[name], samples[CURRENT])]
        s = sorted(diffs)
        lo, hi = s[int(len(s) * 0.025)], s[int(len(s) * 0.975)]
        gap = point[name] - point[CURRENT]
        if lo > 0:
            verdict = "YES -- reliably better"
        elif hi < 0:
            verdict = "no -- reliably worse"
        else:
            verdict = "no -- gap spans zero"
        print(f"  {name:30s} {gap:>+9.2f}  [{lo:>+6.2f}, {hi:>+6.2f}]   {verdict}")

    print("\n  A rule is adopted only when the gap CI excludes zero. Everything")
    print("  else is a tie, and a tie is a reason to change nothing.")


if __name__ == "__main__":
    main()
