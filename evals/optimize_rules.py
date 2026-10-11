"""RAW SCORER -- the unguarded table, kept for quick looks.

This file is the RAW scorer: it prints point lifts for every candidate rule
over the whole corpus and applies NO multiplicity bar, NO bootstrap and NO
session split. Rules that topped this table were read as signal and were not
-- use `autoresearch/` (score/validate/report) for anything a decision rests
on. The banner is here so nobody repeats that reading.

The rule predicates and the outcome label live in ONE place
(`autoresearch.rules`, `autoresearch.harvest`). Three copies of this scoring
code existed -- this file, optimize_rules_ci.py and optimize_rules_final.py --
and they drifted; the two derived copies are deleted and everything here is
imported from the shared modules.

THE LABEL. `repair_next` is repair language in the FOLLOWING turn's assistant
prose (word-bounded; tool args and tool results are not read) -- measured, not
guessed; see sidecar/NEGATIVE-RESULTS.md section 6 for the re-baseline. It is
a proxy: it measures that repair FOLLOWED, not that a review would have
prevented it.

lift    = P(repair | trigger) / P(repair)      how much more likely
volume  = fraction of turns triggered          what it costs
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from autoresearch.harvest import load_turns  # noqa: E402
from autoresearch.rules import (  # noqa: E402
    CANDIDATES, FAILURE, REPAIR, lift_on,
)

# Re-exported on purpose: the shared definitions live in autoresearch/, and
# anything that still imports them from this module gets the current ones
# rather than a stale copy.
__all__ = ["CANDIDATES", "FAILURE", "REPAIR", "lift_on", "load_turns"]


def _fmt(v: Any) -> str:
    return f"{v:.2f}" if v is not None else "n/a"


def main() -> None:
    turns = load_turns()
    n = len(turns)
    if not n:
        print("no turns harvested")
        return
    base = sum(1 for t in turns if t["repair_next"]) / n
    base_f = sum(1 for t in turns if t["fail_next"]) / n
    print(f"turns: {n}")
    print(f"base rate: repair-next {base:.1%}   failure-next {base_f:.1%}\n")

    rows: List[Dict[str, Any]] = []
    for name, fn in CANDIDATES.items():
        mask = [fn(t["calls"], t["says"], t["edits"]) for t in turns]
        fired = sum(mask)
        if not fired:
            continue
        hits = sum(1 for t, m in zip(turns, mask) if m and t["repair_next"])
        p = hits / fired
        recall = hits / max(1, sum(1 for t in turns if t["repair_next"]))
        fail_hits = sum(1 for t, m in zip(turns, mask) if m and t["fail_next"])
        rows.append({
            "rule": name, "volume": fired / n, "precision": p, "recall": recall,
            "lift": lift_on(turns, mask),
            "lift_fail": (fail_hits / fired / base_f) if base_f else None,
        })

    print(f"  {'rule':30s} {'volume':>7s} {'prec':>6s} {'recall':>7s} "
          f"{'lift':>6s} {'lift_fail':>10s}")
    for r in sorted(rows, key=lambda r: (r["lift"] is None, -(r["lift"] or 0.0))):
        print(f"  {r['rule']:30s} {r['volume']:>6.1%} {r['precision']:>6.1%} "
              f"{r['recall']:>6.1%} {_fmt(r['lift']):>6s} {_fmt(r['lift_fail']):>10s}")

    print("\n  A rule earns its place on LIFT (how much more likely repair is)")
    print("  at LOW volume. High lift + high volume = per-turn review.")
    print("  High volume + lift ~1.0 = noise.")
    print("  NOTE: no bar is applied here -- the top row is not evidence.")

    best = [r for r in rows if (r["lift"] or 0.0) > 1.2 and r["volume"] < 0.25]
    print(f"\n  rules with lift > 1.2 AND volume < 25%: {len(best)}")
    for r in sorted(best, key=lambda r: -(r["lift"] or 0.0)):
        print(f"    {r['rule']:30s} volume {r['volume']:.1%}  lift {_fmt(r['lift'])}")


if __name__ == "__main__":
    main()
