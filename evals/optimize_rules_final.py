"""Multiplicity correction, and a held-out check, before adopting anything.

The previous pass tested ~12 candidate rules against the shipped one at 95% each.
At that rate P(at least one false winner) is 1 - 0.95^12 ~= 46%, so a gap whose
interval barely excludes zero is exactly what resampling luck looks like. Two
independent guards, both pre-committed:

  1. BONFERRONI. Re-read the same bootstrap samples at 1 - 0.05/12 ~= 99.6%.
     Cheap, uses no new data, and simply makes the bar honest for having looked
     at twelve things.
  2. HELD-OUT SPLIT. Split sessions (not turns -- turns within a session share
     their author, their project and their habits, so a turn-level split leaks)
     into fit and validation halves. A winner must beat the shipped rule on the
     validation half too, or it was chosen by the fit half alone.

A rule is adopted only if it clears BOTH. Otherwise the shipped rule stands, and
that is the finding.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from optimize_rules import CANDIDATES, DB, load_turns
import sqlite3

CURRENT = "destructive_or_external"
BOOTSTRAPS = 4000
SEED = 20261010
ALPHA = 0.05


def lift_on(turns, mask) -> float:
    fired = [t for t, m in zip(turns, mask) if m]
    if not fired:
        return 0.0
    base = sum(1 for t in turns if t["repair_next"]) / max(1, len(turns))
    if not base:
        return 0.0
    return (sum(1 for t in fired if t["repair_next"]) / len(fired)) / base


def load_turns_grouped():
    """Turns WITH their session id, so a split can be made by session."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sessions = [r["session_id"] for r in con.execute(
        "select session_id from messages where role in ('assistant','tool') "
        "group by session_id having count(*) >= 20")]
    grouped = {}
    for s in sessions:
        rows = list(con.execute(
            "select role, content, tool_calls from messages where session_id=? order by id",
            (s,)))
        turns, cur = [], None
        for r in rows:
            if r["role"] == "user":
                cur = {"calls": [], "says": [], "edits": 0}
                turns.append(cur)
                continue
            if cur is None:
                cur = {"calls": [], "says": [], "edits": 0}
                turns.append(cur)
            if r["role"] == "assistant":
                if r["content"]:
                    cur["says"].append(r["content"])
                if r["tool_calls"]:
                    cur["calls"].append(r["tool_calls"])
                    cur["edits"] += 1
            elif r["role"] == "tool":
                if r["content"]:
                    cur["calls"].append(r["content"])
        if len(turns) < 6:
            continue
        from optimize_rules import REPAIR, FAILURE
        out = []
        for i, t in enumerate(turns):
            nxt = ""
            if i + 1 < len(turns):
                nxt = (" ".join(turns[i + 1]["says"]) + "\n"
                       + "\n".join(turns[i + 1]["calls"]))
            out.append({"calls": "\n".join(t["calls"]), "says": "\n".join(t["says"]),
                        "edits": t["edits"],
                        "repair_next": bool(REPAIR.search(nxt)) if nxt else False})
        grouped[s] = out
    return grouped


def main() -> None:
    rng = random.Random(SEED)
    turns = load_turns()
    n = len(turns)
    masks = {name: [fn(t["calls"], t["says"], t["edits"]) for t in turns]
             for name, fn in CANDIDATES.items()}
    point = {k: lift_on(turns, m) for k, m in masks.items()}
    n_tests = len(masks) - 1                       # candidates vs the shipped rule
    lo_q, hi_q = ALPHA / n_tests / 2, 1 - ALPHA / n_tests / 2

    idx = range(n)
    samples = {k: [] for k in masks}
    for _ in range(BOOTSTRAPS):
        pick = [rng.randrange(n) for _ in range(n)]
        sub = [turns[i] for i in pick]
        for k, m in masks.items():
            samples[k].append(lift_on(sub, [m[i] for i in pick]))

    print(f"turns: {n}  bootstraps: {BOOTSTRAPS}  candidates tested: {n_tests}")
    print(f"Bonferroni: each gap CI at {100 * (1 - ALPHA / n_tests):.2f}% "
          f"(alpha {ALPHA}/{n_tests} = {ALPHA / n_tests:.4f})\n")
    print(f"  {'rule':30s} {'gap':>7s}  {'Bonferroni CI':>18s}   survives?")
    survivors = []
    for name in sorted(masks, key=lambda k: -(point[k] - point[CURRENT])):
        if name == CURRENT:
            continue
        d = sorted(a - b for a, b in zip(samples[name], samples[CURRENT]))
        lo = d[int(len(d) * lo_q)]
        hi = d[int(len(d) * hi_q) - 1]
        ok = lo > 0
        if ok:
            survivors.append(name)
        print(f"  {name:30s} {point[name] - point[CURRENT]:>+7.2f}  "
              f"[{lo:>+6.2f}, {hi:>+6.2f}]   {'YES' if ok else 'no'}")

    print(f"\n  survivors of the Bonferroni bar: {survivors or 'NONE'}")

    # ---- held-out: split by SESSION so turns from one author do not leak ----
    grouped = load_turns_grouped()
    ids = sorted(grouped)
    rng.shuffle(ids)
    half = len(ids) // 2
    fit = [t for s in ids[:half] for t in grouped[s]]
    val = [t for s in ids[half:] for t in grouped[s]]
    print(f"\n  held-out check: {len(ids)} sessions -> fit {len(fit)} turns, "
          f"val {len(val)} turns")
    print(f"  {'rule':30s} {'fit gap':>9s} {'val gap':>9s}   confirmed on val?")
    for name in (survivors or []):
        fm = [CANDIDATES[name](t['calls'], t['says'], t['edits']) for t in fit]
        cm = [CANDIDATES[CURRENT](t['calls'], t['says'], t['edits']) for t in fit]
        vm = [CANDIDATES[name](t['calls'], t['says'], t['edits']) for t in val]
        cv = [CANDIDATES[CURRENT](t['calls'], t['says'], t['edits']) for t in val]
        fg = lift_on(fit, fm) - lift_on(fit, cm)
        vg = lift_on(val, vm) - lift_on(val, cv)
        print(f"  {name:30s} {fg:>+9.2f} {vg:>+9.2f}   "
              f"{'YES' if vg > 0 else 'NO -- did not hold'}")
    if not survivors:
        print("  (nothing to validate)")

    print("\n  ADOPT only on: Bonferroni survives AND holds on the validation half.")
    print("  Otherwise the shipped rule stands -- and a null result is the finding.")


if __name__ == "__main__":
    main()
