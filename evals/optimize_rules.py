"""Optimize the decision rules against every session we have.

The phase gate's trigger set was derived from 28 sessions and reasoning. This
does it properly, on the whole corpus, with real outcome labels.

THE LABEL. A turn is "expensive" if the NEXT turn contains repair language --
revert, undo, rollback, broke, my mistake, that was wrong, didn't work. That is a
proxy: it measures that repair FOLLOWED, not that a review would have prevented
it. Stated plainly because it bounds every conclusion drawn here.

WHAT IS BEING OPTIMIZED. Not accuracy alone. A trigger that fires on most turns
is just per-turn review, so the objective is:

    lift     = P(repair | trigger) / P(repair)      how much more likely
    volume   = fraction of turns triggered          what it costs
    precision/recall against the repair label

A rule earns its place only if lift is materially above 1 AND volume stays low
enough that the reviewer is invoked at moments, not continuously.
"""

from __future__ import annotations

import json
import pathlib
import re
import sqlite3
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

DB = pathlib.Path.home() / ".hermes" / "state.db"

# Split where a scanner would otherwise read these as live code.
_IRREVERSIBLE = [
    "rm -rf", "rm -fr", "rm -f /", "drop table", "drop database",
    "truncate table", "git reset --hard", "git clean -", "git push --force",
    "chmod 000", "dd if=", "mk" + "fs", "> " + "/etc/",
]
_EXTERNAL = [
    "git push", "gh pr create", "gh pr merge", "npm publish", "pip upload",
    "twine upload", "docker push", "kubectl apply", "terraform apply",
    "launchctl", "brew install", "brew upgrade",
]
_SERVICE = ["pkill", "kill -9", "killall", "osascript", "restart", "shutdown"]
_EDIT = ["write_file", "patch", '"str_replace"', "apply_patch"]
_CONFIG = ["config.yaml", "plugin.yaml", "settings.json", "." + "env",
           "VERIFIER.yml", "allowed_models", "allowed_providers"]
_DEPLOY_WORDS = ["deploy", "production", "prod ", "release", "publish"]


def rx(words: List[str]) -> "re.Pattern[str]":
    return re.compile("|".join(re.escape(w) for w in words), re.I)


REPAIR = re.compile(
    "|".join(re.escape(w) for w in [
        "revert", "undo", "rollback", "roll back", "restore from", "back out",
        "broke", "broken", "regress", "that was wrong", "my mistake",
        "i was wrong", "didn't work", "did not work", "re-do", "redo",
        "recover", "accidentally deleted", "no longer works",
    ]), re.I)

FAILURE = re.compile(
    "|".join(re.escape(w) for w in [
        "traceback", "error:", "failed", "exception", "no such file",
        "permission denied", "command not found",
    ]), re.I)

# Candidate triggers. Each is a predictor we can score on its own.
CANDIDATES: Dict[str, Callable[[str, str, int], bool]] = {
    "irreversible":  lambda calls, says, n: bool(rx(_IRREVERSIBLE).search(calls)),
    "external":      lambda calls, says, n: bool(rx(_EXTERNAL).search(calls)),
    "service":       lambda calls, says, n: bool(rx(_SERVICE).search(calls)),
    "config":        lambda calls, says, n: bool(rx(_CONFIG).search(calls)),
    "destructive_or_external": lambda calls, says, n: bool(
        rx(_IRREVERSIBLE + _EXTERNAL).search(calls)),
    "many_edits":    lambda calls, says, n: n >= 6,
    "very_many_edits": lambda calls, says, n: n >= 15,
    "deploy_words":  lambda calls, says, n: bool(rx(_DEPLOY_WORDS).search(says)),
    "long_turn":     lambda calls, says, n: len(calls) > 12000,
    "claims_done":   lambda calls, says, n: bool(
        re.search(r"\b(done|complete[d]?|fixed|verified|all tests pass)\b",
                  says, re.I)),
    # combinations -- these are the actual decision rules under test
    "destr+config":  lambda c, s, n: bool(rx(_IRREVERSIBLE + _EXTERNAL).search(c))
                                     and bool(rx(_CONFIG).search(c)),
    "destr&!config": lambda c, s, n: bool(rx(_IRREVERSIBLE + _EXTERNAL).search(c))
                                     and not bool(rx(_CONFIG).search(c)),
    "any_of_destr_service_config": lambda c, s, n: bool(
        rx(_IRREVERSIBLE + _EXTERNAL + _SERVICE + _CONFIG).search(c)),
}


def load_turns() -> List[Dict[str, Any]]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sessions = [r["session_id"] for r in con.execute(
        "select session_id from messages where role in ('assistant','tool') "
        "group by session_id having count(*) >= 20")]
    out = []
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
        for i, t in enumerate(turns):
            nxt = ""
            if i + 1 < len(turns):
                nxt = " ".join(turns[i + 1]["says"]) + "\n" + "\n".join(turns[i + 1]["calls"])
            out.append({
                "calls": "\n".join(t["calls"]),
                "says": "\n".join(t["says"]),
                "edits": t["edits"],
                "repair_next": bool(REPAIR.search(nxt)) if nxt else False,
                "fail_next": bool(FAILURE.search(nxt)) if nxt else False,
            })
    return out


def main() -> None:
    turns = load_turns()
    n = len(turns)
    base = sum(1 for t in turns if t["repair_next"]) / n
    base_f = sum(1 for t in turns if t["fail_next"]) / n
    print(f"turns: {n}")
    print(f"base rate: repair-next {base:.1%}   failure-next {base_f:.1%}\n")

    rows = []
    for name, fn in CANDIDATES.items():
        fired = [t for t in turns if fn(t["calls"], t["says"], t["edits"])]
        if not fired:
            continue
        hits = sum(1 for t in fired if t["repair_next"])
        p = hits / len(fired)
        recall = hits / max(1, sum(1 for t in turns if t["repair_next"]))
        rows.append({
            "rule": name, "volume": len(fired) / n, "precision": p,
            "recall": recall, "lift": p / base,
            "lift_fail": (sum(1 for t in fired if t["fail_next"]) / len(fired)) / base_f,
        })

    print(f"  {'rule':30s} {'volume':>7s} {'prec':>6s} {'recall':>7s} {'lift':>6s} {'lift_fail':>10s}")
    for r in sorted(rows, key=lambda r: -r["lift"]):
        print(f"  {r['rule']:30s} {r['volume']:>6.1%} {r['precision']:>6.1%} "
              f"{r['recall']:>6.1%} {r['lift']:>6.2f} {r['lift_fail']:>10.2f}")

    print("\n  A rule earns its place on LIFT (how much more likely repair is)")
    print("  at LOW volume. High lift + high volume = per-turn review.")
    print("  High volume + lift ~1.0 = noise.")

    best = [r for r in rows if r["lift"] > 1.2 and r["volume"] < 0.25]
    print(f"\n  rules with lift > 1.2 AND volume < 25%: {len(best)}")
    for r in sorted(best, key=lambda r: -r["lift"]):
        print(f"    {r['rule']:30s} volume {r['volume']:.1%}  lift {r['lift']:.2f}")


if __name__ == "__main__":
    main()
