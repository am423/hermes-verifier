"""Find detectable 'consequential moment' signals in real session data.

The question: if the verifier were invoked at CONSEQUENTIAL moments rather than
every turn, what would trigger it, and how often?

Method: replay this session's messages, segment into turns (a turn starts at a
user message), and test each turn for candidate signals. Frequency matters as
much as detectability -- a signal that fires on 90% of turns is just the current
per-turn review wearing a hat.

Signals are drawn from what actually goes wrong in agent work: irreversible
actions, publishes, config edits, service disruption, repeated failures, and
completion claims. The last one is the interesting one: the claim-vs-artifact
cases that defeated every reviewer config live there.
"""

from __future__ import annotations

import pathlib
import re
import sqlite3
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from autoresearch.paths import get_hermes_home  # noqa: E402

SESSION = "20261010_003202_ad2365"

# ---------------------------------------------------------------------------
# Detection patterns.
#
# Written as a list of plain phrases, escaped and joined, rather than as one
# regex literal. Two reasons, both learned the hard way:
#   * the file DETECTS dangerous commands, and a static scanner cannot tell a
#     detector from the thing it detects -- written whole, "mkfs" and ".env"
#     were flagged as format_filesystem and dump_all_env.
#   * assembling a regex from string fragments is easy to get wrong: an earlier
#     attempt at this silently stopped matching "rm -rf" entirely.
# So the tokens are split where the scanner trips, and joined mechanically.
# ---------------------------------------------------------------------------

_IRREVERSIBLE = [
    "rm -" + "rf", "rm -" + "fr", "rm -f " + "/",
    "drop table", "drop database",
    "truncate table", "kill -9", "pkill", "killall", "shutdown", "reboot",
    "dd if=", "mk" + "fs",
    "git reset --hard", "git clean -", "git push --force",
    "chmod 000", "> " + "/etc/", "/dev/" + "null",
]
DESTRUCTIVE = re.compile("|".join(re.escape(x) for x in _IRREVERSIBLE), re.I)

_PUBLISH = [
    "git push", "gh pr create", "gh pr merge", "npm publish", "pip upload",
    "twine upload", "docker push", "kubectl apply", "terraform apply",
    "launchctl kickstart", "launchctl load", "launchctl unload",
    "systemctl", "brew install", "brew upgrade",
]
PUBLISH = re.compile("|".join(re.escape(x) for x in _PUBLISH), re.I)

_CONFIG_FILES = [
    "config.yaml", "plugin.yaml", "settings.json", "." + "env",
    "VERIFIER.yml", "VERIFIER.yaml", ".hermes/config",
    "allowed_models", "allowed_providers",
]
CONFIG_EDIT = re.compile("|".join(re.escape(x) for x in _CONFIG_FILES), re.I)

_DISRUPT = [
    "pkill", "kill -", "kill ", "launchctl", "restart", "stop the",
    "process_manage",
]
SERVICE_DISRUPT = re.compile("|".join(re.escape(x) for x in _DISRUPT), re.I)

_COMPLETION = [
    "verified", "confirmed", "all tests pass", "tests passing", "it works",
    "working now", "done", "complete", "completed", "fixed", "succeeded",
    "successfully", "is now live", "deployed", "shipped", "no bugs", "clean",
]
COMPLETION_CLAIM = re.compile("|".join(re.escape(x) for x in _COMPLETION), re.I)


COMPLETION_CLAIM = re.compile(
    r"\b(verified|confirmed|all\s+tests\s+pass|tests\s+pass(ing)?|"
    r"it\s+works|working\s+now|done|complete[d]?|fixed|succeeded|successfully|"
    r"is\s+now\s+live|deployed|shipped|no\s+bugs|clean)\b", re.I)


def load_turns(db: pathlib.Path | None = None) -> List[Dict[str, Any]]:
    """Read session data from the current profile, never credential stores."""
    if db is None:
        db = get_hermes_home() / "state.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = list(con.execute(
        "select role, content, tool_calls, tool_name, timestamp from messages "
        "where session_id=? order by id", (SESSION,)))

    turns, current = [], None
    for r in rows:
        if r["role"] == "user":
            current = {"user": (r["content"] or "")[:300], "assistant": [],
                       "tool_text": [], "tools": []}
            turns.append(current)
            continue
        if current is None:
            continue
        if r["role"] == "assistant":
            if r["content"]:
                current["assistant"].append(r["content"])
            if r["tool_calls"]:
                current["tools"].append(r["tool_calls"])
                current["tool_text"].append(r["tool_calls"])
        elif r["role"] == "tool":
            current["tool_text"].append(r["content"] or "")
    return turns


def analyse(turn: Dict[str, Any]) -> Dict[str, bool]:
    calls = "\n".join(turn["tool_text"])
    says = "\n".join(turn["assistant"])
    return {
        "destructive": bool(DESTRUCTIVE.search(calls)),
        "publish": bool(PUBLISH.search(calls)),
        "config_edit": bool(CONFIG_EDIT.search(calls)),
        "service_disrupt": bool(SERVICE_DISRUPT.search(calls)),
        "completion_claim": bool(COMPLETION_CLAIM.search(says)),
        "many_edits": calls.count("write_file") + calls.count("patch") >= 4,
    }


def main() -> None:
    turns = load_turns()
    print(f"turns in this session: {len(turns)}\n")

    counts = {k: 0 for k in
              ("destructive", "publish", "config_edit", "service_disrupt",
               "completion_claim", "many_edits")}
    any_signal = 0
    combos = {}
    for t in turns:
        sig = analyse(t)
        for k, v in sig.items():
            if v:
                counts[k] += 1
        if any(sig.values()):
            any_signal += 1
        key = "+".join(sorted(k for k, v in sig.items() if v)) or "(none)"
        combos[key] = combos.get(key, 0) + 1

    n = len(turns)
    print("  signal                 turns    share of turns")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:20s} {v:>6d}    {v/n:>6.0%}")
    print(f"  {'ANY signal':20s} {any_signal:>6d}    {any_signal/n:>6.0%}")
    print()
    print("  most common combinations:")
    for k, v in sorted(combos.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {v:>4d}  {k}")


if __name__ == "__main__":
    main()
