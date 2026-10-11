"""Collect the plugin's decision logs and the session corpus into one table.

Two sources, one shape:
  * decisions the plugin made (gate / phase / agreement jsonl)
  * what the agent actually did (state.db messages), segmented into turns

A "turn" starts at a user message and runs until the next one. The outcome label
(`repair_next`) is computed from the FOLLOWING turn's assistant PROSE only;
`repair_next_any` keeps the older, broad reading (prose + tool args + tool
results) so the two can be compared.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import Any, Dict, List

from .rules import FAILURE, REPAIR

DEFAULT_DATA = pathlib.Path.home() / ".hermes" / "plugin-data" / "hermes-verifier"
DEFAULT_DB = pathlib.Path.home() / ".hermes" / "state.db"


def read_decisions(data_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Every decision row, tagged with which log it came from.

    A torn final line (the process died mid-append) is skipped rather than
    raising: these files are appended to by a daemon thread, and the loop must
    not fall over on the last row.
    """
    out: List[Dict[str, Any]] = []
    for kind in ("gate", "phase", "agreement"):
        for path in sorted(data_dir.glob(f"{kind}.*.jsonl")):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    row["kind"] = kind
                    out.append(row)
    return out


def load_turns(db: pathlib.Path) -> List[Dict[str, Any]]:
    """Segment every substantial session into turns, with outcome labels."""
    if not db.exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    # active=0 rows were compacted away or rewound by the user; the host's own
    # reader hides them, and the analysis must too. Older DBs have no such
    # column, so guard before filtering rather than breaking the loop.
    cols = {r[1] for r in con.execute("PRAGMA table_info(messages)")}
    active = " and active=1" if "active" in cols else ""
    sessions = [r["session_id"] for r in con.execute(
        "select session_id from messages where role in ('assistant','tool')"
        + active + " group by session_id having count(*) >= 20")]
    out: List[Dict[str, Any]] = []
    for sid in sessions:
        rows = list(con.execute(
            "select role, content, tool_calls from messages "
            "where session_id=?" + active + " order by id", (sid,)))
        turns: List[Dict[str, Any]] = []
        cur: Dict[str, Any] = None  # type: ignore[assignment]
        for r in rows:
            if r["role"] == "user" or cur is None:
                cur = {"calls": [], "says": [], "edits": 0}
                turns.append(cur)
            if r["role"] == "assistant":
                if r["content"]:
                    cur["says"].append(r["content"])
                if r["tool_calls"]:
                    cur["calls"].append(r["tool_calls"])
                    cur["edits"] += 1
            elif r["role"] == "tool" and r["content"]:
                cur["calls"].append(r["content"])
        if len(turns) < 6:
            continue
        for i, t in enumerate(turns):
            nxt_prose = ""
            nxt_any = ""
            if i + 1 < len(turns):
                nxt_prose = " ".join(turns[i + 1]["says"])
                nxt_any = nxt_prose + "\n" + "\n".join(turns[i + 1]["calls"])
            out.append({
                "session_id": sid, "turn": i,
                "calls": "\n".join(t["calls"]), "says": "\n".join(t["says"]),
                "edits": t["edits"],
                # Prose only. Including tool args and results made the label a
                # proxy for "the next turn ran tests or printed repo text".
                "repair_next": bool(REPAIR.search(nxt_prose)) if nxt_prose else False,
                # The old, broad reading (prose + tool args + tool results),
                # kept so the two labels can be compared.
                "repair_next_any": bool(REPAIR.search(nxt_any)) if nxt_any else False,
                "fail_next": bool(FAILURE.search(nxt_any)) if nxt_any else False,
            })
    return out


def harvest(data_dir: pathlib.Path = DEFAULT_DATA,
            db: pathlib.Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    """Turns, with any decisions that belong to the same session attached."""
    turns = load_turns(db)
    decisions = read_decisions(data_dir)
    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for d in decisions:
        by_session.setdefault(str(d.get("session_id") or ""), []).append(d)
    for t in turns:
        t["decisions"] = by_session.get(t["session_id"], [])
    return turns
