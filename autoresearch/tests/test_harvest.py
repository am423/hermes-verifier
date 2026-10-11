"""Harvest joins the plugin's decisions to what happened in the session."""

from __future__ import annotations

import json
import sqlite3

from autoresearch.harvest import harvest, read_decisions


def test_read_decisions_reads_every_kind(tmp_path):
    for kind, row in (("gate", {"engage": True}),
                      ("phase", {"verdict": "block"}),
                      ("agreement", {"agreed": False})):
        (tmp_path / f"{kind}.s1.jsonl").write_text(json.dumps(row) + "\n")
    out = read_decisions(tmp_path)
    assert {r["kind"] for r in out} == {"gate", "phase", "agreement"}


def test_read_decisions_skips_a_torn_line(tmp_path):
    p = tmp_path / "gate.s1.jsonl"
    p.write_text('{"engage": true}\n{"engage": tru\n')
    out = read_decisions(tmp_path)
    assert len(out) == 1, "a half-written row must be skipped, not raise"


def test_harvest_returns_turns_with_labels(tmp_path):
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.execute("create table messages (id integer primary key, session_id text,"
                " role text, content text, tool_calls text, tool_name text)")
    for _ in range(6):
        con.execute("insert into messages (session_id, role, content) values"
                    " ('sess1','user','do a thing')")
        for i in range(8):
            con.execute("insert into messages (session_id, role, content) values"
                        " ('sess1','assistant',?)", (f"step {i}",))
    con.execute("insert into messages (session_id, role, content) values"
                " ('sess1','assistant','I need to revert that')")
    con.commit()
    con.close()
    rows = harvest(tmp_path, db)
    assert rows, "no turns harvested"
    assert "repair_next" in rows[0]


def test_the_label_reads_the_next_turns_prose_only(tmp_path):
    """The label must mean "the agent said repair", not "a tool printed the
    word revert". Searching tool args and tool results made the label a proxy
    for "the next turn ran tests or printed repo text" -- git-log output
    matched `recover`, a regression suite matched `regress`."""
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.execute("create table messages (id integer primary key, session_id text,"
                " role text, content text, tool_calls text, tool_name text)")
    for _ in range(7):
        con.execute("insert into messages (session_id, role, content) values"
                    " ('s1','user','do a thing')")
        for j in range(8):
            con.execute("insert into messages (session_id, role, content) values"
                        " ('s1','assistant',?)", (f"step {j}",))
    # last turn: clean prose, but a tool RESULT carries repair language
    con.execute("insert into messages (session_id, role, content) values"
                " ('s1','assistant','everything looks fine')")
    con.execute("insert into messages (session_id, role, content) values"
                " ('s1','tool','revert 3 files -- recovery log')")
    con.commit()
    con.close()
    rows = harvest(tmp_path, db)
    sixth = next(r for r in rows if r["turn"] == 5)
    assert sixth["repair_next"] is False, "tool output leaked into the label"
    assert sixth["repair_next_any"] is True, (
        "the broad comparison label must keep the old, tools-included reading"
    )


def test_inactive_messages_are_excluded(tmp_path):
    """61% of rows in the segmented sessions are active=0: compacted away, or
    rewound by the user. The host's own reader hides them; the analysis scored
    them.

    The plan's sketch (user + 2 assistant rows per turn) leaves the session
    below the >= 20 assistant/tool rows a session needs to be segmented at
    all, so each turn here carries 4 kept assistant rows plus one compacted
    row that must never surface."""
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.execute("create table messages (id integer primary key, session_id text,"
                " role text, content text, tool_calls text, tool_name text,"
                " active integer)")
    for _ in range(8):
        con.execute("insert into messages (session_id, role, content, active)"
                    " values ('s1','user','do a thing',1)")
        for _ in range(4):
            con.execute("insert into messages (session_id, role, content, active)"
                        " values ('s1','assistant','kept',1)")
        con.execute("insert into messages (session_id, role, content, active)"
                    " values ('s1','assistant','COMPACTED-AWAY',0)")
    con.commit()
    con.close()
    rows = harvest(tmp_path, db)
    assert rows
    assert "COMPACTED-AWAY" not in " ".join(r["says"] for r in rows)
