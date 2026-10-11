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
