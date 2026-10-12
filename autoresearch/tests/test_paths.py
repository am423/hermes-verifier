"""Profile-aware offline analysis; never inspect the developer's real home."""
from __future__ import annotations

import importlib
import sqlite3
import sys
import types

import pytest


def _corpus(home, session):
    home.mkdir(parents=True)
    db = home / "state.db"
    with sqlite3.connect(db) as con:
        con.execute("create table messages (id integer primary key, session_id text,"
                    " role text, content text, tool_calls text)")
        for _ in range(6):
            con.execute("insert into messages (session_id, role, content) values (?, 'user', 'task')",
                        (session,))
            for _ in range(4):
                con.execute("insert into messages (session_id, role, content) values"
                            " (?, 'assistant', 'fixture prose')", (session,))
    data = home / "plugin-data" / "hermes-verifier"
    data.mkdir(parents=True)
    (data / f"gate.{session}.jsonl").write_text('{"origin_turn": 1, "engage": true}\n')


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    # Force the standalone branch even on a machine with Hermes installed.
    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    connect = sqlite3.connect

    def isolated_connect(database, *args, **kwargs):
        assert str(database).removeprefix("file:").startswith(str(tmp_path) + "/"), (
            "analysis attempted to open a database outside its fixture"
        )
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", isolated_connect)
    return home


def test_harvest_and_load_turns_follow_environment_changes(isolated_home, tmp_path, monkeypatch):
    module = importlib.import_module("autoresearch.harvest")
    for name in ("first-profile", "second-profile"):
        profile = tmp_path / name
        _corpus(profile, name)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        turns = module.load_turns()
        assert len(turns) == 6
        assert {turn["session_id"] for turn in turns} == {name}
        rows = module.harvest()
        assert len(rows) == 6
        assert rows[0]["decisions"][0]["engage"] is True
        assert sum(len(row["decisions"]) for row in rows) == 1


def test_host_home_accessor_wins_and_is_resolved_lazily(isolated_home, tmp_path, monkeypatch):
    module = importlib.import_module("autoresearch.harvest")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "not-the-active-profile"))
    host = types.ModuleType("hermes_constants")
    monkeypatch.setitem(sys.modules, "hermes_constants", host)
    calls = []
    for name in ("host-first", "host-second"):
        profile = tmp_path / name
        _corpus(profile, name)

        def current_home():
            calls.append(profile)
            return profile

        monkeypatch.setattr(host, "get_hermes_home", current_home, raising=False)
        rows = module.harvest()
        assert len(rows) == 6
        assert {row["session_id"] for row in rows} == {name}
        assert rows[0]["decisions"][0]["engage"] is True
    assert calls == [tmp_path / "host-first", tmp_path / "host-second"]


def test_standalone_default_retains_the_default_home(isolated_home):
    module = importlib.import_module("autoresearch.harvest")
    _corpus(isolated_home / ".hermes", "default-home")
    assert {row["session_id"] for row in module.harvest()} == {"default-home"}


def test_explicit_paths_need_neither_host_nor_home_resolution(isolated_home, tmp_path, monkeypatch):
    module = importlib.import_module("autoresearch.harvest")
    profile = tmp_path / "explicit"
    _corpus(profile, "explicit")
    host = types.ModuleType("hermes_constants")

    def unavailable():
        raise AssertionError("an explicit database must not consult the host")

    monkeypatch.setattr(host, "get_hermes_home", unavailable, raising=False)
    monkeypatch.setitem(sys.modules, "hermes_constants", host)
    assert len(module.load_turns(profile / "state.db")) == 6
    rows = module.harvest(profile / "plugin-data" / "hermes-verifier", profile / "state.db")
    assert rows[0]["decisions"][0]["engage"] is True


def test_existing_host_accessor_failure_is_not_hidden(isolated_home, monkeypatch):
    module = importlib.import_module("autoresearch.harvest")
    host = types.ModuleType("hermes_constants")

    def broken():
        raise RuntimeError("fixture host failure")

    monkeypatch.setattr(host, "get_hermes_home", broken, raising=False)
    monkeypatch.setitem(sys.modules, "hermes_constants", host)
    with pytest.raises(RuntimeError, match="fixture host failure"):
        module.harvest()
