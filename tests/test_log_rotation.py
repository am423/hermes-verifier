"""Telemetry files must stay bounded on disk and cheap to poll.

The JSONL rows carry up to 2000-char delta excerpts and are append-only, so in a
long-lived gateway the data directory grew for as long as the process lived.
Every read also loaded and split the WHOLE file, and the dashboard polls these
files every few seconds -- O(file) per poll, on files that only ever grew.
"""

from __future__ import annotations

import json

from verifier_pkg import runtime

from .test_runtime import ctx  # noqa: F401


def _rows(path, n, pad=200):
    for i in range(n):
        runtime._append_jsonl(path, {"ts": i, "i": i, "pad": "x" * pad})


def test_the_tail_read_returns_the_newest_records(tmp_path):
    path = tmp_path / "verify.s.default.jsonl"
    _rows(path, 50)
    out = runtime.read_jsonl_tail(path, limit=5)
    assert [r["i"] for r in out] == [45, 46, 47, 48, 49]


def test_the_tail_read_is_bounded_not_whole_file(tmp_path, monkeypatch):
    """Seeking must actually happen; reading everything defeats the point."""
    path = tmp_path / "verify.s.default.jsonl"
    _rows(path, 200)
    assert path.stat().st_size > 40_000

    read_bytes = {"n": 0}
    real_open = open

    class _Counting:
        def __init__(self, fh):
            self._fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self._fh.__exit__(*a)

        def seek(self, *a):
            return self._fh.seek(*a)

        def read(self, *a):
            data = self._fh.read(*a)
            read_bytes["n"] += len(data)
            return data

    monkeypatch.setattr("builtins.open",
                        lambda *a, **k: _Counting(real_open(*a, **k)))
    monkeypatch.setattr(runtime, "_TAIL_WINDOW_BYTES", 4096)

    out = runtime.read_jsonl_tail(path, limit=3)
    assert [r["i"] for r in out] == [197, 198, 199]
    assert read_bytes["n"] <= 8192, (
        f"read {read_bytes['n']} bytes for a 3-row tail; it is loading the file"
    )


def test_a_tiny_file_still_reads_correctly(tmp_path):
    path = tmp_path / "verify.s.default.jsonl"
    _rows(path, 2)
    out = runtime.read_jsonl_tail(path, limit=10)
    assert len(out) == 2


def test_a_missing_file_is_not_an_error(tmp_path):
    assert runtime.read_jsonl_tail(tmp_path / "nope.jsonl") == []


def test_the_log_rotates_instead_of_growing_without_bound(tmp_path, monkeypatch):
    path = tmp_path / "verify.s.default.jsonl"
    monkeypatch.setattr(runtime, "_MAX_LOG_BYTES", 4096)
    _rows(path, 60, pad=200)          # well past the ceiling
    rotated = tmp_path / "verify.s.default.jsonl.1"
    assert rotated.exists(), "the file grew past its ceiling without rotating"
    assert path.stat().st_size < 4096, "the live file must restart below the ceiling"


def test_rotation_keeps_only_one_generation(tmp_path, monkeypatch):
    """A growing pile of generations would defeat the point of rotating."""
    path = tmp_path / "verify.s.default.jsonl"
    monkeypatch.setattr(runtime, "_MAX_LOG_BYTES", 2048)
    _rows(path, 200, pad=200)
    extra = sorted(p.name for p in tmp_path.iterdir() if p.name.endswith((".1", ".2", ".3")))
    assert extra == ["verify.s.default.jsonl.1"], extra


def test_a_rotated_file_is_still_valid_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "verify.s.default.jsonl"
    monkeypatch.setattr(runtime, "_MAX_LOG_BYTES", 4096)
    _rows(path, 60, pad=200)
    rotated = tmp_path / "verify.s.default.jsonl.1"
    for line in rotated.read_text().splitlines():
        json.loads(line)          # raises if rotation split a row
