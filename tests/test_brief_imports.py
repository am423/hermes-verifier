"""@import expansion: no cycles, no blocking reads.

`expand_imports` promises "cycles are skipped". Without seeding `seen` with the
file being expanded, an A<->B cycle re-included A: B's `@A` did not find A in
`seen`, because only A had been walked from the top, so A was expanded a second
time underneath B.
"""

from __future__ import annotations

import os

import pytest

from verifier_pkg import brief


def _fence_off(path, body):
    path.write_text(body)
    return path


def test_a_mutual_cycle_does_not_duplicate_the_entry(tmp_path):
    a = _fence_off(tmp_path / "A.md", "A-HEAD\n@B.md\nA-TAIL\n")
    _fence_off(tmp_path / "B.md", "B-HEAD\n@A.md\nB-TAIL\n")

    out = brief.expand_imports(a.read_text(), tmp_path, entry=a)

    assert out.count("A-HEAD") == 1, f"A was included twice:\n{out}"
    assert out.count("B-HEAD") == 1
    assert "B-TAIL" in out and "A-TAIL" in out


def test_a_self_import_is_not_duplicated(tmp_path):
    a = _fence_off(tmp_path / "S.md", "S-HEAD\n@S.md\nS-TAIL\n")
    out = brief.expand_imports(a.read_text(), tmp_path, entry=a)
    assert out.count("S-HEAD") == 1


def test_a_three_way_cycle_terminates(tmp_path):
    _fence_off(tmp_path / "A.md", "A\n@B.md\n")
    _fence_off(tmp_path / "B.md", "B\n@C.md\n")
    _fence_off(tmp_path / "C.md", "C\n@A.md\n")
    out = brief.expand_imports((tmp_path / "A.md").read_text(), tmp_path, entry=tmp_path / "A.md")
    assert out.count("A") >= 1 and "B" in out and "C" in out


def test_a_normal_import_still_expands(tmp_path):
    _fence_off(tmp_path / "B.md", "IMPORTED-CONTENT")
    out = brief.expand_imports("before\n@B.md\nafter\n", tmp_path)
    assert "IMPORTED-CONTENT" in out


def test_a_missing_import_stays_literal(tmp_path):
    out = brief.expand_imports("x\n@nope.md\n", tmp_path)
    assert "@nope.md" in out


def test_an_oversized_import_is_refused_not_read(tmp_path, monkeypatch):
    big = tmp_path / "big.md"
    big.write_text("y" * 5000)
    monkeypatch.setattr(brief, "_MAX_IMPORT_BYTES", 1000)
    out = brief.expand_imports("@big.md\n", tmp_path)
    assert "@big.md" in out, "an oversized import must stay literal"
    assert "yyyy" not in out


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no mkfifo on this platform")
def test_a_fifo_is_refused_rather_than_blocking(tmp_path):
    """Reading a FIFO with no writer would block forever, stalling discovery and
    therefore the session."""
    fifo = tmp_path / "pipe.md"
    os.mkfifo(fifo)
    try:
        out = brief.expand_imports("@pipe.md\n", tmp_path)
    except Exception as exc:  # pragma: no cover - a hang is the failure mode
        raise AssertionError(f"a FIFO import raised instead of being refused: {exc}")
    assert "@pipe.md" in out
