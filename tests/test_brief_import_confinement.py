"""A brief's @path imports must not reach outside the brief's own tree.

NousResearch/hermes-agent#135996: "`@path` imports in a brief can read any file.
`brief.py:165-176` follows `@path` from a project `VERIFIER.md` to any absolute
or `..` path, and the content goes to the verifier's provider. A cloned repo
could pull in `~/.ssh/id_rsa`."

Confirmed in the current tree before fixing: the `~` branch resolves via
`Path(raw).expanduser()`, so `@~/.ssh/id_rsa` was read and shipped verbatim.

The bound is the importing file's repo root (its own directory when it is not in
a repo) -- so the existing tests, which use a plain tmp_path, keep working.
"""

from __future__ import annotations

import pathlib

import pytest

from verifier_pkg import brief


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    return r


def test_a_tilde_path_is_refused(repo, tmp_path, monkeypatch):
    """The live path: `~` expands to whatever HOME is, outside any repo.

    HOME is redirected so the test is hermetic and actually exercises the
    expansion. Without the redirect this passed vacuously on machines that have
    no ~/.ssh/id_rsa -- the import stayed literal because the file was MISSING,
    not because anything confined it.
    """
    fake_home = tmp_path / "home"
    (fake_home / ".ssh").mkdir(parents=True)
    (fake_home / ".ssh" / "id_rsa").write_text("-----BEGIN PRIVATE KEY-----\nLEAKED\n")
    monkeypatch.setenv("HOME", str(fake_home))
    out = brief.expand_imports("@~/.ssh/id_rsa\n", repo, entry=repo / "V.md")
    assert "LEAKED" not in out, "a home-dir secret was read into the brief"
    assert "@~/.ssh/id_rsa" in out, "the @path must stay literal"


def test_an_escaping_relative_path_is_refused(repo):
    secret = repo.parent / "outside.txt"
    secret.write_text("TOP-SECRET-OUTSIDE")
    out = brief.expand_imports("@../outside.txt\n", repo, entry=repo / "V.md")
    assert "TOP-SECRET-OUTSIDE" not in out, "a .. escape was followed"
    assert "@../outside.txt" in out


def test_a_symlink_out_of_the_tree_is_refused(repo, tmp_path):
    """Resolved comparison: the link's own name looks contained."""
    secret = tmp_path / "target.txt"
    secret.write_text("VIA-SYMLINK")
    (repo / "link.md").symlink_to(secret)
    out = brief.expand_imports("@link.md\n", repo, entry=repo / "V.md")
    assert "VIA-SYMLINK" not in out, "a symlink escaped the root"


def test_an_absolute_path_is_refused(repo):
    out = brief.expand_imports("@/etc/hosts\n", repo, entry=repo / "V.md")
    assert "@/etc/hosts" in out
    assert "localhost" not in out


def test_an_in_tree_import_still_works(repo):
    (repo / "RULES.md").write_text("IN-REPO-CONTENT")
    out = brief.expand_imports("@RULES.md\n", repo, entry=repo / "V.md")
    assert "IN-REPO-CONTENT" in out, "a legitimate in-tree import must still expand"


def test_a_nested_import_cannot_escape_via_its_own_directory(repo, tmp_path):
    """The confinement must survive recursion.

    Each recursive call rebases `base_dir` on the file it is expanding, so a
    freshly-derived root per level would let a chain walk out: A imports
    sub/B, B imports ../../outside. The root has to be threaded, not recomputed.
    """
    secret = repo.parent / "outside.txt"
    secret.write_text("REACHED-BY-CHAIN")
    sub = repo / "sub"
    sub.mkdir()
    (sub / "B.md").write_text("B-BODY\n@../../outside.txt\n")
    (repo / "A.md").write_text("@sub/B.md\n")
    out = brief.expand_imports((repo / "A.md").read_text(), repo, entry=repo / "A.md")
    assert "B-BODY" in out, "the legitimate nested import should still expand"
    assert "REACHED-BY-CHAIN" not in out, "a nested import escaped the root"


def test_the_user_level_file_is_bounded_by_its_own_directory(tmp_path, monkeypatch):
    """HERMES_HOME for the user-level file: imports inside it are fine, ~/ out is not."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "SHARED.md").write_text("HERMES-SHARED")
    secret = pathlib.Path.home() / ".ssh" / "id_rsa"
    v = home / "VERIFIER.md"
    out = brief.expand_imports("@SHARED.md\n@~/.ssh/id_rsa\n", home, entry=v)
    assert "HERMES-SHARED" in out
    if secret.exists():  # pragma: no cover - depends on the machine
        assert "PRIVATE KEY" not in out
