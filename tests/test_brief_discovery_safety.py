"""Discovery must confine prompt sources before reading any of their bytes."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import types

import pytest

from verifier_pkg import brief


NAMES = ["VERIFIER.md", "VERIFIER.yml", "VERIFIER.yaml"]
CAP = 256 * 1024


@pytest.fixture()
def layout(tmp_path, monkeypatch):
    home = tmp_path / "home"
    hermes = home / ".hermes"
    repo = home / "repo"
    hermes.mkdir(parents=True)
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.chdir(repo)
    return types.SimpleNamespace(home=home, hermes=hermes, repo=repo, tmp=tmp_path)


@pytest.fixture(autouse=True)
def file_safety(monkeypatch):
    """Install the actual imported host API, not a lookalike on brief.

    The default explicitly ALLOWS fixture files, so confinement tests cannot
    pass because an unrelated host policy happened to reject their targets.
    """
    calls = []
    module = types.ModuleType("agent.file_safety")

    def allow(path):
        calls.append(path)
        return None

    module.get_read_block_error = allow
    monkeypatch.setitem(sys.modules, "agent.file_safety", module)
    monkeypatch.setattr(sys.modules["agent"], "file_safety", module, raising=False)
    return module, calls


def _payload(name, marker):
    if name == "VERIFIER.md":
        return marker + "\n"
    return (f"instructions: SHARED-{marker}\nverifiers:\n"
            f"  - name: Audit\n    instructions: INLINE-{marker}\n")


def _contents(layout, name, start=None):
    start = layout.repo if start is None else start
    if name == "VERIFIER.md":
        return brief.build_system_prompt(start, layout.hermes)
    return repr(brief.load_roster(start, layout.hermes))


def _put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("directory", [".", ".hermes"])
def test_project_symlink_cannot_put_outside_content_in_prompt(layout, name, directory):
    # Synthetic secret: it exists, and would be valid inline YAML as well as MD.
    outside = _put(layout.tmp / name, _payload(name, "OUTSIDE-CANARY"))
    candidate = layout.repo / directory / name
    candidate.parent.mkdir(exist_ok=True)
    candidate.symlink_to(outside)
    assert outside.is_file() and "OUTSIDE-CANARY" in outside.read_text()
    assert "OUTSIDE-CANARY" not in _contents(layout, name)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("directory", [".", ".hermes"])
def test_in_root_symlink_is_allowed(layout, file_safety, name, directory):
    target = _put(layout.repo / "rules" / name, _payload(name, "IN-ROOT-CANARY"))
    candidate = layout.repo / directory / name
    candidate.parent.mkdir(exist_ok=True)
    candidate.symlink_to(target)
    assert "IN-ROOT-CANARY" in _contents(layout, name)
    assert str(target.resolve()) in file_safety[1]


@pytest.mark.parametrize("name", NAMES)
def test_symlinked_project_parent_cannot_escape(layout, name):
    outside = layout.tmp / "outside"
    _put(outside / name, _payload(name, "PARENT-ESCAPE"))
    (layout.repo / ".hermes").symlink_to(outside, target_is_directory=True)
    assert "PARENT-ESCAPE" not in _contents(layout, name)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("start_in_hermes", [False, True])
def test_user_candidate_keeps_hermes_bound_inside_larger_repo(layout, name, start_in_hermes):
    (layout.home / ".git").mkdir()
    outside = _put(layout.home / "outside" / name, _payload(name, "USER-ESCAPE"))
    (layout.hermes / name).symlink_to(outside)
    start = layout.hermes / "work" if start_in_hermes else layout.repo
    start.mkdir(exist_ok=True)
    # Starting inside HERMES_HOME also encounters the user file as a project
    # candidate. That second discovery must not widen its bound to the outer git.
    assert "USER-ESCAPE" not in _contents(layout, name, start)


@pytest.mark.parametrize("name", NAMES)
def test_user_import_bound_is_hermes_not_enclosing_git(layout, name):
    (layout.home / ".git").mkdir()
    outside = _put(layout.home / "outside.txt", _payload(name, "IMPORT-ESCAPE"))
    inside = _put(layout.hermes / "inside.txt", _payload(name, "USER-LOCAL"))
    candidate = _put(layout.hermes / name, "@../outside.txt\n")
    assert outside.is_file()
    assert "IMPORT-ESCAPE" not in _contents(layout, name)
    candidate.write_text("@inside.txt\n")
    assert inside.is_file()
    assert "USER-LOCAL" in _contents(layout, name)


@pytest.mark.parametrize("name", NAMES)
def test_user_directory_alias_does_not_widen_discovery_bound(layout, name):
    (layout.home / ".git").mkdir()
    profile = layout.home / "profiles" / "current"
    profile.mkdir(parents=True)
    layout.hermes.rmdir()
    layout.hermes.symlink_to(profile, target_is_directory=True)
    outside = _put(layout.home / "outside.txt", _payload(name, "PROFILE-ALIAS-ESCAPE"))
    (profile / name).symlink_to(outside)
    start = layout.home / "work"
    start.mkdir()
    # Explicit user discovery sees the real profile; the project walk sees its
    # .hermes alias. Both must apply the same, narrower profile root.
    assert "PROFILE-ALIAS-ESCAPE" not in _contents(layout, name, start)


def test_import_root_comes_from_unresolved_entry_location(layout):
    outside = layout.tmp / "other-repo"
    (outside / ".git").mkdir(parents=True)
    target = _put(outside / "source.md", "unused synthetic content")
    secret = _put(outside / "canary.md", "LEXICAL-ROOT-ESCAPE")
    entry = layout.repo / "VERIFIER.md"
    entry.symlink_to(target)
    assert brief._import_root(entry, entry.parent) == layout.repo.resolve()
    out = brief.expand_imports(f"@{secret}\n", entry.parent, entry=entry)
    assert "LEXICAL-ROOT-ESCAPE" not in out


def test_in_root_entry_symlink_does_not_rebase_relative_imports(layout):
    nested = layout.repo / "nested"
    (nested / ".git").mkdir(parents=True)
    target = _put(nested / "source.md", "@RULES.md\n")
    _put(layout.repo / "RULES.md", "LEXICAL-IMPORT-BASE")
    (layout.repo / "VERIFIER.md").symlink_to(target)
    assert "LEXICAL-IMPORT-BASE" in brief.build_system_prompt(layout.repo, layout.hermes)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("outside_home", [False, True])
def test_no_repo_walk_is_bounded_by_actual_home_or_start(layout, monkeypatch, name, outside_home):
    monkeypatch.setattr(brief, "_find_repo_root", lambda start: None)
    layout.hermes = layout.tmp / "profiles" / "custom"
    layout.hermes.mkdir(parents=True)

    def unique_payload(marker):
        # Distinct slugs prevent a later, allowed roster entry from hiding an
        # earlier forbidden read via ordinary override behavior.
        return _payload(name, marker).replace("name: Audit", f"name: {marker}")

    if outside_home:
        start = layout.tmp / "workspace" / "leaf"
        forbidden = start.parent / name
    else:
        start = layout.home / "work" / "leaf"
        forbidden = layout.home.parent / name
        _put(layout.home / name, unique_payload("AT-ACTUAL-HOME"))
    start.mkdir(parents=True)
    _put(start / name, unique_payload("AT-START"))
    _put(forbidden, unique_payload("ABOVE-BOUND"))
    # Baseline discovery walks to /. Never let a failing test read any host
    # ancestor files: only our synthetic fixture files may actually be opened.
    read_candidate = brief._read_candidate

    def fixture_only(candidate, *args, **kwargs):
        if not candidate.is_relative_to(layout.tmp):
            return None
        return read_candidate(candidate, *args, **kwargs)

    monkeypatch.setattr(brief, "_read_candidate", fixture_only)
    contents = _contents(layout, name, start)
    assert "ABOVE-BOUND" not in contents
    assert "AT-START" in contents
    if not outside_home:
        assert "AT-ACTUAL-HOME" in contents


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("extra", [0, 1])
def test_candidate_byte_cap_is_256_kib(layout, name, extra):
    body = _payload(name, "SIZE-CANARY").encode("utf-8")
    body += b"\n#" + b"x" * (CAP + extra - len(body) - 2)
    candidate = layout.repo / name
    candidate.write_bytes(body)
    assert candidate.stat().st_size == CAP + extra
    assert ("SIZE-CANARY" in _contents(layout, name)) is (extra == 0)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("kind", ["directory", "fifo", "device"])
def test_nonregular_candidate_is_not_opened(layout, monkeypatch, name, kind):
    candidate = layout.repo / name
    if kind == "directory":
        candidate.mkdir()
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("no mkfifo on this platform")
        os.mkfifo(candidate)
    else:
        # No privileged mknod or real device read: model a character device's
        # metadata on an ordinary synthetic file, and forbid opening it below.
        candidate.write_text("synthetic device")
        real_stat = Path.stat

        def device_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if path == candidate:
                fields = list(result)
                fields[0] = stat.S_IFCHR | 0o600
                return os.stat_result(fields)
            return result

        monkeypatch.setattr(Path, "stat", device_stat)
    real_open, real_os_open = Path.open, os.open

    def no_path_open(path, *args, **kwargs):
        assert path != candidate, "attempted to open a non-regular candidate"
        return real_open(path, *args, **kwargs)

    def no_os_open(path, *args, **kwargs):
        assert Path(path) != candidate, "attempted to open a non-regular candidate"
        return real_os_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_path_open)
    monkeypatch.setattr(os, "open", no_os_open)
    assert _contents(layout, name) == (brief.SYSTEM_PROMPT if name == "VERIFIER.md" else "[]")


@pytest.mark.parametrize("name", NAMES)
def test_file_safety_denial_checks_resolved_target(layout, file_safety, name):
    module, calls = file_safety
    target = _put(layout.repo / "rules" / name, _payload(name, "POLICY-CANARY"))
    (layout.repo / name).symlink_to(target)
    assert "POLICY-CANARY" in _contents(layout, name)
    assert str(target.resolve()) in calls
    calls.clear()

    def deny(path):
        calls.append(path)
        return "fixture denied" if path == str(target.resolve()) else None

    module.get_read_block_error = deny
    assert "POLICY-CANARY" not in _contents(layout, name)
    assert str(target.resolve()) in calls


def test_file_safety_failure_does_not_allow_candidate(layout, file_safety):
    module, calls = file_safety
    candidate = _put(layout.repo / "VERIFIER.md", "FAILED-POLICY-CANARY")

    def fail(path):
        calls.append(path)
        raise RuntimeError("synthetic policy failure")

    module.get_read_block_error = fail
    assert "FAILED-POLICY-CANARY" not in brief.build_system_prompt(layout.repo, layout.hermes)
    assert str(candidate.resolve()) in calls


def test_import_uses_same_host_policy_and_keeps_refused_line_literal(layout, file_safety):
    module, calls = file_safety
    _put(layout.repo / "VERIFIER.md", "TOP-LEVEL-ALLOWED\n@rules/blocked.md\n")
    target = _put(layout.repo / "rules" / "blocked.md", "IMPORTED-POLICY-CANARY")
    assert "IMPORTED-POLICY-CANARY" in brief.build_system_prompt(layout.repo, layout.hermes)
    assert str(target.resolve()) in calls

    def deny(path):
        calls.append(path)
        return "fixture denied" if path == str(target.resolve()) else None

    module.get_read_block_error = deny
    calls.clear()
    out = brief.build_system_prompt(layout.repo, layout.hermes)
    assert "TOP-LEVEL-ALLOWED" in out
    assert "IMPORTED-POLICY-CANARY" not in out
    assert "@rules/blocked.md" in out
    assert str(target.resolve()) in calls


@pytest.mark.parametrize("replacement", ["fifo", "symlink", "regular"])
def test_replaced_candidate_is_refused_before_reading(layout, file_safety, monkeypatch, replacement):
    candidate = _put(layout.repo / "VERIFIER.md", "ORIGINAL-CANARY")
    target = _put(layout.tmp / "replacement.md", "REPLACEMENT-CANARY")
    if replacement == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("no mkfifo on this platform")
    changed = []

    def replace_after_stat(path):
        assert path == str(candidate.resolve())
        changed.append(path)
        if replacement == "regular":
            os.replace(target, candidate)
        else:
            candidate.unlink()
            if replacement == "fifo":
                os.mkfifo(candidate)
            else:
                candidate.symlink_to(target)
        return None

    file_safety[0].get_read_block_error = replace_after_stat
    real_open = os.open
    opened = []

    def checked_open(path, flags, *args, **kwargs):
        if replacement == "fifo":
            # Fail fast rather than letting a regressed blocking open hang CI.
            assert flags & os.O_NONBLOCK
        fd = real_open(path, flags, *args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", checked_open)
    assert brief.build_system_prompt(layout.repo, layout.hermes) == brief.SYSTEM_PROMPT
    assert changed == [str(candidate)]
    # A policy exception also refuses the source. Prove replacement really
    # completed so that this is not passing through that adjacent failure gate.
    if replacement == "fifo":
        assert stat.S_ISFIFO(candidate.stat().st_mode)
        assert opened
    elif replacement == "symlink":
        assert candidate.is_symlink() and candidate.resolve() == target
    else:
        assert candidate.read_text() == "REPLACEMENT-CANARY"
        assert opened
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)  # rejected descriptors were closed too


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("kind", ["broken", "loop"])
def test_unresolvable_candidate_is_skipped(layout, name, kind):
    candidate = layout.repo / name
    candidate.symlink_to(candidate if kind == "loop" else layout.repo / "missing")
    assert _contents(layout, name) == (brief.SYSTEM_PROMPT if name == "VERIFIER.md" else "[]")


def test_read_is_bounded_even_if_candidate_grows_after_stat(layout, monkeypatch):
    candidate = layout.repo / "VERIFIER.md"
    candidate.write_bytes(b"GREW-CANARY\n" + b"x" * CAP)
    real_stat, real_fstat = Path.stat, os.fstat
    inode = candidate.stat().st_ino

    def undersized(result):
        if result.st_ino == inode:
            fields = list(result)
            fields[6] = 1
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(Path, "stat", lambda path, *a, **kw: undersized(real_stat(path, *a, **kw)))
    monkeypatch.setattr(os, "fstat", lambda fd: undersized(real_fstat(fd)))
    reads = []

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, size=-1):
            assert 0 <= size <= CAP + 1, "candidate read was not byte-bounded"
            reads.append(size)
            return self.stream.read(size)

    real_open, real_fdopen = Path.open, os.fdopen
    monkeypatch.setattr(Path, "open", lambda path, *a, **kw: BoundedReader(real_open(path, *a, **kw)))
    monkeypatch.setattr(os, "fdopen", lambda fd, *a, **kw: BoundedReader(real_fdopen(fd, *a, **kw)))
    assert "GREW-CANARY" not in brief.build_system_prompt(layout.repo, layout.hermes)
    assert reads, "the cap must be checked on the bytes read, not just stat"


def test_nested_and_ancestor_imports_remain_inside_repo(layout):
    leaf = layout.repo / "src" / "component"
    leaf.mkdir(parents=True)
    _put(layout.repo / "VERIFIER.md", "ROOT-BRIEF\n@rules/first.md\n")
    _put(layout.repo / "rules" / "first.md", "FIRST-IMPORT\n@nested/second.md\n")
    _put(layout.repo / "rules" / "nested" / "second.md", "SECOND-IMPORT")
    _put(leaf / ".hermes" / "VERIFIER.md", "LEAF-BRIEF\n@../../../rules/first.md\n")
    out = brief.build_system_prompt(leaf, layout.hermes)
    for marker in ["ROOT-BRIEF", "FIRST-IMPORT", "SECOND-IMPORT", "LEAF-BRIEF"]:
        assert marker in out
    assert out.index("ROOT-BRIEF") < out.index("LEAF-BRIEF")


def test_no_repo_import_scope_is_not_widened_to_home(layout, monkeypatch):
    monkeypatch.setattr(brief, "_find_repo_root", lambda start: None)
    _put(layout.home / "outside.md", "OUTSIDE-ENTRY-DIRECTORY")
    _put(layout.repo / "VERIFIER.md", "LOCAL-BRIEF\n@../outside.md\n@empty.md\n")
    _put(layout.repo / "empty.md", "")
    out = brief.build_system_prompt(layout.repo, layout.hermes)
    assert "LOCAL-BRIEF" in out
    assert "OUTSIDE-ENTRY-DIRECTORY" not in out
    assert "@../outside.md" in out
    assert "@empty.md" not in out


def test_roster_override_order_and_imports_are_preserved(layout):
    leaf = layout.repo / "leaf"
    leaf.mkdir()
    _put(layout.hermes / "VERIFIER.yml", "verifiers:\n  - name: Audit\n    model: user\n  - name: User only\n")
    _put(layout.repo / "VERIFIER.yml", "verifiers:\n  - name: Audit\n    model: ancestor\n  - name: Root only\n")
    _put(leaf / "VERIFIER.yml", "verifiers:\n  - name: Audit\n    model: leaf-yml\n")
    _put(leaf / "VERIFIER.yaml", "verifiers:\n  - name: Audit\n    model: leaf-yaml\n")
    _put(leaf / ".hermes" / "VERIFIER.yaml", "@../../rules/roster.yml\n")
    _put(layout.repo / "rules" / "roster.yml", "@nested/roster.yml\n")
    _put(layout.repo / "rules" / "nested" / "roster.yml",
         "instructions: Shared rules\nverifiers:\n  - name: Audit\n    model: leaf-hermes\n    instructions: Inline rules\n")
    roster = brief.load_roster(leaf, layout.hermes)
    assert [e["slug"] for e in roster] == ["audit", "user-only", "root-only"]
    assert roster[0]["model"] == "leaf-hermes"
    assert roster[0]["shared_instructions"] == "Shared rules"
    assert roster[0]["instructions"] == "Inline rules"
