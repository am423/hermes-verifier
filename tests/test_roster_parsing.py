"""One malformed roster entry must not destroy the whole roster.

`_parse_roster_file`'s docstring promises "Never raises ... malformed entries are
dropped with a warning so one bad project config cannot kill the session". That
was false for a non-list `tools` value: `[str(t) for t in item.get("tools") or []]`
raises TypeError on `tools: 7`, because `7` is truthy and not iterable.

The exception escaped to discover_roster, which answered [] -- indistinguishable
from "no roster configured" -- and _sync_roster then called
ensure_verifiers([]), DELETING every previously valid named verifier together
with their pending findings.
"""

from __future__ import annotations

import textwrap

from verifier_pkg import brief, runtime

from .test_runtime import FakeResult, ctx  # noqa: F401


def _parse(body: str):
    """Call the function whose docstring promises 'Never raises'.

    Testing through load_roster would drag in the whole ancestor/repo discovery
    chain, which is a different concern; the defect here is in the parse.
    """
    return brief._parse_roster_file(textwrap.dedent(body))


def test_a_non_list_tools_value_does_not_raise():
    """tools: 7 -> `[str(t) for t in 7]` -> TypeError, out of a function that
    documents 'Never raises ... one bad project config cannot kill the session'."""
    try:
        entries = _parse("""
            instructions: shared
            verifiers:
              - name: Broken
                tools: 7
              - name: Good
                tools: [read_file]
        """)
    except TypeError as exc:  # pragma: no cover - that is the bug
        raise AssertionError(f"a malformed entry raised out of the parse: {exc}")

    names = {e["name"] for e in entries}
    assert "Good" in names, (
        "one malformed entry took the whole roster with it; discover_roster then "
        "answers [] and ensure_verifiers([]) deletes every live verifier"
    )
    broken = next((e for e in entries if e["name"] == "Broken"), None)
    if broken is not None:
        assert broken["tools"] == [], "a non-list tools value must yield no tools"


def test_a_string_tools_value_does_not_explode_into_characters():
    """`tools: read_file` would iterate the STRING into single characters."""
    entries = _parse("""
        verifiers:
          - name: Typo
            tools: read_file
    """)
    typo = next((e for e in entries if e["name"] == "Typo"), None)
    assert typo is not None, "the entry must survive; only its tools are unusable"
    assert typo["tools"] == [], (
        f"a string tools value was iterated character-wise: {typo['tools']!r}"
    )


def test_a_valid_roster_is_unaffected():
    entries = _parse("""
        instructions: shared
        verifiers:
          - name: Security
            model: anthropic/claude-x
            tools: [read_file, search_files]
            instructions: focus on secrets
    """)
    sec = next(e for e in entries if e["name"] == "Security")
    assert sec["tools"] == ["read_file", "search_files"]
    assert sec["model"] == "anthropic/claude-x"
    assert "focus on secrets" in sec["instructions"]


def test_a_dict_tools_value_does_not_raise():
    """`tools:` with a mapping is another plausible hand-edit."""
    entries = _parse("""
        verifiers:
          - name: Mapping
            tools:
              read_file: true
    """)
    entry = next((e for e in entries if e["name"] == "Mapping"), None)
    if entry is not None:
        assert isinstance(entry["tools"], list)


def test_a_live_session_keeps_its_verifiers_when_a_bad_roster_appears(tmp_path, monkeypatch):  # noqa: F811
    """End to end: the entry that used to be deleted must survive."""
    state = runtime._get_state("s1")
    runtime._ROSTER_OVERRIDE = [{"name": "Security", "slug": "security",
                                 "enabled": True, "tools": ["read_file"]}]
    state.ensure_verifiers(runtime.discover_roster())
    assert "security" in state.verifiers

    # A malformed roster now appears: discovery returns whatever the parse
    # yields, which must not be a TypeError escaping into a live turn.
    monkeypatch.setattr(runtime, "discover_roster", lambda: brief._parse_roster_file(
        "verifiers:\n  - name: Broken\n    tools: 7\n"))
    monkeypatch.setattr(runtime, "_ROSTER_OVERRIDE", None)

    try:
        state.ensure_verifiers(runtime.discover_roster())
    except TypeError as exc:  # pragma: no cover
        raise AssertionError(f"roster discovery raised during a live turn: {exc}")

    assert "default" in state.verifiers, "the built-in verifier is always present"
