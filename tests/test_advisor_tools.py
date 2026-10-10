"""Advisor tool-grant tests: allowlist enforcement and call parsing."""

from __future__ import annotations

from verifier_pkg import advisor_tools


def test_allowlist_refuses_mutating_and_unknown_tools():
    granted = advisor_tools.resolve_grant(["read_file", "terminal", "write_file", "nope"])
    assert granted == ["read_file"]


def test_empty_grant_stays_empty():
    assert advisor_tools.resolve_grant(None) == []
    assert advisor_tools.resolve_grant([]) == []
    assert advisor_tools.resolve_grant(["terminal"]) == []


def test_grant_is_capped():
    assert len(advisor_tools.resolve_grant(["read_file", "search_files"])) <= advisor_tools.MAX_TOOL_CALLS


def test_render_available_empty_without_grant():
    assert advisor_tools.render_available([]) == ""


def test_render_available_documents_the_protocol():
    block = advisor_tools.render_available(["read_file"])
    assert "TOOL_CALL:" in block
    assert "read_file" in block


def test_parse_valid_call():
    call = advisor_tools.parse_tool_call(
        'TOOL_CALL: read_file {"path": "src/jobs/queue.py"}', ["read_file"])
    assert call == {"tool": "read_file", "args": {"path": "src/jobs/queue.py"}}


def test_parse_rejects_ungranted_tool():
    assert advisor_tools.parse_tool_call(
        'TOOL_CALL: terminal {"command": "rm -rf /"}', ["read_file"]) is None


def test_parse_rejects_malformed_json():
    assert advisor_tools.parse_tool_call("TOOL_CALL: read_file {not json}", ["read_file"]) is None


def test_parse_rejects_missing_required_arg():
    assert advisor_tools.parse_tool_call("TOOL_CALL: read_file {}", ["read_file"]) is None


def test_parse_returns_none_without_a_call():
    assert advisor_tools.parse_tool_call("Just some prose, no call here.", ["read_file"]) is None


def test_dispatch_truncates_and_never_raises():
    class Ctx:
        def dispatch_tool(self, name, args):
            assert name == "read_file"
            return "x" * (advisor_tools.MAX_RESULT_CHARS + 500)

    out = advisor_tools.dispatch(Ctx(), {"tool": "read_file", "args": {"path": "f"}})
    assert out.endswith("…[truncated]")
    assert len(out) < advisor_tools.MAX_RESULT_CHARS + 200


def test_dispatch_swallows_exceptions():
    class Boom:
        def dispatch_tool(self, name, args):
            raise RuntimeError("host refused")

    assert "failed" in advisor_tools.dispatch(Boom(), {"tool": "read_file", "args": {"path": "f"}})
