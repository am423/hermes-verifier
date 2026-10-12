"""Quoted shell/program bodies must remain visible to the phase trigger.

All commands here are inert inputs to the detector; none are executed.
"""
from __future__ import annotations

import pytest

from verifier_pkg import runtime
from .test_runtime import FakeResult, ctx  # noqa: F401

DELETE = "rm -" + "rf /srv/example/cache"


@pytest.mark.parametrize("command,risk", [
    (f"bash -c '{DELETE}'", "irreversible"),
    (f"sh -c '{DELETE}'", "irreversible"),
    ("bash -lc 'git push origin main'", "external"),
    ("sh -c 'npm publish'", "external"),
    ("psql -c 'DROP TABLE example'", "irreversible"),
    ('python -c \'print("git push")\'', "external"),
])
def test_quoted_bodies_are_not_hidden(command, risk):
    assert runtime._action_is_high_risk("terminal", {"command": command}) == risk


def test_quoted_shell_body_reaches_the_reviewer(ctx):  # noqa: F811
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "block", "finding": "Check the deletion target first."})
    directive = runtime.handle_pre_tool_call(
        tool_name="terminal", args={"command": f"bash -c '{DELETE}'"}, session_id="s1")
    assert ctx.llm.calls, "the detector must schedule review, not merely return a label"
    assert directive and directive["action"] == "block"


def test_quotes_on_other_tools_do_not_hide_risk():
    assert runtime._action_is_high_risk(
        "execute_code", {"code": "run('git push origin main')"}) == "external"
