"""The loop reports. It does not change the plugin's configuration.

Writes to the loop's OWN outputs -- the markdown report -- are expected and
allowed; forbidding `write_text` outright would fail on correct code and push
whoever hit it next to weaken the test. What must never happen is a write that
reaches the plugin's settings or the Hermes config.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[1]

# The API that mutates plugin settings, by any name it might be called.
FORBIDDEN_CALLS = {"set_config", "safe_dump"}

# Paths the loop must never mention outside its own detector. rules.py lists
# "config.yaml" because it DETECTS config edits -- that is the one exemption.
FORBIDDEN_PATH_TOKENS = ("config.yaml", ".hermes/config", "plugin.yaml",
                         "settings.json")
DETECTOR = "rules.py"


def _sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        if "tests" in path.parts:
            continue
        yield path, path.read_text()


def test_no_module_calls_a_config_writer():
    for path, src in _sources():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            assert name not in FORBIDDEN_CALLS, (
                f"{path.name} calls {name!r}; the loop must not write configuration"
            )


def test_config_paths_appear_only_in_the_detector():
    for path, src in _sources():
        if path.name == DETECTOR:
            continue
        for token in FORBIDDEN_PATH_TOKENS:
            assert token not in src, (
                f"{path.name} mentions {token!r}. Only {DETECTOR} may: it detects "
                "config edits. Everywhere else the loop reads logs, it does not "
                "touch settings."
            )


def test_the_loop_writes_only_its_own_report_tree():
    """Guard the guard: the report writer must target an output directory."""
    from autoresearch import report
    src = inspect.getsource(report.write_report)
    assert "out_dir" in src or "reports" in src, (
        "write_report no longer targets an output directory; the allowlist "
        "reasoning in this file is stale"
    )
