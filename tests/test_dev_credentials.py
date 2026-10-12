"""Development scripts use explicit env credentials, never host stores.

All credentials here are invented fixtures. Every provider transport is blocked
unless a test replaces it with a local fake; no Hermes CLI process is launched.
"""
from __future__ import annotations

import ast
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types
import urllib.request

import pytest

from tests.test_evals_run import REPO, _run_module


def _module(relative, monkeypatch):
    name = "dev_contract_" + Path(relative).stem
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path, monkeypatch):
    """Create the stores before guarding reads, so absence cannot pass a test."""
    home = tmp_path / "home"
    profile = tmp_path / "active-profile"
    stores = []
    for directory in (home / ".hermes", profile):
        directory.mkdir(parents=True)
        env_file = directory / ".env"
        env_file.write_text("DEEPSEEK_API_KEY=fixture-store-key\n")
        auth_file = directory / "auth.json"
        auth_file.write_text(json.dumps({"providers": {"openai-codex": {
            "tokens": {"access_token": "fixture-oauth", "account_id": "fixture-account"}
        }}}))
        stores.extend((env_file, auth_file))
    assert all(path.is_file() for path in stores)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    host = types.ModuleType("hermes_constants")
    monkeypatch.setattr(host, "get_hermes_home", lambda: profile, raising=False)
    monkeypatch.setitem(sys.modules, "hermes_constants", host)

    reads = []

    def guard_open(opener):
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(file)).resolve()
                if path.name in {".env", "auth.json"}:
                    reads.append(path)
                    raise AssertionError("development code attempted a credential-store read")
            return opener(file, *args, **kwargs)
        return guarded

    monkeypatch.setattr(builtins, "open", guard_open(builtins.open))
    monkeypatch.setattr(io, "open", guard_open(io.open))
    monkeypatch.setattr(os, "open", guard_open(os.open))

    def no_transport(*args, **kwargs):
        pytest.fail("test attempted a real provider request or CLI invocation")

    monkeypatch.setattr(urllib.request, "urlopen", no_transport)
    monkeypatch.setattr(subprocess, "run", no_transport)
    yield {"home": home, "profile": profile, "reads": reads, "stores": stores}
    assert reads == [], "a credential-store read was attempted, even if the error was caught"


def _key_loader(which, monkeypatch):
    if which == "eval":
        return lambda: _run_module()._key("DEEPSEEK_API_KEY"), RuntimeError
    return _module("sidecar/llm_gate.py", monkeypatch).load_key, SystemExit


@pytest.mark.parametrize("which", ["eval", "sidecar"])
def test_only_process_environment_supplies_the_key(which, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-environment-key")
    load_key, _ = _key_loader(which, monkeypatch)
    assert load_key() == "fixture-environment-key"


@pytest.mark.parametrize("which", ["eval", "sidecar"])
@pytest.mark.parametrize("value", [None, "", " \t "])
def test_missing_or_blank_env_key_fails_despite_populated_stores(which, value, monkeypatch):
    if value is not None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", value)
    load_key, error_type = _key_loader(which, monkeypatch)
    with pytest.raises(error_type, match="DEEPSEEK_API_KEY.*process environment"):
        load_key()


def test_deepseek_missing_key_fails_before_any_request():
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY.*process environment"):
        _run_module()._complete_deepseek([], "deepseek/deepseek-flash")


@pytest.mark.parametrize("script", [
    "llm_gate.py", "llm_gate_heldout.py", "llm_gate_fresh3.py",
    "gate_consequence_test.py", "tool_protocol_test.py",
])
def test_all_sidecar_entrypoints_fail_without_an_exported_key(script, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "sidecar"))
    # Isolate the script-style imports from other tests, without importing any
    # top-level experiment until the file/network guards are in place.
    for name in ("llm_gate", "llm_gate_heldout", "hv_brief", "verifier_pkg"):
        if name in sys.modules:
            monkeypatch.setitem(sys.modules, name, sys.modules[name])
        else:
            monkeypatch.setitem(sys.modules, name, None)
            del sys.modules[name]
    with pytest.raises(SystemExit, match="DEEPSEEK_API_KEY.*process environment"):
        runpy.run_path(str(REPO / "sidecar" / script), run_name="__main__")


def test_deepseek_request_uses_only_the_exported_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-environment-key")
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        assert request.full_url == "https://api.deepseek.com/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer fixture-environment-key"
        return io.BytesIO(b'{"choices": [{"message": {"content": "fixture response"}}]}')

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    out = _run_module()._complete_deepseek([], "deepseek/deepseek-flash")
    assert out["text"] == "fixture response"
    assert len(requests) == 1


def test_eval_cli_missing_key_exits_before_running_cases(monkeypatch, capsys):
    module = _run_module()
    monkeypatch.setattr(sys, "argv", ["evals/run.py"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "DEEPSEEK_API_KEY is required in the process environment" in capsys.readouterr().err


def test_direct_codex_function_and_cli_option_are_removed(monkeypatch, capsys):
    module = _run_module()
    assert not hasattr(module, "complete_codex_direct")
    monkeypatch.setattr(sys, "argv", ["evals/run.py", "--provider", "codex", "--list"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_codex_eval_routes_through_the_host_cli_without_a_deepseek_key(monkeypatch):
    module = _run_module()
    monkeypatch.setattr(module, "COMPLETER", None)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module.case_mod, "all_cases", lambda: [{
        "id": "fixture", "defect": True, "keywords": ["fixture"],
        "user": "fixture", "actions": [], "final": "fixture",
    }])
    monkeypatch.setattr(sys, "argv", [
        "evals/run.py", "--provider", "cli", "--cli-provider", "openai-codex",
        "--model", "fixture-model",
    ])
    calls = []
    prompts = []

    def fake_cli(command, **kwargs):
        calls.append(command)
        prompt = Path(command[-1])
        prompts.append(prompt)
        assert "fixture" in prompt.read_text()
        assert kwargs == {"capture_output": True, "text": True, "timeout": 600}
        return types.SimpleNamespace(stdout='{"severity": "none", "finding": ""}', returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_cli)
    module.main()
    assert len(calls) == 1
    assert calls[0][:-1] == [
        "hermes", "chat", "--oneshot", "-Q", "-m", "fixture-model",
        "--provider", "openai-codex", "--query-file",
    ]
    assert not prompts[0].exists(), "the temporary query must be removed"


def test_moment_analysis_resolves_the_active_home_at_call_time(monkeypatch, tmp_path):
    module = _module("evals/moment_analysis.py", monkeypatch)
    profile = tmp_path / "later-profile"
    monkeypatch.setattr(sys.modules["hermes_constants"], "get_hermes_home", lambda: profile)
    opened = []

    def fake_connect(path, *, uri):
        opened.append((path, uri))
        return types.SimpleNamespace(execute=lambda *a: [], row_factory=None)

    monkeypatch.setattr(module.sqlite3, "connect", fake_connect)
    assert module.load_turns() == []
    assert opened == [(f"file:{profile / 'state.db'}?mode=ro", True)]


def test_eval_tools_cannot_escape_to_stored_credentials(isolated_credentials, tmp_path):
    module = _run_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "fixture.py").write_text("fixture code")
    store = isolated_credentials["profile"] / ".env"
    (workspace / "linked-store").symlink_to(store)
    for target in (str(store), "../active-profile/.env", "linked-store"):
        out = module.dispatch("read_file", {"path": target}, workspace)
        assert out.startswith("[error]"), target
    assert module.dispatch("search_files", {"pattern": "fixture"}, workspace) == "fixture.py"
    assert module.dispatch("read_file", {"path": "fixture.py"}, workspace) == "fixture code"


def test_development_sources_have_no_host_credential_transport():
    """Pin removed APIs as well as behavior; include every shipped dev script."""
    forbidden_imports = ("hermes_cli.auth", "agent.credential", "dotenv", "keyring")
    forbidden_literals = ("auth.json", "access_token", "refresh_token",
                          "chatgpt-account-id", "chatgpt.com/backend-api/")
    for directory in ("evals", "sidecar", "autoresearch"):
        for path in (REPO / directory).rglob("*.py"):
            if "tests" in path.relative_to(REPO).parts:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imports = [node.module or ""]
                else:
                    imports = []
                assert not any(name.startswith(forbidden_imports) for name in imports), path
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert not any(token in node.value for token in forbidden_literals), path
