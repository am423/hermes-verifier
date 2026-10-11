"""The dashboard websocket must authorize before it accepts.

NousResearch/hermes-agent#135996, required fix: `dashboard/plugin_api.py`
accepted the socket without checking the session, so any local process could
open it and read the verifier's live event stream. The reviewer asked for
`hermes_cli.web_server_chat._ws_auth_ok` (the dashboard's canonical gate, which
understands `?token=` / `?ticket=` / `?internal=`) and a 1008 close on failure,
as `plugins/kanban/dashboard/plugin_api.py` does.

NOTE on coverage: the ordering assertion below is a SOURCE assertion, not a
behavioural one -- it pins that the call exists and precedes accept(), which is
the property that regressed. Exercising a real rejected socket needs a live
websocket through the host's auth stack, which this suite does not stand up;
saying the endpoint is "covered" would overstate it.
"""

from __future__ import annotations

import pathlib

SRC = pathlib.Path("dashboard/plugin_api.py").read_text()


def test_the_socket_authorizes_before_accepting():
    assert "_ws_auth_ok(" in SRC, (
        "the websocket endpoint does not consult hermes_cli.web_server_chat._ws_auth_ok"
    )
    accept_at = SRC.index("await websocket.accept()")
    guard_at = SRC.index("_ws_upgrade_authorized(")
    assert guard_at < accept_at, "authorization must run BEFORE accept()"


def test_a_rejected_socket_closes_with_1008():
    assert "1008" in SRC, "a rejected socket must close with policy-violation 1008"


def test_the_helper_denies_when_the_host_gate_denies(monkeypatch):
    """The helper must report the host's verdict, not its own opinion."""
    import importlib

    from verifier_pkg.dashboard import plugin_api

    importlib.reload(plugin_api)

    class _FakeWs:
        pass

    class _Deny:
        @staticmethod
        def _ws_auth_ok(_ws):
            return False

    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.web_server_chat", _Deny)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli",
                        type("m", (), {"web_server_chat": _Deny}))
    assert plugin_api._ws_upgrade_authorized(_FakeWs()) is False

    class _Allow:
        @staticmethod
        def _ws_auth_ok(_ws):
            return True

    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.web_server_chat", _Allow)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli",
                        type("m", (), {"web_server_chat": _Allow}))
    assert plugin_api._ws_upgrade_authorized(_FakeWs()) is True


def test_an_unimportable_dashboard_does_not_brick_the_socket(monkeypatch):
    """A bare-FastAPI harness has no hermes_cli; refusing there would break every
    test host. This matches the bundled kanban plugin's own choice."""
    import builtins

    from verifier_pkg.dashboard import plugin_api

    real_import = builtins.__import__

    def _boom(name, *a, **k):
        if name.startswith("hermes_cli"):
            raise ImportError("no host here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert plugin_api._ws_upgrade_authorized(object()) is True
