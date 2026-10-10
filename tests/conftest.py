"""Test bootstrap: import the plugin package (repo root) as a package.

The plugin package lives at the repository root (canonical single-plugin repo
layout), which is not importable by path under its own name. We symlink it
under a synthetic package name — in the SYSTEM TEMP DIR, never inside the
repo, so the link cannot recurse into its own parent during collection.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG_ROOT = Path(tempfile.gettempdir()) / "hermes-verifier-test-pkg"
PKG_ROOT.mkdir(exist_ok=True)
link = PKG_ROOT / "verifier_pkg"
try:
    if link.is_symlink() and link.resolve() != REPO:
        link.unlink()  # stale target (repo restructure) — recreate below
except (OSError, RuntimeError):
    pass
if not link.exists():
    os.symlink(REPO, link)

if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# runtime.py imports ``agent.plugin_llm`` to detect trust-gate errors. That module
# only exists inside a Hermes install, so stub it here: the suite must pass on a
# bare checkout (CI, forks) without Hermes present. The stub exposes a distinct
# PluginLlmTrustError so trust-error classification stays testable.
if "agent" not in sys.modules:
    import types

    _agent = types.ModuleType("agent")
    _plugin_llm = types.ModuleType("agent.plugin_llm")

    class PluginLlmTrustError(Exception):
        """Stand-in for the host's trust-gate error."""

    _plugin_llm.PluginLlmTrustError = PluginLlmTrustError
    _agent.plugin_llm = _plugin_llm
    sys.modules.setdefault("agent", _agent)
    sys.modules.setdefault("agent.plugin_llm", _plugin_llm)
