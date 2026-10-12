"""Resolve analysis data paths without loading settings or credentials."""
from __future__ import annotations

import os
from pathlib import Path


def get_hermes_home() -> Path:
    """Use the host's active profile at call time, not a cached import-time path.

    Standalone checkouts need not install Hermes to run offline tests. Only when
    the host module is absent, use HERMES_HOME (or the conventional default home).
    An installed host's import/accessor errors must propagate rather than silently
    selecting the wrong profile. Resolving the path does not open any files.
    """
    try:
        from hermes_constants import get_hermes_home as host_home
    except ModuleNotFoundError as exc:
        if exc.name != "hermes_constants":
            raise
        return Path(os.environ.get("HERMES_HOME") or "~/.hermes").expanduser()
    return Path(host_home())
