"""Test bootstrap: import the plugin directory as a package.

The plugin lives at ``<repo>/plugin`` and uses intra-package relative
imports, but a directory named ``plugin`` is too generic to import by path
directly. We symlink it under a synthetic package name instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO / "plugin"
PKG_ROOT = Path(__file__).resolve().parent / "_pkg"

PKG_ROOT.mkdir(exist_ok=True)
link = PKG_ROOT / "verifier_pkg"
if not link.exists() and not link.is_symlink():
    os.symlink(PLUGIN_DIR, link)

if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
