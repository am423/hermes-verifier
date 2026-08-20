"""hermes-verifier — an independent reviewer for your Hermes sessions.

The verifier reviews each primary-agent turn delta, raises at most one
finding per review (note / caution / blocker), and delivers accepted findings
back into the session:

- ``post_llm_call``  → extract the transcript delta, schedule a background review
- ``pre_llm_call``   → inject a pending finding into the next turn's context
- ``pre_verify``     → a pending blocker keeps a coding turn alive (steer gate)
- ``on_session_reset`` → re-prime the verifier (clear cursor + dedupe state)

NOTE: submodule imports live inside ``register()``. This module must stay
import-safe when loaded WITHOUT package context — pytest's collection imports
the package root's ``__init__.py`` by file basename (the repo directory is a
hyphenated package name), and any module-level relative import would crash
that import. Hermes always calls ``register()`` through the PluginManager
namespace, where the package context is proper.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("hermes-verifier")


def register(ctx) -> None:
    """Wire hooks and the slash command. Called once at plugin load."""
    from . import commands, runtime

    runtime.bind_ctx(ctx)

    ctx.register_hook("post_llm_call", runtime.handle_post_llm_call)
    ctx.register_hook("pre_llm_call", runtime.handle_pre_llm_call)
    ctx.register_hook("pre_verify", runtime.handle_pre_verify)
    ctx.register_hook("on_session_reset", runtime.handle_session_reset)

    ctx.register_command(
        "verify",
        handler=commands.make_handler(ctx),
        description="Control the session verifier: status, on/off, reset, dump.",
        args_hint="[on|off|status|reset|dump]",
    )
    logger.info("hermes-verifier registered (subsystem default: %s)",
                bool(ctx.get_config("enabled", False)))
