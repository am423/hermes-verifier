"""hermes-verifier — an independent reviewer for your Hermes sessions.

The verifier reviews each primary-agent turn delta, raises at most one
finding per review (note / caution / blocker), and delivers accepted findings
back into the session:

- ``post_llm_call``  → extract the transcript delta, schedule a background review
- ``pre_llm_call``   → inject a pending finding into the next turn's context
- ``pre_verify``     → a pending blocker keeps a coding turn alive (steer gate)
- ``on_session_reset`` → re-prime the verifier (clear cursor + dedupe state)
"""

from __future__ import annotations

import logging

from . import commands, runtime

logger = logging.getLogger("hermes-verifier")


def register(ctx) -> None:
    """Wire hooks and the slash command. Called once at plugin load."""
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
