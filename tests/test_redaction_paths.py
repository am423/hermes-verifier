"""Tool results must be redacted before they reach the reviewer.

The reviewer's prompt promises "you never see secrets and must not ask for them"
(brief.py). That held for the transcript delta, which passes through
`delta.redact`, but the advisor's TOOL RESULTS were spliced straight into the
next prompt. With `advisor_tools` on -- and `default_tools: [read_file,
search_files]` is a common configuration -- a `read_file` over a config, a `.env`
or a key file shipped the secret verbatim to the reviewer model.

This is a live path, not a theoretical one: it is one of the two settings this
plugin ships that read arbitrary files.
"""

from __future__ import annotations

from verifier_pkg import runtime

from .test_runtime import FakeResult, _review_sync, _turn, ctx  # noqa: F401

SECRET = "sk-proj-LEAKCANARY1234567890abcdef"


def _arm_tool_path(ctx):
    """Give the DEFAULT verifier a read_file grant and script a tool call."""
    ctx.config["advisor_tools"] = True
    ctx.config["default_tools"] = ["read_file"]
    missing = object()

    def _complete(**kwargs):
        ctx.llm.calls.append(kwargs)
        if "TOOL_CALL" in kwargs.get("system_prompt", ""):
            if getattr(_complete, "used", False):
                return FakeResult({"severity": "note", "finding": "secrets are stored here",
                                   "reasoning": ""})
            _complete.used = True
            return FakeResult({"severity": "none", "finding": "", "reasoning": ""},
                              text='TOOL_CALL: read_file {"path": "config/.env"}')
        return FakeResult({"severity": "none", "finding": "", "reasoning": ""})

    ctx.llm.complete_structured = _complete
    # What the host would return for that read.
    ctx.dispatch_tool = lambda name, args: f"OPENAI_API_KEY={SECRET}"


def test_tool_result_reaching_the_reviewer_is_redacted(ctx, monkeypatch):  # noqa: F811
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    _arm_tool_path(ctx)
    _turn(ctx)

    tool_calls = [c for c in ctx.llm.calls
                  if "TOOL_CALL" in c.get("system_prompt", "")]
    assert len(tool_calls) >= 2, (
        "the reviewer should have made a verification call and then answered; "
        f"got {len(tool_calls)}"
    )
    follow_up = tool_calls[-1]["input"][0]["text"]
    assert SECRET not in follow_up, (
        "an unredacted secret from a tool result reached the reviewer model:\n"
        f"{follow_up[:400]}"
    )
    assert "REDACTED" in follow_up, "expected the redaction marker in its place"


def test_tool_results_are_still_bounded(ctx, monkeypatch):  # noqa: F811
    """Redaction must not remove the cap that keeps tool payloads out of the bill."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    _arm_tool_path(ctx)
    ctx.dispatch_tool = lambda name, args: "\n".join(
        f"line {i} of a very long tool result" for i in range(4000))
    _turn(ctx)

    tool_calls = [c for c in ctx.llm.calls
                  if "TOOL_CALL" in c.get("system_prompt", "")]
    follow_up = tool_calls[-1]["input"][0]["text"]
    assert len(follow_up) < 20000, "the transcript must stay bounded"
