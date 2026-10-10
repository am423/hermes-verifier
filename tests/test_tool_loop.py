"""Contract of the advisor tool loop.

The prompt tells the advisor: "You get at most MAX_TOOL_CALLS tool calls per
review." Three things must hold, and none were pinned before:

  1. never dispatch more than that, even if the model keeps asking
  2. once the budget is spent, still OBTAIN the review envelope -- it is the
     whole deliverable, and dropping it loses the finding
  3. when the model never complies, record the ACTUAL cause rather than blaming
     a token cap that was not involved

The tool feature was also verified reachable in production: against the real
provider and the real composed prompt (host's JSON directive included), the
advisor does emit the `TOOL_CALL:` protocol line, so a mismatch there is not the
failure mode.
"""

from __future__ import annotations

from verifier_pkg import advisor_tools, runtime

from .test_runtime import FakeResult, _review_sync, _turn, ctx  # noqa: F401

ENVELOPE = FakeResult({"severity": "caution", "finding": "the write path bypasses the queue",
                       "reasoning": ""})
# parsed=None is the realistic shape: a model emitting the tool protocol is NOT
# emitting a schema-valid envelope, so the host's JSON parse yields nothing.
TOOL_REPLY = FakeResult(None, text='TOOL_CALL: read_file {"path": "a.py"}')


def _arm(ctx, replies):
    """Grant the default verifier read_file and serve `replies` in order."""
    ctx.config["advisor_tools"] = True
    ctx.config["default_tools"] = ["read_file"]
    queue = list(replies)

    def _complete(**kwargs):
        ctx.llm.calls.append(kwargs)
        if "TOOL_CALL" not in kwargs.get("system_prompt", ""):
            return FakeResult({"severity": "none", "finding": "", "reasoning": ""})
        return queue.pop(0) if queue else ENVELOPE

    ctx.llm.complete_structured = _complete
    dispatches = []
    ctx.dispatch_tool = lambda name, args: dispatches.append(name) or "file contents"
    return dispatches


def _tool_calls(ctx):
    return [c for c in ctx.llm.calls if "TOOL_CALL" in c.get("system_prompt", "")]


def test_never_dispatches_more_than_the_advertised_cap(ctx, monkeypatch):  # noqa: F811
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    dispatches = _arm(ctx, [TOOL_REPLY] * 10)      # never stops asking
    _turn(ctx)
    assert len(dispatches) <= advisor_tools.MAX_TOOL_CALLS, (
        f"dispatched {len(dispatches)}; the advisor is promised at most "
        f"{advisor_tools.MAX_TOOL_CALLS}"
    )


def test_the_envelope_is_still_obtained_after_the_budget_is_spent(ctx, monkeypatch):  # noqa: F811
    """A greedy model must not cost the session its review."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    # It asks MAX+1 times (the last asking is refused) and then complies.
    dispatches = _arm(ctx, [TOOL_REPLY] * (advisor_tools.MAX_TOOL_CALLS + 1) + [ENVELOPE])
    _turn(ctx)

    assert len(dispatches) == advisor_tools.MAX_TOOL_CALLS
    prompts = [c["input"][0]["text"] for c in _tool_calls(ctx)]
    assert any("Tool budget exhausted" in p for p in prompts), (
        "the advisor must be told the budget is spent so it can answer"
    )
    # The review actually landed.
    state = runtime._get_state("s1")
    assert state.verifiers["default"].pending, (
        "the finding was lost: the envelope was never requested"
    )


def test_a_model_stuck_on_tools_records_the_real_cause(ctx, monkeypatch):  # noqa: F811
    """Blaming advisor_max_tokens sent operators to raise a cap that was not the
    problem."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    _arm(ctx, [TOOL_REPLY] * 20)
    _turn(ctx)

    state = runtime._get_state("s1")
    err = state.last_error or ""
    assert "tool" in err.lower(), f"expected the tool cause, got: {err!r}"
    assert "advisor_max_tokens" not in err, (
        "misdiagnosed as truncation; that is not what happened"
    )


def test_a_clean_review_makes_no_extra_calls(ctx, monkeypatch):  # noqa: F811
    """No tools requested -> exactly one call, no loop overhead."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    _arm(ctx, [ENVELOPE])
    _turn(ctx)
    assert len(_tool_calls(ctx)) == 1


def test_every_verification_result_is_kept_not_just_the_last(ctx, monkeypatch):  # noqa: F811
    """The follow-up transcript was REBUILT each iteration as
    `delta + latest tool result`, so after two calls the advisor saw only the
    second piece of evidence and reviewed on partial information.
    """
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["advisor_tools"] = True
    ctx.config["default_tools"] = ["read_file"]

    calls = {'n': 0}

    def _complete(**kwargs):
        ctx.llm.calls.append(kwargs)
        if "TOOL_CALL" not in kwargs.get("system_prompt", ""):
            return FakeResult({"severity": "none", "finding": "", "reasoning": ""})
        calls['n'] += 1
        if calls['n'] <= 2:
            return FakeResult(None, text='TOOL_CALL: read_file {"path": "a.py"}')
        return ENVELOPE

    ctx.llm.complete_structured = _complete
    ctx.dispatch_tool = lambda name, args: f"EVIDENCE_{args['path']}"

    _turn(ctx)

    prompts = [c["input"][0]["text"] for c in _tool_calls(ctx)]
    final = prompts[-1]
    assert "EVIDENCE_a.py" in final
    # The second read is the one that would have wiped the first.
    assert final.count("EVIDENCE_") >= 2, (
        "only the most recent tool result survived; the advisor is reviewing "
        "with partial evidence"
    )
