"""The /ask path: its own schema, its own validation.

/ask reused REVIEW_SCHEMA, which demands a severity/finding/reasoning review
envelope -- so asking a question produced a review verdict, and the UI rendered
the raw envelope as the answer. And an empty completion returned
{"ok": True, "answer": ""}, making "the model said nothing" indistinguishable
from a real answer: the exact failure class the review path was hardened against.
"""

from __future__ import annotations

from verifier_pkg import runtime

from .test_runtime import FakeResult, ctx  # noqa: F401


def _armed(ctx):
    runtime._get_state("s1")            # ask_verifier needs an existing session
    return ctx


def test_ask_requests_the_answer_schema_not_the_review_schema(ctx):  # noqa: F811
    _armed(ctx)
    ctx.llm.next = FakeResult({"answer": "the write path bypasses the queue"})
    runtime.ask_verifier(session_id="s1", question="what changed?")

    kwargs = ctx.llm.calls[-1]
    assert kwargs["schema_name"] == "verifier_answer"
    assert set(kwargs["json_schema"]["properties"]) == {"answer"}, (
        "ask is still asking the model for a review envelope"
    )
    assert kwargs["purpose"] == "hermes-verifier ask"


def test_ask_returns_the_parsed_answer(ctx):  # noqa: F811
    _armed(ctx)
    ctx.llm.next = FakeResult({"answer": "the retry loop swallows errors"})
    out = runtime.ask_verifier(session_id="s1", question="what is wrong?")
    assert out["ok"] is True
    assert out["answer"] == "the retry loop swallows errors"


def test_ask_falls_back_to_raw_text_when_parsing_yields_nothing(ctx):  # noqa: F811
    _armed(ctx)
    ctx.llm.next = FakeResult(None, text="a plain prose answer, not JSON")
    out = runtime.ask_verifier(session_id="s1", question="what is wrong?")
    assert out["ok"] is True
    assert out["answer"].startswith("a plain prose answer")


def test_an_empty_answer_is_a_failure_not_an_empty_success(ctx):  # noqa: F811
    _armed(ctx)
    ctx.llm.next = FakeResult({"answer": "   "})
    out = runtime.ask_verifier(session_id="s1", question="what is wrong?")
    assert out["ok"] is False, out
    assert out["error"]


def test_a_whitespace_only_text_answer_is_also_a_failure(ctx):  # noqa: F811
    _armed(ctx)
    ctx.llm.next = FakeResult(None, text="   \n  ")
    out = runtime.ask_verifier(session_id="s1", question="what is wrong?")
    assert out["ok"] is False


def test_the_review_path_still_uses_the_review_schema(ctx):  # noqa: F811
    """The parameterisation must not have changed the review contract."""
    from .test_runtime import _review_sync, _turn
    import pytest

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
        ctx.llm.next = FakeResult({"severity": "none", "finding": "", "reasoning": ""})
        _turn(ctx, history=[{"role": "user", "content": "hi"}])
    kwargs = ctx.llm.calls[-1]
    assert kwargs["schema_name"] == "verifier_review"
    assert set(kwargs["json_schema"]["properties"]) == {"severity", "finding", "reasoning"}
