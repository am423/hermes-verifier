"""K-of-N agreement: deliver a finding only when repeats agree.

Measured on the plugin's own case set: at >=3/3 the deepseek x baseline arm went
from 67% false alarms to 17% while keeping 72% of findings -- the only operating
point measured below the ~30% rate at which the Coverity data says users stop
reading a reviewer at all.

The subtle part is INDEPENDENCE. Each pass must be a fresh judgement; if the
guard's dedupe sees pass 1's finding it suppresses the identical text on pass 2,
every repeat answers "none", and agreement becomes unsatisfiable by
construction -- the filter would suppress everything, silently.
"""

from __future__ import annotations

from verifier_pkg import runtime

from .test_runtime import FakeResult, _review_sync, ctx  # noqa: F401

ENVELOPE = {"severity": "caution", "finding": "the write path bypasses the queue"}
QUIET = {"severity": "none", "finding": ""}


def _arm(ctx, envelopes):
    """Serve one envelope per review pass, in order."""
    queue = list(envelopes)

    def _complete(**kwargs):
        ctx.llm.calls.append(kwargs)
        parsed = queue.pop(0) if queue else QUIET
        return FakeResult(parsed)

    ctx.llm.complete_structured = _complete
    return _complete


def _state():
    state = runtime._get_state("s1")
    state.verifiers["default"].pending.clear()
    return state, state.verifiers["default"]


def test_off_by_default_runs_the_review_once(ctx):  # noqa: F811
    state, verifier = _state()
    _arm(ctx, [ENVELOPE])
    runtime._review_one(state, verifier, "delta", "deep", 1)
    assert len(ctx.llm.calls) == 1, "agreement must be opt-in"
    assert verifier.pending, "a single run should still deliver"


def test_unanimous_agreement_delivers(ctx, monkeypatch):  # noqa: F811
    ctx.config["agreement_reviews"] = 3
    ctx.config["agreement_min"] = 3
    state, verifier = _state()
    _arm(ctx, [ENVELOPE, ENVELOPE, ENVELOPE])
    runtime._review_one(state, verifier, "delta", "deep", 1)
    assert len(ctx.llm.calls) == 3, "three passes must actually run"
    assert verifier.pending, "unanimous findings must be delivered"


def test_split_verdict_suppresses_the_finding(ctx, monkeypatch):  # noqa: F811
    """2 of 3 is below a 3-of-3 bar -- this is the false-alarm filter working."""
    ctx.config["agreement_reviews"] = 3
    ctx.config["agreement_min"] = 3
    state, verifier = _state()
    _arm(ctx, [ENVELOPE, QUIET, QUIET])
    runtime._review_one(state, verifier, "delta", "deep", 1)
    assert len(ctx.llm.calls) == 3
    assert not verifier.pending, (
        "a finding only one pass raised was delivered anyway; the filter is inert"
    )


def test_passes_are_independent_not_deduped(ctx, monkeypatch):  # noqa: F811
    """The regression that would silently disable the whole feature.

    Without resetting the guard between passes, pass 1 records the finding, pass
    2 sees identical text and answers 'deduped' (i.e. nothing), so a unanimous
    case would score 1-of-3 and be suppressed forever.
    """
    ctx.config["agreement_reviews"] = 3
    ctx.config["agreement_min"] = 3
    state, verifier = _state()
    _arm(ctx, [ENVELOPE, ENVELOPE, ENVELOPE])
    runtime._review_one(state, verifier, "delta", "deep", 1)
    assert verifier.pending, (
        "three identical findings were NOT delivered: the passes are not "
        "independent, so agreement can never be satisfied"
    )


def test_two_of_three_bar_accepts_a_majority(ctx, monkeypatch):  # noqa: F811
    ctx.config["agreement_reviews"] = 3
    ctx.config["agreement_min"] = 2
    state, verifier = _state()
    _arm(ctx, [ENVELOPE, ENVELOPE, QUIET])
    runtime._review_one(state, verifier, "delta", "deep", 1)
    assert verifier.pending, "2-of-3 must clear a 2-of-3 bar"


def test_agreement_is_observable(ctx, monkeypatch):  # noqa: F811
    ctx.config["agreement_reviews"] = 3
    ctx.config["agreement_min"] = 3
    state, verifier = _state()
    _arm(ctx, [ENVELOPE, QUIET, QUIET])
    runtime._review_one(state, verifier, "delta", "deep", 1)
    events = [e for e in runtime.EVENT_LOG if e.get("kind") == "review_agreement"]
    assert events, "suppression must be visible, not silent"
    assert events[-1]["raised"] == 1 and events[-1]["required"] == 3
    assert events[-1]["agreed"] is False


def test_a_bad_bar_falls_back_to_single_run(ctx, monkeypatch):  # noqa: F811
    """min > reviews is nonsense; it must not suppress everything."""
    ctx.config["agreement_reviews"] = 2
    ctx.config["agreement_min"] = 5
    state, verifier = _state()
    _arm(ctx, [ENVELOPE])
    runtime._review_one(state, verifier, "delta", "deep", 1)
    assert len(ctx.llm.calls) == 1
    assert verifier.pending


def test_agreement_cost_is_accounted(ctx, monkeypatch):  # noqa: F811
    """Agreement is N times the price. The roll-up must show that."""
    ctx.config["agreement_reviews"] = 3
    ctx.config["agreement_min"] = 3
    state, verifier = _state()
    before = verifier.stats["reviews"]
    _arm(ctx, [ENVELOPE, ENVELOPE, ENVELOPE])
    runtime._review_one(state, verifier, "delta", "deep", 1)
    assert verifier.stats["reviews"] >= before + 3, (
        "three passes ran but fewer than three were counted"
    )
