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


def test_agreement_leaves_the_session_dedupe_intact(ctx, monkeypatch):  # noqa: F811
    """A delivered finding must still be IN the guard's history afterwards.

    The filter called guard.reset() on the SHARED guard -- once between passes
    and again before every delivery -- which clears dedupe for the whole
    SESSION, not just the passes' own bookkeeping. After any agreement review
    the guard had forgotten everything it had ever delivered, so the same
    finding was free to come back on a later turn.

    Asserts on the public accept() rather than on a delivered-again outcome: a
    repeat CAN also be suppressed by failing to clear the vote bar, which would
    make the test pass while the history was still being wiped.
    """
    ctx.config["agreement_reviews"] = 3
    ctx.config["agreement_min"] = 3
    state, verifier = _state()

    first = {"severity": "caution", "finding": "the write path bypasses the queue"}
    second = {"severity": "caution", "finding": "the retry loop is untested"}

    # Turn 1 delivers `first`. Everything in _seen now is {first}.
    _arm(ctx, [first, first, first])
    runtime._review_one(state, verifier, "delta one", "deep", 1)
    assert verifier.pending, "precondition: the first finding was delivered"
    verifier.pending.clear()

    # Turn 2 delivers a DIFFERENT finding. Under the bug this reset()s the shared
    # guard on the way through, which is what forgets `first`.
    _arm(ctx, [second, second, second])
    runtime._review_one(state, verifier, "delta two", "deep", 2)
    assert verifier.pending, "precondition: the second finding was delivered"

    # begin_cycle() before each probe: accept() checks dedupe and THEN a
    # one-finding-per-cycle budget, so without this both probes return False
    # from the rate limit and the assertion would pass while the history was
    # still being wiped. Opening the cycle makes the result mean "already seen".
    verifier.guard.begin_cycle()
    assert verifier.guard.accept("caution", first["finding"]) is False, (
        "the guard forgot an EARLIER delivered finding: the second agreement "
        "review reset the session history, so turn 1's finding can ship again"
    )
    verifier.guard.begin_cycle()
    assert verifier.guard.accept("caution", second["finding"]) is False, (
        "the guard forgot the finding it had just delivered"
    )


def test_the_passes_do_not_pollute_the_shared_guard(ctx, monkeypatch):  # noqa: F811
    """Independence must come from scratch guards, not from amnesia.

    The old design reset the shared guard so pass 2 would not see pass 1's
    finding. A scratch guard per pass gives the same independence while leaving
    the shared guard's history and counters alone -- which is what makes the
    accept count meaningful and dedupe survive.
    """
    ctx.config["agreement_reviews"] = 3
    ctx.config["agreement_min"] = 3
    state, verifier = _state()
    before = dict(verifier.guard.stats)

    _arm(ctx, [ENVELOPE, ENVELOPE, ENVELOPE])
    runtime._review_one(state, verifier, "delta", "deep", 1)

    assert verifier.guard.stats["accepted"] == before["accepted"] + 1, (
        "three probe passes plus one delivery recorded "
        f"{verifier.guard.stats['accepted'] - before['accepted']} accepts, not 1"
    )


def test_agreement_rows_name_the_verifier(ctx, monkeypatch):  # noqa: F811
    """With a roster, agreement rows were indistinguishable from each other."""
    import json
    ctx.config["agreement_reviews"] = 2
    ctx.config["agreement_min"] = 2
    state, verifier = _state()
    _arm(ctx, [ENVELOPE, ENVELOPE])
    runtime._review_one(state, verifier, "delta", "deep", 1)
    path = runtime.data_dir() / f"agreement.{runtime._slug_session('s1')}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert rows[-1].get("verifier") == "default", (
        "agreement rows carry no verifier: with a roster you cannot tell which "
        "entry produced a suppression"
    )
