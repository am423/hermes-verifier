"""Session memory: the verifier sees its own earlier findings for the same session.

Each review only receives the newest transcript delta. Without its earlier findings
the verifier cannot tell that a claim like "the verifier flagged this twice" refers to
its own prior output, and it raises a false "invented claim" caution against the
primary agent. These tests pin the memory block: same-session only, accepted
findings only, bounded, and never able to break a review.
"""

from __future__ import annotations

import json

from verifier_pkg import runtime

from .test_runtime import FakeResult, _review_sync, ctx  # noqa: F401


def _write(session: str, rows) -> None:
    path = runtime._log_path(session, "default")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write((row if isinstance(row, str) else json.dumps(row)) + "\n")


def _state(session: str):
    state = runtime._get_state(session)
    return state, state.verifiers["default"]


def test_a_new_session_has_no_memory(ctx):  # noqa: F811
    state, verifier = _state("fresh")
    assert runtime._prior_findings_block(state, verifier) == ""


def test_prior_findings_are_listed_for_the_same_session_only(ctx):  # noqa: F811
    _write("s1", [{"accepted": True, "severity": "caution", "finding": "claimed X unverified",
                   "origin_turn": 2}])
    _write("other", [{"accepted": True, "severity": "blocker", "finding": "other session",
                      "origin_turn": 1}])
    state, verifier = _state("s1")
    block = runtime._prior_findings_block(state, verifier)
    assert "turn 2 [caution]: claimed X unverified" in block
    assert "other session" not in block


def test_only_accepted_real_findings_are_kept_and_capped(ctx):  # noqa: F811
    rows = [{"accepted": True, "severity": "note", "finding": f"f{i}", "origin_turn": i}
            for i in range(12)]
    rows += [{"accepted": False, "severity": "caution", "finding": "rejected", "origin_turn": 50},
             {"accepted": True, "severity": "none", "finding": "nothing", "origin_turn": 51},
             "not json"]
    _write("s1", rows)
    state, verifier = _state("s1")
    lines = runtime._prior_findings_block(state, verifier).splitlines()[1:]
    assert len(lines) == 12  # 12 accepted findings fit under the cap; rejected/none/bad dropped
    assert "f0" in lines[0] and "f11" in lines[-1]
    assert not any("rejected" in l or "nothing" in l or "omitted" in l for l in lines)


def test_a_long_session_keeps_its_first_and_latest_findings(ctx):  # noqa: F811
    _write("s1", [{"accepted": True, "severity": "caution", "finding": f"f{i}", "origin_turn": i}
                  for i in range(30)])
    state, verifier = _state("s1")
    lines = runtime._prior_findings_block(state, verifier).splitlines()[1:]
    first, recent = runtime._PRIOR_FINDINGS_FIRST, runtime._PRIOR_FINDINGS_RECENT
    assert len(lines) == first + recent + 1  # plus the "omitted" marker
    assert [l.split(": ", 1)[1] for l in lines[:first]] == [f"f{i}" for i in range(first)]
    assert lines[first] == f"- ({30 - first - recent} older findings omitted)"
    assert lines[-1].endswith(": f29")
    assert lines[first + 1].endswith(f": f{30 - recent}")


def test_an_unreadable_log_never_breaks_memory(ctx, tmp_path):  # noqa: F811
    runtime._DATA_DIR_OVERRIDE = tmp_path / "missing" / "dir"
    state, verifier = _state("s1")
    assert runtime._prior_findings_block(state, verifier) == ""


def test_the_review_prompt_carries_the_memory(ctx):  # noqa: F811
    _write("s1", [{"accepted": True, "severity": "caution", "finding": "said fixed, not checked",
                   "origin_turn": 0}])
    state, _ = _state("s1")
    ctx.llm.next = FakeResult({"severity": "none", "finding": "", "reasoning": "ok"})
    _review_sync(state, "USER: hi\nASSISTANT: the verifier flagged that earlier")
    prompt = ctx.llm.calls[-1]["system_prompt"]
    assert prompt.startswith("You are the Verifier")
    assert "Your earlier findings in THIS session" in prompt
    assert "said fixed, not checked" in prompt


def test_no_memory_block_without_prior_findings(ctx):  # noqa: F811
    state, _ = _state("s1")
    ctx.llm.next = FakeResult({"severity": "none", "finding": "", "reasoning": "ok"})
    _review_sync(state, "USER: hi\nASSISTANT: hello")
    assert "Your earlier findings in THIS session" not in ctx.llm.calls[-1]["system_prompt"]
