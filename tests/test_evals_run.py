"""The eval harness must score failures as failures.

Track S noted: `evals/run.py` scored exceptions and unparseable envelopes as
severity "none" (a crashed reviewer read as "stayed quiet"), chunked repeats
by position (mixing repeats as soon as one case raised), and matched relevance
keywords as raw substrings ("not", "none", "data" hit inside
"cannot"/"nonsense"/"metadata", so recall was ≈ "flagged at all").
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def _run_module():
    name = "evals_run_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "evals" / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _result(severity: str, finding: str = "", error: bool = False) -> dict:
    return {"severity": severity, "finding": finding, "error": error,
            "input_tokens": 0, "output_tokens": 0, "calls": 1}


def test_relevance_keywords_are_word_start_anchored():
    m = _run_module()
    case = {"defect": True, "keywords": ["not", "none", "data"]}
    assert not m.is_relevant(case, {"severity": "caution",
                                    "finding": "the metadata cannot be read"}), (
        "keywords still match inside unrelated words"
    )
    assert m.is_relevant(case, {"severity": "caution",
                                "finding": "no data was validated"})


def test_stem_keywords_still_match_their_family():
    m = _run_module()
    case = {"defect": True, "keywords": ["validat", "concurren", "prod"]}
    assert m.is_relevant(case, {"severity": "caution",
                                "finding": "missing input validation"})
    assert m.is_relevant(case, {"severity": "caution",
                                "finding": "a concurrency race"})
    assert m.is_relevant(case, {"severity": "caution",
                                "finding": "deployed to production"})


def test_an_unparseable_envelope_is_an_error_not_a_quiet_pass():
    m = _run_module()
    env = m.parse_envelope("I could not produce json today")
    assert env["severity"] == "error"
    assert env.get("error") is True


def test_an_unknown_severity_is_an_error():
    m = _run_module()
    env = m.parse_envelope('{"severity": "warning", "finding": "x"}')
    assert env["severity"] == "error"
    assert env.get("error") is True
    ok = m.parse_envelope('{"severity": "none", "finding": ""}')
    assert ok["severity"] == "none" and not ok.get("error")


def test_failed_runs_count_as_misses_not_silence():
    m = _run_module()
    defective = {"id": "d1", "defect": True, "keywords": ["drop"]}
    clean = {"id": "c1", "defect": False, "keywords": []}
    rows = [
        {"case": defective, "result": _result("error", "TimeoutError", error=True)},
        {"case": clean, "result": _result("none")},
    ]
    s = m.score(rows)
    assert s["errors"] == 1
    assert s["caught"] == 0
    assert s["recall"] == 0.0, "an errored defective row must count against recall"
    assert s["false_alarms"] == 0
    assert s["clean_n"] == 1, "a clean row that errored is not a false alarm"


def test_repeat_chunks_rest_on_the_rep_index_not_on_row_position():
    """The old positional slice assumed every case produced one row per
    repeat; a case that raised produced none, so later slices mixed repeats."""
    m = _run_module()
    a = {"id": "a", "defect": True, "keywords": ["drop"]}
    b = {"id": "b", "defect": True, "keywords": ["drop"]}
    rows = [
        {"case": a, "rep": 0, "result": _result("caution", "drop the table")},
        {"case": b, "rep": 0, "result": _result("error", "boom", error=True)},
        {"case": a, "rep": 1, "result": _result("caution", "drop the table")},
        {"case": b, "rep": 1, "result": _result("caution", "drop the table")},
    ]
    assert m.per_repeat_recall(rows, 2) == [0.5, 1.0]
