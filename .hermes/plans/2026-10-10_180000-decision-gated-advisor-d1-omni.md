> **SUPERSEDED.** Kept for the record. The D1-omni / dynamic-tier direction
> this describes was abandoned on measured evidence -- `d1-omni-600M` could not
> separate a paraphrase from a distinct finding, and the semantic dedupe stage it
> proposed is disabled and documented UNVALIDATED in `sidecar/NEGATIVE-RESULTS.md`.
> The current work is in `2026-10-11_074500-closeout-v0.4.0.md`.

# Plan — Decision-Gated Advisor with a Continuous Evaluation Loop

## Goal

Gate the advisor with a local decision model (LiquidAI `d1-omni-600M`), then **continuously measure whether its decisions are good** and let per-family request-format preferences be **learned from accumulated outcomes** rather than fixed by assumption.

## Current context / assumptions

HEAD `f09ff95`, 114 tests, gates clean. Advisor on `deepseek/deepseek-flash`.

### Feasibility, verified

| Question | Answer | Evidence |
| --- | --- | --- |
| Does `ctx.llm` accept reasoning/effort? | **No** | literal search on the resolved path `~/.hermes/hermes-agent/agent/plugin_llm.py` (30,353 bytes) → **0** matches |
| Does effort reach the DeepSeek wire? | **Yes** | `thinking_toggle_extras` → `plugins/model-providers/deepseek/__init__.py:13,41`; emits top-level `reasoning_effort` |
| DeepSeek effort vocabulary | `low, medium, high, max` | `DEEPSEEK_V4_EFFORTS` |
| Plugin-reachable effort route | **Yes** | `_AUX_ROUTE_KEYS` includes `reasoning_effort`; `ctx.register_auxiliary_task()` |
| Plugin-owned isolated runtime | **Yes** | `python_runtime` manifest field (`plugins_manifest.py:38`) |
| Durable plugin storage | **Yes** | `plugin_data_dir()`, `plugin_db()` |
| Hermes core depends on `transformers`? | Not directly | only via optional TTS extras (`pyproject.toml:565`) |
| Installed vs needed | `transformers 5.7.0`, needs **`>=5.15`** — confirmed on PyPI (5.15.1 exists; latest 5.19.0) | d1-omni card + PyPI |

### d1-omni API shape — verified against the model source, not the card prose

Checked this session by fetching the repo directly. The model ships custom code
(`modeling_d1.py`, `prompt.py`, `encoder.py`, `audio.py`, `vision.py`), so
`trust_remote_code=True` is required and correct.

| Claim | Source | Status |
| --- | --- | --- |
| `system_one(state, questions: Mapping, images=None, audio=None)` | `modeling_d1.py:66` | ✓ |
| returns `{"answers": {…}, "usage": {"input_tokens": n, "output_tokens": 0}}` | `modeling_d1.py:78` | ✓ zero output tokens confirmed in source |
| `probabilities(state, questions: **Sequence**, …)` | `modeling_d1.py:81` | ✓ — **see bug 1** |
| `QTYPES = {"choice": 0, "score": 1, "noul": 2}` | `prompt.py:19` | ✓ |
| `noul` answer → `{"type": "noul", "noul": probs[0]}` | `prompt.py:117` | ✓ matches the gate predicate |
| `score` answer → `{"score": sum(i * p), "confidence": …}` | `prompt.py:124` | ✓ — **see bug 2** |

**Bug 1 — `probabilities` does not take the same shape as `system_one`.**
`system_one` takes a `Mapping` of named questions; `probabilities` takes a
`Sequence`. Passing the dict would iterate its keys and feed bare strings through
`as_question`.

**Bug 2 — `score` is an expected value, not an index.** `sum(i * p for i, p in
enumerate(probs))` yields a float such as 1.4, so tier mapping must bucket it,
not switch on an integer.

Both are corrected in the tasks below.

### What d1-omni-600M is, and why it fits

A **587M decision model**: give it a state (text/JSON, optionally image or audio) and named questions, and it returns typed answers — `noul` (yes/no), `choice`, or `score` — read from the distribution. **Zero output tokens, no generation, no parsing.** One forward pass answers several questions, and `probabilities()` exposes the raw distribution for confidence thresholds.

A gate, not a judge. Decision Index is modest (15.95) but the triage-relevant sub-scores are strong: Civil Comments 95.8, MASSIVE intent 86.1, PAWS-X 79.5. Recommended for "routing and triage, moderation, intent classification, agent guardrails".

| User ask | d1-omni question | type |
| --- | --- | --- |
| **When** to review | "Is there anything in this delta a reviewer should examine?" | `noul` |
| **How hard** | "How much review effort does this warrant?" | `score` (0–2) |
| **Which formatting** | "Which format does this advisor family handle best?" | `choice` |

### Why this revision exists

The first draft decided things — skip or review, which tier, which format — and never checked whether any decision was good. That is the failure mode this project already hit repeatedly: acting on an assumption and reporting it as a result.

Gating has a structural trap of its own: **a skipped review produces no evidence.** If the gate says "nothing here" and is wrong, nothing observable happens. The loop below exists to manufacture that missing evidence deliberately.

---

## Architecture / proposed approach

Run d1-omni as a **local sidecar** (own venv, `python_runtime: external`) behind a small HTTP endpoint, because it needs `transformers>=5.15` and must not force an upgrade of Hermes's own environment. Around it, a closed loop:

```
  decide ──▶ act ──▶ observe ──▶ score ──▶ adapt
    │         │         │          │         │
 journal   gate /   shadow     proxies   per-family
  + conf   tier /   sampling   (parse,   format
           format              accept,   preference,
                             dedupe)    behind a floor
```

1. **Journal** every decision with its inputs, answer, confidence, and raw probabilities.
2. **Shadow-sample** a fraction of *skips*, overriding them to review anyway — the only source of ground truth about skipped turns.
3. **Score** outcomes with honest proxies; never "was this finding useful", which is unobservable.
4. **Adapt** format-per-family from accumulated scores, with a minimum-evidence floor, exploration, a recency window, and a report-only default.

---

## Step-by-step tasks

### Phase 0 — Baseline

```bash
cd ~/hermes-verifier && python -m pytest -q     # expect: 114 passed
```

### Phase 1 — Sidecar (prove it runs, or stop)

#### Task 1.1 — d1-omni in its own venv

```bash
mkdir -p ~/hermes-verifier/sidecar && cd ~/hermes-verifier/sidecar
python3 -m venv .venv && . .venv/bin/activate
pip install "transformers>=5.15" torch torchvision pillow
python -c "import transformers; print('transformers', transformers.__version__)"
```

Expected: `transformers 5.15+` **in this venv only**. The shared environment keeps 5.7.0, so Hermes's optional TTS path is untouched.

#### Task 1.2 — Decision round-trip with probabilities

**File: `sidecar/probe.py`**

```python
"""Prove d1-omni answers a multi-question set, with confidence, at zero output cost."""
from transformers import AutoModel

model = AutoModel.from_pretrained("LiquidAI/d1-omni-600M",
                                  trust_remote_code=True, dtype="float32")
state = "[assistant] rewrite /etc/hosts then chmod 000 it"
qs = {
    "needs_review": {"type": "noul",
                     "instructions": "Is there anything here a reviewer should examine?"},
    "effort": {"type": "score",
               "instructions": "How much review effort does this warrant?",
               "criteria": ["Routine", "Worth a look", "Needs a hard look"]},
}
print(model.system_one(state, qs))
# NOTE: probabilities takes a SEQUENCE, not the Mapping system_one takes
# (modeling_d1.py:81 wraps each item with as_question). Passing `qs` directly
# would iterate its keys.
print("probs:", model.probabilities(state, list(qs.values())))
```

```bash
cd ~/hermes-verifier/sidecar && python probe.py
```

Expected: answers plus raw distributions, `output_tokens: 0`. **The probabilities
are load-bearing** — without them the gate cannot be conservative. Stop here if
they are unavailable or the shapes disagree with the table above.

#### Task 1.3 — HTTP endpoint

**File: `sidecar/server.py`** — stdlib `http.server`, one `POST /decide` taking `{"state": "..."}` and returning the answers JSON. Bind `127.0.0.1` only.

```bash
cd ~/hermes-verifier/sidecar && . .venv/bin/activate && python server.py &
sleep 2
curl -s -X POST 127.0.0.1:8765/decide -d '{"state":"chmod 000 /etc/hosts"}' | head -c 400
```

Expected: JSON with the answers and `output_tokens: 0`.

**Commit:** `feat(sidecar): local d1-omni decision endpoint with probabilities`

#### Task 1.4 — Make the endpoint configurable and its death visible

Gap found during the pre-execution scan. The gate fails **open**, so a sidecar
that is not running degrades to "always review" — correct behaviour, but
**silent**. Nothing would distinguish a working gate from a dead one, which is
the same invisible-failure class the decision journal exists to prevent.

**File: `plugin.yaml`** — declare the connection:

```yaml
  gate_url:
    type: str
    default: "http://127.0.0.1:8765/decide"
    description: "Local decision-model endpoint. Empty disables gating entirely (always review)."
  gate_timeout:
    type: float
    default: 2.0
    description: "Seconds to wait for a decision. On timeout the review proceeds."
```

**File: `gate.py`** — read from config rather than the module constant, and track
reachability so the state is reportable:

```python
_LAST_STATUS: str = "unknown"      # "ok" | "unreachable" | "disabled"


def last_status() -> str:
    """Reachability of the last decision attempt, for /verify decisions."""
    return _LAST_STATUS
```

`ask()` sets `_LAST_STATUS = "ok"` on success and `"unreachable"` on any failure,
so Phase 6 can print it. **A gate that has been unreachable for every turn is
indistinguishable from a gate that never fires — the report must say which.**

```python
def test_gate_reports_unreachable(monkeypatch):
    """A dead sidecar must be visibly dead, not just silently permissive."""
    from verifier_pkg import gate
    monkeypatch.setattr(gate, "_url", lambda: "http://127.0.0.1:1/nope")
    assert gate.ask("x", timeout=0.2) is None
    assert gate.last_status() == "unreachable"
    assert gate.decide(None)[0] is True, "still reviews when unreachable"
```

```bash
python -m pytest tests/test_runtime.py -k "gate_reports_unreachable" -q
```

Expected: `1 failed` → then `1 passed`.

**Commit:** `feat: configurable gate endpoint with explicit reachability status`

#### Task 1.5 — Document how the sidecar is run

**File: `sidecar/README.md`** and a `sidecar/run.sh` that starts the server on
`gate_url`'s port. The plan does **not** add a supervisor: a launchd/systemd unit
is deployment-specific, and inventing one here would be configuration the
operator did not ask for. What it does add is the requirement that the sidecar be
started at all, and the reachability status that makes forgetting it detectable.

```bash
cd ~/hermes-verifier/sidecar && ./run.sh &
sleep 3
curl -s -X POST 127.0.0.1:8765/decide -d '{"state":"echo hi"}' | head -c 200
```

Expected: a decision JSON, and `/verify decisions` subsequently reporting
`gate: ok`. If the sidecar is stopped, the same command must report
`gate: unreachable` — verified by stopping it and re-running.

**Commit:** `docs(sidecar): how to run it, and how to tell it is not running`

---

### Phase 2 — Decision journal (make every decision visible)

#### Task 2.1 — Failing test

**File: `tests/test_runtime.py`** — append:

```python
def test_every_gate_decision_is_journalled(ctx, tmp_path):
    """A gate decision must be recorded even when it skips.

    Skipped reviews are unobservable by construction; without a journal there
    is no way to tell a working gate from a stuck one.
    """
    from verifier_pkg import decisions
    decisions.bind_dir(tmp_path)

    decisions.record(kind="gate", session_id="s1", state_digest="abc",
                     answer={"needs_review": {"noul": 0.97}},
                     action="skip", confidence=0.97, decision_id="d1")

    rows = decisions.read_all()
    assert len(rows) == 1
    assert rows[0]["action"] == "skip"
    assert rows[0]["confidence"] == 0.97
    assert rows[0]["decision_id"] == "d1"


def test_journal_records_review_decisions_too(ctx, tmp_path):
    from verifier_pkg import decisions
    decisions.bind_dir(tmp_path)
    decisions.record(kind="gate", session_id="s1", state_digest="d",
                     answer={}, action="review", confidence=None, decision_id="d2")
    assert "review" in [r["action"] for r in decisions.read_all()]
```

```bash
python -m pytest tests/test_runtime.py -k "journalled or journal_records" -q
```

Expected: `2 failed` — no module `decisions`.

#### Task 2.2 — Implement

**File: `decisions.py`**

```python
"""Decision journal: every gating choice, its inputs, confidence, and outcome link.

Append-only JSONL under the plugin data dir. A skipped review leaves no other
trace, so this file is both the evidence that the gate ran and the base every
later score is computed from.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes-verifier.decisions")

_DIR: Optional[Path] = None


def bind_dir(path: Path) -> None:
    """Test seam."""
    global _DIR
    _DIR = Path(path)


def _path() -> Path:
    if _DIR is not None:
        return _DIR / "decisions.jsonl"
    from plugins.plugin_storage import plugin_data_dir
    return plugin_data_dir("hermes-verifier") / "decisions.jsonl"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def record(*, kind: str, session_id: str, state_digest: str,
           answer: Dict[str, Any], action: str, decision_id: str = "",
           confidence: Optional[float] = None, tier: str = "", fmt: str = "",
           shadow: bool = False, **extra: Any) -> None:
    """Append one decision or outcome row. Never raises."""
    row = {
        "ts": time.time(), "kind": kind, "session_id": session_id,
        "state_digest": state_digest[:200], "answer": answer, "action": action,
        "decision_id": decision_id or new_id(), "confidence": confidence,
        "tier": tier, "format": fmt, "shadow": shadow,
    }
    row.update(extra)
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError as exc:
        logger.warning("decision journal write failed: %s", exc)


def read_all(limit: int = 20000) -> List[Dict[str, Any]]:
    try:
        lines = _path().read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return []
    out: List[Dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
```

```bash
python -m pytest tests/test_runtime.py -k "journalled or journal_records" -q
```

Expected: `2 passed`.

**Commit:** `feat: decision journal for every gate choice`

---

### Phase 3 — Shadow sampling (manufacture the missing evidence)

#### Task 3.1 — Failing test

```python
def test_shadow_sampling_overrides_a_skip(ctx, monkeypatch):
    """A sampled skip must actually be reviewed, and marked as shadow.

    Without this a wrong 'skip' is unfalsifiable: nothing observable happens.
    """
    from verifier_pkg import gate
    monkeypatch.setattr(gate, "_shadow_draw", lambda: 0.0)     # always sample
    review, shadow = gate.decide({"needs_review": {"noul": 0.99}}, shadow_rate=0.1)
    assert review is True and shadow is True


def test_shadow_never_overrides_an_existing_review(ctx, monkeypatch):
    """Shadowing is about skips only."""
    from verifier_pkg import gate
    monkeypatch.setattr(gate, "_shadow_draw", lambda: 0.0)
    review, shadow = gate.decide({}, shadow_rate=0.1)         # would already review
    assert review is True and shadow is False


def test_shadow_disabled_by_default(ctx, monkeypatch):
    from verifier_pkg import gate
    monkeypatch.setattr(gate, "_shadow_draw", lambda: 0.0)
    review, shadow = gate.decide({"needs_review": {"noul": 0.99}})   # rate omitted
    assert review is False and shadow is False
```

```bash
python -m pytest tests/test_runtime.py -k "shadow_sampling or shadow_never or shadow_disabled" -q
```

Expected: `3 failed` — `decide` does not exist.

#### Task 3.2 — Implement

**File: `gate.py`** — extend the Task-2 gate:

```python
import random

SHADOW_RATE_DEFAULT = 0.0     # off unless the operator opts in


def _shadow_draw() -> float:
    """Test seam for the sampling coin."""
    return random.random()


def decide(answers: Optional[Dict[str, Any]],
           *, shadow_rate: float = SHADOW_RATE_DEFAULT) -> "tuple[bool, bool]":
    """``(should_review, was_shadow)``.

    A skip is overridden to a review with probability `shadow_rate`, which is
    how skipped turns get ground truth. It costs real money, so it is off by
    default and the rate is the operator's dial.
    """
    if should_review(answers):
        return True, False
    if shadow_rate > 0 and _shadow_draw() < shadow_rate:
        return True, True
    return False, False
```

`should_review` remains the pure predicate from Phase 2.

**Commit:** `feat: shadow sampling gives skipped decisions ground truth`

#### Task 3.3 — Make sampling targeted and self-retiring, not a flat tax

**Why this task exists.** "If we have the decision model, why shadow-sample at all?"
The answer is that they do different jobs: the model **decides**, sampling
**measures whether the decision was right**. d1-omni's probability is its own
confidence — not evidence that skipping is safe — and a *confidently wrong* skip
produces no signal whatsoever. Without sampling the gate is unfalsifiable, and an
unfalsifiable gate is just a cost cut at an unknown exchange rate; the simpler
design would be to not gate at all.

But a flat 10% forever is a bad answer to that, and the objection is right about
this much: sampling should not be a permanent tax. Two changes make it cheap:

1. **Target the uncertain skips.** Errors concentrate just past the threshold.
   Sampling a `p=0.995` skip is nearly wasted; sampling a `p=0.91` skip is where
   the information is.
2. **Retire as trust is earned.** Start high, decay toward a small maintenance
   floor, and re-raise when the measured miss rate drifts.

```python
def shadow_rate_for(*, confidence: Optional[float], base: float,
                    miss_rate: Optional[float], n: int) -> float:
    """Sampling rate for one skip: targeted near the threshold, decaying with evidence.

    * below MIN_SAMPLE_CONFIDENCE the skip is too uncertain to trust: always sample
    * as n grows and miss_rate stays low, decay toward MAINTENANCE_FLOOR
    * a miss_rate above DRIFT_ALARM raises the rate again
    """
```

```python
def test_shadow_targets_uncertain_skips():
    """A marginal skip is sampled far more than a confident one."""
    from verifier_pkg import gate
    marginal = gate.shadow_rate_for(confidence=0.90, base=0.1, miss_rate=0.0, n=1000)
    certain  = gate.shadow_rate_for(confidence=0.999, base=0.1, miss_rate=0.0, n=1000)
    assert marginal > certain, "uncertainty should attract samples"


def test_shadow_retires_with_evidence():
    """Once the gate has proved itself, sampling becomes a maintenance floor."""
    from verifier_pkg import gate
    fresh  = gate.shadow_rate_for(confidence=0.999, base=0.1, miss_rate=0.0, n=0)
    proven = gate.shadow_rate_for(confidence=0.999, base=0.1, miss_rate=0.0, n=500)
    assert proven < fresh, "evidence should reduce the tax"
    assert proven >= gate.MAINTENANCE_FLOOR, "never retire entirely; drift is undetectable without it"


def test_shadow_rises_on_drift():
    """A rising miss rate must raise sampling again."""
    from verifier_pkg import gate
    calm   = gate.shadow_rate_for(confidence=0.999, base=0.1, miss_rate=0.0,  n=500)
    drift  = gate.shadow_rate_for(confidence=0.999, base=0.1, miss_rate=0.4,  n=500)
    assert drift > calm, "drift must be re-measured, not assumed away"
```

Constants, each justified: `MAINTENANCE_FLOOR = 0.01` (never zero — with zero
sampling, drift is undetectable by construction), `MIN_SAMPLE_CONFIDENCE = 0.95`
(anything under this is sampled always), `DRIFT_ALARM = 0.1` (a 10% miss rate
means the gate is guessing).

```bash
python -m pytest tests/test_runtime.py -k "shadow_targets or shadow_retires or shadow_rises" -q
```

Expected: `3 failed` → then `3 passed`.

**Commit:** `feat: targeted, self-retiring shadow sampling instead of a flat rate`

---

### Phase 3b — Effort tier from the gate (the "how hard" answer)

The refinement above lost this phase; the original ask was "when it is called
**and what reasoning level it receives**", so it belongs here.

`score` is an **expected value**: `prompt.py:124` returns
`sum(i * p for i, p in enumerate(probs))`, a float such as 1.4 — **not an index**.
Tier mapping must bucket it and must require confidence before trusting it.

#### Task 3b.1 — Failing tests

```python
def test_effort_buckets_the_expected_score():
    """score is an expected level (float), not an argmax index."""
    from verifier_pkg import gate
    # Low expected level -> routine.
    assert gate.tier_for({"effort": {"score": 0.2, "confidence": 0.9}}) == "routine"
    # High expected level -> deep.
    assert gate.tier_for({"effort": {"score": 1.9, "confidence": 0.9}}) == "deep"
    # Mid, near the boundary -> the safer tier.
    assert gate.tier_for({"effort": {"score": 1.5, "confidence": 0.9}}) == "deep"


def test_effort_requires_confidence():
    """An unsure effort answer must not downgrade the review."""
    from verifier_pkg import gate
    assert gate.tier_for({"effort": {"score": 0.1, "confidence": 0.35}}) == "deep", \
        "a low-confidence 'routine' must not save money by guessing"


def test_effort_missing_falls_back_to_deep():
    """Absent or malformed effort -> the thorough tier, never the cheap one."""
    from verifier_pkg import gate
    assert gate.tier_for({}) == "deep"
    assert gate.tier_for(None) == "deep"
    assert gate.tier_for({"effort": {}}) == "deep"
```

```bash
python -m pytest tests/test_runtime.py -k "effort_buckets or effort_requires or effort_missing" -q
```

Expected: `3 failed` — `tier_for` does not exist.

#### Task 3b.2 — Implement

**File: `gate.py`** — add:

```python
# Expected-score boundaries on a 0..2 ladder. ROUTINE at or below the low mark,
# DEEP at or above the high mark; between them the safer tier wins.
TIER_LOW = 0.75
TIER_HIGH = 1.5
TIER_CONFIDENCE = 0.6      # below this the answer is not trusted


def tier_for(answers: Optional[Dict[str, Any]]) -> str:
    """'routine' | 'deep' from the gate's expected-effort score.

    Fail-thorough: anything missing, malformed, or low-confidence returns 'deep'.
    The asymmetry is deliberate -- an unnecessary deep review costs tokens, a
    wrongly-downgraded review costs findings.
    """
    if not isinstance(answers, dict):
        return "deep"
    eff = answers.get("effort")
    if not isinstance(eff, dict):
        return "deep"
    score = eff.get("score")
    conf = eff.get("confidence")
    if not isinstance(score, (int, float)):
        return "deep"
    if isinstance(conf, (int, float)) and conf < TIER_CONFIDENCE:
        return "deep"
    if score >= TIER_HIGH:
        return "deep"
    if score <= TIER_LOW:
        return "routine"
    return "deep"
```

```bash
python -m pytest tests/test_runtime.py -k "effort_buckets or effort_requires or effort_missing" -q
```

Expected: `3 passed`.

#### Task 3b.3 — Route the tier through an auxiliary task

Register one effort-bearing tier and route the review through it, so the gate's
answer actually changes reasoning effort:

```python
# __init__.py, inside register(ctx)
ctx.register_auxiliary_task(
    "advisor_deep", display_name="Advisor (deep)",
    description="High-effort advisor pass, used when the gate asks for it.")
```

Operator side, documented in `plugin.yaml`:

```
hermes config set auxiliary.advisor_deep.reasoning_effort high
```

**Stop condition:** if the effort does not appear on the wire (no
`reasoning_effort` in the request log for a deep-tier review), the tier route is
inert — record that and do not ship a tier that does nothing.

```bash
python -m pytest -q
```

**Commit:** `feat: gate-driven effort tier via an auxiliary task, fail-thorough`

---

### Phase 4 — Outcome scoring (observable proxies only)

#### Task 4.1 — Failing test

```python
def test_outcomes_use_proxies_not_opinions():
    """Score only what is observable. 'Was the finding useful' is not.

    Proxies: did the response parse, did the guard accept it, was it a
    duplicate, what did it cost.
    """
    from verifier_pkg import score
    rows = [
        {"severity": "caution", "accepted": True,  "parse_ok": True,  "deduped": False},
        {"severity": "none",    "accepted": False, "parse_ok": False, "deduped": False},
        {"severity": "note",    "accepted": False, "parse_ok": True,  "deduped": True},
    ]
    agg = score.aggregate(rows)
    assert agg["n"] == 3
    assert agg["parse_rate"] == pytest.approx(2 / 3)
    assert agg["accept_rate"] == pytest.approx(0.5)     # of the 2 that parsed
    assert agg["dedupe_rate"] == pytest.approx(0.5)


def test_shadow_skip_joins_to_its_outcome():
    """A shadowed skip must be joinable to whatever the review produced.

    shadow_hits is the number the whole loop exists to produce: the measured
    cost of wrong skips.
    """
    from verifier_pkg import score
    rows = [
        {"kind": "gate", "decision_id": "d1", "action": "skip", "shadow": True},
        {"kind": "review", "decision_id": "d1", "severity": "caution",
         "parse_ok": True, "accepted": True},
        {"kind": "gate", "decision_id": "d2", "action": "skip", "shadow": True},
        {"kind": "review", "decision_id": "d2", "severity": "none",
         "parse_ok": True, "accepted": False},
    ]
    joined = score.join(rows)
    assert joined["shadowed"] == 2
    assert joined["shadow_hits"] == 1, "d1 found something: a gate miss"
```

```bash
python -m pytest tests/test_runtime.py -k "outcomes_use_proxies or shadow_skip_joins" -q
```

Expected: `2 failed` — no module `score`.

#### Task 4.2 — Implement

**File: `score.py`**

```python
"""Outcome scoring from observable proxies, and the shadow-hit join.

Deliberately narrow. The honest signal set is parse / accept / dedupe / tokens.
'Was the finding useful to the primary agent' is NOT observable from inside the
plugin and must not be invented.
"""

from __future__ import annotations

from typing import Any, Dict, List


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    parsed = [r for r in rows if r.get("parse_ok")]
    accepted = [r for r in parsed if r.get("accepted")]
    deduped = [r for r in parsed if r.get("deduped")]
    outs = [r.get("output_tokens") or 0 for r in rows]
    return {
        "n": n,
        "parse_rate": (len(parsed) / n) if n else 0.0,
        "accept_rate": (len(accepted) / len(parsed)) if parsed else 0.0,
        "dedupe_rate": (len(deduped) / len(parsed)) if parsed else 0.0,
        "mean_out_tokens": (sum(outs) / n) if n else 0.0,
    }


def join(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Link gate decisions to the reviews they produced.

    `shadow_hits` counts shadowed skips that turned out to have something in
    them — i.e. gate misses, measured rather than assumed.
    """
    outcomes: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("kind") == "review":
            outcomes[r.get("decision_id") or ""] = r
    shadowed = hits = 0
    for r in rows:
        if r.get("kind") == "gate" and r.get("action") == "skip" and r.get("shadow"):
            shadowed += 1
            out = outcomes.get(r.get("decision_id") or "")
            if out and out.get("accepted"):
                hits += 1
    return {"shadowed": shadowed, "shadow_hits": hits,
            "shadow_hit_rate": (hits / shadowed) if shadowed else 0.0}
```

**Commit:** `feat: proxy scoring plus the shadow-hit join (measured gate miss rate)`

#### Task 4.3 — Wire the proxies into the review path

Record `parse_ok` (did `parsed` arrive as a dict), `accepted` (guard verdict),
`deduped`, and `output_tokens` on every review row, and carry the originating
`decision_id` so the join works in production, not only in tests.

```bash
python -m pytest -q
```

**Commit:** `feat: attach observable outcome proxies and decision_id to reviews`

---

### Phase 5 — Per-family format preferences, learned

#### Task 5.1 — Failing tests: floor, exploration, window

```python
def test_format_preference_requires_evidence(tmp_path):
    """Do not prefer a format until a floor of samples exists.

    MIN_N=5, set from this project's own failure: a 3-sample experiment was
    read as a result and was not.
    """
    from verifier_pkg import prefs
    prefs.bind_dir(tmp_path)

    prefs.observe(family="deepseek", fmt="tagged", parse_ok=True, accepted=True)
    assert prefs.best_format("deepseek") is None, "n=1 must not decide"

    for _ in range(6):
        prefs.observe(family="deepseek", fmt="tagged", parse_ok=True, accepted=True)
        prefs.observe(family="deepseek", fmt="plain", parse_ok=True, accepted=False)
    assert prefs.best_format("deepseek", explore=False) == "tagged"


def test_unknown_family_falls_back(tmp_path):
    from verifier_pkg import prefs
    prefs.bind_dir(tmp_path)
    assert prefs.best_format("never-seen") is None


def test_stale_observations_are_ignored(tmp_path, monkeypatch):
    """Preferences expire: providers change models under us."""
    from verifier_pkg import prefs
    prefs.bind_dir(tmp_path)
    monkeypatch.setattr(prefs.time, "time", lambda: 0.0)          # all rows ancient
    for _ in range(6):
        prefs.observe(family="f", fmt="tagged", parse_ok=True, accepted=True)
    monkeypatch.undo()
    assert prefs.best_format("f") is None, "rows outside the window must not count"
```

```bash
python -m pytest tests/test_runtime.py -k "format_preference or unknown_family or stale_observations" -q
```

Expected: `3 failed` — no module `prefs`.

#### Task 5.2 — Implement

**File: `prefs.py`**

```python
"""(family, format) outcome tallies with a floor, exploration, and a recency window.

Each rule traces to a specific past failure:
  * MIN_N = 5        -- below the floor, return None            (the n=3 mistake)
  * EPSILON explore  -- never stop sampling the alternative, or a lucky early
                        win locks the choice in permanently
  * WINDOW_DAYS      -- preferences expire; providers change under us
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

MIN_N = 5
EPSILON = 0.15
WINDOW_DAYS = 30

_DIR: Optional[Path] = None


def bind_dir(path: Path) -> None:
    global _DIR
    _DIR = Path(path)


def _path() -> Path:
    if _DIR is not None:
        return _DIR / "format_obs.jsonl"
    from plugins.plugin_storage import plugin_data_dir
    return plugin_data_dir("hermes-verifier") / "format_obs.jsonl"


def observe(*, family: str, fmt: str, parse_ok: bool, accepted: bool) -> None:
    row = {"ts": time.time(), "family": family, "format": fmt,
           "parse_ok": bool(parse_ok), "accepted": bool(accepted)}
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        return


def _load(family: str) -> List[Dict[str, Any]]:
    try:
        lines = _path().read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return []
    cutoff = time.time() - WINDOW_DAYS * 86400
    out = []
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("family") == family and r.get("ts", 0) >= cutoff:
            out.append(r)
    return out


def _score(rows: List[Dict[str, Any]]) -> float:
    """A format that fails to parse is worse than one that parses weakly."""
    if not rows:
        return 0.0
    parse = sum(1 for r in rows if r.get("parse_ok")) / len(rows)
    parsed = [r for r in rows if r.get("parse_ok")]
    acc = (sum(1 for r in parsed if r.get("accepted")) / len(parsed)) if parsed else 0.0
    return 0.5 * parse + 0.5 * acc


def tally(family: str) -> Dict[str, Dict[str, Any]]:
    per: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in _load(family):
        per[r.get("format") or ""].append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for fmt, rows in per.items():
        parsed = [r for r in rows if r.get("parse_ok")]
        out[fmt] = {
            "n": len(rows),
            "parse_rate": sum(1 for r in rows if r.get("parse_ok")) / max(1, len(rows)),
            "accept_rate": (sum(1 for r in parsed if r.get("accepted")) / len(parsed))
                           if parsed else 0.0,
            "score": _score(rows),
        }
    return out


def best_format(family: str, *, explore: bool = True) -> Optional[str]:
    """Learned winner, or None below the evidence floor."""
    stats = tally(family)
    if not stats or max(s["n"] for s in stats.values()) < MIN_N:
        return None
    if explore and random.random() < EPSILON:
        # Sample an alternative rather than locking in the current leader.
        ranked = sorted(stats.items(), key=lambda kv: (-kv[1]["score"], -kv[1]["n"]))
        return ranked[-1][0] if len(ranked) > 1 else ranked[0][0]
    ranked = sorted(stats.items(), key=lambda kv: (-kv[1]["score"], -kv[1]["n"]))
    return ranked[0][0] if ranked else None
```

```bash
python -m pytest -q
```

**Commit:** `feat: per-family format preferences with floor, exploration, and recency window`

#### Task 5.3 — Report-only by default

New setting `format_learning: off | report | apply`, defaulting to **`report`**:
the plugin computes the preference, records it, and shows it in `/verify status`,
but keeps rendering the current format until an operator sets `apply`.
Auto-applying an unmeasured loop is how a bad preference becomes invisible.

**File: `plugin.yaml`** — declare it:

```yaml
  format_learning:
    type: str
    default: "report"
    description: "off | report | apply. Learn which request format works best per advisor family from accumulated parse/accept outcomes. 'report' computes and shows the preference without changing behaviour."
  shadow_sample_rate:
    type: float
    default: 0.0
    description: "Fraction of gate skips to review anyway, for ground truth. 0 = off. Costs real tokens; 0.1 bounds the saving near 90%."
```

**Commit:** `feat: format learning defaults to report-only`

---

### Phase 6 — Reporting with sample sizes

#### Task 6.1 — `/verify decisions`

Show: gate action counts; **shadow hit rate** (measured wrong-skip rate); per-tier
outcomes; per-family format tallies.

**Every rate is printed with its `n` beside it.** A rate without a sample size is
precisely the artefact this project has repeatedly mistaken for a result.

```python
def test_decisions_report_always_shows_n(tmp_path):
    """/verify decisions must never print a bare rate."""
    from verifier_pkg import commands, decisions
    decisions.bind_dir(tmp_path)
    decisions.record(kind="gate", session_id="s1", state_digest="x",
                     answer={}, action="skip", shadow=True, decision_id="d1")
    text = "\n".join(commands._decisions_lines())
    assert "shadow" in text.lower()
    assert "n=" in text, f"a rate without its sample size is not reportable:\n{text}"
```

```bash
python -m pytest -q && hermes plugins validate . --install-deps
```

**Commit:** `feat: /verify decisions reports gate quality with sample sizes`

---

## Tests / validation

| After | `pytest -q` |
| --- | --- |
| Baseline | `114 passed` |
| 2.2 journal | `116 passed` |
| 3.2 shadow | `119 passed` |
| 3.3 adaptive shadow | `122 passed` |
| 3b.2 effort tier | `125 passed` |
| 4.2 scoring | `127 passed` |
| 5.2 format prefs | `130 passed` |
| 6.1 reporting | `131 passed` |

Gates: `python -m pytest -q`, `hermes plugins validate . --install-deps`, `hermes plugins doctor . --ci`.

## Risks, tradeoffs, and open questions

1. **Outcome labels are proxies, not truth.** `accepted` means the emission guard
   passed it, not that the primary agent found it useful. Every learning rule
   optimizes a proxy and can therefore optimize the wrong thing. This is a
   standing limitation to state in the README, not a bug to fix later.

2. **Shadow sampling costs the money the gate exists to save — but it is not a
   flat tax.** Task 3.3 makes it targeted (marginal skips attract samples,
   confident ones do not) and self-retiring (it decays toward a 1% maintenance
   floor as the gate earns trust, and rises again if the measured miss rate
   drifts). The floor is never zero: with zero sampling, drift is undetectable by
   construction and the loop silently stops being an evaluation loop.

3. **Learning is slow and per-family sparse.** MIN_N=5 per (family × format) cell
   across 2–4 formats and several families is dozens of reviews before anything is
   interesting. Early output will be "insufficient data" for a long time — which is
   the correct output, and the report must say so rather than show a rate.

4. **Non-stationarity.** Providers change models without notice; a 30-day window is
   a guess. If a preference scores differently across windows, it is not real.

5. **Self-reinforcing bias — the most serious one.** If the gate skips a class of
   turn and shadow sampling is off, that class is never reviewed and never scored,
   so the loop cannot discover it was wrong. **Shadow sampling is not a nicety; it
   is the only thing keeping the gate from hiding its own failures.** With it off,
   the evaluation loop measures nothing about skips and should say so.

6. **The gate sees a digest, not the delta.** Context is 16,384 tokens and text is
   cut to 896 with images, while deltas reach 20,000 chars. What goes in the digest
   is undefined and is the likeliest source of bad skips.

7. **d1-omni is weak overall** (Decision Index 15.95) though strong on
   classification. Suited to triage, not to judging code quality.

8. **`transformers>=5.15` is the hard blocker.** Installed is 5.7.0; the sidecar
   exists so the shared environment is untouched. If Task 1.2 fails, the design
   stops — it is the gate for the gate.

9. **Explicitly out of scope:** per-call reasoning on `ctx.llm` (still 0 matches),
   and any Hermes core change. Plugin + sidecar only.
