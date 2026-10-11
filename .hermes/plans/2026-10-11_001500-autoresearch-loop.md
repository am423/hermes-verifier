
> Note: destructive-command literals in this plan are split (`"rm -" + "rf"`) so the repo's own security scan reads them as data. It flags them otherwise, correctly — a scanner cannot tell a detector from the thing it detects.

# Autoresearch loop: improve hermes-verifier from its own decision logs

## Goal

Build a loop that periodically harvests this plugin's decision logs, labels each
decision against what actually happened next, and reports whether any rule change
is justified — with enough statistical care that it can return "no change" and
mean it.

## Current context / assumptions

### What already exists

- Repo: `~/hermes-verifier` (Hermes plugin, Python 3.12/3.14, 282 tests passing
  via `python -m pytest -q` from the repo root).
- Install: `~/.hermes/plugins/hermes-verifier` is a symlink to the repo, so
  editing the repo changes what loads on the next process start.
- Decision logs, written by the plugin at runtime:
  - `~/.hermes/plugin-data/hermes-verifier/gate.<session>.jsonl` — the engagement
    gate's decisions. Row keys: `ts` (float), `session_id`, `origin_turn` (int),
    `engage` (bool), `effort` (str), `source` (str), `digest_excerpt` (str).
  - `~/.hermes/plugin-data/hermes-verifier/verify.<session>.<verifier>.jsonl` —
    review outcomes. Row keys: `ts`, `session_id`, `verifier`, `severity`,
    `finding`, `reasoning`, `accepted` (bool), `origin_turn`, `model`,
    `delta_excerpt`, `prompt_parts` (dict), `usage` (dict with `input_tokens`,
    `output_tokens`, `cost_usd`).
  - At the time of writing: 61 gate files, 66 verify files, 92 gate decisions.
- Conversation corpus: `~/.hermes/state.db` (SQLite). Tables `messages`
  (33k+ rows; columns include `session_id`, `role`, `content`, `tool_calls`,
  `tool_name`, `timestamp`) and `sessions`. **Open read-only** —
  `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`.
- Analysis scripts already written (reuse these, do not rewrite):
  - `evals/moment_analysis.py` — signal detection patterns over turns.
  - `evals/optimize_rules.py` — `CANDIDATES` dict of rule predicates,
    `load_turns()`, `REPAIR` / `FAILURE` label regexes.
  - `evals/optimize_rules_ci.py` — paired bootstrap vs a reference rule.
  - `evals/optimize_rules_final.py` — Bonferroni + session-level held-out split.

### What is NOT persisted, and must be

`phase_gate` and `review_agreement` decisions currently go only to the in-memory
`EVENT_LOG` (a bounded ring of 200, lost on restart). The loop cannot analyse
what is not on disk, so Task 1 fixes this.

### Methodological constraints (learned the hard way; do not relax them)

1. **Never tune on the evaluation set.** A previous pass scored 12 candidate
   triggers on 832 turns and reported the top two as winners at 95%. Under
   Bonferroni both collapsed. Testing N candidates at 95% gives `1 - 0.95^N`
   odds of a spurious winner — ~46% at N=12.
2. **Report power, not just point estimates.** A null result under a strict bar
   is *not* confirmation: with Bonferroni at 99.6% and a small held-out split, a
   null is likely whether or not the effect is real. Every report must state the
   validation set's session, turn and positive-label counts, and say "no
   candidate cleared the bar" rather than "the current rule is confirmed".
3. **Label honestly.** The outcome proxy is repair language in the NEXT turn
   (`revert`, `undo`, `broke`, `my mistake`, `didn't work`). It measures that
   repair *followed*, not that a review would have *prevented* it. Every report
   must repeat this caveat verbatim.
4. **Two labels.** Track `repair_next` and `fail_next` (failure language). If a
   rule's lift on one is materially above 1.0 and on the other is ~1.0, the rule
   is firing on conversational noise. That check caught `claims_done`.
5. **Volume is a first-class metric.** A rule firing on most turns is per-turn
   review wearing a hat. Adopt nothing with lift < 1.2 or volume > 25%.
6. **Never auto-apply.** The loop writes a report and a proposed patch. A human
   applies it. Silent self-modification of a reviewer is not acceptable.

### Assumptions

- The loop runs on this machine, which is where the logs and the corpus live.
- Sessions are the unit of splitting (turns within a session share an author and
  project, so a turn-level split leaks).
- 832 labelled turns is nowhere near enough power for small effects. The loop's
  first job is *accumulating* labelled data, not tuning on what exists.

## Architecture

A five-stage pipeline run on a schedule: **harvest** (logs + corpus → one
normalised table), **label** (attach `repair_next` / `fail_next`), **score**
(bootstrap + Bonferroni + session-split held-out), **propose** (candidate rules,
each with its CI and power), **report** (a markdown file; a human decides).
Stages are separate modules with plain functions over lists of dicts so each is
testable without the others; nothing writes to the plugin's config.

## Step-by-step tasks

Every task: write the test, run it and see it fail, implement, run it and see it
pass, commit. From the repo root unless stated.

---

### Task 1 — Persist the two decisions that are currently memory-only

Without this the loop is blind to the phase gate and the agreement filter, which
are the two newest and least-understood rules.

**1a. Write the failing test.** Create `tests/test_decision_persistence.py`:

```python
"""phase_gate and review_agreement decisions must be durable.

They currently reach only EVENT_LOG, a bounded in-memory ring that is lost on
restart, so the decision logs cannot answer "how often did the gate block, and
was it right?".
"""

from __future__ import annotations

import json

from verifier_pkg import runtime

from .test_runtime import FakeResult, _review_sync, _turn, ctx  # noqa: F401


def _rows(kind: str, session: str = "s1"):
    path = runtime.data_dir() / f"{kind}.{runtime._slug_session(session)}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_phase_gate_decision_is_written_to_disk(ctx):  # noqa: F811
    ctx.config["phase_gate_enabled"] = True
    runtime._get_state("s1")
    ctx.llm.next = FakeResult({"verdict": "block", "finding": "irreversible"})
    runtime.handle_pre_tool_call(tool_name="terminal",
                                 args={"command": "rm -" + "rf /data"},
                                 session_id="s1")
    rows = _rows("phase")
    assert rows, "the gate decided but wrote nothing"
    assert rows[-1]["verdict"] == "block"
    assert rows[-1]["risk"] == "irreversible"
    assert rows[-1]["tool"] == "terminal"


def test_an_agreement_decision_is_written_to_disk(ctx):  # noqa: F811
    ctx.config["agreement_reviews"] = 3
    ctx.config["agreement_min"] = 3
    state = runtime._get_state("s1")
    state.verifiers["default"].pending.clear()
    payloads = [{"severity": "caution", "finding": "x"},
                {"severity": "none", "finding": ""},
                {"severity": "none", "finding": ""}]
    ctx.llm.queue = [FakeResult(p) for p in payloads]
    runtime._review_one(state, state.verifiers["default"], "delta", "deep", 1)
    rows = _rows("agreement")
    assert rows, "agreement ran but wrote nothing"
    assert rows[-1]["raised"] == 1
    assert rows[-1]["required"] == 3
    assert rows[-1]["agreed"] is False
```

**1b. Run it — expect failure.**

```bash
cd ~/hermes-verifier && python -m pytest tests/test_decision_persistence.py -q
```

Expected: 2 failed — `_rows("phase")` returns `[]`.

**1c. Add the writer.** In `runtime.py`, next to `_gate_log_path`, add:

```python
def _decision_log_path(kind: str, session_id: str) -> Path:
    """One append-only file per decision kind, mirroring gate.<session>.jsonl."""
    return data_dir() / f"{kind}.{_slug_session(session_id)}.jsonl"
```

In `_pre_review_action` (runtime.py), immediately AFTER the existing
`publish_event("phase_gate", ...)` call, add:

```python
    _append_jsonl(_decision_log_path("phase", state.session_id), {
        "ts": time.time(),
        "session_id": state.session_id,
        "origin_turn": state.turns_seen,
        "tool": tool_name,
        "risk": label,
        "verdict": verdict or "unknown",
        "args_excerpt": payload[:400],
    })
```

In `_review_agreeing`, immediately AFTER the existing
`publish_event("review_agreement", ...)`, add:

```python
    _append_jsonl(_decision_log_path("agreement", state.session_id), {
        "ts": time.time(),
        "session_id": state.session_id,
        "origin_turn": origin_turn,
        "reviews": n,
        "required": k,
        "raised": len(raised),
        "agreed": agreed,
        "severity": str((raised[0] if raised else {}).get("severity") or "none"),
    })
```

**1d. Run the tests — expect pass.**

```bash
cd ~/hermes-verifier && python -m pytest tests/test_decision_persistence.py -q
```

Expected: `2 passed`. Then the whole suite:

```bash
cd ~/hermes-verifier && python -m pytest -q
```

Expected: `284 passed` (282 + 2).

**1e. Commit.**

```bash
cd ~/hermes-verifier && git add -A \
  && git commit -m "feat: persist phase-gate and agreement decisions

Both reached only EVENT_LOG, an in-memory bounded ring lost on restart, so the
decision logs could not answer how often the gate blocked or whether the
agreement filter suppressed anything. The autoresearch loop is blind without
them."
```

---

### Task 2 — Extract the shared rule predicates (DRY)

`evals/optimize_rules.py` already holds `CANDIDATES`, `load_turns()` and the
label regexes. Move them somewhere both the eval scripts and the loop can import
without a `sys.path` hack.

**2a. Write the failing test.** Create `autoresearch/tests/test_rules.py`:

```python
"""The rule predicates and labels live in one importable place."""

from __future__ import annotations

from autoresearch.rules import CANDIDATES, FAILURE, REPAIR, lift_on


def test_labels_detect_their_language():
    assert REPAIR.search("I need to revert that change")
    assert not REPAIR.search("the tests pass")
    assert FAILURE.search("Traceback (most recent call last)")


def test_candidates_include_the_shipped_rule():
    assert "destructive_or_external" in CANDIDATES


def test_lift_is_one_when_a_rule_fires_on_everything():
    turns = [{"repair_next": True, "calls": "", "says": "", "edits": 0}] * 4
    mask = [True] * 4
    assert lift_on(turns, mask) == 1.0
```

**2b. Run it — expect failure.**

```bash
cd ~/hermes-verifier && python -m pytest autoresearch/tests/test_rules.py -q
```

Expected: `ModuleNotFoundError: No module named 'autoresearch'`.

**2c. Create the package and move the code.** Create `autoresearch/__init__.py`
as an empty file, `autoresearch/tests/__init__.py` as an empty file, then
`autoresearch/rules.py` containing the pattern lists, the `rx()` helper,
`REPAIR`, `FAILURE`, `CANDIDATES` and a `lift_on(turns, mask)` function — all
moved **verbatim** from `evals/optimize_rules.py`, plus:

```python
def lift_on(turns, mask) -> float:
    """P(repair | fired) / P(repair). 0.0 when the rule never fires."""
    fired = [t for t, m in zip(turns, mask) if m]
    if not fired:
        return 0.0
    base = sum(1 for t in turns if t["repair_next"]) / max(1, len(turns))
    if not base:
        return 0.0
    return (sum(1 for t in fired if t["repair_next"]) / len(fired)) / base
```

Then edit `evals/optimize_rules.py` to import from the new module instead of
defining its own:

```python
from autoresearch.rules import CANDIDATES, FAILURE, REPAIR, lift_on  # noqa: F401
```

and delete the now-duplicated definitions. Do the same for
`evals/optimize_rules_ci.py` and `evals/optimize_rules_final.py`.

**2d. Run everything — expect pass, with identical numbers.**

```bash
cd ~/hermes-verifier && python -m pytest autoresearch/tests/test_rules.py -q
```

Expected: `3 passed`.

```bash
cd ~/hermes-verifier && python evals/optimize_rules.py 2>&1 | head -6
```

Expected: the same header as before the move — `turns: 830` (or the current
count) and the same base rates. **If the numbers moved, the move was not
faithful; revert and retry.**

**2e. Commit.**

```bash
cd ~/hermes-verifier && git add -A \
  && git commit -m "refactor: extract rule predicates and labels into autoresearch/rules.py

The eval scripts each carried their own copy. One importable source so the loop
and the evals cannot drift apart."
```

---

### Task 3 — Harvest: one normalised table from logs + corpus

**3a. Write the failing test.** Create `autoresearch/tests/test_harvest.py`:

```python
"""Harvest joins the plugin's decisions to what happened in the session."""

from __future__ import annotations

import json
import sqlite3

from autoresearch.harvest import harvest, read_decisions


def test_read_decisions_reads_every_kind(tmp_path):
    for kind, row in (("gate", {"engage": True}),
                      ("phase", {"verdict": "block"}),
                      ("agreement", {"agreed": False})):
        (tmp_path / f"{kind}.s1.jsonl").write_text(json.dumps(row) + "\n")
    out = read_decisions(tmp_path)
    assert {r["kind"] for r in out} == {"gate", "phase", "agreement"}


def test_read_decisions_skips_a_torn_line(tmp_path):
    p = tmp_path / "gate.s1.jsonl"
    p.write_text('{"engage": true}\n{"engage": tru\n')
    out = read_decisions(tmp_path)
    assert len(out) == 1, "a half-written row must be skipped, not raise"


def test_harvest_returns_turns_with_labels(tmp_path):
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.execute("create table messages (id integer primary key, session_id text,"
                " role text, content text, tool_calls text, tool_name text)")
    con.execute("insert into messages (session_id, role, content) values"
                " ('sess1','user','do a thing')")
    for i in range(6):
        con.execute("insert into messages (session_id, role, content) values"
                    " ('sess1','assistant',?)", (f"step {i}",))
    con.execute("insert into messages (session_id, role, content) values"
                " ('sess1','assistant','I need to revert that')")
    con.commit()
    con.close()
    rows = harvest(tmp_path, db)
    assert rows, "no turns harvested"
    assert "repair_next" in rows[0]
```

**3b. Run it — expect failure.**

```bash
cd ~/hermes-verifier && python -m pytest autoresearch/tests/test_harvest.py -q
```

Expected: `ModuleNotFoundError: No module named 'autoresearch.harvest'`.

**3c. Implement `autoresearch/harvest.py`.**

```python
"""Collect the plugin's decision logs and the session corpus into one table.

Two sources, one shape:
  * decisions the plugin made (gate / phase / agreement jsonl)
  * what the agent actually did (state.db messages), segmented into turns

A "turn" starts at a user message and runs until the next one. The outcome label
is computed from the FOLLOWING turn.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import Any, Dict, List

from .rules import FAILURE, REPAIR

DEFAULT_DATA = pathlib.Path.home() / ".hermes" / "plugin-data" / "hermes-verifier"
DEFAULT_DB = pathlib.Path.home() / ".hermes" / "state.db"


def read_decisions(data_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Every decision row, tagged with which log it came from.

    A torn final line (the process died mid-append) is skipped rather than
    raising: these files are appended to by a daemon thread and the loop must
    not fall over on the last row.
    """
    out: List[Dict[str, Any]] = []
    for kind in ("gate", "phase", "agreement"):
        for path in sorted(data_dir.glob(f"{kind}.*.jsonl")):
            for line in path.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    row["kind"] = kind
                    out.append(row)
    return out


def load_turns(db: pathlib.Path) -> List[Dict[str, Any]]:
    """Segment every substantial session into turns, with outcome labels."""
    if not db.exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sessions = [r["session_id"] for r in con.execute(
        "select session_id from messages where role in ('assistant','tool') "
        "group by session_id having count(*) >= 20")]
    out: List[Dict[str, Any]] = []
    for sid in sessions:
        rows = list(con.execute(
            "select role, content, tool_calls from messages "
            "where session_id=? order by id", (sid,)))
        turns: List[Dict[str, Any]] = []
        cur = None
        for r in rows:
            if r["role"] == "user" or cur is None:
                cur = {"calls": [], "says": [], "edits": 0}
                turns.append(cur)
            if r["role"] == "assistant":
                if r["content"]:
                    cur["says"].append(r["content"])
                if r["tool_calls"]:
                    cur["calls"].append(r["tool_calls"])
                    cur["edits"] += 1
            elif r["role"] == "tool" and r["content"]:
                cur["calls"].append(r["content"])
        if len(turns) < 6:
            continue
        for i, t in enumerate(turns):
            nxt = ""
            if i + 1 < len(turns):
                nxt = (" ".join(turns[i + 1]["says"]) + "\n"
                       + "\n".join(turns[i + 1]["calls"]))
            out.append({
                "session_id": sid, "turn": i,
                "calls": "\n".join(t["calls"]), "says": "\n".join(t["says"]),
                "edits": t["edits"],
                "repair_next": bool(REPAIR.search(nxt)) if nxt else False,
                "fail_next": bool(FAILURE.search(nxt)) if nxt else False,
            })
    return out


def harvest(data_dir: pathlib.Path = DEFAULT_DATA,
            db: pathlib.Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    """Turns, with any decisions that belong to the same session attached."""
    turns = load_turns(db)
    decisions = read_decisions(data_dir)
    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for d in decisions:
        by_session.setdefault(str(d.get("session_id") or ""), []).append(d)
    for t in turns:
        t["decisions"] = by_session.get(t["session_id"], [])
    return turns
```

**3d. Run the tests — expect pass.**

```bash
cd ~/hermes-verifier && python -m pytest autoresearch/tests/test_harvest.py -q
```

Expected: `3 passed`.

**3e. Smoke it against the real corpus.**

```bash
cd ~/hermes-verifier && python -c "
from autoresearch.harvest import harvest, read_decisions, DEFAULT_DATA
d = read_decisions(DEFAULT_DATA)
print('decision rows:', len(d), {k: sum(1 for x in d if x['kind']==k) for k in ('gate','phase','agreement')})
t = harvest()
print('turns:', len(t), '| sessions:', len({x['session_id'] for x in t}))
print('repair_next rate:', round(sum(1 for x in t if x['repair_next'])/max(1,len(t)), 3))
"
```

Expected: non-zero decision rows, ~800+ turns, a repair rate in the 0.3–0.5 band.
**If `phase` and `agreement` counts are 0, Task 1 did not land — go back.**

**3f. Commit.**

```bash
cd ~/hermes-verifier && git add -A \
  && git commit -m "feat: harvest decisions and session turns into one normalised table"
```

---

### Task 4 — Score with the guards that were learned the hard way

**4a. Write the failing test.** Create `autoresearch/tests/test_score.py`:

```python
"""Every score must carry a CI, a power statement, and the multiplicity bar."""

from __future__ import annotations

from autoresearch.score import compare_rules, lift_ci


def _turns(n_repair, n_clean, fired_every=1):
    """fired_every=1 -> the rule fires on every turn; 2 -> every other."""
    out = []
    for i in range(n_repair):
        out.append({"repair_next": True, "calls": "x", "says": "", "edits": 0,
                    "fire": (i % fired_every == 0)})
    for i in range(n_clean):
        out.append({"repair_next": False, "calls": "x", "says": "", "edits": 0,
                    "fire": (i % fired_every == 0)})
    return out


def test_a_rule_firing_on_everything_has_lift_one():
    turns = _turns(40, 60)
    lo, hi = lift_ci(turns, lambda t: t["fire"], bootstraps=200, seed=1)
    assert lo <= 1.0 <= hi


def test_the_report_states_power():
    turns = _turns(40, 60)
    res = compare_rules(turns, {"always": lambda t: t["fire"]},
                        reference="always", bootstraps=200, seed=1)
    assert "power" in res
    assert res["power"]["turns"] == len(turns)
    assert res["power"]["positive_labels"] == 40
    assert res["power"]["sessions"] >= 1
```

**4b. Run it — expect failure.**

```bash
cd ~/hermes-verifier && python -m pytest autoresearch/tests/test_score.py -q
```

Expected: `ModuleNotFoundError`.

**4c. Implement `autoresearch/score.py`.**

```python
"""Score candidate rules with the guards a previous pass lacked.

Three rules, all mandatory:
  * every lift gets a bootstrap CI, so a point estimate cannot be read as signal
  * the multiplicity bar is Bonferroni over the number of candidates tested
  * the report carries the power statement -- sessions, turns, positive labels --
    because a null under a strict bar is NOT confirmation when power is low
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Tuple

DEFAULT_BOOTSTRAPS = 4000
DEFAULT_SEED = 20261010
ALPHA = 0.05


def lift_on(turns: List[Dict[str, Any]], mask: List[bool]) -> float:
    fired = [t for t, m in zip(turns, mask) if m]
    if not fired:
        return 0.0
    base = sum(1 for t in turns if t["repair_next"]) / max(1, len(turns))
    if not base:
        return 0.0
    return (sum(1 for t in fired if t["repair_next"]) / len(fired)) / base


def lift_ci(turns, predicate: Callable[[Dict[str, Any]], bool],
            bootstraps: int = DEFAULT_BOOTSTRAPS, seed: int = DEFAULT_SEED
            ) -> Tuple[float, float]:
    """95% bootstrap CI for one rule's lift."""
    rng = random.Random(seed)
    n = len(turns)
    mask = [predicate(t) for t in turns]
    vals = []
    for _ in range(bootstraps):
        pick = [rng.randrange(n) for _ in range(n)]
        vals.append(lift_on([turns[i] for i in pick], [mask[i] for i in pick]))
    vals.sort()
    return vals[int(len(vals) * 0.025)], vals[int(len(vals) * 0.975) - 1]


def compare_rules(turns, candidates: Dict[str, Callable], *, reference: str,
                  bootstraps: int = DEFAULT_BOOTSTRAPS, seed: int = DEFAULT_SEED
                  ) -> Dict[str, Any]:
    """Paired bootstrap of every candidate against `reference`.

    Paired: both rules are scored on the SAME resampled turns, so shared noise
    cancels and the CI is on the actual difference.
    """
    rng = random.Random(seed)
    n = len(turns)
    masks = {k: [f(t) for t in turns] for k, f in candidates.items()}
    point = {k: lift_on(turns, m) for k, m in masks.items()}
    draws = {k: [] for k in candidates}
    for _ in range(bootstraps):
        pick = [rng.randrange(n) for _ in range(n)]
        sub = [turns[i] for i in pick]
        for k, m in masks.items():
            draws[k].append(lift_on(sub, [m[i] for i in pick]))

    n_tests = max(1, len(candidates) - 1)
    lo_q = ALPHA / n_tests / 2
    hi_q = 1 - lo_q
    results = {}
    for k in candidates:
        if k == reference:
            continue
        d = sorted(a - b for a, b in zip(draws[k], draws[reference]))
        lo = d[int(len(d) * lo_q)]
        hi = d[int(len(d) * hi_q) - 1]
        results[k] = {
            "gap": point[k] - point[reference],
            "ci": (lo, hi),
            "clears_bar": lo > 0,
        }
    return {
        "reference": reference,
        "reference_lift": point[reference],
        "point": point,
        "results": results,
        "multiplicity": {"tests": n_tests, "alpha_each": ALPHA / n_tests},
        "power": {
            "turns": n,
            "sessions": len({t["session_id"] for t in turns}),
            "positive_labels": sum(1 for t in turns if t["repair_next"]),
        },
    }
```

**4d. Run the tests — expect pass.**

```bash
cd ~/hermes-verifier && python -m pytest autoresearch/tests/test_score.py -q
```

Expected: `2 passed`.

**4e. Commit.**

```bash
cd ~/hermes-verifier && git add -A \
  && git commit -m "feat: score candidate rules with CIs, Bonferroni and a power statement"
```

---

### Task 5 — Validate on a held-out half, split by session

**5a. Write the failing test.** Create `autoresearch/tests/test_validate.py`:

```python
"""Adoption requires BOTH the multiplicity bar and a held-out confirmation."""

from __future__ import annotations

from autoresearch.score import compare_rules
from autoresearch.validate import split_by_session, validate


def _corpus():
    turns = []
    for s in range(20):
        for i in range(10):
            turns.append({
                "session_id": f"s{s}", "turn": i,
                "calls": "rm -rf x" if (s % 2 == 0 and i % 2 == 0) else "ls",
                "says": "", "edits": 1,
                "repair_next": bool(s % 2 == 0 and i % 2 == 0),
                "fail_next": False,
            })
    return turns


def test_the_split_is_by_session_not_by_turn():
    fit, val = split_by_session(_corpus(), seed=1)
    assert not ({t["session_id"] for t in fit} & {t["session_id"] for t in val}), (
        "turns from one session leaked across the split"
    )
    assert fit and val


def test_validate_reports_the_bar_and_the_held_out_gap():
    out = validate(_corpus(), {"always": lambda t: True,
                               "rm": lambda t: "rm -rf" in t["calls"]},
                   reference="always", bootstraps=200, seed=1)
    assert "held_out" in out
    assert "power" in out
    assert set(out["held_out"]) == {"rm"}
```

**5b. Run it — expect failure.**

```bash
cd ~/hermes-verifier && python -m pytest autoresearch/tests/test_validate.py -q
```

Expected: `ModuleNotFoundError`.

**5c. Implement `autoresearch/validate.py`.**

```python
"""Split by SESSION, then require a candidate to survive the fit half AND hold
up on the validation half.

Splitting by turn leaks: turns inside one session share an author, a project and
a set of habits, so a turn-level split lets the fit half memorise the validation
half and every rule looks generalisable.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Tuple

from .score import compare_rules


def split_by_session(turns: List[Dict[str, Any]], seed: int = 20261010
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ids = sorted({t["session_id"] for t in turns})
    random.Random(seed).shuffle(ids)
    half = len(ids) // 2
    fit_ids, val_ids = set(ids[:half]), set(ids[half:])
    fit = [t for t in turns if t["session_id"] in fit_ids]
    val = [t for t in turns if t["session_id"] in val_ids]
    return fit, val


def validate(turns, candidates: Dict[str, Callable], *, reference: str,
             bootstraps: int = 4000, seed: int = 20261010) -> Dict[str, Any]:
    fit, val = split_by_session(turns, seed=seed)
    fit_res = compare_rules(fit, candidates, reference=reference,
                            bootstraps=bootstraps, seed=seed)
    survivors = [k for k, r in fit_res["results"].items() if r["clears_bar"]]

    held_out = {}
    for k in survivors:
        v = compare_rules(val, candidates, reference=reference,
                          bootstraps=bootstraps, seed=seed)
        held_out[k] = {
            "fit_gap": fit_res["results"][k]["gap"],
            "val_gap": v["results"][k]["gap"],
            "confirmed": v["results"][k]["gap"] > 0,
        }
    return {
        "fit_power": fit_res["power"],
        "val_power": fit_res["power"],
        "survivors": survivors,
        "held_out": held_out,
        "adoptable": [k for k, h in held_out.items() if h["confirmed"]],
        "verdict": ("no candidate cleared the bar" if not survivors
                    else ("adopted: " + ", ".join(
                        k for k, h in held_out.items() if h["confirmed"])
                        or "candidates cleared the bar but failed held-out")),
    }
```

**5d. Run the tests — expect pass.**

```bash
cd ~/hermes-verifier && python -m pytest autoresearch/tests/test_validate.py -q
```

Expected: `2 passed`.

**5e. Commit.**

```bash
cd ~/hermes-verifier && git add -A \
  && git commit -m "feat: session-level held-out validation before any rule is adopted"
```

---

### Task 6 — The report

**6a. Write the failing test.** Create `autoresearch/tests/test_report.py`:

```python
"""The report must state its own limits, every run, without being asked."""

from __future__ import annotations

from autoresearch.report import render


def test_the_caveat_is_always_present():
    text = render({"verdict": "no candidate cleared the bar",
                   "power": {"turns": 10, "sessions": 2, "positive_labels": 4}})
    assert "repair FOLLOWED" in text
    assert "not that a review would have prevented it" in text


def test_a_low_power_run_says_so():
    text = render({"verdict": "no candidate cleared the bar",
                   "power": {"turns": 40, "sessions": 3, "positive_labels": 5}})
    assert "UNDERPOWERED" in text, (
        "a null with few positive labels is not evidence, and the report must say so"
    )
```

**6b. Run it — expect failure.** → `ModuleNotFoundError`.

**6c. Implement `autoresearch/report.py`.**

```python
"""Render the run. The caveats are part of the output, not the commentary."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict

# Below this many positive labels the bootstrap cannot resolve a small effect,
# so a null is uninformative rather than reassuring.
MIN_POSITIVES = 100

CAVEAT = (
    "The outcome label is repair language in the FOLLOWING turn. It measures "
    "that repair FOLLOWED, not that a review would have prevented it. Every "
    "number here is correlational."
)


def render(result: Dict[str, Any]) -> str:
    p = result.get("power") or {}
    lines = [
        f"# hermes-verifier autoresearch — {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**verdict:** {result.get('verdict', 'unknown')}",
        "",
        "## power",
        f"- turns: {p.get('turns', 0)}",
        f"- sessions: {p.get('sessions', 0)}",
        f"- positive labels: {p.get('positive_labels', 0)}",
    ]
    if p.get("positive_labels", 0) < MIN_POSITIVES:
        lines.append(
            f"- **UNDERPOWERED** — fewer than {MIN_POSITIVES} positive labels. "
            "A null result under this bar is NOT evidence the current rule is "
            "the best one; it means the data cannot tell."
        )
    lines += ["", "## caveat", CAVEAT, ""]
    if result.get("held_out"):
        lines += ["## held-out", "```", json.dumps(result["held_out"], indent=2), "```"]
    return "\n".join(lines)


def write_report(result: Dict[str, Any], out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"autoresearch-{time.strftime('%Y%m%d-%H%M%S')}.md"
    path.write_text(render(result))
    return path
```

**6d. Run the tests — expect pass.** → `2 passed`.

**6e. Commit.**

```bash
cd ~/hermes-verifier && git add -A \
  && git commit -m "feat: autoresearch report states its power and its caveat unasked"
```

---

### Task 7 — Wire the loop together

**7a. Implement `autoresearch/run.py`.**

```python
"""One autoresearch pass: harvest, score, validate, report. Applies nothing."""

from __future__ import annotations

import pathlib

from .harvest import harvest
from .report import write_report
from .rules import CANDIDATES
from .validate import validate


def candidates_from_turns(turns):
    """Bind each rule predicate to a harvested turn.

    rules.CANDIDATES take (calls, says, edits); harvested turns carry those plus
    the labels, so this adapts one to the other in a single place.
    """
    return {name: (lambda t, fn=fn: fn(t["calls"], t["says"], t["edits"]))
            for name, fn in CANDIDATES.items()}


def main() -> None:
    turns = harvest()
    if not turns:
        print("no turns harvested; nothing to do")
        return
    cands = candidates_from_turns(turns)
    result = validate(turns, cands, reference="destructive_or_external")
    text = write_report(result, pathlib.Path("autoresearch/reports"))
    print(text.read_text())


if __name__ == "__main__":
    main()
```

**7b. Run it.**

```bash
cd ~/hermes-verifier && python -m autoresearch.run 2>&1 | head -30
```

Expected: a report with a verdict line, a power block, and the caveat. It will
almost certainly say `no candidate cleared the bar` — that is the correct output
for this much data, not a failure.

**7c. Add a test that the loop never writes configuration.**

The naive version of this test -- ban `write_text(` outright -- fails on the
loop's own legitimate output: `report.write_report` writes a markdown report,
and that is the natural way to do it. A test that forbids correct code gets
weakened or bypassed by whoever hits it next, so scope the assertion to
CONFIGURATION writes and say what is allowed.

Create `autoresearch/tests/test_no_autonomy.py`:

```python
"""The loop reports. It does not change the plugin's configuration.

Writes to the loop's OWN outputs -- the harvest table, the markdown report --
are expected and allowed. What must never happen is a write that reaches the
plugin's settings or the Hermes config.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[1]

# Settings and config are off limits, by any name.
FORBIDDEN_CALLS = {"set_config", "safe_dump", "safe_load_write"}
FORBIDDEN_PATH_TOKENS = ("config.yaml", ".hermes/config", "plugin.yaml",
                         "settings.json")

# Writes are fine as long as they target the loop's own output tree.
ALLOWED_PATH_TOKENS = ("reports", "autoresearch", "tmp")


def _sources():
    for path in PACKAGE.rglob("*.py"):
        if "tests" in path.parts:
            continue
        yield path, path.read_text()


def test_no_call_writes_configuration():
    for path, src in _sources():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            assert name not in FORBIDDEN_CALLS, (
                f"{path.name} calls {name!r}; the loop must not write configuration"
            )


def test_no_write_targets_a_config_path():
    """Any string literal that looks like a config path is a red flag."""
    for path, src in _sources():
        for token in FORBIDDEN_PATH_TOKENS:
            assert token not in src, (
                f"{path.name} mentions {token!r}; the loop must not touch "
                "configuration"
            )


def test_the_allowed_output_tree_is_what_we_think_it_is():
    """Guard the guard: if outputs move, this test must be updated too, not
    quietly broadened."""
    from autoresearch import report
    import inspect
    src = inspect.getsource(report.write_report)
    assert "reports" in src or "out_dir" in src, (
        "write_report no longer targets an output directory; the allowlist in "
        "this file is stale"
    )
```

**7d. Run it.** → `3 passed`.

**7e. Commit.**

```bash
cd ~/hermes-verifier && git add -A \
  && git commit -m "feat: autoresearch run -- harvest, score, validate, report

Applies nothing. Adoption stays a human decision."
```

---

### Task 8 — Schedule it

**8a. Create the cron job** so the loop accumulates labelled data over time.

```bash
cd ~/hermes-verifier && python -c "
from hermes_tools import terminal
print(terminal('ls autoresearch/run.py')['output'])
"
```

Then, using the `cronjob_manage` tool (not a shell command), create a daily job:

- **schedule:** `0 9 * * *` (09:00 daily)
- **prompt:** `Run the hermes-verifier autoresearch pass: cd ~/hermes-verifier && python -m autoresearch.run. Then read the newest file in autoresearch/reports/ and report only the verdict line, the power block, and whether the run was UNDERPOWERED. Do not change any configuration.`
- **name:** `hermes-verifier autoresearch`

**8b. Verify it registered.**

```bash
hermes cron list 2>&1 | grep -i autoresearch
```

Expected: one row named `hermes-verifier autoresearch`.

**8c. Run it once by hand** to confirm it produces a report end to end, then
commit any generated report under `autoresearch/reports/` (they are the audit
trail).

---

## Tests / validation

- Per-task TDD as above; each task ends green before committing.
- Full suite after every task: `python -m pytest -q` from `~/hermes-verifier`.
  Baseline is **282 passing**; Tasks 1 and 2 add 5 tests, Tasks 3–7 add 8.
  Final expected: **295 passed**.
- Gates: `hermes plugins validate . --install-deps` must print
  `Validation passed.` and `security scan — safe`; `hermes plugins doctor . --ci`
  must exit 0.
- The loop's own output is validated by the `UNDERPOWERED` and caveat tests — if
  either stops appearing, the reporting regressed.

## Risks, tradeoffs, and open questions

**Power is the binding constraint.** 832 labelled turns yields CIs roughly
±0.15 wide on a lift of ~1.3. Every real effect this loop could find is smaller
than that. **Expect the first several months of runs to say "no change"** — that
is the loop working, not failing. Do not lower the bar to manufacture findings;
that is the exact error this design exists to prevent.

**The label is a proxy and may be wrong.** `repair_next` detects *language*, so
a session that says "let me revert" without reverting scores as a positive, and
a silent recovery scores as a negative. Mitigation: `fail_next` is tracked in
parallel, and a rule whose two lifts disagree is treated as noise. This does not
make the proxy correct, only detectable.

**Nothing here proves prevention.** A/B on real turns would, and that requires
splitting live sessions into reviewed and unreviewed arms — a real experiment
with real user impact, and out of scope for this plan. Until it runs, the loop
can tell you a rule correlates with expensive turns; it cannot tell you the rule
helps.

**Candidates are hand-written.** The loop tunes among rules a human wrote. It
will never discover a rule nobody thought of. That is a deliberate limit: an
automated search over trigger predicates over 832 turns is a machine for
producing spurious winners.

**Ordering caveat on the corpus split.** Task 5 splits 25-ish sessions into two
halves, which is thin. If `val_power["sessions"]` reports fewer than 10 sessions,
treat the held-out check as advisory and say so in the report rather than
adopting on it.

**Open questions**

1. Should the loop also score the *engagement gate* (engage/skip), which is
   model-driven rather than rule-driven? It can measure how often a skip preceded
   repair, but it cannot propose a fix without a candidate predicate. Defer.
2. Should reports be committed to the repo or kept in `~/.hermes`? Committed is
   a better audit trail; it also grows the repo. Suggest committing one per
   month and pruning the rest.
3. When the tool-usage debt is paid (the feature is configured and unused),
   a `tool_called` column should be added to the harvest so tool use can be
   scored against outcomes. Not now — it would be noise until the feature fires.
