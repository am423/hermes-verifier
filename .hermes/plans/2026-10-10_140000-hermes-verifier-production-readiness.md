# Plan — Production Readiness Review for `hermes-verifier`

## Goal

Prove with tests that the advisor engages exactly when it should — and just as importantly, that it does **not** engage when it shouldn't — then close the reliability and observability gaps that review exposes.

## Current context / assumptions

Verified on this machine at HEAD `a4b50f8`:

| Fact | Value |
| --- | --- |
| Repo | `~/hermes-verifier` (repo root IS the installable package) |
| Live install | `~/.hermes/plugins/hermes-verifier` → symlink, `enabled 0.2.0` |
| Suite | **91 passing** (`python -m pytest -q` → `91 passed in 0.55s`) |
| `validate` | `Validation passed.` — security scan `safe`, no core override, desktop surface OK |
| `doctor --ci` | exit 0 |
| Local config | `enabled: true`, `advisor_tools: true`, `sync_backlog: "off"`, `verifier_model: nous/anthropic/claude-opus-5.5` |
| Source | `runtime.py` (939), `brief.py` (303), `advisor_tools.py` (132), `guard.py` (107), `delta.py` (166), `commands.py` (178), `dashboard/plugin_api.py` (315) |

### Note on the advisor feedback in the prompt

All three notes are **stale** — already resolved in `a4b50f8`:

1. `last_review_ts` — **already deleted** (`grep -n "last_review_ts" runtime.py` → no match). The eviction *is* FIFO (`next(iter(_SESSIONS))` on an insertion-ordered dict), so no LRU work is warranted.
2. `sync_backlog` — **already reverted to `"off"`** locally, on the measurement that the advisor call fires 3–30s *after* `Turn ended`.
3. The in-loop retry test — **already exists**: `test_empty_response_retries_then_records` (`tests/test_runtime.py:827`), `test_persistent_empty_response_is_logged_and_counted` (856), `test_no_retry_when_a_real_finding_returns` (870).

**Do not redo any of the three.** This plan covers what they missed.

### The real gap: engagement is under-tested

The suite tests *what happens after* a review well, and barely tests **whether a review should happen at all**. These engagement conditions have **no coverage** (verified by grep against `tests/*.py`, all returning 0 matches):

| # | Condition | Why it matters |
| --- | --- | --- |
| 1 | Empty transcript delta → no review | Wastes a paid provider call on nothing |
| 2 | Hook raising → never crashes the agent | A review bug must not take down the session |
| 3 | `consecutive_failures` release path | A broken provider must not wedge the plugin |
| 4 | `_MAX_SESSIONS` eviction | Unbounded dict growth = slow leak |
| 5 | Guard dedupe **across** turns | Same finding re-raised every turn = nagging |
| 6 | Nudge + pending finding **merged** | One can silently overwrite the other |
| 7 | `pre_verify` immunity cooldown expiry | The cooldown must actually expire |
| 8 | Budget + retry interaction | A retry must not silently double-spend |

**Already covered — do not duplicate:** the disabled-session case is tested at `tests/test_runtime.py:264` (`test_disabled_session_skips_everything`, asserts no session state and `ctx.llm.calls == []`), and the existing immunity/steer-gate assertions at lines 226-236.

Conditions 5 and 8 are the most likely to be real bugs — the guard's `_seen` map is reset on `reset()` but nothing tests cross-turn behaviour, and the retry I just moved adds a second provider call per review with no test that the budget counts it once.

---

## Architecture / proposed approach

No production code changes until the tests say something is broken. Phase 1 writes failing engagement tests against existing behaviour and fixes only what they expose. Phase 2 adds the one observability gap the tests reveal — a way to tell from `/verify status` whether the advisor is actually engaging — and Phase 3 gates the whole thing behind the same checks catalog CI runs. If every new test passes against unchanged code, that is a legitimate and successful outcome: the deliverable is the proof, not the diff.

---

## Step-by-step tasks

All paths relative to `~/hermes-verifier`. Commit after each task.

### Phase 0 — Baseline

```bash
cd ~/hermes-verifier && python -m pytest -q
```

Expected: `91 passed`. If not green, stop and fix that first.

---

### Phase 1 — Engagement tests (TDD)

Match the file's existing conventions exactly: module-level `FakeCtx`/`FakeResult`/`FakeLlm`, the `ctx` fixture, `monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)`, and the `_turn(ctx, ...)` / `_default(state)` helpers. Do **not** invent fixture names.

#### Task 1.1 — Empty delta must not trigger a review

**File: `tests/test_runtime.py`** — append:

```python
def test_empty_delta_does_not_review(ctx, monkeypatch):
    """No new transcript content means no paid provider call."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    # A turn whose content is identical to the last cursor position.
    _turn(ctx, history=[], response="")
    assert ctx.llm.calls == [], "an empty delta must not spend a provider call"


def test_identical_consecutive_turns_do_not_review(ctx, monkeypatch):
    """Replaying the same transcript must not re-review."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    hist = [{"role": "assistant", "content": "same"}]
    _turn(ctx, history=hist, response="same")
    first = len(ctx.llm.calls)
    _turn(ctx, history=hist, response="same")     # no new rows
    assert len(ctx.llm.calls) == first, "no new delta → no second review"
```

```bash
python -m pytest tests/test_runtime.py -k "empty_delta or identical_consecutive" -q
```

Expected: `2 passed`. If either fails, the bug is real — `handle_post_llm_call` returns early on `if not delta_text` (runtime.py:665-667), so these should pass unchanged. **A failure here means the cursor logic regressed.**

**Commit:** `test: empty and replayed deltas must not spend a provider call`

#### Task 1.2 — A raising hook must never crash the agent

This is the highest-value test in the plan. It proves the advisor is safe to enable.

```python
def test_review_exception_never_propagates(ctx, monkeypatch):
    """A failing review must be contained: the primary agent is unaffected."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.raise_error = RuntimeError("provider exploded")

    # Must not raise.
    _turn(ctx)

    state = runtime._get_state("s1", create=False)
    assert state.consecutive_failures >= 1, "the failure must be recorded"
    assert state.last_error, "the failure must be surfaced, not swallowed silently"


def test_failure_streak_releases_backlog_wait(ctx, monkeypatch):
    """A broken advisor must never stall the primary turn."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.raise_error = RuntimeError("down")
    for i in range(4):
        _turn(ctx, history=[{"role": "user", "content": f"t{i}"}], response=f"a{i}")
    state = runtime._get_state("s1", create=False)
    assert state.consecutive_failures >= 3
    # _backlog_wait must return immediately on a failure streak.
    clock = {"t": 0.0}
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(runtime.time, "sleep",
                        lambda s: clock.__setitem__("t", clock["t"] + s))
    runtime._backlog_wait(state, 1)
    assert clock["t"] == 0, "a failing verifier must never stall the session"
```

```bash
python -m pytest tests/test_runtime.py -k "exception_never_propagates or failure_streak" -q
```

Expected: `2 passed`.

**Commit:** `test: a failing review is contained and never stalls the session`

#### Task 1.3 — Guard dedupe across turns (most likely to expose a real bug)

```python
def test_identical_finding_not_raised_on_consecutive_turns(ctx, monkeypatch):
    """The same finding must not re-fire every turn — that is nagging."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    for i in range(3):
        ctx.llm.next = FakeResult({"severity": "caution",
                                   "finding": "Bypasses the durable queue in src/jobs",
                                   "reasoning": "same issue"})
        _turn(ctx, history=[{"role": "user", "content": f"t{i}"}], response=f"a{i}")
        runtime.handle_pre_llm_call(session_id="s1", user_message="m",
                                    conversation_history=[])
    state = runtime._get_state("s1", create=False)
    stats = _default(state).guard.stats
    assert stats["deduped"] >= 2, (
        f"identical findings must be deduped across turns, got {stats}"
    )


def test_escalating_severity_still_delivers(ctx, monkeypatch):
    """A note escalating to a blocker is new information, not a duplicate."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "note",
                               "finding": "queue bypass", "reasoning": ""})
    _turn(ctx)
    runtime.handle_pre_llm_call(session_id="s1", user_message="m",
                                conversation_history=[])
    ctx.llm.next = FakeResult({"severity": "blocker",
                               "finding": "queue bypass", "reasoning": ""})
    _turn(ctx, history=[{"role": "user", "content": "t2"}], response="a2")
    assert _default(runtime._get_state("s1", create=False)).pending is not None, \
        "an escalation must survive dedupe"
```

```bash
python -m pytest tests/test_runtime.py -k "identical_finding_not_raised or escalating_severity" -q
```

Expected: `2 passed`. **A failure in the first test is a real product bug** — the advisor would nag every turn.

**Commit:** `test: findings dedupe across turns but escalations survive`

#### Task 1.4 — Nudge and pending finding must merge, not overwrite

```python
def test_nudge_and_pending_finding_both_delivered(ctx, monkeypatch):
    """A nudge and a real finding in the same turn must both arrive."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["nudge_first_turns"] = 5
    ctx.llm.next = FakeResult({"severity": "caution",
                               "finding": "queue bypass", "reasoning": ""})
    _turn(ctx)
    runtime.handle_pre_llm_call(session_id="s1", user_message="m",
                                conversation_history=[])   # consume the finding
    # Second turn: the nudge is still armed and a new finding arrives.
    ctx.llm.next = FakeResult({"severity": "blocker",
                               "finding": "credential leak", "reasoning": ""})
    _turn(ctx, history=[{"role": "user", "content": "t2"}], response="a2")

    injected = runtime.handle_pre_llm_call(session_id="s1", user_message="m",
                                           conversation_history=[])
    assert injected is not None
    assert "advisor" in injected["context"].lower(), "the nudge must be present"
    assert "credential leak" in injected["context"], "the finding must be present"
```

```bash
python -m pytest tests/test_runtime.py -k "nudge_and_pending" -q
```

Expected: `1 passed`. **A failure means one silently overwrites the other** — `handle_pre_llm_call` builds `blocks` with `if nudge: blocks.insert(0, nudge)` (runtime.py:713-716), so it should pass.

**Commit:** `test: nudge and a pending finding are delivered together`

#### Task 1.5 — `pre_verify` immunity cooldown must actually expire

```python
def test_immunity_cooldown_expires(ctx, monkeypatch):
    """The cooldown is a rate limit, not a permanent mute."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["immunity_turns"] = 1

    # Turn 1: a blocker interrupts and arms the cooldown.
    ctx.llm.next = FakeResult({"severity": "blocker",
                               "finding": "first blocker", "reasoning": ""})
    _turn(ctx)
    steer = runtime.handle_pre_verify(session_id="s1")
    assert steer is not None, "the first blocker must steer"
    state = runtime._get_state("s1", create=False)
    assert state.immunity_left == 1

    # Turn 2: a second blocker arrives while immunity is armed.
    ctx.llm.next = FakeResult({"severity": "blocker",
                               "finding": "second blocker", "reasoning": ""})
    _turn(ctx, history=[{"role": "user", "content": "t2"}], response="a2")
    runtime.handle_pre_llm_call(session_id="s1", user_message="m",
                                conversation_history=[])      # decrements immunity

    # Turn 3: immunity has expired, so a blocker must steer again.
    ctx.llm.next = FakeResult({"severity": "blocker",
                               "finding": "third blocker", "reasoning": ""})
    _turn(ctx, history=[{"role": "user", "content": "t3"}], response="a3")
    runtime.handle_pre_llm_call(session_id="s1", user_message="m",
                                conversation_history=[])
    steer3 = runtime.handle_pre_verify(session_id="s1")
    assert steer3 is not None, "immunity must expire so a real blocker can steer"
```

```bash
python -m pytest tests/test_runtime.py -k "immunity_cooldown_expires" -q
```

Expected: `1 passed`. **A failure is a real bug** — the advisor would mute itself permanently after one interrupt.

**Commit:** `test: the immunity cooldown expires instead of muting permanently`

#### Task 1.6 — Budget counts a review once, retry included

```python
def test_budget_counts_retry_as_one_review(ctx, monkeypatch):
    """The empty-response retry must not double-spend the budget."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["max_reviews_per_session"] = 2
    attempts = {"n": 0}

    def _flaky(**kw):
        attempts["n"] += 1
        if attempts["n"] % 2 == 1:            # first call of each review: empty
            return FakeResult(None, text="")
        return FakeResult({"severity": "note", "finding": "ok", "reasoning": ""})

    ctx.llm.complete_structured = _flaky
    _turn(ctx)
    _turn(ctx, history=[{"role": "user", "content": "t2"}], response="a2")

    state = runtime._get_state("s1", create=False)
    assert state.reviews_used == 2, "two reviews consumed, retries included"
    assert attempts["n"] == 4, "2 reviews x (1 empty + 1 retry) = 4 provider calls"


def test_budget_blocks_after_cap_with_retries(ctx, monkeypatch):
    """No provider call at all once the cap is reached."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.config["max_reviews_per_session"] = 1
    ctx.llm.complete_structured = lambda **kw: FakeResult(None, text="")
    _turn(ctx)
    before = len(ctx.llm.calls)
    _turn(ctx, history=[{"role": "user", "content": "t2"}], response="a2")
    assert len(ctx.llm.calls) == before, "the cap must hold even when retrying"
```

```bash
python -m pytest tests/test_runtime.py -k "budget_counts_retry or budget_blocks_after_cap_with" -q
```

Expected: `2 passed`. **A failure is a real cost bug** — retries would silently double advisory spend.

**Commit:** `test: the review budget counts a retried review exactly once`

#### Task 1.7 — Session eviction stays bounded

```python
def test_session_eviction_is_bounded():
    """_SESSIONS must not grow without bound across many sessions."""
    original = dict(runtime._SESSIONS)
    try:
        runtime._SESSIONS.clear()
        for i in range(runtime._MAX_SESSIONS + 10):
            runtime._get_state(f"session-{i}")
        assert len(runtime._SESSIONS) <= runtime._MAX_SESSIONS, (
            f"_SESSIONS grew to {len(runtime._SESSIONS)} "
            f"(cap {runtime._MAX_SESSIONS})"
        )
    finally:
        runtime._SESSIONS.clear()
        runtime._SESSIONS.update(original)
```

```bash
python -m pytest tests/test_runtime.py -k "session_eviction_is_bounded" -q
```

Expected: `1 passed`.

**Commit:** `test: session state stays bounded at the eviction cap`

#### Task 1.8 — Full gate

```bash
python -m pytest -q && hermes plugins validate . --install-deps 2>&1 | grep -E "Validation passed" && hermes plugins doctor . --ci && echo "DOCTOR OK"
```

Expected: `105 passed` (91 + 14 new), then `Validation passed.` and `DOCTOR OK`.

---

### Phase 2 — Observability: can you tell the advisor is engaging?

Phase 1 proves engagement in tests. This makes it visible in a live session, which is what "is the advisor on?" actually needs.

#### Task 2.1 — Surface engagement counters in `/verify status`

**File: `commands.py`** — in `_status_lines()`, after the existing advisor-tools line, add:

```python
    snap_sessions = snap["sessions"]
    total_reviews = sum((s.get("verifiers") or {}).get("default", {}).get("stats", {}).get("reviews", 0)
                        for s in snap_sessions.values()) if snap_sessions else 0
    total_empty = sum(s.get("empty_responses", 0) for s in snap_sessions.values()) if snap_sessions else 0
    if snap_sessions:
        lines.append(
            f"  engagement: {total_reviews} review(s), {total_empty} empty response(s)"
        )
```

`empty_responses` is already in `snapshot()` via `SessionVerifier`; verify with:

```bash
grep -n "empty_responses" runtime.py | head -3
```

Expected: at least the dataclass field and its increment site.

**Verification:**

```bash
python -m pytest -q
```

Expected: `105 passed` (no behavior change).

**Commit:** `feat: surface engagement counters in /verify status`

#### Task 2.2 — README: document how to confirm engagement

**File: `README.md`** — add under the existing `## Status` section:

```markdown
### Confirming the advisor is engaging

`/verify status` reports an `engagement:` line with the review count and the
empty-response count for this session. A non-zero review count means the
advisor actually ran. `empty response(s)` counts provider completions that
returned no payload — those are retried once, then surfaced in `last_error`.
A rising empty count with a flat review count means the advisor model is not
answering, not that it found nothing.
```

**Verification:**

```bash
grep -c "Confirming the advisor is engaging" README.md    # → 1
```

**Commit:** `docs: how to confirm the advisor is engaging`

---

### Phase 3 — Production gates

#### Task 3.1 — Bump version and re-validate

**File: `plugin.yaml`** — `version: 0.2.0` → `version: 0.3.0`.

```bash
python -m pytest -q && hermes plugins validate . --install-deps 2>&1 | tail -18 && hermes plugins doctor . --ci && echo "DOCTOR OK"
```

Expected: `105 passed`, then every line `✓` except the known `◇ Isolation` advisory, then `DOCTOR OK`.

**Commit:** `chore: release v0.3.0 — production readiness test coverage`

#### Task 3.2 — Host-independence check

The suite must pass on a bare checkout with no Hermes install.

```bash
python - << 'PYEOF'
import sys, subprocess, os
blocker = "/tmp/blocker_site_pr"; os.makedirs(blocker, exist_ok=True)
open(f"{blocker}/agent.py", "w").write("raise ImportError('nope')\n")
r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=".",
                   env=dict(os.environ, PYTHONPATH=blocker), capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1])
PYEOF
```

Expected: `105 passed`.

**Commit:** none (verification only).

#### Task 3.3 — Live smoke test

```bash
cd ~ && timeout 280 hermes chat -q "Create /tmp/pr_check.py with a function that runs os.system('cat ' + name) with no validation." 2>&1 | tail -3; sleep 6
/verify status 2>/dev/null || python -c "print('check engagement line in the session UI')"
grep -E "complete_structured plugin=hermes-verifier" ~/.hermes/logs/agent.log | tail -2
```

Expected: an advisor call logged with `content_type=json` and non-zero `tokens=`, and the engagement line showing a non-zero review count.

**Note:** do not expect a finding on the `os.system` prompt — across every prior attempt the advisor produced no persisted finding for it. That is a model-behavior observation, not a plugin defect, and it is the open question below.

---

## Tests / validation

| After task | `pytest -q` |
| --- | --- |
| Baseline | `91 passed` |
| 1.1 | `93 passed` |
| 1.2 | `95 passed` |
| 1.3 | `97 passed` |
| 1.4 | `98 passed` |
| 1.5 | `99 passed` |
| 1.6 | `101 passed` |
| 1.7 | `102 passed` |
| 1.8 | `105 passed` |
| 2.1 / 2.2 | `105 passed` |

The count is a tripwire: it must never drop below 91. Gate commands, all of which exit non-zero on failure:

```bash
python -m pytest -q
hermes plugins validate . --install-deps
hermes plugins doctor . --ci
```

### What each new test proves

| Test | Engagement guarantee |
| --- | --- |
| `empty_delta_does_not_review` | No paid call on an empty delta |
| `identical_consecutive_turns_do_not_review` | No re-review of replayed content |
| `review_exception_never_propagates` | A review bug cannot crash the session |
| `failure_streak_releases_backlog_wait` | A broken advisor never stalls the turn |
| `identical_finding_not_raised_on_consecutive_turns` | No nagging |
| `escalating_severity_still_delivers` | Escalations are not suppressed as dupes |
| `nudge_and_pending_finding_both_delivered` | Neither channel clobbers the other |
| `immunity_cooldown_expires` | The advisor cannot mute itself permanently |
| `budget_counts_retry_as_one_review` | Retries do not double-spend |
| `budget_blocks_after_cap_with_retries` | The cost cap holds under retry |
| `session_eviction_is_bounded` | No unbounded memory growth |

---

## Risks, tradeoffs, and open questions

1. **The plan may produce no production diff.** If all 14 tests pass against unchanged code, the outcome is a proven-correct plugin with better coverage. That is the intended success case — the deliverable is the proof, not the diff. Tasks 1.3, 1.4, 1.5 and 1.6 are the ones most likely to expose real bugs.

2. **The `os.system` security case is unexplained.** Across every attempt the advisor produced **no persisted finding** for the command-injection prompt, while it reliably reviewed conversational deltas. Two candidate causes, and the plan does not distinguish them: (a) Opus returns `severity: "none"` on code deltas, which the plugin correctly does not persist, or (b) the emission guard or dedupe is suppressing it. **Open question for the owner:** worth adding a debug-level log of *rejected* `severity: none` reviews so this is diagnosable, or leave it silent to avoid log noise?

3. **`engagement:` line is a heuristic.** It counts `reviews` and `empty_responses` from the default verifier only. With a multi-verifier roster, per-advisor counts need a different aggregation. Kept deliberately simple (YAGNI) — extend only if a roster user asks.

4. **No live E2E automation.** The smoke test in 3.3 is manual because it depends on a real provider and a live session. Automating it would mean mocking `ctx.llm`, which the unit tests already do — the manual run exists only to prove the *real* Nous/Opus path still works.

5. **Deliberately out of scope:** model-pairing rank enforcement (Hermes exposes no capability ranking), and the Anthropic server-side `advisor_20260301` tool (no Hermes hook exists). Both were considered and rejected in the upgrade plan; re-litigating them here would be scope creep.
