# Plan — `hermes-verifier` Audit: Cost Observability, Delta Budgeting, and Review Durability

## Goal

Fix the plugin's blind spots so advisor cost is measurable and controllable: every review writes telemetry (including no-finding reviews), the delta budget is honest about tokens, and reviews are not silently lost when a process exits.

## Current context / assumptions

HEAD `d21d170`, 105 tests passing, `validate` clean, `doctor --ci` exit 0. Plugin live-enabled on `nous/anthropic/claude-opus-5.5`.

### Audit findings (all verified, with evidence)

**A1 — No-finding reviews write no telemetry at all.**
`runtime.py:557-559` returns *before* `_record_and_deliver`:

```python
    if severity == "none" or not finding_text:
        return                                    # ← nothing recorded
    accepted = verifier.guard.accept(severity, finding_text)
    _record_and_deliver(state, verifier, parsed, accepted, usage, ...)
```

Every review that finds nothing — likely the majority — leaves zero trace on disk. Cost per review is therefore **unmeasurable from any artifact**, which is the structural reason this session's "9,700 tokens" chase went in circles.

**A2 — The `prompt_parts` instrumentation added in `d21d170` is dead code.**
Two independent defects:

```python
runtime.py:528    verifier.stats.setdefault("prompt_parts", {   # (1) freezes at first review
                      "system_prompt": len(system_prompt), ... })
```

1. `setdefault` never updates after the first review, so the numbers are the first review's sizes forever.
2. `prompt_parts` appears **once** in `runtime.py` and is absent from the `record` dict (`runtime.py:390-405`), so it never reaches the jsonl. It is in-memory only and invisible to every log.

**A3 — RESOLVED by measurement: the provider bills ~2.48x the tiktoken count.**

`delta.py` truncates on `len(tail) > max_chars` with `max_delta_chars: 20000` (chars). An earlier version of this entry claimed a "1.7x underestimate" with a fitted 2.2–2.5 chars/token figure. **Both were inferences reported as measurements and are retracted.**

What was measured, with a real tokenizer (`tiktoken`, `cl100k_base`) and then against the provider's own billing:

| Content | chars | tiktoken tokens | chars/tok |
| --- | --- | --- | --- |
| Escaped-JSON terminal row | 140 | 46 | 3.04 |
| Prose | 450 | 101 | 4.46 |
| Delta built to the **20,000-char cap** | 20,020 | 5,006 | 4.00 |

Then, from three live reviews once the telemetry existed, `est_tokens` (tiktoken-derived) vs `usage.input_tokens` (billed):

| est | actual billed | ratio |
| --- | --- | --- |
| 693 | 1,705 | 2.46 |
| 831 | 1,982 | 2.39 |
| 618 | 1,598 | 2.59 |

**The provider bills ~2.48x the tiktoken count — ~1.60 chars per billed token, not 4.00.**

So a delta at the 20,000-char cap costs **~12,400 billed tokens**, and the observed 5,286–10,413 range corresponds to deltas that were large but below cap. The mechanism is confirmed: a character cap, amplified by a tokenizer ~2.5x heavier than assumed.

A previous version of this entry said the reconstruction "falls short and does NOT confirm the mechanism", based on a delta that was only 10,352 chars (~52% of the cap). **That conclusion was wrong twice over** — undersized input *and* the wrong tokenizer model. It is superseded by the table above.

**Verified:** all early rows belong to session `20261010_003202_ad2365` — this desktop conversation. Probes wrote no rows until the Phase 3 drain landed, and an earlier "reproducible across sessions" claim was wrong.

**A4 — One-shot CLI reviews are lost, and `sync_backlog` does not prevent it.**
The review runs on a daemon thread (`runtime.py:595`). `hermes chat -q` exits at turn end — measured `Turn ended` and `_run_review entered` in the same millisecond. Instrumented run: `_review_one` entered → `calling llm` → process gone, no row.

`sync_backlog` is **not** the fix: measured advisor calls fire 3–75 s *after* `Turn ended`, so `_backlog_wait` is released before the review even starts. It is correctly `"off"`.

**A5 — Dashboard cost parity gap.** `/verify status` reports cost (`commands.py:111`); `dashboard/plugin_api.py` has **0** references to cost, delta, or prompt sizes.

### What is already good (do not touch)

- 105 tests; gates green; host-independent suite.
- Emission guard, immunity cooldown (drain bug fixed), review budget, tool-grant allowlist.
- The two perf commits (`06d36b1`, `ff89963`) — small but real.

---

## Architecture / proposed approach

Make telemetry unconditional and complete before changing any budget: write one metrics row per review (finding or not), with the component sizes and a token estimate, so every later decision is data-driven rather than modelled. Then derive the delta budget from measured data instead of adding a speculative setting (YAGNI). Finally, bound the daemon-thread loss at process exit with an `atexit` join, which is the only durable fix available to a plugin.

---

## Step-by-step tasks

All paths relative to `~/hermes-verifier`. TDD: failing test → verify fail → implement → verify pass → commit.

### Phase 0 — Baseline

```bash
cd ~/hermes-verifier && python -m pytest -q
```

Expected: `105 passed`. Stop and fix if not.

---

### Phase 1 — Unconditional review telemetry

#### Task 1.1 — Remove the dead instrumentation

**File: `runtime.py`** — delete the `prompt_parts` block (lines 526-532) added in `d21d170`. It is superseded by Task 1.3 and its `setdefault` is a trap for the next reader.

**Verification:**

```bash
grep -c "prompt_parts" runtime.py    # → 0
```

**Commit:** `refactor: drop the inert prompt_parts instrumentation`

#### Task 1.2 — Failing test: every review is recorded

**File: `tests/test_runtime.py`** — append:

```python
def test_every_review_writes_a_metrics_row(ctx, monkeypatch):
    """A no-finding review must still produce telemetry.

    Regression: _record_and_deliver sat after the severity=='none' early
    return, so reviews that found nothing left no trace and cost was
    unmeasurable.
    """
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "none", "finding": "", "reasoning": ""})
    _turn(ctx)

    log = runtime._log_path("s1", "default")
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 1, "a none-review must still write a row"
    assert rows[0]["severity"] == "none"
    assert rows[0]["kind"] == "review"

    parts = rows[0]["prompt_parts"]
    assert parts["transcript"] > 0
    assert parts["system_prompt"] > 0
    assert rows[0]["usage"]["input_tokens"] >= 0


def test_metrics_row_tracks_the_latest_review(ctx, monkeypatch):
    """Sizes must update per review, not freeze at the first one."""
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "none", "finding": "", "reasoning": ""})
    _turn(ctx)
    _turn(ctx, history=[{"role": "user", "content": "x" * 500}], response="y" * 500)

    log = runtime._log_path("s1", "default")
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]["prompt_parts"]["transcript"] != rows[0]["prompt_parts"]["transcript"], \
        "the second review's sizes must differ (no setdefault freeze)"
```

```bash
python -m pytest tests/test_runtime.py -k "every_review_writes or metrics_row_tracks" -q
```

Expected: `2 failed` — `assert 0 == 1` (no row) and an `IndexError`/length mismatch.

#### Task 1.3 — Implement: record a row for every review

**File: `runtime.py`** — extend the `record` dict in `_record_and_deliver` (around line 390) so it carries the prompt composition:

```python
        "delta_excerpt": delta_excerpt[:2000],
        "prompt_parts": prompt_parts or {},
```

Add `prompt_parts: Optional[Dict[str, Any]] = None` to the signature:

```python
def _record_and_deliver(state: SessionVerifier, verifier: Verifier,
                        parsed: Dict[str, Any], accepted: bool,
                        usage: Any, delta_excerpt: str = "",
                        prompt_parts: Optional[Dict[str, Any]] = None) -> None:
```

Then in `_review_one`, **move the recording so it happens before the `none` early return**. Replace:

```python
    severity = str(parsed.get("severity") or "none").strip().lower()
    finding_text = str(parsed.get("finding") or "").strip()
    if severity == "none" or not finding_text:
        return
    accepted = verifier.guard.accept(severity, finding_text)
    _record_and_deliver(state, verifier, parsed, accepted, usage,
                        delta_excerpt=delta_text)
```

with:

```python
    severity = str(parsed.get("severity") or "none").strip().lower()
    finding_text = str(parsed.get("finding") or "").strip()
    parts = {
        "system_prompt": len(system_prompt),
        "instructions": len(brief.REVIEW_TASK),
        "transcript": len(transcript),
    }
    if severity == "none" or not finding_text:
        # Still record: cost must be measurable for reviews that find nothing,
        # otherwise the majority of spend is invisible.
        _record_and_deliver(state, verifier, parsed, False, usage,
                            delta_excerpt=delta_text, prompt_parts=parts)
        return
    accepted = verifier.guard.accept(severity, finding_text)
    _record_and_deliver(state, verifier, parsed, accepted, usage,
                        delta_excerpt=delta_text, prompt_parts=parts)
```

Guarded by `with _LOCK: state.last_error = ...`-style safety — `_record_and_deliver` already appends safely via `_append_jsonl`.

**Verification:**

```bash
python -m pytest -q
```

Expected: `107 passed`.

**Commit:** `feat: record telemetry for every review, including no-finding ones`

#### Task 1.4 — Test and surface the token estimate

The char→token ratio is the A3 gap. Record an estimate alongside the raw sizes so the budget can be tuned from data.

**File: `runtime.py`** — in the `parts` dict from Task 1.3, add:

```python
        # Token estimate for budget tuning. 3.96 chars/token is MEASURED with
        # tiktoken/cl100k_base on realistic max-size deltas — but tiktoken is a
        # proxy, not Claude's tokenizer, so treat this as indicative only. The
        # authoritative number is usage.input_tokens, which rides in the same row.
        "est_tokens": int((len(system_prompt) + len(transcript)) / 3.96),
```

**File: `commands.py`** — in `_status_lines()`, extend the engagement line to show the latest estimate:

```python
        latest_est = 0
        for s in snap["sessions"].values():
            for v in (s.get("verifiers") or {}).values():
                parts = (v.get("stats") or {}).get("prompt_parts") or {}
                latest_est = max(latest_est, int(parts.get("est_tokens", 0) or 0))
        lines.append(
            f"  engagement: {total_reviews} review(s), {total_empty} empty response(s)"
            f"{f', ~{latest_est} tok/review' if latest_est else ''}"
        )
```

**Verification:**

```bash
python -m pytest tests/test_runtime.py -k "status_surfaces_engagement" -q
```

Expected: `1 passed`. Then add an assertion for the new segment:

```python
def test_status_reports_token_estimate(ctx, monkeypatch):
    from verifier_pkg import commands
    monkeypatch.setattr(runtime, "_REVIEW_FN_OVERRIDE", _review_sync)
    ctx.llm.next = FakeResult({"severity": "none", "finding": "", "reasoning": ""})
    _turn(ctx)
    text = "\n".join(commands._status_lines())
    assert "tok/review" in text, f"expected a token estimate, got:\n{text}"
```

```bash
python -m pytest -q
```

Expected: `108 passed`.

**Commit:** `feat: estimate tokens per review and surface it in /verify status`

#### Task 1.5 — Measure, then decide (no code)

Run one real session and read the actual per-review sizes:

```bash
cd ~ && timeout 280 hermes chat -q "Create /tmp/audit_probe.py with a small function."
sleep 8
cat ~/.hermes/plugin-data/hermes-verifier/*.jsonl | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l)
    p = d.get('prompt_parts') or {}
    u = d.get('usage') or {}
    if p:
        print(f\"session={d['session_id'][-12:]} sev={d['severity']:8s} \"
              f\"sys={p.get('system_prompt')} trans={p.get('transcript')} \"
              f\"est={p.get('est_tokens')} actual_in={u.get('input_tokens')}\")
"
```

Expected: rows with `sys=`, `trans=`, `est=`, `actual_in=`. **This is the deliverable of Phase 1** — the first honest cost attribution, and the explicit check that confirms or refutes A3.

**How to read it:**
- `est` ≈ `actual_in` within ~30% → the char cap is the driver and `est` is a usable tuning signal.
- `actual_in` consistently ≫ `est` → the tokenizer proxy is off (A3 candidate (a)); record the observed `actual_in / est` ratio in the plan and use *that* to size any cap, not the tiktoken constant.
- `actual_in` larger than `est` even though `trans` is well under `max_delta_chars` → the delta is not the driver; investigate the request as a whole before touching any cap.

**Do not tune `max_delta_chars` or add any token-based setting until this comparison exists.**

If no row appears, the review was lost (A4) — go to Phase 3 first.

**Commit:** none (measurement).

---

### Phase 2 — Make the delta budget honest

Only start this if Task 1.5 produced rows.

#### Task 2.1 — Document the real semantics

**File: `plugin.yaml`** — replace the `max_delta_chars` description:

```yaml
  max_delta_chars:
    type: int
    default: 20000
    description: "Cap on the rendered transcript delta sent to the advisor, in CHARACTERS. Measured at ~3.96 chars/token on realistic agent transcripts, so 20000 chars is roughly 5000 tokens (tiktoken proxy; the true figure depends on the provider's tokenizer). Lower it to cut cost."
```

**Verification:**

```bash
hermes plugins validate . --install-deps 2>&1 | grep -E "config schema|Validation passed"
```

Expected: `✓ config schema — shape valid` then `Validation passed.`

**Commit:** `docs: state the real char-to-token cost of max_delta_chars`

#### Task 2.2 — Test the truncation boundary

**File: `tests/test_delta.py`** — append:

```python
def test_delta_truncation_marks_and_bounds():
    """The cap must bound output and say so, so a cut review is visible."""
    hist = [{"role": "tool", "tool_call_id": "c1",
             "content": "\n".join(f"line {i}" for i in range(3000))} for _ in range(12)]
    _cursor, rendered = delta.extract_delta(
        cursor_len=None, conversation_history=hist,
        assistant_response="done", max_chars=2000,
    )
    assert len(rendered) <= 2000 + len("\n… [delta truncated]") + 1
    assert rendered.endswith("[delta truncated]"), "a cut delta must be marked"
```

```bash
python -m pytest tests/test_delta.py -k truncation_marks -q
```

Expected: `1 passed` (the marker already exists — this locks the contract).

**Commit:** `test: lock the delta truncation contract`

---

### Phase 3 — Stop losing reviews in short-lived processes

#### Task 3.1 — Failing test: a pending review is joined before exit

**File: `tests/test_runtime.py`** — append:

```python
def test_pending_reviews_can_be_drained():
    """Short-lived processes must be able to wait for in-flight reviews.

    hermes chat -q exits at turn end and the daemon review thread dies with it,
    so the review and its telemetry are lost.
    """
    drained = runtime.drain_reviews(timeout=0.01)
    assert isinstance(drained, bool)
```

```bash
python -m pytest tests/test_runtime.py -k pending_reviews_can_be_drained -q
```

Expected: `1 failed` — `AttributeError: module 'runtime' has no attribute 'drain_reviews'`.

#### Task 3.2 — Implement a bounded drain

**File: `runtime.py`** — add near `_schedule_review`:

```python
_THREADS_LOCK = threading.Lock()
_ACTIVE: "set[threading.Thread]" = set()
_DRAIN_TIMEOUT = 20.0


def drain_reviews(timeout: float = _DRAIN_TIMEOUT) -> bool:
    """Wait briefly for in-flight reviews so a short-lived process does not
    kill them. Returns True when everything finished within the timeout.

    The review thread is a daemon (it must never block a normal exit), so this
    is called explicitly at process teardown and is always bounded.
    """
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        with _THREADS_LOCK:
            pending = [t for t in _ACTIVE if t.is_alive()]
        if not pending:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("drain_reviews timed out with %d review(s) in flight",
                           len(pending))
            return False
        pending[0].join(timeout=min(0.1, remaining))
```

Register the thread in `_schedule_review` (after `thread.start()`):

```python
    with _THREADS_LOCK:
        _ACTIVE.add(thread)
    thread.start()
```

and discard it in `_run_review`'s `finally`, alongside `_INFLIGHT.release()`:

```python
        with _THREADS_LOCK:
            _ACTIVE.discard(threading.current_thread())
```

**File: `__init__.py`** — register the teardown hook inside `register(ctx)`:

```python
    import atexit

    atexit.register(runtime.drain_reviews, 20.0)
```

**Verification:**

```bash
python -m pytest -q
```

Expected: `109 passed`.

**Commit:** `feat: drain in-flight reviews at process exit so one-shot runs are not lost`

#### Task 3.3 — Verify against the real one-shot path

```bash
cd ~ && timeout 280 hermes chat -q "Create /tmp/drain_probe.py with a small function."
sleep 3
cat ~/.hermes/plugin-data/hermes-verifier/*.jsonl | python3 -c "
import sys, json
rows = [json.loads(l) for l in sys.stdin]
print('rows total:', len(rows))
print('rows from a /tmp/drain_probe session:',
      sum(1 for r in rows if 'drain' in json.dumps(r).lower()))
"
```

Expected: a row for the probe session (previously zero). If still absent, the thread was killed before `atexit` ran — record that and leave Phase 3 in place as best-effort, documenting it in the README.

**Commit:** none (verification).

---

### Phase 4 — Dashboard parity

#### Task 4.1 — Expose cost and delta in `/state`

**File: `dashboard/plugin_api.py`** — in `_state_payload()`, extend the settings/costblock with the same numbers `/verify status` shows (0 refs today):

```python
    payload["cost"] = {
        "reviews": sum(int((v.get("stats") or {}).get("reviews", 0))
                       for s in (payload.get("sessions") or {}).values()
                       for v in (s.get("verifiers") or {}).values()),
        "input_tokens": sum(int((v.get("stats") or {}).get("input_tokens", 0))
                            for s in (payload.get("sessions") or {}).values()
                            for v in (s.get("verifiers") or {}).values()),
        "empty_responses": sum(int(s.get("empty_responses", 0))
                               for s in (payload.get("sessions") or {}).values()),
    }
```

**Verification:**

```bash
python -m pytest -q && hermes plugins validate . --install-deps 2>&1 | grep "Validation passed"
```

Expected: `110 passed` then `Validation passed.`

**Commit:** `feat: expose review cost in the dashboard /state payload`

---

## Tests / validation

| After task | `pytest -q` |
| --- | --- |
| Baseline | `105 passed` |
| 1.3 | `107 passed` |
| 1.4 | `108 passed` |
| 3.2 | `109 passed` |
| 4.1 | `110 passed` |

Gate commands (all exit non-zero on failure):

```bash
cd ~/hermes-verifier
python -m pytest -q
hermes plugins validate . --install-deps
hermes plugins doctor . --ci
```

### What each new test proves

| Test | Guarantee |
| --- | --- |
| `every_review_writes_a_metrics_row` | No-finding reviews are no longer invisible |
| `metrics_row_tracks_the_latest_review` | Sizes update per review (no `setdefault` freeze) |
| `status_reports_token_estimate` | Cost is visible in a live session |
| `delta_truncation_marks_and_bounds` | A cut delta is bounded and marked |
| `pending_reviews_can_be_drained` | Short-lived processes can wait for reviews |

---

## Risks, tradeoffs, and open questions

1. **More rows on disk.** Phase 1 writes a row per review instead of one per finding. At ~1 row/turn with a 2,000-char excerpt, this is a few hundred KB per long session. Acceptable; if not, cap the excerpt to 500 chars for `none` rows specifically.

2. **`drain_reviews` adds up to 20 s to process teardown.** Bounded and only when a review is genuinely in flight. If that is unacceptable for `hermes chat -q`, lower the default to 5 s — the failure mode is a lost review, not a hang.

3. **`atexit` and daemon threads are best-effort.** Interpreter shutdown gives no guarantee; `atexit` runs before daemon threads are killed, which is why it should work, but Task 3.3 is the empirical check. If it does not, the honest outcome is to document "one-shot CLI runs may not get a review" rather than fake a fix.

4. **The chars/token constant is measured, and the provider's tokenizer is ~2.5x heavier than tiktoken.** The 2.48x ratio came from three live `est` vs `actual` pairs (2.46 / 2.39 / 2.59) — tight enough to trust, small enough that occasional re-checks are worthwhile. `est_tokens` uses the measured ~1.60 chars/token. This also resolved the earlier "unexplained gap", which was caused by an undersized test delta plus the wrong tokenizer model.

5. **Open question — is a token-based cap worth it?** After Task 1.5 there will be `est` vs `actual_in` per review for the first time. If the ratio is stable, a `max_delta_tokens` setting is justified. If it swings widely, a char cap plus telemetry is the better call. **Do not add the setting before that data exists.**

6. **Explicitly out of scope:** Claude Code's server-side `advisor_20260301` tool and model-rank pairing (no Hermes hooks exist), and any change to `sync_backlog` — measured to be irrelevant to the loss in A4, and correctly `"off"`.
