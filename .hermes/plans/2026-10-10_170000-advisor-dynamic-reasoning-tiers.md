> **SUPERSEDED.** Kept for the record. The D1-omni / dynamic-tier direction
> this describes was abandoned on measured evidence -- `d1-omni-600M` could not
> separate a paraphrase from a distinct finding, and the semantic dedupe stage it
> proposed is disabled and documented UNVALIDATED in `sidecar/NEGATIVE-RESULTS.md`.
> The current work is in `2026-10-11_074500-closeout-v0.4.0.md`.

# Plan — Dynamic Reasoning Levels for the Advisor

## Goal

Let the advisor spend reasoning effort in proportion to the situation — cheap effort on routine turns, deep effort when the delta looks risky — using Hermes's auxiliary-task route.

## Current context / assumptions

HEAD `f09ff95`, 114 tests, gates clean. Advisor runs on `deepseek/deepseek-flash`.

### Feasibility — verified by reading the host, not assumed

| Question | Answer | Evidence |
| --- | --- | --- |
| Does `ctx.llm` accept a reasoning/effort arg? | **No** | `grep -cE "reasoning\|effort\|thinking" agent/plugin_llm.py` → **0** |
| Does DeepSeek support effort levels? | **Yes** — `low, medium, high, max` | `DEEPSEEK_V4_EFFORTS` in `agent/reasoning_effort.py` |
| Is `deepseek-flash` thinking-capable? | **Yes** | `_THINKING_CAPABLE_IDS = frozenset({"deepseek-flash"})` in `plugins/model-providers/deepseek/__init__.py` |
| Does effort reach the wire? | **Yes** — top-level `reasoning_effort` | `thinking_toggle_extras` returns `({...}, {"reasoning_effort": clamped})` |
| Can a plugin route through an effort-bearing slot? | **Yes** | `_AUX_ROUTE_KEYS` includes `reasoning_effort`; `_host_kwargs` forwards `task=`; `ctx.register_auxiliary_task()` is documented |
| Is this surface exercised today? | **No** — zero plugins use it | `grep -rn register_auxiliary_task plugins/` → no hits |

**So: dynamic effort is reachable, but only via auxiliary task slots — there is no
per-call reasoning parameter for plugins.** The plugin chooses *which tier* per
review; the tier's effort is operator config.

### Also verified while investigating (relevant to the DeepSeek switch)

DeepSeek's native API declares `unsupported_response_formats=("json_schema",)` — it
implements only `json_object`, and `json_schema` is a guaranteed HTTP 400. The host
downgrades gracefully in `agent/auxiliary_structured_output.py`, and live DeepSeek
reviews parsed correctly (`sev=caution accepted=True`). Worth knowing: on this
provider the plugin's `REVIEW_SCHEMA` is **not enforced by the provider**, only by
local validation.

---

## Architecture / proposed approach

Register two (optionally three) auxiliary tasks as effort tiers, then have the
review pick a tier from signals already in the delta — tool-call count, delta
size, and whether the previous review raised a finding. No host change, no new
dependency: the routing, the effort translation, and the config surface all
already exist.

---

## Step-by-step tasks

TDD throughout; commit per task.

### Phase 0 — Baseline

```bash
cd ~/hermes-verifier && python -m pytest -q
```

Expected: `114 passed`.

### Task 1 — Verify the task route works at all, before building on it

This is the riskiest assumption. Prove it with a spike, not a feature.

**File: `tests/test_runtime.py`** — append:

```python
def test_aux_task_routing_is_available():
    """The plugin must be able to register an auxiliary task.

    The whole dynamic-effort design rests on `task=` routing; if the probe
    refuses it, the design is dead and we should stop here.
    """
    class Ctx:
        def __init__(self):
            self.tasks = []
        def register_auxiliary_task(self, key, **kw):
            self.tasks.append((key, kw))

    ctx = Ctx()
    ctx.register_auxiliary_task("advisor_deep", display_name="Advisor (deep)",
                                description="High-effort advisor pass.")
    assert ctx.tasks[0][0] == "advisor_deep"
```

```bash
python -m pytest tests/test_runtime.py -k aux_task_routing -q
```

Expected: `1 passed`. This only proves the call shape, not host acceptance — Task 3
is the real check.

**Commit:** `test: pin the auxiliary-task registration shape`

### Task 2 — Tier selection from the delta

**File: `tests/test_runtime.py`** — append:

```python
def test_effort_tier_selection(ctx):
    """Routine deltas get the cheap tier; risky ones get the deep tier."""
    from verifier_pkg import brief

    # A tiny delta with no tool calls is routine.
    assert brief.select_effort_tier(
        delta="[user] what time is it\n[assistant final] It is noon.",
        tool_calls=0, prior_severity="",
    ) == "routine"

    # Many tool calls in one delta is where things go wrong.
    assert brief.select_effort_tier(
        delta="x" * 400, tool_calls=9, prior_severity="",
    ) == "deep"

    # A recent finding means the situation is already live.
    assert brief.select_effort_tier(
        delta="x" * 40, tool_calls=0, prior_severity="caution",
    ) == "deep"
```

```bash
python -m pytest tests/test_runtime.py -k effort_tier_selection -q
```

Expected: `1 failed` — `AttributeError: module ... has no attribute 'select_effort_tier'`.

### Task 3 — Implement the selector and register the tiers

**File: `brief.py`** — add:

```python
ROUTINE_TIER = "advisor_routine"
DEEP_TIER = "advisor_deep"

_DEEP_TOOL_CALLS = 5      # a delta with this many tool calls warrants a hard look
_DEEP_DELTA_CHARS = 300


def select_effort_tier(*, delta: str, tool_calls: int = 0,
                       prior_severity: str = "") -> str:
    """Pick an auxiliary-task tier for this review.

    Cheap by default: most turns are routine and a low-effort pass is enough.
    Escalate when the delta is large, touched many tools, or follows a finding.
    """
    if prior_severity in ("caution", "blocker"):
        return DEEP_TIER
    if tool_calls >= _DEEP_TOOL_CALLS:
        return DEEP_TIER
    if len(delta or "") >= _DEEP_DELTA_CHARS:
        return DEEP_TIER
    return ROUTINE_TIER
```

**File: `__init__.py`** — inside `register(ctx)`, after the hooks:

```python
    # Effort tiers. Operators pin each tier's reasoning_effort in
    # `auxiliary.<key>`; deepseek-flash accepts low|medium|high|max.
    ctx.register_auxiliary_task(
        "advisor_routine", display_name="Advisor (routine)",
        description="Low-effort advisor pass for routine turns.")
    ctx.register_auxiliary_task(
        "advisor_deep", display_name="Advisor (deep)",
        description="High-effort advisor pass for risky or busy turns.")
```

**File: `runtime.py`** — in `_review_one`, choose the tier and pass it:

```python
        tool_calls = delta_text.count("[tool") + delta_text.count("tool_call")
        prior = ""
        for v in state.verifiers.values():
            if v.pending:
                prior = str(v.pending.get("severity") or "")
                break
        tier = brief.select_effort_tier(delta=delta_text, tool_calls=tool_calls,
                                        prior_severity=prior)
```

and add `task=tier` to the `_call_structured(...)` call plus a `task` parameter to
`_call_structured` that forwards into its `kwargs`.

**File: `plugin.yaml`** — document the operator side in the description of
`verifier_model`, and note that effort is configured per tier:

```yaml
  # Tiers are registered as auxiliary tasks; set effort with:
  #   hermes config set auxiliary.advisor_routine.reasoning_effort low
  #   hermes config set auxiliary.advisor_deep.reasoning_effort high
```

```bash
python -m pytest -q
```

Expected: `116 passed`.

**Commit:** `feat: effort tiers registered and selected per review`

### Task 4 — Configure the tiers and verify on the wire

```bash
cd ~
hermes config set auxiliary.advisor_routine.reasoning_effort low
hermes config set auxiliary.advisor_deep.reasoning_effort high
hermes config get auxiliary.advisor_routine.reasoning_effort   # -> low
```

Then run one routine and one busy task and read the wire from the log:

```bash
timeout 280 hermes chat -q "Create /tmp/tier_routine.py with a one-line hello function."
grep -E "plugin_llm.*hermes-verifier" ~/.hermes/logs/agent.log | tail -2
```

Expected: `provider=deepseek model=deepseek-flash`, and the review row's
`prompt_parts` present. **If the effort does not appear on the wire** (no
`reasoning_effort` in the request, or the task route is refused), stop — the
design does not hold, and record that rather than shipping tiers that do nothing.

### Task 5 — Record the outcome

Add a short section to `PLAN.md` stating whether tiers took effect, with the
evidence. If they did not, say so plainly and revert Tasks 3-4.

---

## Tests / validation

| After task | `pytest -q` |
| --- | --- |
| Baseline | `114 passed` |
| 1 | `115 passed` |
| 3 | `116 passed` |

Gates:

```bash
python -m pytest -q
hermes plugins validate . --install-deps
hermes plugins doctor . --ci
```

## Risks, tradeoffs, and open questions

1. **The load-bearing assumption is untested.** No plugin uses `register_auxiliary_task`
   today, so Task 3's wire check is the first real exercise of it. If the task
   route is refused or silently ignores `reasoning_effort`, tiers become dead
   config — worse than not shipping them. Task 4 is a stop condition, not a
   formality.

2. **Effort is operator config, not plugin code.** The plugin picks the *tier*;
   the operator pins each tier's effort. A user who never configures it gets two
   identical tiers and no benefit. That is a discovery problem the README must
   solve, not a correctness one.

3. **Escalation is a heuristic, and it is unmeasured.** Tool-call count and delta
   size are proxies for "risky". They could easily be wrong — this plan does not
   claim they predict anything. If tiers ship, the review rows now carry enough
   telemetry (`prompt_parts`, `severity`, `usage`) to check whether deep-tier
   reviews actually find more.

4. **`json_schema` is not enforced on DeepSeek.** The provider implements only
   `json_object`, so structure is guaranteed only by local validation. A tier
   experiment should confirm parse rates rather than assume them.

5. **Cost is not obviously improved.** Reasoning effort raises *output* tokens on
   the deep tier. The bet is that most turns land on the cheap tier, which is
   exactly the bet the prompt-instruction experiment failed to demonstrate. Treat
   "cheaper" as unproven until per-tier telemetry says otherwise.

6. **Out of scope:** a per-call reasoning parameter on `ctx.llm`. It would be the
   cleaner design, but `plugin_llm.py` has no such surface and adding one is a
   Hermes-core change, not a plugin change.
