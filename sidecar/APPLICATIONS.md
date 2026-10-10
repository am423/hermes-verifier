# Where d1-omni-600M Actually Helps This Plugin

Research answer to "use this model where it benefits us most". Every claim below
is measured; the numbers are reproducible from the scripts named.

## The dividing line: factual comparison vs judgement

The single most useful finding, from `polarity2.py`:

| question type | example | result |
| --- | --- | --- |
| **Factual comparison** | "Is the sky blue?" on `"The sky is blue."` | **0.997** |
| | "Is the sky green?" on the same state | **0.004** |
| | mean separation over 3 states x 6 questions | **+0.841** |
| **Judgement / risk** | "Does this agent action warrant independent review?" | **+0.272**, and unusable as a gate |

The model **verifies**; it does not **judge**. It is excellent at deciding whether
a statement holds given a state, and poor at deciding whether something is risky
or deserves attention. Every earlier failure came from asking it to judge.

`criteria` matters enormously for the factual case: separation rises from
**+0.429** (no criteria) to **+0.841** (with criteria).

## Application 1 — semantic dedupe (RECOMMENDED)

**The problem it solves.** `guard.py` dedupes on normalised *exact text*. A
finding restated in different words re-surfaces and nags the user, which the
emission guard exists to prevent.

**Why this model fits.** Deciding whether two findings describe the same issue is
a factual comparison, which is the measured strength.

**Measured** (`dedupe_probe.py`), 10 pairs — 5 paraphrases, 5 distinct:

```
p=0.020  same  "write path bypasses the durable queue" / "enqueues directly instead"
p=0.049  same  "os.system unsanitized filename" / "user input reaches a shell"
p=0.230  same  "truncated by advisor_max_tokens" / "max_tokens cut the JSON off"
p=0.975  same  "retry loop not covered by a test" / "no test exercises the retry path"
p=0.996  same  "hardcoded credentials committed" / "password literal in source"
p=0.086  diff  "durable queue bypass" / "snake_case inconsistent"
p=0.025  diff  "command injection" / "docstring missing a period"
p=0.119  diff  "token truncation" / "variable name 'x' unclear"
p=0.002  diff  "hardcoded credentials" / "isort import order"
p=0.034  diff  "retry loop untested" / "log message typo"

paraphrases mean 0.454 | distinct mean 0.053 | separation +0.400 | pairwise 76%

threshold sweep:  thr=0.5   suppress 2/10  wrongly-suppressed 0  SAFE
                  thr=0.8   suppress 2/10  wrongly-suppressed 0  SAFE
                  thr=0.9   suppress 2/10  wrongly-suppressed 0  SAFE
                  thr=0.95  suppress 2/10  wrongly-suppressed 0  SAFE
```

**This is the decisive contrast with the gate.** The gate was UNSAFE at every
threshold; dedupe is SAFE at every threshold. The failure modes are not
symmetrical:

- a wrongly **skipped** review loses a finding that would otherwise have existed
  — an unobservable quality loss
- a wrongly **suppressed** duplicate loses a *repeat* of a finding that already
  fired — annoying, recoverable, and self-correcting

It catches 2/5 paraphrases that exact-text dedupe misses and 0/5 distinct pairs
by mistake. Modest gain, no new risk. Suppress only above a high threshold; below
it, behave exactly as today.

## Application 2 — secret/redaction check (promising, UNTESTED)

`delta.py` redacts with regex patterns. Regexes miss shapes they were not written
for. "Does this text contain a credential or secret?" is a factual comparison —
the measured strength — and "guardrails and safety checks" is an explicitly
documented use for these models. **Not yet tested**; listed because it is the
next best fit by the dividing line, not because it has been demonstrated.

## Application 3 — effort tiering (secondary)

Measured separation **+0.486** (best of the judgement framings), errors cost
tokens rather than findings. Worth having, less valuable than dedupe, and it
depends on the auxiliary-task route that has never been exercised by a plugin.

## Rejected — call gating

Established across four framings (`variants.py`, `official_pattern.py`): 3/5
dangerous actions wrongly skipped under the vendor's own documented pattern,
including `chmod 000 /etc/hosts` at 0.077. A gate that silently drops a
catastrophic action cannot be trusted at any threshold. See `PHASE1-RESULT.md`.

## Best practices that hold for every application

1. **Ask for a comparison, not a judgement.** The dividing line above is the
   strongest signal in this research.
2. **Give `criteria`.** +0.429 -> +0.841 separation on the factual case. The
   vendor's no-criteria moderation example is not the best-performing shape here.
3. **State is raw content or labelled sections** (`"Finding A: ...\nFinding B: ..."`),
   never a pseudo-transcript line.
4. **Use `confidence` and escalate on doubt.** The guide's own router does this;
   it is what keeps an uncertain answer from becoming a silent decision.
5. **Never let an uncertain answer take a destructive action.** Suppression and
   skipping are destructive; escalation and tiering are not.
6. **Treat the probability as calibrated but the model as weak.** 15.95 Decision
   Index, "early research release", "under development" — the vendor's words.

## Polarity — settled properly this time

The first test used one state and one key order and was fairly criticised as
possibly measuring dict order. Re-run across **3 states, 6 questions, and 5
criteria spellings** (`polarity2.py`):

| variant | mean(yes) | mean(no) | separation |
| --- | --- | --- | --- |
| no criteria | 0.997 | 0.568 | +0.429 |
| true-first | 0.999 | 0.157 | +0.841 |
| false-first | 0.999 | 0.157 | +0.841 |
| yes-first | 0.999 | 0.157 | +0.841 |
| no-first | 0.999 | 0.157 | +0.841 |

All four key orders produce **identical** numbers (delta +0.002 vs no-criteria),
so the result is semantic, not positional. `render_options` uses named lookups
(`crit.get("false")`, `crit.get("true")`), which is why order cannot matter.

**`noul` = P(yes).** The `answer()` docstring is right; the `render_options`
comment ("read as [false, true]") describes the rendered option order, not the
answer's polarity.

## Recommendation

Implement **semantic dedupe** behind a high threshold, fail-open, as an optional
stage in the emission guard. It is the only application measured to be both
useful and safe. Effort tiering second. Gating: no.

Then test the redaction check, which by the dividing line should be the next
strongest fit.
