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

## Application 1 — semantic dedupe (SHIPPED, off by default)

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
```

### The honest scorecard: high precision, LOW RECALL

| metric | value |
| --- | --- |
| **recall — paraphrases caught (p >= 0.5)** | **2 / 5 = 40%** |
| precision — wrongly suppressed distinct pairs | **0 / 5** |
| separation | +0.400 |
| pairwise accuracy | 76% |
| suppressed at threshold 0.5 / 0.8 / 0.9 / 0.95 | 2 / 2 / 2 / 2, all correct |

**Both numbers matter together.** The safety claim ("0 wrongly suppressed at
every threshold") is real, but it is only meaningful next to the recall: **this
detects a minority of restatements, not all of them.** Three of five paraphrases
scored 0.020, 0.049 and 0.230 — well below any usable threshold — so they will
still re-surface exactly as they do today.

So the value is: **a small reduction in repeat findings, at zero added risk.**
Not a fix for duplicate nagging. The exact-text guard remains the primary
mechanism; this only trims a fraction of the remainder.

**Shipped default: `semantic_dedupe_threshold: 0.0` — the stage is OFF.** With it
off, behaviour is byte-identical to before. The owner decides whether a ~40%
catch rate on restatements is worth running the sidecar and enabling it.

### Why acting on it is safe, when gating was not

- a wrongly **suppressed** duplicate loses a *repeat* of a finding that already
  fired — annoying, recoverable, self-correcting
- a wrongly **skipped** review loses a finding that would otherwise have existed
  — an unobservable quality loss

Only the second is irrecoverable, and the gate was unsafe at *every* threshold
while this is safe at every threshold. But "safe" is not "effective": read both
columns.

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

---

## Can we train this model for better call gating?

Short answer: **legally yes, practically yes with custom work, but it is the
wrong first move.** Test a bigger off-the-shelf decision model before building a
training pipeline.

### What is permitted

The blog states the models are *"Open-weight — Download, fine-tune, and deploy
without restrictions."* So there is no licensing barrier.

### What is NOT provided

Liquid's official fine-tuning repo, **LEAP Finetune**, is scoped to *"Liquid
Foundation Models (LFMs)"* — the generative family. Its examples train
`LFM2-1.2B` with SFT/DPO/GRPO, and the blog draws the distinction explicitly:

> "Unlike our generative Liquid Foundation Models (LFMs), our d1 decision models
> don't produce tokens. Instead, they produce an answer in a single forward pass."

A decision model ends in a **classification head over option logits**, not a
token distribution. Token-level SFT/DPO/GRPO does not apply; training one means
writing a cross-entropy objective over the option positions yourself. **There is
no documented recipe or tooling for training decision models.**

### The real bottleneck is labels, not compute

A 587M model is cheap to fine-tune — a single consumer GPU, plausibly a Mac. The
cost is the training set: hundreds to thousands of labelled
`(delta_state, should_review)` pairs. Ten were hand-labelled for this
investigation; a gate needs orders of magnitude more.

**The label pipeline already exists in design, from the evaluation plan:**
shadow sampling. A shadowed skip that later produces a caution or blocker is a
*positive* label — a case the gate got wrong. That is exactly the data a gate
needs, gathered automatically from real traffic. The decision journal records the
input state and the confidence for every decision, so labelling is a join, not a
chore.

So the honest order of operations is:

1. **Shadow sampling must run first**, or there are no labels at all. This is
   also the only mechanism that reveals gate errors, since a wrongly skipped
   review is otherwise invisible.
2. Collect labels until the positive class is large enough to train on. At a 10%
   shadow rate this is slow — many hundreds of sessions.
3. Only then train.

### The cheaper alternative that should be tried first

**Test `d1-3B` zero-shot.** Liquid scores it **48.57** on their Decision Index
against d1-omni-600M's **15.95** — three times the decision quality, and the docs
label it *"Recommended"* versus *"Smallest"*. It is 3.1B parameters, runs on the
same stack, and needs no labels, no training loop, and no eval harness.

`sidecar/official_pattern.py` re-tests it in one command by changing `MODEL_ID`.
If d1-3B gates acceptably, training is unnecessary; if it does not, that is
strong evidence the task needs training rather than a bigger zero-shot model —
and now with a baseline to beat.

### Why this may not be worth doing at all

**A locally trained checkpoint cannot ship in a catalog plugin.** This plugin is
destined for the Hermes plugin catalog, installed at a pinned commit by users who
have no model and no GPU. A custom decision head is a personal optimization, not
a distributable feature — so its value accrues only to this machine.

Weigh that against the alternative already shipped: semantic dedupe at ~40%
recall and zero risk, and effort tiering, both of which work with the model as-is.

### Recommendation

1. Do not train yet.
2. Re-run the harness against **d1-3B** (one-line change).
3. Implement **shadow sampling** if a gate is still wanted after that — it is
   required for labels *and* it is the only way to see gate errors.
4. Train only with a real positive-class count and a d1-3B baseline to beat.

---

## SUPERSEDED: dedupe should use an embedder, not the decision model

Following a size review, the semantic-dedupe application was re-implemented
against a sentence embedder. Same 10 labelled pairs, direct comparison
(`embed_compare.py`):

| | all-MiniLM-L6-v2 | d1-omni-600M |
| --- | --- | --- |
| params | **22.7M** | 587M |
| on-disk (measured) | **87 MB** | 2.2 GB |
| separation | **+0.451** | +0.401 |
| pairwise accuracy | **25/25 = 100%** | 19/25 = 76% |
| recall at 0 wrong suppressions | **5/5** | 3/5 |

**The 25x smaller model wins on every metric.** The decision model was also the
wrong tool on principle:

- **206M of its 587M params are vision and audio encoders** this plugin never
  touches — the card itself splits it as *"381M shared trunk, 94M vision, 112M
  audio"*. For a text-only plugin that is provably dead weight.
- It required a bespoke question with hand-written `criteria` that had to be
  tuned and measured.
- It required `trust_remote_code`, executing model-supplied Python.
- Cosine similarity needs none of the above and is deterministic.

**Thresholds do not transfer.** Cosine is a *similarity*, not a probability.
Paraphrase pairs score 0.355-0.780 and distinct pairs 0.064-0.191, so the
operating point is **0.30** — the decision model's 0.9 would suppress nothing at
all, silently defeating the stage. A test pins the recommended threshold against
the measured scores and asserts 0.9 suppresses nothing, so this cannot regress.

Decision-model artifacts (`probe.py`, `polarity*.py`, `official_pattern.py`,
`dedupe_probe.py`, `discriminate.py`, `variants.py`, `threshold_check.py`) are
kept as the evidence trail for why gating was rejected and why the model was
dropped. `embed_server.py` supersedes `server.py` for the shipped feature.
