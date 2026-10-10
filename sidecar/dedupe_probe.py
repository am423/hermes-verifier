"""Where d1-omni actually helps: SEMANTIC DEDUPE of findings.

Everything tested so far asked the model to make a JUDGMENT ("is this risky?",
"should a reviewer look?"). It is bad at those. The polarity run showed what it
is good at: FACTUAL COMPARISON with criteria ("Is the sky blue?" -> 0.997,
"Is the sky green?" -> 0.004).

That matters because the plugin's emission guard currently dedupes on
normalised EXACT text. A restated finding -- same underlying issue, different
words -- therefore re-surfaces and nags. Deciding whether two findings describe
the same issue is a factual comparison, which is the strength just measured.

This tests that directly, on paraphrase pairs and distinct pairs.
"""

from transformers import AutoModel

MODEL_ID = "LiquidAI/d1-omni-600M"

# (first, second, same_issue)
PAIRS = [
    # Same underlying issue, restated.
    ("The write path bypasses the durable queue in src/jobs/",
     "src/jobs/write.py enqueues directly instead of going through the queue", True),
    ("os.system with an unsanitized filename is command injection",
     "user input reaches a shell unescaped, allowing arbitrary commands", True),
    ("The response was truncated by advisor_max_tokens",
     "a max_tokens cap cut the JSON off mid-object", True),
    ("The retry loop is not covered by a test",
     "no test exercises the retry path", True),
    ("Hardcoded credentials were committed",
     "a password literal appears in the committed source", True),
    # Different issues.
    ("The write path bypasses the durable queue in src/jobs/",
     "the function name uses snake_case inconsistently", False),
    ("os.system with an unsanitized filename is command injection",
     "the docstring is missing a period", False),
    ("The response was truncated by advisor_max_tokens",
     "the variable name 'x' is unclear", False),
    ("Hardcoded credentials were committed",
     "the import order does not follow isort", False),
    ("The retry loop is not covered by a test",
     "the log message has a typo", False),
]

Q = {
    "same": {
        "type": "noul",
        "instructions": "Do these two review findings describe the same underlying issue?",
        "criteria": {"yes": "they report the same problem, even in different words",
                     "no": "they report different, unrelated problems"},
    },
}


def main() -> None:
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype="float32")

    rows = []
    for first, second, same in PAIRS:
        state = f"Finding A: {first}\nFinding B: {second}"
        p = (model.system_one(state, Q).get("answers", {}).get("same") or {}).get("noul")
        rows.append((first[:38], second[:38], same, p))
        print(f"p={p:.3f} same={same!s:5s}  A:{first[:40]}  B:{second[:40]}")

    same_p = [r[3] for r in rows if r[2]]
    diff_p = [r[3] for r in rows if not r[2]]
    ms, md = sum(same_p) / len(same_p), sum(diff_p) / len(diff_p)
    pairs = [(a, b) for a in same_p for b in diff_p]
    good = sum(1 for a, b in pairs if a > b)
    print(f"\nparaphrases mean p = {ms:.3f}  (n={len(same_p)})")
    print(f"distinct    mean p = {md:.3f}  (n={len(diff_p)})")
    print(f"separation = {ms - md:+.3f}   pairwise accuracy = {good}/{len(pairs)}"
          f" = {good / len(pairs):.0%}")

    # The threshold that matters: only suppress a repeat when confident.
    print("\nthreshold sweep (suppress-only-when-confident):")
    for thr in (0.5, 0.8, 0.9, 0.95):
        suppressed = [r for r in rows if r[3] >= thr]
        wrong = [r for r in suppressed if not r[2]]     # distinct issue wrongly suppressed
        good_sup = [r for r in suppressed if r[2]]
        print(f"  thr={thr:<5} suppress {len(suppressed):2d}/10  "
              f"correctly-suppressed {len(good_sup)}  WRONGLY-suppressed {len(wrong)}"
              + ("  SAFE" if not wrong else "  UNSAFE"))


if __name__ == "__main__":
    main()
