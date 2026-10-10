"""Prove d1-omni answers a multi-question set, with confidence, at zero output cost.

API shape verified against the model's own source before writing this. All
citations below refer to the file in the Hugging Face repository --
huggingface.co/LiquidAI/d1-omni-600M/raw/main/prompt.py (131 lines) and
.../modeling_d1.py -- not to any file in this plugin:

  * modeling_d1.py system_one(state, questions: Mapping, images=None, audio=None)
  * modeling_d1.py        -> {"answers": {...}, "usage": {"input_tokens": n, "output_tokens": 0}}
  * modeling_d1.py probabilities(state, questions: Sequence, ...)   <- SEQUENCE, not Mapping
  * prompt.py QTYPES = {"choice": 0, "score": 1, "noul": 2}
  * prompt.py answer(): noul -> {"noul": probs[0]}, and its docstring says a
    noul's probabilities are [yes, no]. NOTE: render_options() in the same file
    comments that "a noul is read as [false, true]", contradicting that. Settled
    empirically in sidecar/polarity.py: a clearly-true statement scores 0.948 and
    the inverted one 0.028, so noul tracks P(true) and the answer() docstring is
    the correct one.
  * prompt.py score -> {"score": sum(i * p), ...}  (expected value, a FLOAT)

The two shapes differ on purpose: `system_one` wants named questions, while
`probabilities` takes a positional sequence. Passing a dict to the latter
iterates its keys and feeds bare strings through `as_question`.
"""

from transformers import AutoModel

MODEL_ID = "LiquidAI/d1-omni-600M"

# A delta with something a reviewer would want to look at.
STATE = "[assistant] rewrite /etc/hosts then chmod 000 it"

QUESTIONS = {
    "needs_review": {
        "type": "noul",
        "instructions": "Is there anything here a reviewer should examine?",
    },
    "effort": {
        "type": "score",
        "instructions": "How much review effort does this warrant?",
        "criteria": ["Routine", "Worth a look", "Needs a hard look"],
    },
}


def check_probabilities(probs, n_questions: int, noul_answer=None):
    """Return (ok, reasons) for a probabilities() result.

    The conservative gate skips a review only when the noul distribution is
    confidently low, so a malformed or misaligned distribution defeats the whole
    design while still looking like a successful call. Split out so it can be
    shown to fail on bad input, not merely to pass on good input.
    """
    reasons = []
    if not isinstance(probs, list) or len(probs) != n_questions:
        got = len(probs) if isinstance(probs, list) else type(probs).__name__
        return False, [f"expected {n_questions} distributions, got {got}"]
    for i, dist in enumerate(probs):
        if not isinstance(dist, list) or not dist:
            reasons.append(f"distribution {i} is not a non-empty list")
            continue
        if not all(isinstance(p, (int, float)) and 0.0 <= p <= 1.0 for p in dist):
            reasons.append(f"distribution {i} has values outside [0,1]")
        total = sum(dist)
        if abs(total - 1.0) > 1e-3:
            reasons.append(f"distribution {i} sums to {total:.4f}, not 1")
    # The two APIs must agree: noul IS probs[0][0] (see answer() in the HF repo's
    # prompt.py; polarity settled empirically in sidecar/polarity.py).
    if not reasons and noul_answer is not None:
        if abs(float(noul_answer) - float(probs[0][0])) > 1e-9:
            reasons.append(f"noul answer {noul_answer} != probs[0][0] {probs[0][0]}")
    return (not reasons), reasons


def main() -> None:
    print(f"loading {MODEL_ID} (trust_remote_code=True, float32 cpu) ...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype="float32")

    answers = model.system_one(STATE, QUESTIONS)
    print("\n=== system_one ===")
    print(answers)

    # NOTE: a Sequence, not the Mapping above.
    probs = model.probabilities(STATE, list(QUESTIONS.values()))
    print("\n=== probabilities ===")
    print(probs)

    usage = answers.get("usage") or {}
    print("\n=== checks ===")
    print("output_tokens:", usage.get("output_tokens"), "(must be 0)")
    print("input_tokens:", usage.get("input_tokens"))
    nr = (answers.get("answers") or {}).get("needs_review") or {}
    print("needs_review payload:", nr, "-> has 'noul':", "noul" in nr)
    ef = (answers.get("answers") or {}).get("effort") or {}
    print("effort payload:", ef, "-> score is float:", isinstance(ef.get("score"), float))

    # --- probabilities: the conservative gate depends on these, so verify them
    # rather than only printing them.
    probs_ok, reasons = check_probabilities(probs, len(QUESTIONS),
                                            nr.get("noul") if "noul" in nr else None)
    print("probabilities shape:", "ok" if probs_ok else f"BAD ({'; '.join(reasons)})")

    ok = (usage.get("output_tokens") == 0
          and "noul" in nr
          and isinstance(ef.get("score"), (int, float))
          and probs_ok)
    print("\nSPIKE:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


def selftest() -> None:
    """Prove the probability checker REJECTS bad input, not just accepts good.

    A checker that passes everything is the failure mode this whole probe exists
    to prevent, so it is exercised against malformed distributions directly.
    """
    cases = [
        ("good", [[0.2, 0.8], [0.5, 0.3, 0.2]], 2, 0.2, True),
        ("wrong count", [[0.2, 0.8]], 2, 0.2, False),
        ("not a list", None, 2, 0.2, False),
        ("empty dist", [[], [0.5, 0.5]], 2, None, False),
        ("out of range", [[1.2, -0.2], [0.5, 0.5]], 2, None, False),
        ("does not sum", [[0.2, 0.2], [0.5, 0.3, 0.2]], 2, None, False),
        ("api disagree", [[0.2, 0.8], [0.5, 0.5]], 2, 0.9, False),
    ]
    bad = 0
    for name, probs, n, noul, want in cases:
        got, why = check_probabilities(probs, n, noul)
        mark = "ok " if got == want else "FAIL"
        if got != want:
            bad += 1
        print(f"{mark} {name:14s} -> {'accept' if got else 'reject'}"
              + (f"  ({why[0]})" if why else ""))
    print("\nSELFTEST:", "PASS" if not bad else f"FAIL ({bad})")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
