"""Prove d1-omni answers a multi-question set, with confidence, at zero output cost.

API shape verified against the model source before writing this
(huggingface.co/LiquidAI/d1-omni-600M):

  * modeling_d1.py:66  system_one(state, questions: Mapping, images=None, audio=None)
  * modeling_d1.py:78  -> {"answers": {...}, "usage": {"input_tokens": n, "output_tokens": 0}}
  * modeling_d1.py:81  probabilities(state, questions: Sequence, ...)   <- SEQUENCE, not Mapping
  * prompt.py:19       QTYPES = {"choice": 0, "score": 1, "noul": 2}
  * prompt.py:117      noul  -> {"noul": probs[0]}
  * prompt.py:124      score -> {"score": sum(i * p), "confidence": ...}  (expected value, a FLOAT)

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

    ok = (usage.get("output_tokens") == 0
          and "noul" in nr
          and isinstance(ef.get("score"), (int, float)))
    print("\nSPIKE:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
