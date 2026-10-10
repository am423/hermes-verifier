"""Definitive polarity test: does `noul` mean P(yes) or P(no)?

The earlier single-state test ("igloo") was confounded: it could have been
measuring dict key order, and "igloo" is a poor probe because the model may
simply not know the referent. A polarity error would invert the optimizer, so
this settles it with unambiguous states, both key spellings, both insertion
orders, and several states per condition.

Design:
  * truth is not in question -- the sky is blue, grass is green, water is wet
  * each (state, question) pair has an obvious correct polarity
  * criteria given as true/false AND as yes/no
  * keys inserted in BOTH orders, to detect order-dependence rather than semantics
  * 3 states per condition

If `noul` tracks P(yes), the TRUE questions score high and the FALSE ones low,
and swapping key order changes nothing.
"""

from transformers import AutoModel

MODEL_ID = "LiquidAI/d1-omni-600M"

# (state, question, expected) -- expected is True when the honest answer is yes.
PAIRS = [
    ("The sky is blue on a clear day.", "Is the sky blue?", True),
    ("The sky is blue on a clear day.", "Is the sky green?", False),
    ("Grass is typically green.", "Is grass usually green?", True),
    ("Grass is typically green.", "Is grass usually purple?", False),
    ("Water is wet.", "Is water wet?", True),
    ("Water is wet.", "Is water dry?", False),
]


def ask(model, state, question, criteria):
    q = {"v": {"type": "noul", "instructions": question}}
    if criteria is not None:
        q["v"]["criteria"] = criteria
    ans = model.system_one(state, q)
    return (ans.get("answers", {}).get("v") or {}).get("noul")


def variants(question):
    """Every spelling/order combination that must agree if polarity is semantic."""
    return {
        "no criteria": None,
        "true-first": {"true": "yes, it does", "false": "no, it does not"},
        "false-first": {"false": "no, it does not", "true": "yes, it does"},
        "yes-first": {"yes": "yes, it does", "no": "no, it does not"},
        "no-first": {"no": "no, it does not", "yes": "yes, it does"},
    }


def main() -> None:
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype="float32")

    names = list(variants("").keys())
    # scores[name][expected] = list of p
    scores = {n: {True: [], False: []} for n in names}

    for state, question, expected in PAIRS:
        row = []
        for name, crit in variants(question).items():
            p = ask(model, state, question, crit)
            scores[name][expected].append(p)
            row.append(f"{name}={p:.3f}")
        print(f"{question:22s} exp={'YES' if expected else 'no ':3s}  " + "  ".join(row))

    print("\n=== summary: does each spelling track P(yes)? ===")
    print(f"{'variant':14s} {'mean(yes)':>10} {'mean(no)':>10} {'sep':>8}  verdict")
    for name in names:
        my = sum(scores[name][True]) / len(scores[name][True])
        mn = sum(scores[name][False]) / len(scores[name][False])
        ok = my > mn
        print(f"{name:14s} {my:10.3f} {mn:10.3f} {my - mn:+8.3f}  "
              f"{'P(yes)' if ok else 'INVERTED -> P(no)' if my < mn and abs(my - mn) > 0.2 else 'no separation'}")

    # If any variant flips relative to "no criteria", polarity is order-dependent.
    base = names[0]
    my_b = sum(scores[base][True]) / len(scores[base][True])
    print(f"\norder-dependence check (vs {base!r}):")
    for name in names[1:]:
        my = sum(scores[name][True]) / len(scores[name][True])
        delta = my - my_b
        flag = "OK" if abs(delta) < 0.25 else "DIFFERS -- order or spelling matters"
        print(f"  {name:14s} mean(yes)={my:.3f}  delta={delta:+.3f}  {flag}")


if __name__ == "__main__":
    main()
