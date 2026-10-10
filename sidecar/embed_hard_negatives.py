"""Validate the dedupe threshold against HARD negatives from the same session.

The first comparison used negatives that were maximally unrelated
(cos 0.064-0.191). Real dedupe compares findings from the SAME session, which
share file paths, identifiers and topic -- so genuinely distinct findings will
score far higher. If they reach the positive range (0.355-0.780), a fixed 0.30
over-suppresses and the documented "operating point" is wrong.

This set is all DISTINCT issues, sharing vocabulary the way real same-session
findings do. Every pair here should NOT be suppressed.
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Distinct issues that share identifiers/domain -- the realistic negative case.
HARD_NEGATIVES = [
    ("The write path bypasses the durable queue in src/jobs/",
     "src/jobs/queue.py has no test coverage"),
    ("The write path bypasses the durable queue in src/jobs/",
     "src/jobs/ names its worker threads inconsistently"),
    ("os.system with an unsanitized filename is command injection",
     "os.system is also called unsafely in the retry helper"),
    ("The response was truncated by advisor_max_tokens",
     "advisor_max_tokens is not documented in the README"),
    ("Hardcoded credentials were committed to the repo",
     "the credentials file is not in .gitignore"),
    ("The retry loop is not covered by a test",
     "the retry loop has an off-by-one in its max attempt count"),
    ("The delta cap uses characters rather than tokens",
     "the delta renderer truncates messages mid-word"),
    ("The emission guard dedupes on exact text only",
     "the guard stats are not exposed in /verify status"),
]

# The original easy negatives, for contrast.
EASY_NEGATIVES = [
    ("The write path bypasses the durable queue in src/jobs/",
     "the function name uses snake_case inconsistently"),
    ("Hardcoded credentials were committed",
     "the import order does not follow isort"),
]

# Positives, from the original set.
POSITIVES = [
    ("The retry loop is not covered by a test", "no test exercises the retry path", 0.780),
    ("Hardcoded credentials were committed",
     "a password literal appears in the committed source", 0.585),
    ("The write path bypasses the durable queue in src/jobs/",
     "src/jobs/write.py enqueues directly instead of going through the queue", 0.632),
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(EMBED_ID)
    model = AutoModel.from_pretrained(EMBED_ID)
    model.eval()

    def cos(a: str, b: str) -> float:
        batch = tok([a, b], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(**batch)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return float(torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0))

    print("=== HARD negatives (distinct, same session, shared vocabulary) ===")
    hard = []
    for a, b in HARD_NEGATIVES:
        c = cos(a, b)
        hard.append(c)
        print(f"  cos={c:.3f}  {a[:40]} / {b[:40]}")

    print("\n=== EASY negatives (the original set) ===")
    easy = [cos(a, b) for a, b in EASY_NEGATIVES]
    for (a, b), c in zip(EASY_NEGATIVES, easy):
        print(f"  cos={c:.3f}  {a[:40]} / {b[:40]}")

    print("\n=== positives ===")
    pos = [cos(a, b) for a, b, _ in POSITIVES]
    for (a, b, _), c in zip(POSITIVES, pos):
        print(f"  cos={c:.3f}  {a[:40]} / {b[:40]}")

    print("\n=== the verdict ===")
    print(f"  positives   min={min(pos):.3f} max={max(pos):.3f}")
    print(f"  hard negs   min={min(hard):.3f} max={max(hard):.3f}")
    print(f"  easy negs   min={min(easy):.3f} max={max(easy):.3f}")
    sep_hard = min(pos) - max(hard)
    sep_easy = min(pos) - max(easy)
    print(f"\n  margin vs EASY negatives: {sep_easy:+.3f}")
    print(f"  margin vs HARD negatives: {sep_hard:+.3f}")
    if sep_hard <= 0:
        print("\n  OVERLAP: no threshold separates them. A fixed cut cannot work here.")
        # What threshold would be needed, and what it costs.
        for thr in (0.30, 0.40, 0.50, 0.60, 0.70):
            fp = sum(1 for c in hard if c >= thr)
            tp = sum(1 for c in pos if c >= thr)
            print(f"    thr={thr:.2f}  catches {tp}/{len(pos)} paraphrases, "
                  f"WRONGLY suppresses {fp}/{len(hard)} distinct")
    else:
        print(f"\n  separable with margin {sep_hard:+.3f}")


if __name__ == "__main__":
    main()
