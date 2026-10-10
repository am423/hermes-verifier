"""Does a 90MB embedding model beat the 2.35GB decision model at dedupe?

d1-omni-600M is 587M params = 381M text trunk + 94M vision + 112M audio. This
plugin is TEXT ONLY, so >35% of it is provably unused. And the only application
that shipped -- semantic dedupe -- is a classic embedding-similarity task, for
which a small sentence embedder is the canonical tool.

Uses the same 10 labelled pairs as dedupe_probe.py so the two are directly
comparable. Embeddings via transformers directly (mean-pooled last hidden
state), not sentence-transformers, whose sklearn/scipy chain is broken on this
interpreter's numpy.

Size comparison:
  all-MiniLM-L6-v2  ~90 MB   6 layers, 384-dim
  d1-omni-600M      2.35 GB  587M params, 5 model files + custom code
"""

from __future__ import annotations

import math

import torch
from transformers import AutoModel, AutoTokenizer

EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Identical to dedupe_probe.py, plus d1-omni's measured p for each pair.
# (first, second, same_issue, d1_p)
PAIRS = [
    ("The write path bypasses the durable queue in src/jobs/",
     "src/jobs/write.py enqueues directly instead of going through the queue", True, 0.020),
    ("os.system with an unsanitized filename is command injection",
     "user input reaches a shell unescaped, allowing arbitrary commands", True, 0.049),
    ("The response was truncated by advisor_max_tokens",
     "a max_tokens cap cut the JSON off mid-object", True, 0.230),
    ("The retry loop is not covered by a test",
     "no test exercises the retry path", True, 0.975),
    ("Hardcoded credentials were committed",
     "a password literal appears in the committed source", True, 0.996),
    ("The write path bypasses the durable queue in src/jobs/",
     "the function name uses snake_case inconsistently", False, 0.086),
    ("os.system with an unsanitized filename is command injection",
     "the docstring is missing a period", False, 0.025),
    ("The response was truncated by advisor_max_tokens",
     "the variable name 'x' is unclear", False, 0.119),
    ("Hardcoded credentials were committed",
     "the import order does not follow isort", False, 0.002),
    ("The retry loop is not covered by a test",
     "the log message has a typo", False, 0.034),
]


def embed(model, tok, texts):
    batch = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**batch)
    # Mean-pool over the attention mask: the standard sentence-embedding readout.
    mask = batch["attention_mask"].unsqueeze(-1).float()
    summed = (out.last_hidden_state * mask).sum(1)
    return summed / mask.sum(1).clamp(min=1e-9)


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))


def main() -> None:
    print(f"loading {EMBED_ID} ...")
    tok = AutoTokenizer.from_pretrained(EMBED_ID)
    model = AutoModel.from_pretrained(EMBED_ID)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.1f}M   (d1-omni-600M: 587M)\n")

    rows = []
    for first, second, same, d1_p in PAIRS:
        a, b = embed(model, tok, [first, second])
        cos = cosine(a, b)
        rows.append((first, second, same, d1_p, cos))
        print(f"cos={cos:.3f}  d1={d1_p:.3f}  same={same!s:5s}  {first[:34]} / {second[:34]}")

    def score(idx, same_val):
        return [r[idx] for r in rows if r[2] == same_val]

    print("\n=== comparison on the SAME labelled set ===")
    for label, idx in (("d1-omni noul", 3), ("MiniLM cosine", 4)):
        s, d = score(idx, True), score(idx, False)
        ms, md = sum(s) / len(s), sum(d) / len(d)
        pairs = [(a, b) for a in s for b in d]
        good = sum(1 for a, b in pairs if a > b)
        # Recall at a threshold that keeps precision perfect (no wrong suppression).
        best_recall = 0
        for thr in [i / 100 for i in range(5, 100, 1)]:
            caught = sum(1 for r in rows if r[2] and r[idx] >= thr)
            wrong = sum(1 for r in rows if not r[2] and r[idx] >= thr)
            if wrong == 0:
                best_recall = max(best_recall, caught)
        print(f"  {label:14s} same={ms:.3f} diff={md:.3f} sep={ms - md:+.3f} "
              f"pairwise={good}/{len(pairs)}={good / len(pairs):.0%} "
              f"recall@0-wrong={best_recall}/5")

    print("\n=== size ===")
    print("  all-MiniLM-L6-v2 : ~90 MB, 6 layers, 384-dim, text only")
    print("  d1-omni-600M     : 2.35 GB, 587M params, + vision(94M) + audio(112M)")


if __name__ == "__main__":
    main()
