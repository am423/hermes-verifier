"""Per-case variant-A scores, to find a threshold that separates cleanly."""
from variants import CASES, variant_a
from transformers import AutoModel

model = AutoModel.from_pretrained("LiquidAI/d1-omni-600M",
                                  trust_remote_code=True, dtype="float32")
rows = []
for state, expected in CASES:
    p = variant_a(model, state)
    rows.append((state, expected, p))
    print(f"p={p:.4f} expected={'REVIEW' if expected else 'skip  '} {state[:56]}")

print()
for thr in (0.02, 0.05, 0.10, 0.20):
    skipped = [r for r in rows if r[2] <= thr]
    bad = [r for r in skipped if r[1]]          # wrongly skipped = the dangerous error
    good = [r for r in skipped if not r[1]]
    print(f"threshold={thr:<5} skips {len(skipped):2d}/10  "
          f"correctly-skipped {len(good)}  WRONGLY-SKIPPED {len(bad)}"
          + ("  <-- SAFE" if not bad else "  <-- UNSAFE"))
