"""Emission guard: the filter on the path from verifier output to delivery.

Enforces three rules from the design:

1. **Normalization.** Lowercase, NFKC, collapse every run of non-alphanumeric
   characters to one space, then trim.
2. **Content-free phrase filter.** Short phrases with no concrete reason are
   suppressed.
3. **Exact-text dedupe.** A normalized note already accepted by this verifier
   in this session is dropped — unless it re-raises at a strictly higher
   severity (a real escalation). FIFO history holds at most ``max_history``
   entries.
4. **Per-cycle rate limit.** At most one finding per review cycle.

Pure stdlib, no Hermes imports: unit-testable in isolation.
"""

from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from typing import Dict, Optional

SEVERITY_RANK: Dict[str, int] = {"note": 1, "caution": 2, "blocker": 3}

# Content-free phrases: short, no concrete reason. Suppressed entirely.
_PHRASES = {
    "stop", "done", "complete", "no issue continue", "lgtm", "nothing to add",
    "looks good", "looks fine", "all good", "ok", "okay", "fine", "none",
    "no issues", "no findings", "no problems", "no concerns", "continue",
    "everything looks good", "all clear", "nothing to report", "no action",
    "no action needed", "seems fine", "seems good", "nothing of note",
}

_ALNUM_RUN = re.compile(r"[^0-9a-z]+")


def normalize(note: str) -> str:
    """Lowercase, NFKC, collapse non-alphanumeric runs, trim."""
    folded = unicodedata.normalize("NFKC", note or "")
    lowered = folded.lower()
    return _ALNUM_RUN.sub(" ", lowered).strip()


def is_content_free(normalized: str) -> bool:
    return not normalized or len(normalized) <= 2 or normalized in _PHRASES


class EmissionGuard:
    """Per-verifier dedupe + rate-limit state.

    Guard-level suppression is silent by design: callers get a boolean, and
    rejected noise never consumes the per-cycle budget.
    """

    def __init__(self, max_history: int = 4096) -> None:
        self._seen: "OrderedDict[str, int]" = OrderedDict()
        self._max_history = max_history
        self._cycle_used = False
        self.stats = {"accepted": 0, "deduped": 0, "phrase_filtered": 0,
                      "rate_limited": 0}

    def begin_cycle(self) -> None:
        """Call once at the start of each review; opens the one-finding budget."""
        self._cycle_used = False

    def accept(self, severity: str, note: str) -> bool:
        """Return True when this finding may be delivered.

        Order: normalization, phrase filter, escalation-aware dedupe, then the
        per-cycle rate limit.
        """
        severity = severity if severity in SEVERITY_RANK else "note"
        normalized = normalize(note)
        if is_content_free(normalized):
            self.stats["phrase_filtered"] += 1
            return False

        previous_rank = self._seen.get(normalized)
        new_rank = SEVERITY_RANK[severity]
        if previous_rank is not None and new_rank <= previous_rank:
            self.stats["deduped"] += 1
            return False

        if self._cycle_used:
            self.stats["rate_limited"] += 1
            return False

        # Record BEFORE the rate-limit decision so an escalation that loses the
        # rate limit is still remembered at its new severity.
        self._seen[normalized] = max(previous_rank or 0, new_rank)
        self._seen.move_to_end(normalized)
        while len(self._seen) > self._max_history:
            self._seen.popitem(last=False)

        self._cycle_used = True
        self.stats["accepted"] += 1
        return True

    def reset(self) -> None:
        """Clear dedupe history and the cycle gate (verifier re-prime)."""
        self._seen.clear()
        self._cycle_used = False

    def __len__(self) -> int:
        return len(self._seen)
