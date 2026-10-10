"""Semantic dedupe: suppress a RESTATED finding the exact-text guard would miss.

The emission guard dedupes on normalised exact text, so a finding reported again
in different words re-surfaces and nags. Deciding whether two findings describe
the same issue is a factual comparison, which is the one thing the local decision
model is measurably good at (sidecar/APPLICATIONS.md).

Why suppression is safe here, when skipping a review was not: a wrongly
suppressed duplicate loses a *repeat* of a finding that already fired. A wrongly
skipped review loses a finding that would otherwise have existed. Only the second
is an unobservable quality loss.

Fail-open and off by default. Every uncertain path -- unreachable sidecar,
timeout, malformed reply, low confidence -- returns "keep the finding", which is
exactly today's behaviour.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Optional

logger = logging.getLogger("hermes-verifier.dedupe")

DEFAULT_URL = "http://127.0.0.1:8765/same-issue"
# Suppress only at or above this. 0 disables the stage entirely.
DEFAULT_THRESHOLD = 0.0

# The question is a comparison with both sides defined; measured separation
# +0.400 / 76% pairwise, with 0 wrongly-suppressed pairs from 0.5 to 0.95.
QUESTION = {
    "same_issue": {
        "type": "noul",
        "instructions": "Do these two review findings describe the same underlying issue?",
        "criteria": {"yes": "they report the same problem, even in different words",
                     "no": "they report different, unrelated problems"},
    },
}


def ask(first: str, second: str, *, url: str = DEFAULT_URL,
        timeout: float = 2.0) -> Optional[float]:
    """Probability that two findings describe the same issue. None on any failure."""
    state = f"Finding A: {first}\nFinding B: {second}"
    try:
        req = urllib.request.Request(
            url, data=json.dumps({"state": state}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload: Any = json.load(r)
    except Exception as exc:  # unreachable, timeout, malformed -- keep the finding
        logger.debug("semantic dedupe unavailable (%s); keeping the finding", exc)
        return None
    ans = (payload or {}).get("answers") or {}
    p = (ans.get("same_issue") or {}).get("noul")
    return float(p) if isinstance(p, (int, float)) else None


def should_suppress(p: Optional[float], *, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Suppress only on an explicit, confident 'same issue'.

    Destructive action, so the default for every other case is False -- keep.
    """
    if not threshold or threshold <= 0:
        return False
    if not isinstance(p, (int, float)):
        return False
    return float(p) >= float(threshold)
