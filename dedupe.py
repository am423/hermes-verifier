"""Semantic dedupe: suppress a RESTATED finding the exact-text guard would miss.

The emission guard dedupes on normalised exact text, so a finding reported again
in different words re-surfaces and nags. Deciding whether two findings describe
the same issue is a similarity question, and a small sentence embedder is the
canonical tool for it.

Why EMBEDDINGS and not the decision model this started with -- measured on the
same 10 labelled pairs (sidecar/embed_compare.py):

    all-MiniLM-L6-v2   22.7M params   ~90 MB   sep +0.451   pairwise 100%   recall 5/5
    d1-omni-600M        587M params   2.35 GB   sep +0.401   pairwise  76%   recall 3/5

26x smaller and better on every metric. The decision model was also a poor fit
on principle: 206M of its 587M params are vision and audio encoders this plugin
never touches, it needed a bespoke question with hand-written criteria, and it
required `trust_remote_code`. Cosine similarity needs none of that and is
deterministic.

Why suppression is safe here, when skipping a review was not: a wrongly
suppressed duplicate loses a *repeat* of a finding that already fired. A wrongly
skipped review loses a finding that would otherwise have existed. Only the second
is an unobservable quality loss.

Fail-open and off by default. Every uncertain path -- unreachable sidecar,
timeout, malformed reply, score below threshold -- returns "keep the finding",
which is exactly today's behaviour.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Optional

logger = logging.getLogger("hermes-verifier.dedupe")

DEFAULT_URL = "http://127.0.0.1:8766/similarity"
# Cosine similarity, NOT a probability. Measured operating point: paraphrase pairs
# scored 0.355-0.780 and distinct pairs 0.064-0.191, so 0.30 catches 5/5 with 0/5
# wrongly suppressed. 0 disables the stage entirely.
DEFAULT_THRESHOLD = 0.0
RECOMMENDED_THRESHOLD = 0.30


def ask(first: str, second: str, *, url: str = DEFAULT_URL,
        timeout: float = 2.0) -> Optional[float]:
    """Cosine similarity between two findings, or None on any failure.

    None is indistinguishable from 'not similar' to the caller, and both mean
    keep the finding -- so a broken sidecar degrades to today's behaviour.
    """
    try:
        req = urllib.request.Request(
            url, data=json.dumps({"a": first, "b": second}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload: Any = json.load(r)
    except Exception as exc:  # unreachable, timeout, malformed -- keep the finding
        logger.debug("semantic dedupe unavailable (%s); keeping the finding", exc)
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("similarity")
    return float(value) if isinstance(value, (int, float)) else None


def should_suppress(score: Optional[float], *, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Suppress only when the score clears the threshold.

    Destructive action, so the default for every other case is False -- keep.
    """
    if not threshold or threshold <= 0:
        return False
    if not isinstance(score, (int, float)):
        return False
    return float(score) >= float(threshold)
