"""Render the run. The caveats are part of the output, not the commentary."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict

# Below this many positive labels the bootstrap cannot resolve a small effect,
# so a null is uninformative rather than reassuring.
MIN_POSITIVES = 100

CAVEAT = (
    "The outcome label is repair language in the FOLLOWING turn's assistant "
    "prose -- tool arguments and tool results are not read. It measures that "
    "repair FOLLOWED, not that a review would have prevented it. Every number "
    "here is correlational."
)


def render(result: Dict[str, Any]) -> str:
    p = result.get("power") or {}
    lines = [
        f"# hermes-verifier autoresearch — {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**verdict:** {result.get('verdict', 'unknown')}",
        "",
        "## power",
        f"- turns: {p.get('turns', 0)}",
        f"- sessions: {p.get('sessions', 0)}",
        f"- positive labels: {p.get('positive_labels', 0)}",
    ]
    if p.get("positive_labels", 0) < MIN_POSITIVES:
        lines.append(
            f"- **UNDERPOWERED** — fewer than {MIN_POSITIVES} positive labels. "
            "A null result under this bar is NOT evidence the current rule is "
            "the best one; it means the data cannot tell."
        )
    lines += ["", "## caveat", CAVEAT, ""]
    if result.get("held_out"):
        lines += ["## held-out",
                  "```", json.dumps(result["held_out"], indent=2), "```", ""]
    if result.get("survivors") is not None:
        lines += [f"## survivors\n{result['survivors'] or 'none'}", ""]
    return "\n".join(lines)


def write_report(result: Dict[str, Any], out_dir: pathlib.Path) -> pathlib.Path:
    """Write the report. This is the loop's own output tree — not config."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"autoresearch-{time.strftime('%Y%m%d-%H%M%S')}.md"
    path.write_text(render(result))
    return path
