"""Render the run. The caveats are part of the output, not the commentary."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, Optional

# Below this many positive labels the bootstrap cannot resolve a small effect,
# so a null is uninformative rather than reassuring.
MIN_POSITIVES = 100

CAVEAT = (
    "The outcome label is repair language in the FOLLOWING turn's assistant "
    "prose -- tool arguments and tool results are not read. It measures that "
    "repair FOLLOWED, not that a review would have prevented it. Every number "
    "here is correlational."
)


def _power_line(label: str, p: Optional[Dict[str, Any]]) -> Optional[str]:
    if not p:
        return None
    return (f"- {label}: turns {p.get('turns', 0)}, sessions {p.get('sessions', 0)}, "
            f"positive labels {p.get('positive_labels', 0)}")


def render(result: Dict[str, Any]) -> str:
    p = result.get("power") or {}
    lines = [
        f"# hermes-verifier autoresearch — {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**verdict:** {result.get('verdict', 'unknown')}",
    ]
    if result.get("reason"):
        lines.append(f"**why:** {result['reason']}")
    lines += [
        "",
        "## power",
        f"- turns: {p.get('turns', 0)}",
        f"- sessions: {p.get('sessions', 0)}",
        f"- positive labels: {p.get('positive_labels', 0)}",
    ]
    lines += [l for l in (_power_line("fit half", result.get("fit_power")),
                          _power_line("validation half", result.get("val_power"))) if l]

    # The UNDERPOWERED banner follows the half that actually decided the
    # verdict -- the validation half when anything survived the fit bar,
    # otherwise the fit half. A thin unreached half must not cry wolf, and a
    # thin deciding half must not pass in silence.
    deciding = result.get("deciding_half") or ""
    powers = {"fit": result.get("fit_power"), "validation": result.get("val_power")}
    deciding_power = powers.get(deciding) or p
    if deciding_power.get("positive_labels", 0) < MIN_POSITIVES:
        where = f"the {deciding} half" if deciding else "this run"
        lines.append(
            f"- **UNDERPOWERED** — {where} has fewer than {MIN_POSITIVES} positive "
            "labels. A null result under this bar is NOT evidence the current rule "
            "is the best one; it means the data cannot tell."
        )

    mult = result.get("multiplicity") or {}
    if mult:
        lines += ["", "## multiplicity",
                  f"- candidates tested: {mult.get('tests')}",
                  f"- alpha each (Bonferroni): {mult.get('alpha_each')}"]
        if mult.get("bootstraps") is not None:
            lines.append(f"- bootstraps: {mult.get('bootstraps')} "
                         f"({mult.get('quantile_draws')} draws below the lower quantile)")
        if mult.get("warning"):
            lines.append(f"- **WARNING**: {mult['warning']}")

    fit_results = result.get("fit_results") or {}
    if fit_results:
        lines += ["", "## candidates (fit half)"]
        for name, r in fit_results.items():
            lines.append(f"- {name}: gap {r.get('gap')}, ci {r.get('ci')}, "
                         f"cluster_ci {r.get('cluster_ci')}, "
                         f"clears_bar {r.get('clears_bar')}")
            if r.get("unscorable"):
                lines.append(f"  unscorable: {r.get('reason')}")
            if r.get("ineligible"):
                lines.append(f"  ineligible: {r['ineligible']}")

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
