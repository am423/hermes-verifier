"""One autoresearch pass: harvest, score, validate, report. Applies nothing."""

from __future__ import annotations

import pathlib

from .harvest import harvest
from .report import write_report
from .rules import CANDIDATES
from .validate import validate

REFERENCE = "destructive_or_external"
REPORT_DIR = pathlib.Path(__file__).resolve().parent / "reports"


def candidates_from_turns(turns):
    """Bind each rule predicate to a harvested turn.

    rules.CANDIDATES take (calls, says, edits); harvested turns carry those plus
    the labels, so this adapts one to the other in exactly one place.
    """
    return {name: (lambda t, fn=fn: fn(t["calls"], t["says"], t["edits"]))
            for name, fn in CANDIDATES.items()}


def main() -> None:
    turns = harvest()
    if not turns:
        print("no turns harvested; nothing to do")
        return
    result = validate(turns, candidates_from_turns(turns), reference=REFERENCE)
    path = write_report(result, REPORT_DIR)
    print(path.read_text())


if __name__ == "__main__":
    main()
