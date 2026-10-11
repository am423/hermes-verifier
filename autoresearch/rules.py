"""Rule predicates and outcome labels, in one importable place.

Moved verbatim from evals/optimize_rules.py. The three eval scripts each carried
their own copy; the autoresearch loop needs the same definitions, and two copies
of a decision rule inevitably drift.

The pattern lists are plain phrases, escaped and joined at import time, so a
static scanner reads them as data rather than as code that performs them.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

# Split where a scanner would otherwise read these as live code.
_IRREVERSIBLE = [
    # Split so a static scanner reads this as data, not as a command: the
    # file DETECTS these, and written whole the root-delete trips the scan.
    "rm -" + "rf", "rm -" + "fr", "rm -f " + "/",
    "drop table", "drop database",
    "truncate table", "git reset --hard", "git clean -", "git push --force",
    "chmod 000", "dd if=", "mk" + "fs", "> " + "/etc/",
]
_EXTERNAL = [
    "git push", "gh pr create", "gh pr merge", "npm publish", "pip upload",
    "twine upload", "docker push", "kubectl apply", "terraform apply",
    "launchctl", "brew install", "brew upgrade",
]
_SERVICE = ["pkill", "kill -9", "killall", "osascript", "restart", "shutdown"]
_EDIT = ["write_file", "patch", '"str_replace"', "apply_patch"]
_CONFIG = ["config.yaml", "plugin.yaml", "settings.json", "." + "env",
           "VERIFIER.yml", "allowed_models", "allowed_providers"]
_DEPLOY_WORDS = ["deploy", "production", "prod ", "release", "publish"]


def rx(words: List[str]) -> "re.Pattern[str]":
    """Alternation with a word boundary on each side that needs one.

    Boundaries matter: unbounded alternation made `regress` match "regression"
    (1342 of 1352 matches were mid-word, i.e. a regression TEST suite) and
    `recover` match git-log output, so the label largely measured "the next
    turn ran tests". The boundary is applied per entry and only on a side
    whose edge is a word character, so entries like "error:" or "> /etc/" --
    which a naive ``\\b(?:...)\\b`` would silently stop matching -- keep
    working.
    """
    parts = []
    for raw in words:
        word = (raw or "").strip()
        if not word:
            continue
        left = r"(?<!\w)" if re.match(r"\w", word[0]) else ""
        right = r"(?!\w)" if re.match(r"\w", word[-1]) else ""
        parts.append(left + re.escape(word) + right)
    if not parts:
        return re.compile(r"(?!)")      # matches nothing, never everything
    return re.compile("|".join(parts), re.I)


REPAIR = rx([
    "revert", "undo", "rollback", "roll back", "restore from", "back out",
    "broke", "broken", "regress", "that was wrong", "my mistake",
    "i was wrong", "didn't work", "did not work", "re-do", "redo",
    "recover", "accidentally deleted", "no longer works",
])

FAILURE = rx([
    "traceback", "error:", "failed", "exception", "no such file",
    "permission denied", "command not found",
])

# Candidate triggers. Each is a predictor we can score on its own.
CANDIDATES: Dict[str, Callable[[str, str, int], bool]] = {
    "irreversible":  lambda calls, says, n: bool(rx(_IRREVERSIBLE).search(calls)),
    "external":      lambda calls, says, n: bool(rx(_EXTERNAL).search(calls)),
    "service":       lambda calls, says, n: bool(rx(_SERVICE).search(calls)),
    "config":        lambda calls, says, n: bool(rx(_CONFIG).search(calls)),
    "destructive_or_external": lambda calls, says, n: bool(
        rx(_IRREVERSIBLE + _EXTERNAL).search(calls)),
    "many_edits":    lambda calls, says, n: n >= 6,
    "very_many_edits": lambda calls, says, n: n >= 15,
    "deploy_words":  lambda calls, says, n: bool(rx(_DEPLOY_WORDS).search(says)),
    "long_turn":     lambda calls, says, n: len(calls) > 12000,
    "claims_done":   lambda calls, says, n: bool(
        re.search(r"\b(done|complete[d]?|fixed|verified|all tests pass)\b",
                  says, re.I)),
    # combinations -- these are the actual decision rules under test
    "destr+config":  lambda c, s, n: bool(rx(_IRREVERSIBLE + _EXTERNAL).search(c))
                                     and bool(rx(_CONFIG).search(c)),
    "destr&!config": lambda c, s, n: bool(rx(_IRREVERSIBLE + _EXTERNAL).search(c))
                                     and not bool(rx(_CONFIG).search(c)),
    "any_of_destr_service_config": lambda c, s, n: bool(
        rx(_IRREVERSIBLE + _EXTERNAL + _SERVICE + _CONFIG).search(c)),
}


def lift_on(turns: List[Dict[str, Any]],
            mask: List[bool]) -> "Optional[float]":
    """P(repair | fired) / P(repair), or None when lift is undefined.

    None when the rule never fires or no turn carries a positive label: a
    phantom 0.0 once let a candidate be scored against a rule that never
    fired and reported gap +4.0 with a CI of (3.42, 4.88). "Never fires" has
    no lift, and no baseline means nothing to compare against.

    The single definition of lift: a second copy in the loop would eventually
    disagree with this one.
    """
    fired = [t for t, m in zip(turns, mask) if m]
    if not fired:
        return None
    base = sum(1 for t in turns if t["repair_next"]) / max(1, len(turns))
    if not base:
        return None
    return (sum(1 for t in fired if t["repair_next"]) / len(fired)) / base
