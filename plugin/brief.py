"""Review-brief discovery and verifier prompt assembly.

``VERIFIER.md`` files carry review priorities for the verifier only — they are
appended to the verifier's system prompt and never reach the primary agent's
context. Discovery loads the user-level file first, then project files while
walking from the working directory up to the git repository root (or the home
directory when no repo root is found), mirroring how project context files are
discovered but without stopping at the nearest file.

Only the standard library is imported at module level: this package is loaded
inside the Hermes plugin harness and must stay import-light.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("hermes-verifier")

# Reviewer persona. Kept byte-stable per process so provider prompt caching
# is not churned by dynamic content: brief files append AFTER this base.
SYSTEM_PROMPT = """\
You are the Verifier for a Hermes Agent session. You review what the primary \
agent just did and raise findings that steer it before it goes wrong.

You are a reviewer, not an executor. You never run tools, never edit files, \
and never take actions of your own. You only look at the transcript delta you \
are given and decide whether to raise a finding.

Rules:
- At most ONE finding per review. If nothing matters, report severity "none".
- A finding must be concrete and actionable: name the exact risk, the exact \
message or tool call, and what to do instead. Vague praise or a bare \
"looks good" is not a finding — report "none" instead.
- Severity:
  * note    - small cleanup, simplification, low-risk edge case. Non-blocking.
  * caution - material risk: wrong direction, missing constraint, hallucinated \
              API or behavior, wasted work about to happen.
  * blocker - continuing will clearly waste work or ship broken output.
- Weigh, don't blindly obey: the primary agent weighs your finding. You do not \
control it.
- The delta has been redacted; you never see secrets and must not ask for them.
- Review only the DELTA since your last review, not the whole conversation.
- Keep findings under 200 words.
"""

REVIEW_TASK = (
    "Review the transcript delta below (since your last review). Respond with "
    "a single review envelope: severity, finding, and private reasoning."
)

ATTENTION_HEADER = "Especially pay attention to:\n<attention>"
ATTENTION_FOOTER = "</attention>"


def _find_repo_root(start: Path) -> Optional[Path]:
    """Walk upward to the nearest .git entry; None when not inside a repo."""
    current = start
    for _ in range(64):
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent
    return None


def _candidates(start: Path, user_home: Path) -> List[Path]:
    """Ordered candidate files: user-level first, then ancestors down to cwd.

    Mirrors the "broad guidance first, narrow guidance last" prompt ordering
    so project files closer to the working directory land nearest the end of
    the system prompt.
    """
    out: List[Path] = [user_home / "VERIFIER.md"]

    anchor = _find_repo_root(start) or user_home
    chain = []
    current = start
    for _ in range(64):
        if current == anchor.parent:
            break
        chain.append(current)
        if current == current.parent:
            break
        current = current.parent
    chain.reverse()  # farthest ancestor -> cwd
    for directory in chain:
        out.append(directory / "VERIFIER.md")
        out.append(directory / ".hermes" / "VERIFIER.md")
    return out


def load_brief_blocks(start_dir: Optional[Path] = None) -> List[str]:
    """Read every readable VERIFIER.md block, in prompt order.

    Unreadable or missing files are skipped silently. Returns rendered
    ``Especially pay attention to:`` blocks.
    """
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
    except Exception:  # pragma: no cover - outside Hermes (tests)
        home = Path.home()
    start = Path(start_dir) if start_dir else Path(os.getcwd())
    if not start.is_dir():
        start = Path(os.getcwd())

    blocks: List[str] = []
    for candidate in _candidates(start, home):
        try:
            if not candidate.is_file():
                continue
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
        except OSError as exc:  # pragma: no cover - permission races
            logger.debug("skipping %s: %s", candidate, exc)
            continue
        blocks.append(f"{ATTENTION_HEADER}\n{text}\n{ATTENTION_FOOTER}")
    return blocks


def build_system_prompt(start_dir: Optional[Path] = None) -> str:
    """Base persona plus every discovered brief block, in prompt order."""
    parts = [SYSTEM_PROMPT]
    parts.extend(load_brief_blocks(start_dir))
    return "\n\n".join(parts)
