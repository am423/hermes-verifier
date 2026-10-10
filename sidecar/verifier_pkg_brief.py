"""Review-brief + roster discovery and verifier prompt assembly.

``VERIFIER.md`` files carry review priorities for the verifier only — they are
appended to the verifier's system prompt and never reach the primary agent's
context. ``VERIFIER.yml`` declares the verifier roster: one entry per name,
each with its own enable flag, model, tool grant, and specialization prompt.

Discovery loads the user-level file first, then project files while walking
from the working directory up to the git repository root (or the home
directory when no repo root is found), mirroring how project context files are
discovered but without stopping at the nearest file. For roster entries with
duplicate slugs, more-specific files (project leaf > project ancestor > user)
replace earlier entries.

Only the standard library is imported at module level: this package is loaded
inside the Hermes plugin harness and must stay import-light.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

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
- Keep reasoning to one short sentence: it is diagnostics, not part of the finding, \
and it is billed like everything else.
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


def _ancestor_chain(start: Path, anchor: Path) -> List[Path]:
    """Directories from the farthest ancestor down to ``start`` (inclusive).

    Stops at the discovery anchor (repo root or home). Ordered farthest-first
    so callers get user/broad guidance before project/narrow guidance.
    """
    chain: List[Path] = []
    current = start
    for _ in range(64):
        if current == anchor.parent:
            break
        chain.append(current)
        if current == current.parent:
            break
        current = current.parent
    chain.reverse()
    return chain


def _discovery_anchor(start: Path, user_home: Path) -> Path:
    return _find_repo_root(start) or user_home


def _candidates(start: Path, user_home: Path, names: List[str]) -> List[Path]:
    """Ordered candidate files: user-level first, then ancestors down to cwd.

    Mirrors the "broad guidance first, narrow guidance last" prompt ordering
    so project files closer to the working directory land nearest the end of
    the system prompt. For roster files, later files override earlier entries
    with the same slug.
    """
    anchor = _discovery_anchor(start, user_home)
    out: List[Path] = []
    for name in names:
        out.append(user_home / name)
    for directory in _ancestor_chain(start, anchor):
        for name in names:
            out.append(directory / name)
            out.append(directory / ".hermes" / name)
    return out


def _read_candidate(candidate: Path) -> Optional[str]:
    try:
        if not candidate.is_file():
            return None
        text = candidate.read_text(encoding="utf-8", errors="replace").strip()
        return text or None
    except OSError as exc:  # pragma: no cover - permission races
        logger.debug("skipping %s: %s", candidate, exc)
        return None


# ---------------------------------------------------------------------------
# @path import expansion
# ---------------------------------------------------------------------------

_IMPORT_LINE = re.compile(r"^@([^\s`\"'<>]+)\s*$")
_FENCE = re.compile(r"^```|^~~~")


def expand_imports(text: str, base_dir: Path, seen: Optional[set] = None,
                   depth: int = 0) -> str:
    """Expand ``@path`` import lines, mirroring context-file import rules.

    Relative imports resolve from the importing file's directory, ``~/`` from
    the home directory; imports inside fenced code blocks and inline code
    spans stay literal; cycles are skipped; missing or unreadable imports
    leave the original ``@path`` text in place.
    """
    if depth > 8:
        return text
    seen = seen if seen is not None else set()
    out: List[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        match = _IMPORT_LINE.match(line.strip())
        if not match:
            out.append(line)
            continue
        raw = match.group(1)
        target = Path(raw).expanduser() if raw.startswith("~") else (base_dir / raw)
        try:
            target = target.resolve()
        except OSError:
            out.append(line)
            continue
        if target in seen:
            out.append(line)
            continue
        try:
            imported = target.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            out.append(line)  # missing import stays literal
            continue
        seen.add(target)
        out.append(expand_imports(imported, target.parent, seen, depth + 1))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Review briefs (VERIFIER.md)
# ---------------------------------------------------------------------------

def load_brief_blocks(start_dir: Optional[Path] = None,
                      user_home: Optional[Path] = None) -> List[str]:
    """Read every readable VERIFIER.md block, in prompt order.

    Unreadable or missing files are skipped silently. Returns rendered
    ``Especially pay attention to:`` blocks with ``@path`` imports expanded.
    """
    home = Path(user_home) if user_home else _default_home()
    start = Path(start_dir) if start_dir else Path(os.getcwd())
    if not start.is_dir():
        start = Path(os.getcwd())

    blocks: List[str] = []
    for candidate in _candidates(start, home, ["VERIFIER.md"]):
        text = _read_candidate(candidate)
        if text is None:
            continue
        expanded = expand_imports(text, candidate.parent)
        blocks.append(f"{ATTENTION_HEADER}\n{expanded}\n{ATTENTION_FOOTER}")
    return blocks


def _default_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:  # pragma: no cover - outside Hermes (tests)
        return Path.home()


# ---------------------------------------------------------------------------
# Roster (VERIFIER.yml)
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "verifier"


def _parse_roster_file(text: str) -> List[Dict[str, object]]:
    """Parse one VERIFIER.yml into roster entries. Never raises.

    Unknown fields are ignored; malformed entries are dropped with a warning
    so one bad project config cannot kill the session.
    """
    try:
        import yaml  # lazy: only needed when a roster file exists
    except Exception:  # pragma: no cover
        logger.warning("PyYAML unavailable; VERIFIER.yml roster skipped")
        return []
    try:
        parsed = yaml.safe_load(text)
    except Exception as exc:
        logger.warning("VERIFIER.yml parse failed: %s", exc)
        return []
    if not isinstance(parsed, dict):
        return []
    entries: List[Dict[str, object]] = []
    shared = parsed.get("instructions")
    verifiers = parsed.get("verifiers")
    if not isinstance(verifiers, list):
        return []
    for item in verifiers:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"]).strip()
        if not name:
            continue
        entry = {
            "name": name,
            "slug": _slugify(name),
            "enabled": bool(item.get("enabled", True)),
            "model": str(item.get("model") or "").strip(),
            "tools": [str(t) for t in item.get("tools") or [] if isinstance(t, str)],
            "instructions": str(item.get("instructions") or "").strip(),
        }
        if isinstance(shared, str) and shared.strip():
            entry["shared_instructions"] = shared.strip()
        entries.append(entry)
    return entries


def load_roster(start_dir: Optional[Path] = None,
                user_home: Optional[Path] = None) -> List[Dict[str, object]]:
    """Discover every VERIFIER.yml / VERIFIER.yaml entry, in discovery order.

    Later files (project leaf > project ancestor > user) replace earlier
    entries with the same slug.
    """
    home = Path(user_home) if user_home else _default_home()
    start = Path(start_dir) if start_dir else Path(os.getcwd())
    if not start.is_dir():
        start = Path(os.getcwd())

    merged: Dict[str, Dict[str, object]] = {}
    for candidate in _candidates(start, home, ["VERIFIER.yml", "VERIFIER.yaml"]):
        text = _read_candidate(candidate)
        if text is None:
            continue
        expanded = expand_imports(text, candidate.parent)
        for entry in _parse_roster_file(expanded):
            slug = str(entry["slug"])
            merged[slug] = entry  # later (more specific) wins
    return list(merged.values())


def build_system_prompt(start_dir: Optional[Path] = None,
                        user_home: Optional[Path] = None,
                        extra_instructions: str = "") -> str:
    """Base persona + brief blocks + optional verifier specialization."""
    parts = [SYSTEM_PROMPT]
    if extra_instructions and extra_instructions.strip():
        parts.append(extra_instructions.strip())
    parts.extend(load_brief_blocks(start_dir, user_home))
    return "\n\n".join(parts)
