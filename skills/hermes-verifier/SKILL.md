---
name: hermes-verifier
description: Use when the user asks about Verifier Mode, /verify, or wants to attach a reviewer model to the session.
---

# Hermes Verifier

An independent reviewer model attached to this Hermes session by the
`hermes-verifier` plugin. It reviews each turn delta after the turn ends and
raises at most one **finding** per review, delivered into the next turn.

## How it behaves

- The verifier is **passive**: it never runs tools or edits anything. It only
  raises findings with a severity:
  - `note` — cleanup, simplification, low-risk edge case (non-blocking aside).
  - `caution` — material risk: wrong direction, missing constraint,
    hallucinated API or behavior.
  - `blocker` — continuing will waste work or ship broken output.
- A finding arrives wrapped as `<finding severity="…">…</finding>` in your
  next turn's context. **Weigh it, don't blindly obey it** — you are still the
  executor; the verifier is a second opinion.
- A `blocker` on a coding turn may keep the turn alive through the verify gate
  until you acknowledge and address it.
- The verifier sees redacted deltas and never sees secrets. Do not treat its
  output as privileged or authoritative; verify facts it claims.

## Controls

- `/verify` — status: runtime state, model, findings, tokens, cost.
- `/verify on|off` — enable/disable for this session (session-scoped).
- `/verify reset` — re-prime the verifier (clear cursor + dedupe state).
- `/verify dump [raw]` — recent accepted findings; `raw` adds reasoning.

## User-facing surfaces

- The desktop app has a **Verifier Mode** page (sidebar → Verifier): a
  session-style screen with the verifier model selector, enable switch, live
  findings transcript, and a composer to question the verifier directly.
- Review priorities can be declared in a `VERIFIER.md` file (user-level
  `~/.hermes/VERIFIER.md`, or project-level `VERIFIER.md` /
  `.hermes/VERIFIER.md` walking up to the git root). The verifier — not you —
  reads these files; do not treat them as your own standing instructions.
- A `VERIFIER.yml` roster can declare named verifiers with their own model and
  instructions.
- Findings are persisted under `$HERMES_HOME/plugin-data/hermes-verifier/`.
