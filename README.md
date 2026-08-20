# Hermes Verifier

A desktop plugin for [Hermes Agent](https://hermes-agent.nousresearch.com/) that attaches
an independent **verifier** to your session. The verifier is a second model that reviews
each update to the primary agent's transcript, inspects the workspace with its own
read-only tools, and raises **findings** that get steered back into the session before
the primary agent runs off a cliff.

## What it does

- **Second-opinion review** — every primary turn delta is rendered (reasoning, tool
  intent, tool results) and sent to the verifier model you choose.
- **Findings, not control** — the verifier never mutates session state. It raises
  findings with one of three severities:
  - `note` — a non-blocking aside, batched into the transcript at the next step boundary.
  - `caution` — an interrupting steer: material risk, wrong direction, hallucinated API.
  - `blocker` — the work is headed somewhere broken; the session pauses for you.
- **Read-only by default** — the verifier's default grant is `read_file`,
  `search_files`, and `glob`-style file search. Roster entries may widen the grant,
  but never by default.
- **Review briefs** — drop a `VERIFIER.md` in a project (or `~/.hermes/`) to tell the
  verifier what to watch for: dangerous APIs, architectural boundaries, quality bars.
- **Clean delivery** — an emission guard normalizes, dedupes, and rate-limits findings
  so the verifier never nags.
- **First-class UI** — a Verifier Mode screen that looks and feels like a normal
  session: transcript view, composer, and a model selector for the verifier model.

## Repository layout

```
hermes-verifier/
├── README.md          ← you are here
├── PLAN.md            ← milestones and task checklist
├── docs/
│   └── design.md      ← full architecture and behavior spec
├── plugin/            ← agent-side Hermes plugin (plugin.yaml, hooks, runtime)
├── dashboard/         ← plugin_api.py backend (REST + websocket namespace)
├── desktop/           ← desktop plugin (Verifier Mode UI)
└── skills/            ← bundled skill teaching Hermes how to talk about verifier mode
```

## Status

**M1 implemented.** The agent-side verifier runtime is built, tested (29 tests),
validated with `hermes plugins doctor`, and installed in this machine's Hermes.
The desktop UI lands in milestone 3 — see [PLAN.md](PLAN.md).

Activate after the next gateway restart: `/verify on` (session-scoped) or
`plugins.entries.hermes-verifier.settings.enabled: true` (default on).

## License

MIT
