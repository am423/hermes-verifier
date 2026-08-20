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
  A `VERIFIER.yml` roster declares named verifiers with their own model, tool grant,
  and specialization.
- **Clean delivery** — an emission guard normalizes, dedupes, and rate-limits findings
  so the verifier never nags.
- **First-class UI** — a Verifier Mode screen that looks and feels like a normal
  session: transcript view, composer, and the composer's own model catalog as the
  verifier model selector.

## Repository layout (the `plugin/` folder is the installable package)

```
hermes-verifier/
├── plugin/               ← install root (~/.hermes/plugins/hermes-verifier/)
│   ├── plugin.yaml       ← manifest: 4 hooks, llm.model_override capability, schema
│   ├── __init__.py       ← register(ctx)
│   ├── brief.py          ← VERIFIER.md / VERIFIER.yml discovery, @imports, prompts
│   ├── delta.py          ← transcript delta extraction + redaction + finding-strip
│   ├── guard.py          ← emission guard (normalize, phrase filter, dedupe, rate limit)
│   ├── runtime.py        ← per-session verifier runtime, delivery, roster, events
│   ├── commands.py       ← /verify slash command
│   ├── dashboard/
│   │   ├── manifest.json
│   │   └── plugin_api.py ← REST + websocket backend (/api/plugins/hermes-verifier/)
│   ├── desktop/
│   │   └── plugin.js     ← Verifier Mode page (route /verifier)
│   └── skills/hermes-verifier/SKILL.md
├── tests/                ← 41 tests: guard, delta, runtime, roster, backlog, events
├── docs/design.md        ← full architecture and behavior spec
└── PLAN.md               ← milestone checklist
```

## Install

Clone and symlink the package, then enable it:

```bash
git clone git@github.com:am423/hermes-verifier.git
ln -s "$(pwd)/hermes-verifier/plugin" ~/.hermes/plugins/hermes-verifier
hermes plugins enable hermes-verifier
# restart the gateway, then per session:
#   /verify on
```

Or use the one-click install link from the Hermes desktop app:

```
hermes://plugin/install?repo=am423/hermes-verifier&enable=1
```

(The deep link probes what the repo ships; if its probe expects `plugin.yaml` at the
repo root rather than under `plugin/`, restructure the repo accordingly first. The
repo is private, so the link works for the owner only until it is made public.)

## Activating

- Session-scoped: `/verify on` (or the Verifier Mode page's switch).
- Default-on: `plugins.entries.hermes-verifier.settings.enabled: true` in `config.yaml`.
- Verifier model: `settings.model` as `provider/model`, or the dropdown in Verifier Mode.
- Findings: `~/.hermes/plugin-data/hermes-verifier/verify.<session>.<verifier>.jsonl`.

## Status

All four milestones are implemented: verifier runtime, backend API, Verifier Mode UI,
and roster/briefs. The desktop half ships opt-in — enable it in Settings → Plugins
after a gateway restart.

## License

MIT
