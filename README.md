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
- **First-class UI** — a **Verifier Mode** that rides on a real session: the
  sidebar's Verifier entry opens a regular session screen with the verifier
  attached, the titlebar shows "Verifier Mode" plus the verifier model
  selector (the composer's own model catalog) while that session is focused,
  and a Findings pane shows the verifier's live activity.

## What's new vs Claude Code and OMP

This plugin implements the **advisor pattern** — a second, independent model that
reviews the primary agent's work mid-task. It takes the best of both reference
implementations:

| Capability | Claude Code advisor | OMP advisor | Hermes Verifier |
| --- | --- | --- | --- |
| Advisor tools (verify claims, not prose) | — | `read`/`grep`/`glob` | **Yes** — `read_file`, `search_files` via a read-only allowlist |
| Per-session review budget | `max_uses` (per request) | — | **Yes** — `max_reviews_per_session` |
| Advisory output cap | `max_tokens` | — | **Yes** — `advisor_max_tokens` |
| Multi-advisor roster | — | `WATCHDOG.yml` | **Yes** — `VERIFIER.yml`, per-advisor model |
| Project review guidance | — | `WATCHDOG.md` | **Yes** — `VERIFIER.md` |
| First-turn consultation nudge | Yes (+7pp on Haiku-class) | — | **Yes** — `nudge_first_turns`, off by default |
| Catch-up pacing | — | `syncBacklog` | **Yes** — `sync_backlog` |
| Immunity / rate limit | — | `immuneTurns` | **Yes** — `immunity_turns` |
| Cost + token transparency | `/usage` | `/advisor status` | **Yes** — `/verify status` |

**Neither** Claude Code nor OMP ships a session-level review budget: Claude Code
caps per request, OMP does not cap at all, so a noisy roster silently runs up
bill. `max_reviews_per_session` is the honest session-scoped version, and it
counts *attempts* rather than successes, so a failing review still consumes
budget — that is what makes it a cost control.

### The nudge caveat

Claude Code measured the turn-2 consultation reminder at **+7 percentage points
on Haiku-class executors, no measurable effect on Sonnet, and slightly lower
pass rates on Opus**. It is therefore off by default. Enable it deliberately,
and not on a strong main model.

## Advisor tools and their limits

With `advisor_tools: true`, an advisor whose roster entry (or `default_tools`)
grants a tool may **verify a claim instead of reviewing prose** — reading the
file the primary agent just edited, or grepping for the API it called. That is
OMP's core advantage and the main reason an advisor beats a same-model
rubber-stamp review.

The grant is deliberately narrow:

- Only `read_file` and `search_files` are allowlisted. `terminal`, `write_file`,
  and `patch` are refused, so a mistyped or over-broad roster entry cannot
  widen the advisor's reach.
- Dispatch goes through `ctx.dispatch_tool`, so the host's approval, redaction,
  and result-budget pipelines still apply exactly as they would for a
  model-initiated call.
- At most 3 tool calls per review, each result truncated to 6000 characters.
- Grants are **per verifier**: only an advisor that was granted a tool is told
  tools exist, and only it can call one.

> **Privacy:** a tool-granted advisor reads files your session can read, and
> those reads go to the advisor model's provider. That is the same tradeoff OMP
> makes, and the reason the feature ships off by default.

## Configuration

Settings live under `plugins.entries.hermes-verifier.settings`. Every new
capability is **off by default**, so the zero-config behavior is unchanged.

| Key | Default | Effect |
| --- | --- | --- |
| `enabled` | `false` | Run the verifier after every primary turn. |
| `verifier_model` | `""` | Advisor model as `provider/model`; empty = the session's primary model. |
| `advisor_tools` | `false` | Let granted advisors verify claims with read-only tools. |
| `default_tools` | `[]` | Read-only tools for the default verifier. Non-allowlisted names are refused. |
| `max_reviews_per_session` | `0` | Hard cap on reviews per session; `0` = unlimited. |
| `advisor_max_tokens` | `0` | Cap advisor output per review; `0` = provider default, minimum `1024` when set. |
| `nudge_first_turns` | `0` | Inject a consultation reminder into the first N turns. See the caveat above. |
| `sync_backlog` | `"off"` | Catch-up pacing when the advisor falls behind: `off`/`1`/`3`/`5`. |
| `immunity_turns` | `3` | Turns after an interrupting finding before another may interrupt. |
| `max_delta_chars` | `20000` | Cap on the rendered transcript delta sent to the advisor. |
| `blocker_steer` | `true` | Let a blocker keep a coding turn alive via the pre-verify gate. |

## Repository layout (the repo root IS the installable package)

```
hermes-verifier/               ← install root (~/.hermes/plugins/hermes-verifier/)
├── plugin.yaml                ← manifest: 4 hooks, llm.model_override capability, schema
├── __init__.py                ← register(ctx)
├── brief.py                   ← VERIFIER.md / VERIFIER.yml discovery, @imports, prompts
├── advisor_tools.py           ← read-only tool grant: allowlist, TOOL_CALL protocol, dispatch
├── delta.py                   ← transcript delta extraction + redaction + finding-strip
├── guard.py                   ← emission guard (normalize, phrase filter, dedupe, rate limit)
├── runtime.py                 ← per-session verifier runtime, delivery, roster, events
├── commands.py                ← /verify slash command
├── dashboard/
│   ├── manifest.json
│   └── plugin_api.py          ← REST + websocket backend (/api/plugins/hermes-verifier/)
├── desktop/
│   └── plugin.js              ← Verifier Mode page (route /verifier)
├── skills/hermes-verifier/SKILL.md
├── tests/                     ← 88 tests: guard, delta, runtime, roster, backlog,
│                                budget, nudge, advisor tools, gaps, events
├── pytest.ini                 ← importlib mode: keeps pytest off the hyphenated root package
├── docs/design.md             ← full architecture and behavior spec
└── PLAN.md                    ← milestone checklist
```

The docs/ and tests/ folders ride along inertly — Hermes imports only
`__init__.py` and its submodules, and the install probe keys off
`plugin.yaml` + `__init__.py` + `desktop/plugin.js` at this root.

## Install

Clone and symlink the repo itself, then enable it:

```bash
git clone git@github.com:am423/hermes-verifier.git
ln -s "$(pwd)/hermes-verifier" ~/.hermes/plugins/hermes-verifier
hermes plugins enable hermes-verifier
# restart the gateway, then per session:
#   /verify on
```

Or use the one-click install link from the Hermes desktop app:

```
hermes://plugin/install?repo=am423/hermes-verifier&enable=1
```

(The repo is private, so the link works for the owner only until it is made
public. The probe detects both halves at the repo root: agent via
`plugin.yaml` + `__init__.py`, desktop via `desktop/plugin.js`.)

## Activating

- Session-scoped: `/verify on` (or the Verifier Mode page's switch).
- Default-on: `plugins.entries.hermes-verifier.settings.enabled: true` in `config.yaml`.
- Verifier model: `settings.model` as `provider/model`, or the dropdown in Verifier Mode.
- Findings: `~/.hermes/plugin-data/hermes-verifier/verify.<session>.<verifier>.jsonl`.

## Status

All four milestones are implemented: verifier runtime, backend API, Verifier Mode UI,
and roster/briefs. The desktop half ships opt-in — enable it in Settings → Plugins
after a gateway restart.

### Confirming the advisor is engaging

`/verify status` reports an `engagement:` line with the review count and the
empty-response count for this session. A non-zero review count means the advisor
actually ran. `empty response(s)` counts provider completions that returned no
payload — those are retried once, then surfaced in `last_error`. A rising empty
count with a flat review count means the advisor model is not answering, not that
it found nothing.

## License

MIT
