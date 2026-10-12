# Hermes Verifier

A desktop plugin for [Hermes Agent](https://hermes-agent.nousresearch.com/) that attaches
an independent **verifier** to your session. The verifier is a second model that reviews
each update to the primary agent's transcript and raises **findings** that get steered
back into the session before the primary agent runs off a cliff. (A read-only tool grant
for claim-checking exists, but it is opt-in and measured near-inert — see
[Advisor tools](#advisor-tools-and-their-limits).)

## What it does

- **Second-opinion review** — every primary turn delta is rendered (reasoning, tool
  intent, tool results) and sent to the verifier model you choose.
- **Findings, not control** — the verifier never mutates session state and nothing it
  does pauses the session. It raises findings with one of three severities:
  - `note` — a non-blocking aside, injected with the next turn's context.
  - `caution` — material risk, wrong direction, hallucinated API; delivered with the
    next turn's context (no mid-turn interruption).
  - `blocker` — the work is headed somewhere broken, and the only severity that can
    steer: a pending blocker keeps a coding turn alive via the pre-verify gate
    (bounded by `immunity_turns` and `blocker_steer`).
- **Tools off by default** — `advisor_tools` defaults `false` and `default_tools`
  defaults `[]`, so a verifier starts with no tool grant at all. When the feature is
  enabled, only `read_file` and `search_files` can be granted (the latter does the
  glob-style file search), and a roster entry may grant them to its own verifier.
- **Review briefs** — drop a `VERIFIER.md` in a project (or `~/.hermes/`) to tell the
  verifier what to watch for: dangerous APIs, architectural boundaries, quality bars.
  A `VERIFIER.yml` roster declares named verifiers with their own model, tool grant,
  and specialization.
- **Clean delivery — with measured limits** — an emission guard normalizes, dedupes,
  and rate-limits findings, but it is not a no-nag guarantee: false-alarm rates
  measured **44–100%** across this project's configurations (one exception: the
  3-of-3 agreement filter on deepseek at 17%, which kept 72% of findings). The
  `agreement_reviews` / `agreement_min` settings (see `plugin.yaml`) are the filter.
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
| Advisor tools (verify claims, not prose) | — | `read`/`grep`/`glob` | **Yes, opt-in** — `read_file`/`search_files` allowlist; measured near-inert (4 calls in 216 reviews) |
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
OMP's core advantage — but it is plumbing, not yet a working capability:

> **Measured:** across 216 recorded reviews the reviewer made ~4 tool calls in
> total. Treat tool use as opt-in plumbing, not a working capability — expect
> the verifier to review prose first, with claim-checking reserved for the cases
> where the transcript alone cannot settle a claim.

The grant is deliberately narrow:

- Only `read_file` and `search_files` are allowlisted. `terminal`, `write_file`,
  and `patch` are refused, so a mistyped or over-broad roster entry cannot
  widen the advisor's reach.
- Dispatch goes through `ctx.dispatch_tool`, so the host's approval, redaction,
  and result-budget pipelines still apply exactly as they would for a
  model-initiated call.
- At most 3 tool calls per review. Each tool result is truncated to **1500**
  characters (`delta._TOOL_RESULT_CAP` — one result row), and all tool evidence
  forwarded in a single review is capped at **6000** characters total
  (`runtime._MAX_EVIDENCE_CHARS`). Both are separate from `max_delta_chars`
  (default **20000**), which caps the whole rendered delta sent for review.
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
| `verifier_session` | `""` | Designated Verifier Mode session (managed by the desktop UI; do not edit by hand). |
| `advisor_tools` | `false` | Let granted advisors verify claims with read-only tools. Only this switch enables the feature; a roster entry can only grant tools while it is on. |
| `default_tools` | `[]` | Read-only tools for the default verifier. Non-allowlisted names are refused. |
| `max_reviews_per_session` | `0` | Hard cap on reviews per session; `0` = unlimited. |
| `advisor_max_tokens` | `0` | Cap advisor output per review; `0` = provider default, minimum `1024` when set. |
| `advisor_routine_tokens` | `1024` | Output-token budget for a review the gate tiered `routine`; never applied below the `1024` floor. |
| `nudge_first_turns` | `0` | Inject a consultation reminder into the first N turns. See the caveat above. |
| `sync_backlog` | `"off"` | Catch-up pacing when the advisor falls behind: any digit `1`–`5` (`off`/`0` disables). |
| `immunity_turns` | `3` | Turns after an interrupting finding before another may interrupt. |
| `max_delta_chars` | `20000` | Cap on the rendered transcript delta sent to the advisor. |
| `blocker_steer` | `true` | Let a blocker keep a coding turn alive via the pre-verify gate. |
| `gate_enabled` | `false` | A cheap model call decides whether each turn is reviewed at all, and at what effort. Fail-safe: any error or unparseable reply engages the review at deep effort. |
| `phase_gate_enabled` | `false` | Review an impending high-risk tool call and veto it before it runs. Fails open: if the gate cannot reach its model, the call proceeds. |
| `phase_max_per_turn` | `1` | Cap on phase-gate checks per turn. |
| `agreement_reviews` | `1` | Run each review this many times; deliver a finding only when `agreement_min` of the runs raise it. Costs that many times the review price. |
| `agreement_min` | `1` | How many of `agreement_reviews` must raise a finding for it to be delivered. Set equal for the strictest filter. |

### Brief safety and phase-gate limits

Top-level `VERIFIER.md`, `VERIFIER.yml` and `VERIFIER.yaml` files, and their
`@path` imports, share a confined reader. Resolved paths must stay inside the
project discovery root (or `HERMES_HOME` for user-level files). The reader rejects
non-regular files, caps each file at 256 KiB, and consults Hermes's file-read safety
policy. When no repository is found, discovery stops at the login home directory;
outside that home, it checks only the working directory plus user-level files.

The optional phase gate is a conservative phrase trigger, not a shell parser or
security sandbox. Quoted shell and SQL bodies remain visible to it. A search that
merely mentions a risky action can therefore trigger review too; the reviewer can
allow it. Quoting must not hide an executable action from review.

### Development scripts

The development/evaluation directories are opt-in and not imported by the plugin
runtime. Direct provider experiments require an explicitly supplied
`DEEPSEEK_API_KEY` in the process environment; they do not read Hermes credential
stores. Codex evaluations delegate authentication and transport to Hermes's CLI.
Offline analysis resolves the active profile through `get_hermes_home()` and reads
session data only when explicitly run. See [development authentication and
validation](evals/DEVELOPMENT.md) for details and network/billing implications.

## Repository layout (the repo root IS the installable package)

```
hermes-verifier/               ← install root (~/.hermes/plugins/hermes-verifier/)
├── plugin.yaml                ← manifest: 5 hooks, llm.model_override capability, schema
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
├── tests/                     ← guard, delta, runtime, roster, backlog,
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

The probe detects both halves at the repo root: agent via `plugin.yaml` +
`__init__.py`, desktop via `desktop/plugin.js`.

## Activating

- Session-scoped: `/verify on` (or the Verifier Mode page's switch).
- Default-on: `plugins.entries.hermes-verifier.settings.enabled: true` in `config.yaml`.
- Verifier model: `settings.verifier_model` as `provider/model`, or the dropdown in Verifier Mode.
- Findings: `~/.hermes/plugin-data/hermes-verifier/verify.<session>.<verifier>.jsonl`.

## Status

All four milestones are implemented: verifier runtime, backend API, Verifier Mode UI,
and roster/briefs. The desktop half ships opt-in — enable its Desktop switch in
Capabilities → Plugins. Settings → Plugins opens the configuration form.

### Confirming the advisor is engaging

`/verify status` reports an `engagement:` line, aggregated across every session the
process has seen, with the review count and the empty-response count. A non-zero
review count means the advisor actually ran. `empty response(s)` counts provider
completions that returned no payload — those are retried once, then surfaced in
`last_error`. A rising empty count with a flat review count means the advisor model
is not answering, not that it found nothing.

## Disclosure

When enabled, every primary turn's transcript delta (regex-redacted, up to 20,000
chars) plus any VERIFIER.md/VERIFIER.yml review briefs found from the working
directory up to the repo root are sent to the verifier model's provider on your
normal billing (one or more calls per turn per roster entry); a tool-granted advisor
can read files through Hermes's read_file/search_files; a project VERIFIER.yml can
add named verifiers and instructions whose findings are injected into your session
and can keep a coding turn going; findings, reasoning and short delta excerpts are
kept in HERMES_HOME/plugin-data/hermes-verifier/*.jsonl; adds a dashboard API and a
Desktop Verifier page. No telemetry, no shell, no stored credentials.

Additionally, from v0.4.0 — *added by the plugin authors, covering what the merge
disclosure predates:* the engagement gate (off by default) sends a bounded digest
(≤1,500 chars) to decide whether a turn is reviewed; the phase gate (off by default)
sends a bounded excerpt of an impending high-risk tool call's **redacted** arguments;
both write decisions to jsonl under the same directory; JSONL logs rotate at 8 MB
keeping one `.1` generation.

## License

MIT
