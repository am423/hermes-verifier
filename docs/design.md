# Hermes Verifier — Design

## 1. Purpose

The verifier subsystem attaches one or more optional reviewer models to a Hermes
session. Each verifier reviews primary-agent transcript updates, can inspect the
workspace with its own tools, and injects concise **findings** back into the primary
session.

A verifier does not approve actions or mutate primary session state directly. Its
default investigative toolset is `read_file`, `search_files` (content), and
`search_files` (files/glob), but a `VERIFIER.yml` roster entry may grant any built-in —
including mutating tools such as `write_file`, `patch`, `terminal`, and `browser`.
Those tools run against the workspace in an isolated verifier context, and they honor
the session's normal approval mode and per-tool policies; grant them only when the
verifier model and workspace are trusted (see §7).

## 2. Vocabulary

| Term | Meaning |
|------|---------|
| **Verifier** | A reviewer model attached to a session. Passive: it never edits state, only raises findings. |
| **Finding** | A single piece of advice raised by the verifier, with a severity. |
| **Review brief** | `VERIFIER.md` — verifier-only guidance: risks to watch, traps, quality bars. |
| **Roster** | `VERIFIER.yml` — declares the verifiers themselves (name, model, tools, instructions). |
| **Delta** | The new primary transcript content since the verifier's last review. |
| **Emission guard** | The filter on the finding path: normalize, dedupe, rate-limit. |

## 3. Enabling and model selection

The subsystem requires the plugin to be enabled (Settings → Plugins, plus the
`plugins.enabled` gate in `config.yaml`). Model selection then depends on the roster:

- Without any discovered `VERIFIER.yml` entries, the plugin creates the **default
  verifier** and resolves its model from plugin settings (`plugins.entries.<id>.settings.model`)
  or the desktop UI's model selector.
- With a roster, each enabled entry uses its explicit `model` when present, otherwise
  the default. An unresolvable entry is reported as `no_model` without preventing other
  entries from running.
- `verifiers[].enabled: false` keeps an entry visible as paused but does not build its
  runtime.

Model selectors use normal role/model resolution, including provider-prefixed ids,
canonical ids, fallback lists, and thinking suffixes. The desktop UI drives the default
verifier's model through a dropdown (see §10); roster entries override per entry.

### Headless runs

`--verify "Review this task."` enables the verifier for one print-mode process without
persisting settings. Verifier findings and blockers continue to steer the live turn;
after the final prompt settles, print mode preserves late findings without starting
hidden primary turns, then waits up to ten minutes for final reviews before disposing
the session. Error exits use a 30-second drain budget.

### Slash commands

| Command | Effect |
| --- | --- |
| `/verify` | Toggle the verifier subsystem for this session (session-scoped override). |
| `/verify on` | Enable the configured/default verifier runtimes for this session. |
| `/verify off` | Disable the subsystem for this session and stop its runtimes. |
| `/verify status` | Show each verifier's runtime state, model, context usage, token usage, and cost. |
| `/verify dump` | Copy the compact transcript (all active verifiers when a roster is present) to the clipboard. |
| `/verify dump raw` | Copy the full dump, including system prompt, tools, thinking, and calls. |

## 4. What the verifier sees

At each primary update, the verifier runtime receives **only the new transcript delta**
since its previous update. Deltas are rendered with reasoning, tool intent, and role
markers so verifiers can review assistant reasoning as well as user-visible text, tool
calls, and tool results. Provider-bound messages and tool arguments/results pass through
the session secret obfuscator before reaching the verifier model.

Most hidden internal messages collapse to a one-line summary in the delta. The primary
agent's injected plan-mode context is rendered verbatim inside an XML-escaped
`<primary-context kind="…">` wrapper, with repeated copies deduplicated. Verifiers also
receive the primary's discovered project context files (`AGENTS.md` and related standing
instructions) in a `<project-context>` system-prompt block. If the session cwd is outside
Git with exactly one direct child repository, an additional block tells the verifier
which child is the active project.

Verifier findings already injected into the primary transcript are filtered out before
the next delta is rendered. This prevents the verifier from recursively reviewing its own
output.

### Resets

When the primary transcript is rewritten, the verifier runtime is reset:

- compaction
- session switch/resume
- branch/fork-style history replacement
- context-maintenance re-prime when the verifier's own context cannot fit

A reset clears the verifier's private in-memory transcript and rewinds its cursor; the
next update replays the current bounded primary transcript. When the verifier is enabled
mid-session, the cursor seeds to the current primary transcript length so the whole old
conversation is not replayed.

## 5. Findings and delivery

The verifier has one output channel: a **finding** — a note plus an optional severity.

| Severity | Delivery | Intended use |
| --- | --- | --- |
| omitted / `note` | Non-interrupting aside, batched into the primary transcript at the next step boundary. | Cleanup, simplification, low-risk edge cases. |
| `caution` | Interrupting steering message when the delivery constraints below permit it. A late terminal-answer caution is preserved as a visible card instead. | Material risk, likely wrong direction, missing constraint, hallucinated API. |
| `blocker` | Interrupting steering message when the delivery constraints below permit it. Unlike a caution, a terminal answer alone does not prevent it from triggering a turn. | Continuing would clearly waste work or produce broken output. |

Accepted findings are rendered into the primary transcript as XML-escaped `<finding>`
elements. Named roster verifiers add a `verifier` attribute:

```xml
<finding verifier="Architecture" severity="caution" guidance="weigh, don't blindly obey">
note text
</finding>
```

### Delivery rules

When the user deliberately interrupts the agent (Esc, or a cancel from any surface), the
verifier stops auto-resuming it. An interrupting caution/blocker raised while the run is
stopped is recorded as a visible finding card instead of restarting the turn; the finding
re-enters context on the next resume (a new message or steer/follow-up).

A normal yield the agent drove itself is treated differently:

- **While the loop is still streaming**, the note normally steers into the live turn.
- **Once the loop has yielded and gone idle**, delivery keys on how the turn ended:
  - If the primary's tail is a **terminal text answer with no queued work**, a late
    caution is preserved as a visible card rather than waking the agent to restate a
    completed turn. A blocker is the exception: it normally steers a triggered turn,
    because the agent handed off broken or unexercised work.
  - Otherwise (the agent yielded mid-work), an idle caution/blocker normally triggers a
    fresh turn so the finding is acted on immediately.

Two session/client constraints can still preserve a note whose normal delivery path is
steering:

- **Plan mode:** every would-be steer is preserved as a visible card, because only
  user-driven turns converge on ask/resolve.
- **Deferred agent-initiated turns:** when the client cannot represent a triggered turn
  as busy, an idle would-be steer is preserved. Findings raised while the primary loop
  is already streaming can still steer into that live turn.

`immunity_turns` limits interruption frequency: after a verifier successfully delivers a
caution or blocker through the steering channel, later cautions/blockers route as
non-interrupting asides until the configured number of primary turns has completed.
Default `3`. Notes are unaffected.

While a verifier update is reviewing work still in progress, the finding tool withholds
`note` and `caution` calls; only a `blocker` may interrupt partial work. The tool also
suppresses the same whitespace-normalized note at an equal or lower severity while
allowing a real escalation (`note` → `caution` → `blocker`).

## 6. Emission guard

Each verifier has an emission guard on the route from the finding tool to the delivery
channel. It enforces the system prompt's "at most one accepted finding per update" and
no-repeat rules:

1. **Normalization.** Lowercase, NFKC, collapse every run of non-alphanumeric characters
   to one space, then trim.
2. **Content-free phrase filter.** Short phrases with no concrete reason — "stop",
   "done", "complete", "no issue continue", "lgtm", "nothing to add" — are suppressed.
3. **Exact-text dedupe.** Any normalized note already accepted by this verifier in this
   session is dropped. FIFO history holds at most 4096 entries.
4. **Per-update rate limit.** At most one finding per verifier model `prompt()` cycle is
   accepted. Suppressed noise never consumes the budget.

Guard-level suppression is invisible to the model (the finding tool has already returned
`Recorded.`). The tool's earlier equal-or-lower-severity duplicate check is intentionally
visible as `Duplicate finding ignored.`; in-progress non-blockers return `Recorded.`
without routing. Guard state clears on every verifier reset.

## 7. Tools and isolation

The verifier is a full agent with its own agent instance and a distinct tool context
whose id is suffixed `-verifier`. It does not share the primary agent's file snapshots,
seen-lines tracking, conflict state, or summary cache.

Every verifier has the finding tool for surfacing notes into the primary transcript.
When `tools` is omitted, the investigative grant is:

- `read_file`
- `search_files` (content)
- `search_files` (files / glob)

A `VERIFIER.yml` entry may select any subset of built-ins actually constructed for the
session. An explicit empty `tools: []` grants no investigative tools; the finding tool
remains available. Unknown-only lists are dropped with a warning and fall back to the
default subset. Grantable names include mutating tools such as `write_file`, `patch`,
`terminal`, `browser`, and `delegate_task`.

Verifier tools run against the isolated verifier context and are wrapped so approval
mode, per-tool approval policies, and auto-approve apply exactly as they do to registry
tools. A browser bridge exposes delete/edit/search capabilities only when the
corresponding verifier grant exists.

## 8. Review brief (`VERIFIER.md`) and roster (`VERIFIER.yml`)

### VERIFIER.md

`VERIFIER.md` is verifier-only guidance. It is appended to the verifier system prompt;
it is not injected into the primary agent's normal context and does not behave like
`AGENTS.md`.

Use it for review priorities: risks the verifier should watch for, project-specific
traps, dangerous APIs, architectural boundaries, and quality bars that are useful to a
reviewer but too noisy for the main executor.

```markdown
# Review brief

Especially watch for:

- Changes that bypass the durable queue in `src/jobs/`.
- UI renderer paths that display unsanitized tool output.
- New worker spawns that do not re-enter the CLI host.
```

### Discovery locations

Discovery loads every readable candidate:

1. user level: `$HERMES_HOME/VERIFIER.md`
2. project levels while walking from `cwd` upward to the git repository root, or to the
   home directory when no repo root is found:
   - `<dir>/VERIFIER.md`
   - `<dir>/.hermes/VERIFIER.md`

Unlike native context files, discovery does not stop at the nearest project file —
multiple project files can load together. Candidates in hidden owner directories are
ignored unless the file is inside an `.hermes` directory.

### `@` imports

`VERIFIER.md` content is expanded with the `@path` import helper: relative imports
resolve from the importing file's directory, `~/` from the home directory, imports
inside fenced code blocks and inline code spans stay literal, cycles are skipped, and
missing or unreadable imports leave the original `@path` text in place.

### Prompt order

Loaded blocks are sorted: user-level file first, then project-level files from farther
ancestors down toward `cwd`. Each is appended to the verifier system prompt as:

```
Especially pay attention to:
<attention>
...expanded review brief content...
</attention>
```

Later project files sit closer to the end of the prompt, so narrower directory guidance
is more prominent than broad ancestor guidance.

### VERIFIER.yml

`VERIFIER.yml` (or `VERIFIER.yaml`) is the roster — one entry per name, each with its
own enable flag, model, tool grant, and specialization prompt. Files that fail to parse
or fail schema validation are logged and skipped.

```yaml
instructions: |
  Everyone: prefer diffs that keep tests unified.

verifiers:
  - name: Architecture
    enabled: true
    model: anthropic/claude-sonnet-4-5:medium
    tools: [read_file, search_files]
    instructions: |
      Watch cross-module coupling and public-API growth.

  - name: Fixer
    enabled: false
    model: anthropic/claude-sonnet-4-5:high
    tools: [read_file, search_files, patch, terminal]
    instructions: |
      You may edit and run tests to prove a fix locally, then advise.
```

Fields:

- `instructions` (top level): shared prompt prepended to every verifier's system prompt
  alongside `VERIFIER.md`. Concatenated across all discovered `VERIFIER.yml` files.
- `verifiers[].name`: human label; slugified for the session id and its
  `__verify.<slug>.jsonl` filename. Duplicate slugs across files resolve by the same
  specificity rule as `VERIFIER.md` discovery (project leaf > project ancestor > user).
- `verifiers[].enabled`: optional per-verifier switch, default `true`.
- `verifiers[].model`: optional model selector with optional thinking suffix. Omitted →
  uses the default verifier model (settings / UI selector).
- `verifiers[].tools`: optional list of built-in tool names. Omitted → default read-only
  subset; explicit `[]` → no investigative tools. Unknown names dropped with a warning.
- `verifiers[].instructions`: this verifier's specialization, appended after the shared
  baseline. Both instruction fields expand `@path` imports.

Discovery follows the same user + project search path as `VERIFIER.md`.

## 9. Cost, context, and pacing

Verifier usage is separate model usage. `/verify status` reports verifier token counts
and cost from the verifier's own transcript.

The verifier has its own append-only context. Before each verifier prompt, the session
estimates incoming tokens and may maintain verifier context:

1. try model-level context promotion when enabled and a larger compatible model is
   available
2. if promotion cannot fit enough context, compact the verifier's own message history
3. if compaction has no candidates or still cannot fit, re-prime from the current
   bounded primary transcript

The live context is in-memory and append-only, retained while the session runs so
`/verify dump` can inspect it. It is not a replacement for the primary persisted
transcript.

### Catch-up window (`sync_backlog`)

`sync_backlog` is not lockstep turn execution — it is a bounded catch-up delay for the
primary agent when the verifier falls behind. Allowed values: `off`, `1`, `3`, `5`.

On primary turn end:

1. the primary turn delta is queued for the verifier
2. the verifier drain loop starts or continues in the background
3. if not `off`, the primary waits only while verifier backlog is at or above the
   configured threshold
4. the wait is capped at 30 seconds
5. if the verifier catches up below the threshold, the primary continues immediately
6. if the cap expires, the primary continues anyway

Verifier failures do not permanently stall the primary. Retriable failures are attempted
up to three times before that backlog is dropped; three dropped-backlog cycles halt the
runtime until an explicit reset, and a permanent request rejection can halt it after one
cycle. A quota/usage-limit failure pauses the verifier with its batch retained until
`/verify` rebuilds it, configuration is reloaded, a new session starts, or the process
restarts. Catch-up waiters are released as soon as a verifier is failing.

## 10. Desktop UI — Verifier Mode

Verifier Mode rides on a **real session**. The sidebar's Verifier entry
(visible once the plugin is enabled in Settings → Plugins) routes to
`/verifier`, which resolves the designated verifier session — creating one
via `session.create` on first use — and hands it to the native chat surface
through `host.openSession`. The user runs the session normally; the verifier
is attached to it through a session-scoped override, so every turn is
reviewed and findings are delivered into the transcript.

While the designated session holds focus, the titlebar shows the Verifier
Mode chrome:

```
[Verifier Mode]  model [anthropic/claude-sonnet-4-5 ▾]  ● verifying
```

- **Verifier Mode** — fixed label; present only while the verifier session is
  focused.
- **Model selector** — the composer's own model catalog menu; the selection is
  written through `PUT /model` (persisted as `settings.verifier_model`).
- **Status chip** — `active` / `verifying` / `paused` / `error`, from the live
  runtime snapshot.

A right-side **Findings** pane shows the verifier's activity transcript
(findings with severity chips, suppressed items, direct questions and
answers), streaming over the `/events` socket with a polling fallback.

**Wiring.**

- Backend state via `ctx.rest` (`GET /state`, `GET/POST /verifier-session`,
  `PUT /model`, `POST /ask`)
- Live updates via `ctx.socket('/events')`; polling fallback with React Query
- Model selector writes through `PUT /model`, applied via
  `settings.verifier_model` (`model` alone is a reserved settings root)
- Styling strictly with `var(--ui-*)` theme variables; UI kit components so
  the screen matches the app pixel-for-pixel

## 11. Persistence and observability

Every finalized verifier turn is appended to JSONL inside the plugin's durable
data root (`$HERMES_HOME/plugin-data/hermes-verifier/`, the sanctioned per-plugin
storage convention — survives plugin updates and removal):

- default verifier: `verify.<session>.default.jsonl`
- named verifier: `verify.<session>.<slug>.jsonl`

Each record carries the finding, severity, verifier slug, model, usage/cost, and
a bounded excerpt of the reviewed delta, so the transcript is reconstructable for
observability. Relocating these logs into the owning session's artifacts directory
(as the original design specifies) needs a session-file lookup through the session
store — deferred as a small spike.

The on-disk log is append-only and independent of the in-memory context — re-primes
and compaction never truncate it.

The verifier is never a peer. Its registry identity is excluded from every
agent-facing surface — peer rosters, broadcast targets, subagent peer prompts, and
history index/lookup — and cannot be messaged or revived/killed from any hub surface.
It is not addressable as a peer regardless of what tools it has been granted.

## 12. Backend API surface (implemented)

All routes mount under `/api/plugins/hermes-verifier/` in the gateway dashboard.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/state` | Verifier runtime state: enabled, model, status, backlog, token usage, cost |
| GET | `/transcript` | Bounded view of the verifier's transcript (findings + reviewed deltas) |
| GET | `/findings` | Accepted findings for the current session |
| POST | `/ask` | Send the verifier a direct question |
| PUT | `/model` | Set the default verifier model (persisted to plugin settings) |
| POST | `/pause` · `/resume` | Suspend / resume the verifier for the session |
| WS | `/events` | Push: delta-reviewed, finding, status-change, usage |

## 13. Security posture

- The verifier is **passive by default** and its default grant is strictly read-only.
- Findings pass through the session secret obfuscator; the verifier never sees raw
  secrets.
- Verifier tool calls honor the session's approval mode and per-tool policies; mutating
  grants require an explicit, trusted roster entry.
- The Python backend is gated by `plugins.enabled` in `config.yaml` (renderer-side
  enable alone never imports Python). The desktop half ships opt-in
  (`defaultEnabled: false`).
- The emission guard caps verifier output volume (one finding per cycle, dedupe,
  content-free suppression), and unsafe verifier output (instruction overrides,
  destructive directives) is quarantined and discarded rather than dispatched.
- A quarantined turn silently resets and re-primes the verifier once; a second
  consecutive quarantine emits a host warning, drops the batch, and resets the context
  to break the loop.

## 14. Spikes — resolved during M1

1. **Steering channel.** Resolved: `pre_llm_call` returns `{"context": str}` which the
   host injects into the next primary turn's user message (the note/caution path), and
   the `pre_verify` hook accepts `{"action": "continue", "message": str}` to keep a
   coding turn alive (the blocker path). Both verified against the installed agent core.
2. **`ctx.llm` surface.** Resolved: `ctx.llm.complete_structured(instructions=…,
   input=[{"type":"text","text":…}], json_schema=…, json_mode=True, system_prompt=…,
   model=…, provider=…)` returns a result with `.parsed`, `.usage`, `.model`. Overrides
   are trust-gated by `plugins.entries.<id>.llm.allow_model_override` (or the
   `llm.model_override` capability); on denial the plugin degrades to the primary model.
3. **Plan-mode detection.** Open. Not required for the M1 delivery paths; revisit when
   the preserve-as-card rule is implemented.
