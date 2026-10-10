# PLAN.md

Execution plan for the Hermes Verifier desktop plugin. Full behavior spec lives in
[docs/design.md](docs/design.md). Milestones are ordered so that a working end-to-end
slice exists as early as possible.

## M0 — Scaffold (this commit)

- [x] Repository skeleton: README, PLAN, design doc, license, folder layout
- [x] Private GitHub repository
- [ ] CI smoke job: YAML parse of `plugin.yaml` (no runtime yet)

## M1 — Verifier runtime (agent-side)

Goal: the verifier sees every turn delta and can raise findings; findings reach the
primary session through context injection. No UI yet — slash commands only.

- [x] `plugin.yaml` manifest (hooks: `post_llm_call`, `pre_llm_call`,
      `pre_verify`, `on_session_reset`; capability: `llm.model_override`)
- [x] Delta recorder: capture and render transcript deltas (assistant text, tool calls,
      tool results, reasoning markers) with secret redaction
- [x] Verifier loop: single default verifier, model resolved from plugin settings /
      `ctx.llm.complete_structured`, own append-only context
- [x] Finding channel: structured review envelope (`severity`/`finding`/`reasoning`)
      instead of a registered model tool — keeps the primary toolset untouched
- [x] Emission guard: normalization, content-free phrase filter, exact-text dedupe,
      one accepted finding per review cycle
- [x] Delivery: `note`/`caution` via `pre_llm_call` context injection (next step
      boundary); `blocker` additionally steers via the `pre_verify` gate on coding turns
      (spike #1 resolved)
- [x] Reset semantics: compaction/session-switch re-prime, mid-session enable seeds
      cursor to current transcript length
- [x] Slash commands: `/verify`, `/verify on|off|status|dump`, `/verify reset`
- [x] Test: 29 unit + integration tests (fake ctx), `hermes plugins doctor` clean,
      installed + enabled in this machine's Hermes (`~/.hermes/plugins/hermes-verifier`)
- [ ] Optional: real-session smoke test after the next gateway restart

## M2 — Backend API

Goal: the desktop UI can talk to the runtime.

- [x] `dashboard/manifest.json` + `dashboard/plugin_api.py` (mounts under
      `/api/plugins/hermes-verifier/` at gateway startup)
- [x] REST: `GET /state`, `GET /transcript`, `GET /findings`, `POST /ask`,
      `POST /pause`, `POST /resume`, `PUT /model`
- [x] Websocket `/events` (findings, deltas, status changes; seq-cursor replay,
      polling fallback on the client)
- [x] Live-module sharing: plugin_api reaches the running runtime through the
      PluginManager namespace (`hermes_plugins.<slug>`); degrades to on-disk
      JSONL reads when the plugin is not loaded
- [x] Per-session persistence: `verify.<session>.<verifier>.jsonl` under the
      sanctioned `plugin-data/hermes-verifier/` root (the session-artifacts
      relocation from the design needs a `hermes_state` lookup spike — deferred)

## M3 — Desktop UI (Verifier Mode)

Goal: a real session screen with the verifier attached.

- [x] `desktop/plugin.js`: route `/verifier` + sidebar nav (shield) + palette command
- [x] The sidebar entry opens the designated verifier session — a REAL session
      (created via `session.create`, opened through `host.openSession`) in the
      native chat surface, with the verifier attached via a session override
- [x] Titlebar overlay while that session is focused: **Verifier Mode** title,
      model selector dropdown (the composer's own `ModelCatalogMenu`), live
      status chip (active / verifying / paused / error)
- [x] Findings pane (right side): live verifier activity transcript
- [x] Live updates via `ctx.socket('/events')` + React Query polling fallback
- [x] Theme-correct styling via `var(--ui-*)` + inline tokens only
- [x] Ships opt-in (`defaultEnabled: false`) — enable in Settings → Plugins,
      then the sidebar entry appears
- [x] Fixed: `settings.model` was a reserved settings root (rejected on read and
      write) — renamed to `settings.verifier_model`; designated session persists
      under `settings.verifier_session`

## M4 — Review briefs, roster, polish

- [x] `VERIFIER.md` discovery (user + project walk to git root), prompt ordering,
      `@path` import expansion (fences stay literal, cycles skipped, missing
      imports stay literal)
- [x] `VERIFIER.yml` roster: named verifiers with per-entry model / tools /
      instructions; specificity rule (project leaf > ancestor > user); each runs
      with its own guard, stats, model, and JSONL; findings deliver together
- [ ] Verifier-side tool execution (roster `tools` grants are parsed, validated,
      and surfaced; the plugin LLM surface has no tool loop — needs an
      agent-loop design)
- [x] Catch-up window (`off | 1 | 3 | 5`) with a 30s cap and failure-release
- [x] Interruption cooldown (`immunity_turns`)
- [x] Usage/cost attribution surfaced in `/verify status` (aggregated per session)
- [x] Bundled skill (`skills/hermes-verifier/SKILL.md`) teaching Hermes how to
      talk about Verifier Mode
- [x] Install link documented in README (deep link; repo layout caveat noted)
- [ ] Real-session smoke test — requires a gateway restart (would kill the
      active session; next natural restart)

## Risks / spikes

| # | Risk | Status |
|---|------|--------|
| 1 | Mid-turn steering mechanism for `caution`/`blocker` may not exist as a first-class hook | **Resolved** — `pre_verify` hook returns `{"action":"continue","message":…}` to keep coding turns alive; `pre_llm_call` returns `{"context":…}` for next-turn injection |
| 2 | `ctx.llm` exact surface (model override, streaming) needs a spike against the running gateway | **Resolved** — `ctx.llm.complete_structured(…, json_schema=…)` with trust-gated `model=`/`provider=` overrides (`plugins.entries.<id>.llm.allow_model_override`) |
| 3 | Desktop transcript fidelity: matching the native session screen look is component-by-component; core session components may not be exported to the SDK | M3 |
| 4 | Secret redaction parity: verifier must never see raw secrets that the primary's transcript already obfuscates | M1 ships a regex scrubber + long-token collapse; parity with the host obfuscator is a hardening item |

## M5 — Advisor-mode upgrade (best of Claude Code + OMP)

- [x] Declare new settings: `advisor_tools`, `default_tools`,
      `max_reviews_per_session`, `advisor_max_tokens`, `nudge_first_turns`
- [x] **Fix `sync_backlog`** — the `.isdigit()` call crashed on an int setting,
      was swallowed by the handler's broad except, and silently dropped that
      turn's finding. Now normalized via `_parse_sync_backlog`.
- [x] **Advisor tools** (`advisor_tools.py`) — a read-only allowlist
      (`read_file`, `search_files`) with a `TOOL_CALL:` line protocol, dispatched
      through `ctx.dispatch_tool` so host approval/redaction/budget still apply.
      Grants are per verifier (roster `tools:` or `default_tools`).
- [x] **Review budget** — `max_reviews_per_session` (Claude Code's `max_uses`,
      session-scoped). Counts attempts, not successes, so failures still cost.
- [x] **Advisory output cap** — `advisor_max_tokens` forwarded to the provider.
- [x] **First-turn nudge** — `nudge_first_turns`, injected via `pre_llm_call`
      (cache-safe). Off by default; carries Claude Code's measured caveat
      (+7pp Haiku, neutral Sonnet, slightly negative Opus).
- [x] `/verify status` surfaces budget, tool grant, and nudge state.
- [x] Tests: 43 → 84.

### Deliberately out of scope

- Claude Code's server-side `advisor_20260301` tool (encrypted
  `advisor_redacted_result`, `pause_turn` resumption, `max_uses` per request) is
  an Anthropic server-side primitive with no Hermes equivalent hook.
- Model-pairing rank tables ("advisor must rank ≥ main model") need a capability
  ranking Hermes does not expose.
- OMP's `tier.advisor` service tiers have no Hermes surface.
