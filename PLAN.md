# PLAN.md

Execution plan for the Hermes Verifier desktop plugin. Full behavior spec lives in
[docs/design.md](docs/design.md). Milestones are ordered so that a working end-to-end
slice exists as early as possible.

## M0 — Scaffold (this commit)

- [x] Repository skeleton: README, PLAN, design doc, license, folder layout
- [x] Private GitHub repository
- [ ] CI smoke job: YAML parse of `plugin/plugin.yaml` (no runtime yet)

## M1 — Verifier runtime (agent-side)

Goal: the verifier sees every turn delta and can raise findings; findings reach the
primary session through context injection. No UI yet — slash commands only.

- [ ] `plugin/plugin.yaml` manifest (hooks: `post_llm_call`, `post_tool_call`,
      `on_session_start`, `on_session_reset`; capabilities: `llm.model_override`)
- [ ] Delta recorder: capture and render transcript deltas (assistant text, tool calls,
      tool results, reasoning markers) with secret redaction
- [ ] Verifier loop: single default verifier, model resolved from plugin settings /
      `ctx.llm`, own append-only context
- [ ] Finding tool: `raise_finding(note, severity)` with `note` / `caution` / `blocker`
- [ ] Emission guard: normalization, content-free phrase filter, exact-text dedupe,
      one accepted finding per review cycle
- [ ] Delivery: `note` → next step boundary via `pre_llm_call` injection;
      `caution` / `blocker` → steering (mechanism spike — see design doc §9)
- [ ] Reset semantics: compaction/session-switch re-prime, mid-session enable seeds
      cursor to current transcript length
- [ ] Slash commands: `/verify`, `/verify on|off|status|dump`
- [ ] Test: scripted session → assert findings JSONL emitted and injected

## M2 — Backend API

Goal: the desktop UI can talk to the runtime.

- [ ] `dashboard/manifest.json` + `dashboard/plugin_api.py`
- [ ] REST: `GET /state`, `GET /transcript`, `GET /findings`, `POST /pause`,
      `POST /resume`, `POST /ask`, `PUT /model`
- [ ] Websocket `/events` (findings, deltas, status changes)
- [ ] Per-session persistence: `__verify.jsonl` in session artifacts dir

## M3 — Desktop UI (Verifier Mode)

Goal: the session-like screen.

- [ ] `desktop/plugin.js`: route `/verifier` + sidebar nav + palette command
- [ ] Header: **Verifier Mode** title, model selector dropdown for the verifier model,
      enable switch, status badge
- [ ] Body: session-style transcript (watched-session deltas + finding cards),
      streaming indicator
- [ ] Composer: send the verifier a question; `data-hermes-send` style interaction
- [ ] Live updates via `ctx.socket('/events')` with React Query polling fallback
- [ ] Theme-correct styling via `var(--ui-*)` only

## M4 — Review briefs, roster, polish

- [ ] `VERIFIER.md` discovery (user + project walk to git root), prompt ordering,
      `@path` import expansion
- [ ] `VERIFIER.yml` roster: named verifiers with per-entry model/tools/instructions
- [ ] Catch-up window (`off | 1 | 3 | 5`) for primary-turn pacing
- [ ] Interruption cooldown (`immunity_turns`)
- [ ] Usage/cost attribution surfaced in `/verify status`
- [ ] Install link (`hermes://plugin/install?repo=…`) in README

## Risks / spikes

| # | Risk | Where |
|---|------|-------|
| 1 | Mid-turn steering mechanism for `caution`/`blocker` may not exist as a first-class hook; may need gateway event / session message dispatch | M1 |
| 2 | `ctx.llm` exact surface (model override, streaming) needs a spike against the running gateway | M1 |
| 3 | Desktop transcript fidelity: matching the native session screen look is component-by-component; core session components may not be exported to the SDK | M3 |
| 4 | Secret redaction parity: verifier must never see raw secrets that the primary's transcript already obfuscates | M1 |
