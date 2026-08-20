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

- [x] `plugin/plugin.yaml` manifest (hooks: `post_llm_call`, `pre_llm_call`,
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

| # | Risk | Status |
|---|------|--------|
| 1 | Mid-turn steering mechanism for `caution`/`blocker` may not exist as a first-class hook | **Resolved** — `pre_verify` hook returns `{"action":"continue","message":…}` to keep coding turns alive; `pre_llm_call` returns `{"context":…}` for next-turn injection |
| 2 | `ctx.llm` exact surface (model override, streaming) needs a spike against the running gateway | **Resolved** — `ctx.llm.complete_structured(…, json_schema=…)` with trust-gated `model=`/`provider=` overrides (`plugins.entries.<id>.llm.allow_model_override`) |
| 3 | Desktop transcript fidelity: matching the native session screen look is component-by-component; core session components may not be exported to the SDK | M3 |
| 4 | Secret redaction parity: verifier must never see raw secrets that the primary's transcript already obfuscates | M1 ships a regex scrubber + long-token collapse; parity with the host obfuscator is a hardening item |
