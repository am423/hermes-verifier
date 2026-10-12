# Development and evaluation authentication

The `evals/`, `sidecar/`, and `autoresearch/` directories contain opt-in
experiments and offline analysis. They are not imported by the plugin runtime.
Keeping these scripts here does **not** grant them access to Hermes's credential
stores.

## Provider experiments

- Direct DeepSeek experiments read **only** `DEEPSEEK_API_KEY` from the process
  environment. Supply it securely in the launching environment; the scripts do
  not source or parse any environment file, settings file, OAuth store, or
  credential pool. A missing, empty, or whitespace-only key fails explicitly
  before a provider request, even if Hermes has a saved key.
- Codex evaluation runs **only through the host-owned Hermes CLI**. The harness
  passes prompts, provider and model options to `hermes chat`; authentication,
  token refresh and provider transport remain entirely with Hermes. There is no
  direct Codex transport or `--provider codex` option.
- Provider experiments are paid/network operations when explicitly launched.
  The unit tests below use invented credentials and fake transports, not a live
  provider or an actual Hermes CLI process.

From the repository root, after arranging the environment key for a direct run:

```bash
python evals/run.py --configs baseline,notools --model deepseek/deepseek-flash
```

For a provider configured in the host, replace `MODEL` with a model supported by
that provider (no DeepSeek environment key is needed for this route):

```bash
python evals/run.py --provider cli --cli-provider openai-codex --model MODEL
```

Use `--provider cli` for **every** arm of a CLI-based model comparison: mixing a
bare DeepSeek API request and a CLI agent turn confounds the model with agent
scaffolding. CLI results do not expose token usage; they are quality comparisons,
not cost comparisons. `--list` and `--help` do not require credentials.

The synthetic eval's `read_file`/`search_files` tools are confined to its
materialised case workspace, including resolution of symlink targets. Model
output must not become a route to reading host credential files.

All sidecar provider experiments share `sidecar/llm_gate.py:load_key`, with the
same environment-only rule. See [the sidecar notes](../sidecar/NEGATIVE-RESULTS.md#4-rebuilding-the-numbers).

## Offline profile-aware analysis

`autoresearch/harvest.py` and `evals/moment_analysis.py` resolve data paths **at
call time** through `hermes_constants.get_hermes_home()`. That preserves the active
host profile, including profile changes after import. When Hermes is not
installed, a small lazy adapter uses `HERMES_HOME`, or the conventional
`~/.hermes` default only when no override is set. An installed host's errors are
not silently converted to a different profile. Explicit database/data-directory
arguments bypass default-home resolution.

The only default analysis inputs are the profile's read-only `state.db` and
`plugin-data/hermes-verifier` decision logs. Resolving a home directory does not
read settings or credentials. Autoresearch makes no provider calls, applies no
configuration changes, and still writes only its own reports. Its invocation and
report location are unchanged:

```bash
python -m autoresearch.run
```

No scheduled job needs migration for this change. No job, schedule, profile
configuration or running plugin is changed by these source edits.

## Offline regression checks

```bash
python -m pytest -q
python -m pytest -q autoresearch/tests
```

Give parallel worktrees separate `TMPDIR` directories before running tests: the
plugin test bootstrap creates a package symlink under that directory.

`tests/test_dev_credentials.py` creates populated fake stores in both a default
home and an active profile, then guards filesystem opens to prove neither store
is read. It blocks live HTTP/CLI calls, exercises the sidecar entrypoints,
checks env-only DeepSeek authentication and host-owned Codex routing, and scans
all development Python sources for removed credential APIs/private transport.
`autoresearch/tests/test_paths.py` uses synthetic databases and logs to test
redirected homes, host profiles, standalone fallback and explicit input paths.
