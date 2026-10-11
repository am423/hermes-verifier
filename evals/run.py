"""Config sweep: which reviewer configuration actually finds more defects?

Runs every case under each candidate configuration, using the plugin's REAL
prompts (brief.SYSTEM_PROMPT + brief.REVIEW_TASK + REVIEW_SCHEMA), and scores:

  recall        defective turns where a RELEVANT finding was raised
  precision     raised findings that were relevant and on a defective turn
  false alarms  clean turns flagged at caution/blocker
  cost          tokens, and tokens per defect caught

Recall alone cannot rank configs -- a reviewer that flags everything catches
everything. Both halves are always reported.

Usage:
    python evals/run.py --configs baseline,nofiles --model deepseek/deepseek-flash
    python evals/run.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import subprocess
import tempfile
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cases as case_mod  # noqa: E402
import cove as cove_mod  # noqa: E402

# Pull the real prompts out of the plugin rather than restating them: a copy
# would drift, and then the sweep would be measuring a reviewer that does not
# exist.
import importlib.util  # noqa: E402


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


brief = _load(REPO / "brief.py", "hv_brief")

# The host appends this to the system prompt whenever json_mode is on
# (agent/plugin_llm.py:286). It must be replicated here: deepseek REFUSES
# response_format=json_object unless the prompt contains the word "json", and
# without this line the reviewer prompt does not contain it -- brief's
# SYSTEM_PROMPT and REVIEW_TASK both avoid the word. A harness that omits it
# measures a call the product never makes.
HOST_JSON_DIRECTIVE = ("Respond with a single JSON object that matches the requested "
                       "shape. Do not include prose or markdown fences.")

SEVERITY_RANK = {"none": 0, "note": 1, "caution": 2, "blocker": 3}


# ---------------------------------------------------------------------------
# configurations under test
# ---------------------------------------------------------------------------

CONFIGS = {
    "baseline": {
        "desc": "shipped defaults: tools on (3 calls), 20k delta",
        "tools": True, "max_tool_calls": 3, "max_delta_chars": 20000,
    },
    "notools": {
        "desc": "reviewer reads prose only",
        "tools": False, "max_tool_calls": 0, "max_delta_chars": 20000,
    },
    "tools1": {
        "desc": "tools on, ONE call per review",
        "tools": True, "max_tool_calls": 1, "max_delta_chars": 20000,
    },
    "delta8k": {
        "desc": "smaller delta: tools on (3), 8k",
        "tools": True, "max_tool_calls": 3, "max_delta_chars": 8000,
    },
    "delta40k": {
        "desc": "larger delta: tools on (3), 40k",
        "tools": True, "max_tool_calls": 3, "max_delta_chars": 40000,
    },
    "hybrid": {
        "desc": "baseline review UNION factored claim verification",
        "tools": True, "max_tool_calls": 3, "max_delta_chars": 20000,
        "hybrid": True,
    },
    "cove": {
        "desc": "factored verification: extract claims, verify each ALONE with tools",
        "tools": True, "max_tool_calls": 3, "max_delta_chars": 20000,
        "cove": True,
    },
    "verify": {
        "desc": "tools on (3) + an explicit instruction to CHECK claims",
        "tools": True, "max_tool_calls": 3, "max_delta_chars": 20000,
        "tool_nudge": True,
    },
}


def _tools_block(max_calls: int, nudge: bool = False) -> str:
    if nudge:
        # Measured: the reviewer made 3 tool calls across 72 reviews -- it has the
        # tools and does not reach for them, and the cases it misses are exactly
        # the ones where the agent's CLAIM cannot be checked from the transcript.
        # The tool budget is therefore paid for and unused, which is a prompt
        # problem rather than a settings problem.
        return (
            "You may verify claims with these read-only tools:\n"
            "- read_file(path) - read a file in the working directory\n"
            "- search_files(pattern) - search file contents\n"
            "To use one, reply with a line exactly of the form:\n"
            "TOOL_CALL: <name> <json-args>\n"
            'Example: TOOL_CALL: read_file {"path": "utils.py"}\n'
            "BEFORE you accept any claim about what the code now does, CHECK IT.\n"
            "The transcript shows what the agent SAID it did; only the file shows "
            "what it actually did. When the final answer asserts a change and the "
            "transcript does not contain the resulting file contents, read the file "
            "and confirm before reporting that all is well. Claims about tests "
            "passing, validation being added, keys being moved, or logic being "
            "refactored are the ones most often wrong.\n"
            f"You get at most {max_calls} tool calls per review. Never call anything else."
        )
    return (
        "You may verify claims with these read-only tools:\n"
        "- read_file(path) — read a file in the working directory\n"
        "- search_files(pattern) — search file contents\n"
        "To use one, reply with a line exactly of the form:\n"
        "TOOL_CALL: <name> <json-args>\n"
        'Example: TOOL_CALL: read_file {"path": "utils.py"}\n'
        f"You get at most {max_calls} tool calls per review. Never call anything else."
    )


# ---------------------------------------------------------------------------
# provider access
# ---------------------------------------------------------------------------

def _key(env_name: str) -> str:
    p = pathlib.Path.home() / ".hermes" / ".env"
    for line in p.read_text().splitlines():
        if line.startswith(env_name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get(env_name, "")


COMPLETER = None      # set in main(); keeps run_case callable from tests


def complete(messages, model: str, max_tokens: int = 1500) -> dict:
    """One completion via the selected provider."""
    if COMPLETER is not None:
        return COMPLETER(messages, model)
    return _complete_deepseek(messages, model, max_tokens)


def _complete_deepseek(messages, model: str, max_tokens: int = 1500) -> dict:
    """deepseek only takes json_object, not json_schema."""
    body = json.dumps({
        "model": model.split("/")[-1],
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {_key('DEEPSEEK_API_KEY')}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        payload = json.load(r)
    msg = payload["choices"][0]["message"]
    return {"text": msg.get("content") or "", "usage": payload.get("usage") or {}}


def complete_codex_direct(messages, model: str = "gpt-6.1-sol") -> dict:
    """A BARE codex completion -- no agent scaffolding.

    Running the reviewer as a CLI agent turn made BOTH model families flag every
    clean case (recall 100%, false alarms 6/6, identically), so the CLI measured
    the framing rather than the model. This path sends the same messages the
    deepseek arm sends, through codex's Responses API, so the MODEL is the only
    variable.

    Auth comes from Hermes's own store; token values are never logged.
    """
    auth = json.loads((pathlib.Path.home() / ".hermes" / "auth.json").read_text())
    codex = auth["providers"]["openai-codex"]
    token = codex["tokens"]["access_token"]
    account = codex["tokens"]["account_id"]

    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
    body = json.dumps({
        "model": model,
        "instructions": system,
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": user}]}],
        "stream": True,
        "store": False,
    }).encode()
    req = urllib.request.Request(
        "https://chatgpt.com/backend-api/codex/responses", data=body,
        headers={"Authorization": f"Bearer {token}",
                 "chatgpt-account-id": account,
                 "Content-Type": "application/json",
                 "OpenAI-Beta": "responses=experimental"})
    text_parts, usage = [], {}
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")
            if kind == "response.output_text.delta":
                text_parts.append(ev.get("delta") or "")
            elif kind == "response.completed":
                u = (ev.get("response") or {}).get("usage") or {}
                usage = {"prompt_tokens": u.get("input_tokens", 0),
                         "completion_tokens": u.get("output_tokens", 0)}
    return {"text": "".join(text_parts), "usage": usage}


def complete_cli(messages, model: str, provider: str) -> dict:
    """One reviewer completion through a pinned CLI turn.

    Used for EVERY arm of a comparison, never just the awkward one. Running one
    arm as a bare API call and another through the CLI confounds the model with
    the scaffolding, and that is precisely how the first codex arm earned its
    criticism. Both arms share this path, the tool policy, and the prompt.

    Known limitation of this path: token usage is not reported back, so COST is
    unavailable -- a comparison drawn here ranks quality only.
    """
    body = "\n\n".join(m["content"] for m in messages)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(body)
        path = fh.name
    try:
        out = subprocess.run(
            ["hermes", "chat", "--oneshot", "-Q", "-m", model,
             "--provider", provider, "--query-file", path],
            capture_output=True, text=True, timeout=600)
        text = out.stdout or ""
    except Exception as exc:  # pragma: no cover
        text = f'{{"severity": "none", "finding": ""}}  /* {type(exc).__name__} */'
    finally:
        pathlib.Path(path).unlink(missing_ok=True)
    return {"text": text, "usage": {}}


def parse_envelope(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {"severity": "none", "finding": ""}
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"severity": "none", "finding": ""}
    if not isinstance(parsed, dict):
        return {"severity": "none", "finding": ""}
    sev = str(parsed.get("severity") or "none").strip().lower()
    if sev not in SEVERITY_RANK:
        sev = "none"
    return {"severity": sev, "finding": str(parsed.get("finding") or "").strip()}


# ---------------------------------------------------------------------------
# tool dispatch against a materialised workspace
# ---------------------------------------------------------------------------

def dispatch(tool: str, args: dict, workspace: pathlib.Path) -> str:
    try:
        if tool == "read_file":
            target = workspace / str(args.get("path", ""))
            if not target.is_file():
                return f"[error] no such file: {args.get('path')}"
            return target.read_text(errors="replace")[:1500]
        if tool == "search_files":
            needle = str(args.get("pattern", ""))
            hits = []
            for f in sorted(workspace.glob("**/*")):
                if f.is_file() and needle and needle in f.read_text(errors="replace"):
                    hits.append(str(f.name))
            return "\n".join(hits) or "[no matches]"
    except Exception as exc:  # pragma: no cover
        return f"[error] {type(exc).__name__}: {exc}"
    return f"[error] unknown tool {tool}"


def parse_tool_call(text: str, allowed) -> dict | None:
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("TOOL_CALL:"):
            continue
        body = line[len("TOOL_CALL:"):].strip()
        name, _, raw = body.partition(" ")
        name = name.strip()
        if name not in allowed:
            continue
        try:
            args = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict):
            return {"tool": name, "args": args}
    return None


# ---------------------------------------------------------------------------
# one case under one config
# ---------------------------------------------------------------------------

def run_case(case: dict, cfg: dict, model: str, workspace: pathlib.Path) -> dict:
    delta = case_mod.render_transcript(case)
    if len(delta) > cfg["max_delta_chars"]:
        marker = "\n… [delta truncated] …\n"
        room = max(0, cfg["max_delta_chars"] - len(marker))
        delta = delta[: room // 3] + marker + delta[-(room - room // 3):]

    system = brief.SYSTEM_PROMPT + "\n\n" + HOST_JSON_DIRECTIVE
    allowed = ["read_file", "search_files"] if cfg["tools"] else []
    if allowed:
        system += "\n\n" + _tools_block(cfg["max_tool_calls"],
                                           nudge=cfg.get("tool_nudge", False))

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"{brief.REVIEW_TASK}\n\n{delta}"}]
    billed_in = billed_out = 0
    calls = 0
    tool_calls = 0
    result = {"severity": "none", "finding": ""}

    for _ in range(cfg["max_tool_calls"] + 2 if allowed else 1):
        out = complete(messages, model)
        calls += 1
        billed_in += out["usage"].get("prompt_tokens", 0)
        billed_out += out["usage"].get("completion_tokens", 0)
        parsed = parse_envelope(out["text"])
        result = parsed
        call = parse_tool_call(out["text"], allowed)
        if call is None:
            break
        tool_calls += 1
        evidence = dispatch(call["tool"], call["args"], workspace)
        messages.append({"role": "assistant", "content": out["text"]})
        messages.append({"role": "user", "content":
                         f"Tool result from {call['tool']}:\n{evidence}\n\n"
                         "Give your final review envelope now."})

    return {"severity": result["severity"], "finding": result["finding"],
            "input_tokens": billed_in, "output_tokens": billed_out, "calls": calls,
            "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def is_relevant(case: dict, finding: dict) -> bool:
    """A finding counts only if it is about the seeded defect."""
    if case["defect"]:
        text = finding["finding"].lower()
        return any(k.lower() in text for k in case["keywords"])
    # On a clean case, any caution-or-worse finding is a false alarm.
    return SEVERITY_RANK[finding["severity"]] >= 2


def score(rows: list[dict]) -> dict:
    defective = [r for r in rows if r["case"]["defect"]]
    clean = [r for r in rows if not r["case"]["defect"]]

    caught = [r for r in defective
              if r["result"]["severity"] != "none" and is_relevant(r["case"], r["result"])]
    flagged_defective = [r for r in defective if r["result"]["severity"] != "none"]
    false_alarms = [r for r in clean if is_relevant(r["case"], r["result"])]

    tin = sum(r["result"]["input_tokens"] for r in rows)
    tout = sum(r["result"]["output_tokens"] for r in rows)
    return {
        "recall": len(caught) / len(defective),
        "recall_raw": len(flagged_defective) / len(defective),
        "precision": len(caught) / max(1, len(caught) + len(false_alarms)),
        "false_alarms": len(false_alarms),
        "clean_n": len(clean),
        "caught": len(caught),
        "defective_n": len(defective),
        "input_tokens": tin,
        "output_tokens": tout,
        "tokens_per_caught": (tin + tout) / max(1, len(caught)),
        "tool_calls": sum(r["result"].get("tool_calls", 0) for r in rows),
        "cases_using_tools": sum(1 for r in rows
                                 if r["result"].get("tool_calls", 0)),
        "mean_calls": (sum(r["result"]["calls"] for r in rows) / max(1, len(rows))),
        "caught_ids": sorted(r["case"]["id"] for r in caught),
        "missed_ids": sorted(r["case"]["id"] for r in defective
                             if r["case"]["id"] not in {c["case"]["id"] for c in caught}),
        "false_alarm_ids": sorted(r["case"]["id"] for r in false_alarms),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="baseline")
    ap.add_argument("--model", default="deepseek/deepseek-flash")
    ap.add_argument("--provider", default="deepseek",
                    choices=("deepseek", "codex", "cli"),
                    help="cli runs EVERY arm through a pinned CLI turn so the "
                         "model is the only variable")
    ap.add_argument("--cli-provider", default="deepseek",
                    help="Hermes provider name for --provider cli")
    ap.add_argument("--out", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--rows", default="", help="dump per-run rows so per-case "
                    "agreement and per-repeat spread can be computed; pooled rates "
                    "alone cannot distinguish a real gap from binomial noise")
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per case. The provider is NOT deterministic even at "
                         "temperature=0: identical cases under different configs drew "
                         "different misses, so one run per config cannot rank them.")
    args = ap.parse_args()

    if args.list:
        for name, cfg in CONFIGS.items():
            print(f"  {name:10s} {cfg['desc']}")
        return

    global COMPLETER
    if args.provider == "codex":
        COMPLETER = lambda msgs, mdl: complete_codex_direct(msgs, mdl)
    if args.provider == "cli":
        COMPLETER = lambda msgs, mdl: complete_cli(msgs, args.model,
                                                   args.cli_provider)

    all_cases = case_mod.all_cases()
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    summary = {}
    all_rows: dict = {}

    for name in names:
        cfg = CONFIGS[name]
        print(f"\n=== {name}: {cfg['desc']} ===")
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            for case in all_cases:
                ws = pathlib.Path(tmp) / case["id"]
                ws.mkdir(parents=True, exist_ok=True)
                for tool, payload in case["actions"]:
                    if tool == "write_file":
                        (ws / f"{case['id']}.py").write_text(payload)
                # artifacts exist on DISK only -- they are what a reviewer with
                # tools can go and read. A case whose artifact also appeared in
                # the transcript would let a prose-only reviewer check the claim,
                # which is exactly what the tool dimension is meant to measure.
                # NOT `name`: that is the enclosing CONFIG loop's variable, and
                # reusing it here silently rewrote every config's summary key to
                # this artifact's filename -- so four configs collapsed into one
                # entry, each overwriting the last.
                for art_name, content in case.get("artifacts") or []:
                    target = ws / art_name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content)
                for _rep in range(args.repeat):
                    try:
                        if cfg.get("hybrid"):
                            def _baseline(d, ws_):
                                return run_case(case, {"tools": True,
                                                       "max_tool_calls": 3,
                                                       "max_delta_chars": 20000},
                                                args.model, ws_)
                            hr = cove_mod.run_hybrid(complete, parse_tool_call,
                                                     dispatch,
                                                     case_mod.render_transcript(case),
                                                     ws, args.model, _baseline)
                            res = {"severity": hr["severity"],
                                   "finding": hr["finding"],
                                   "input_tokens": hr["input_tokens"],
                                   "output_tokens": hr["output_tokens"],
                                   "calls": hr["calls"],
                                   "tool_calls": sum(
                                       1 for v in (hr.get("verdicts") or [])
                                       if v.get("evidence"))}
                        elif cfg.get("cove"):
                            cr = cove_mod.run(complete, parse_tool_call, dispatch,
                                              case_mod.render_transcript(case),
                                              ws, args.model)
                            res = {"severity": cr["severity"], "finding": cr["finding"],
                                   "input_tokens": cr["input_tokens"],
                                   "output_tokens": cr["output_tokens"],
                                   "calls": 1 + len(cr.get("claims") or []),
                                   "tool_calls": sum(
                                       1 for v in (cr.get("verdicts") or [])
                                       if v.get("evidence"))}
                        else:
                            res = run_case(case, cfg, args.model, ws)
                    except Exception as exc:
                        print(f"  ERROR {case['id']}: {type(exc).__name__}: {exc}")
                        continue
                    rows.append({"case": case, "result": res})
                mark = "CAUGHT" if (case["defect"]
                                    and res["severity"] != "none"
                                    and is_relevant(case, res)) else (
                       "false-alarm" if (not case["defect"]
                                         and is_relevant(case, res)) else (
                       "MISSED" if case["defect"] else "ok"))
                print(f"  {case['id']:22s} {res['severity']:8s} {mark:12s} "
                      f"{res['input_tokens']:>6d}in {res['output_tokens']:>5d}out "
                      f"({res['calls']} call)")
        s = score(rows)
        s["per_repeat_recall"] = []
        reps = args.repeat if args.repeat > 1 else 1
        for r in range(reps):
            chunk = rows[r * len(all_cases):(r + 1) * len(all_cases)]
            if chunk:
                cs = score(chunk)
                s["per_repeat_recall"].append(round(cs["recall"], 4))
        s["per_case"] = {}
        for row in rows:
            cid = row["case"]["id"]
            hit = (row["result"]["severity"] != "none"
                   and is_relevant(row["case"], row["result"]))
            s["per_case"].setdefault(cid, []).append(bool(hit))
        if args.rows:
            all_rows.setdefault(name, []).extend([
                {"config": name, "case": r["case"]["id"], "defect": r["case"]["defect"],
                 "severity": r["result"]["severity"],
                 "finding": r["result"]["finding"][:300],
                 "calls": r["result"]["calls"],
                 "tool_calls": r["result"].get("tool_calls", 0),
                 "input_tokens": r["result"]["input_tokens"],
                 "output_tokens": r["result"]["output_tokens"]}
                for r in rows])
        summary[name] = s
        print(f"  -> recall {s['recall']:.0%} ({s['caught']}/{s['defective_n']})  "
              f"precision {s['precision']:.0%}  false alarms {s['false_alarms']}/{s['clean_n']}  "
              f"tokens {s['input_tokens'] + s['output_tokens']}")
        if s["missed_ids"]:
            print(f"     missed: {', '.join(s['missed_ids'])}")
        if s["false_alarm_ids"]:
            print(f"     false alarms: {', '.join(s['false_alarm_ids'])}")
        time.sleep(0.5)

    print("\n=== comparison ===")
    print(f"  {'config':10s} {'recall':>8s} {'prec':>6s} {'FP':>5s} "
          f"{'tokens':>9s} {'tok/caught':>11s}")
    for name, s in summary.items():
        print(f"  {name:10s} {s['recall']:>7.0%} {s['precision']:>6.0%} "
              f"{s['false_alarms']:>2d}/{s['clean_n']:<2d} "
              f"{s['input_tokens'] + s['output_tokens']:>9d} "
              f"{s['tokens_per_caught']:>11.0f}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out}")
    if args.rows:
        pathlib.Path(args.rows).write_text(json.dumps(all_rows, indent=2))
        print(f"wrote {args.rows}")


if __name__ == "__main__":
    main()
