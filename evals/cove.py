"""Factored verification: CoVe applied to code review.

WHY THIS SHAPE
--------------
The measured failure: the reviewer made 4 tool calls across 216 reviews, and the
cases it missed were the ones where the agent's claim cannot be checked from the
transcript. Bolting an instruction onto the tool block did not fix it -- it cost
34 points of recall (sweep4), consistent with the literature's finding that
naive self-review prompts are unreliable.

Chain-of-Verification (Meta, arXiv 2309.11495) isolates the mechanism that does
work: plan verification questions, then answer them INDEPENDENTLY, "so the
answers are not biased by other responses". The factored variant is the
important one -- when the model can attend to its own draft it "tends to repeat
the hallucinations".

So the bias to avoid is structural, not motivational: a reviewer shown the
agent's claim AND asked to judge it will tend to agree with it. Dividing the
work means the verification step never sees the reasoning that produced the
claim -- only the claim and the artifact.

This also targets the bottleneck the evidence identifies. LLMs "cannot find
reasoning errors, but can correct them given the error location" (ACL 2024), so
finding -- not judging -- is what needs help. Extracting a claim turns one hard
finding problem into N small checkable ones, and each check has an artifact to
read.

Stages:
    1. EXTRACT   the turn's factual claims about what changed. No judgement.
    2. VERIFY    each claim ALONE, with tools, against the artifact. Factored:
                 this call never sees the transcript that produced the claim.
    3. SYNTHESIZE  one review envelope from the verification verdicts. A claim
                 that survives verification produces no finding.
"""

from __future__ import annotations

EXTRACT_SYSTEM = """\
You extract factual claims from a coding agent's turn. You do not judge them.

The transcript shows what an agent was asked to do, the commands it ran, and what
it says it accomplished.

List every claim the agent makes about WHAT CHANGED or WHAT NOW WORKS. Examples
of the shape wanted:
  - "retry() now uses exponential backoff"
  - "the API key was moved to an environment variable"
  - "the test suite passes"
  - "validation was added before the insert"

Rules:
- Claims only. Do NOT say whether a claim is true. You have not checked anything.
- Do NOT invent claims the agent did not make.
- Prefer claims that assert a change to code, config, or test state.
- If the turn makes no such claim, return an empty list.
- Between zero and four claims.

Respond with JSON only: {"claims": ["...", "..."]}"""

VERIFY_SYSTEM = """\
You verify ONE claim about a coding agent's turn. You have read-only tools.

You see the turn's transcript AND the single claim you must check. You are NOT
told whether the claim is true, and no other claim is shown to you.

Decide from the transcript AND the files:

- A claim about what the CODE NOW DOES is checked by READING the file.
- A claim about what the agent DID (a command it ran, a deploy it performed) is
  checked against the transcript, which shows the actual command.
- A claim contradicted by either is "unsupported".

Use the tools when a file is what would settle it:

  TOOL_CALL: read_file {"path": "somefile.py"}
  TOOL_CALL: search_files {"pattern": "some_text"}

Read whatever file the claim is about. Do not decide from the wording of the
claim -- it is the thing under test.

Verdicts:
  supported    - you read the artifact and it matches the claim
  unsupported  - you read the artifact and it CONTRADICTS the claim
  unverifiable - neither the transcript nor any readable file can settle it

Only "unsupported" is actionable. Do not return it because a claim is vague or
could be better worded; return it when the evidence disagrees with the claim.

Respond with JSON only:
{"verdict": "supported|unsupported|unverifiable", "evidence": "what you read and
what it showed", "why": "how the artifact contradicts the claim, if it does"}"""

SYNTH_SYSTEM = """\
You are the Verifier. Verification of one turn's claims has already been done;
you are given the verdicts.

Raise a finding ONLY if a claim was found UNSUPPORTED -- the artifact contradicts
what the agent said it did. That is a real defect: the user is about to believe
something false about their code.

If every claim was supported, or nothing could be verified, report severity
"none". Do not manufacture a finding from a vague claim, a style preference, or
the general quality of the work.

Severity:
  note    - the contradiction is minor or cosmetic
  caution - the user would be misled about meaningful behaviour
  blocker - the agent claims work is done that is not done

Respond with JSON only:
{"severity": "none|note|caution|blocker", "finding": "concrete and actionable,
naming what the agent claimed and what the artifact shows"}"""


def _parse_claims(text: str) -> list[str]:
    import json
    import re
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    claims = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(claims, list):
        return []
    out = []
    for c in claims:
        if isinstance(c, str) and c.strip():
            out.append(c.strip())
    return out[:4]


def _parse_verdict(text: str) -> dict:
    import json
    import re
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {"verdict": "unverifiable", "evidence": "", "why": ""}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"verdict": "unverifiable", "evidence": "", "why": ""}
    if not isinstance(data, dict):
        return {"verdict": "unverifiable", "evidence": "", "why": ""}
    v = str(data.get("verdict") or "unverifiable").strip().lower()
    if v not in ("supported", "unsupported", "unverifiable"):
        v = "unverifiable"
    return {"verdict": v,
            "evidence": str(data.get("evidence") or "")[:400],
            "why": str(data.get("why") or "")[:400]}


def run_hybrid(complete, parse_tool_call, dispatch, delta: str, workspace,
               model: str, baseline_fn, verbose: bool = False) -> dict:
    """Baseline review UNION factored claim verification.

    The two are complementary, and the measurement says so: the baseline already
    catches 13-16 of 18 seeded defects reliably, and its misses are exactly the
    claim-vs-artifact cases. Replacing it would trade a working detector for an
    unproven one -- the first attempt at this pipeline scored ZERO because
    stripping the transcript removed the evidence for every non-file claim.

    So: run the baseline, run claim verification, and keep the more severe
    verdict. This can only ADD findings, and the eval measures whether the
    additions are worth their cost.
    """
    base = baseline_fn(delta, workspace)
    cov = run(complete, parse_tool_call, dispatch, delta, workspace, model,
              verbose=verbose)
    rank = {"none": 0, "note": 1, "caution": 2, "blocker": 3}
    if rank[cov["severity"]] > rank[base["severity"]]:
        winner, src = cov, "claim-verification"
    else:
        winner, src = base, "baseline"
    return {"severity": winner["severity"],
            "finding": winner["finding"],
            "source": src,
            "baseline_severity": base["severity"],
            "cove_severity": cov["severity"],
            "claims": cov.get("claims") or [],
            "verdicts": cov.get("verdicts") or [],
            "input_tokens": base["input_tokens"] + cov["input_tokens"],
            "output_tokens": base["output_tokens"] + cov["output_tokens"],
            "calls": base["calls"] + 1 + len(cov.get("claims") or [])}


def run(complete, parse_tool_call, dispatch, delta: str, workspace,
        model: str, max_verify_calls: int = 2, verbose: bool = False) -> dict:
    """Run the three stages. Returns the envelope plus token accounting.

    `complete`, `parse_tool_call` and `dispatch` are injected so this module
    stays independent of how the harness talks to a provider.
    """
    billed_in = billed_out = 0

    # ---- stage 1: extract claims (no judgement) ---------------------------
    out = complete([{"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": delta}], model)
    billed_in += out["usage"].get("prompt_tokens", 0)
    billed_out += out["usage"].get("completion_tokens", 0)
    claims = _parse_claims(out["text"])
    if verbose:
        print(f"       claims: {claims}")

    if not claims:
        # Nothing asserted, nothing to contradict. No finding -- which is the
        # correct answer for a turn that only ran a command or read a file.
        return {"severity": "none", "finding": "", "claims": [], "verdicts": [],
                "input_tokens": billed_in, "output_tokens": billed_out}

    # ---- stage 2: verify each claim ALONE, with tools ---------------------
    verdicts = []
    for claim in claims:
        messages = [{"role": "system", "content": VERIFY_SYSTEM},
                    {"role": "user",
                     "content": f"TURN TRANSCRIPT:\n{delta}\n\n---\n"
                                f"THE ONE CLAIM TO VERIFY:\n{claim}"}]
        res = {"text": "", "usage": {}}
        for _ in range(max(1, max_verify_calls + 2)):
            res = complete(messages, model)
            billed_in += res["usage"].get("prompt_tokens", 0)
            billed_out += res["usage"].get("completion_tokens", 0)
            call = parse_tool_call(res["text"], ["read_file", "search_files"])
            if call is None:
                break
            evidence = dispatch(call["tool"], call["args"], workspace)
            messages.append({"role": "assistant", "content": res["text"]})
            messages.append({"role": "user",
                             "content": f"Tool result from {call['tool']}:\n{evidence}\n\n"
                                        "Now give your verdict as JSON."})
        verdicts.append({"claim": claim, **_parse_verdict(res["text"])})
        if verbose:
            print(f"       {verdicts[-1]['verdict']:12s} <- {claim[:70]}")

    # ---- stage 3: one envelope from the verdicts -------------------------
    unsupported = [v for v in verdicts if v["verdict"] == "unsupported"]
    if not unsupported:
        # Every claim held up. No finding, and no synthesis call needed.
        return {"severity": "none", "finding": "", "claims": claims,
                "verdicts": verdicts,
                "input_tokens": billed_in, "output_tokens": billed_out}

    report = "\n\n".join(
        f"CLAIM: {v['claim']}\nVERDICT: {v['verdict']}\n"
        f"EVIDENCE: {v['evidence']}\nWHY IT CONTRADICTS: {v['why']}"
        for v in verdicts)
    res = complete([{"role": "system", "content": SYNTH_SYSTEM},
                    {"role": "user", "content": report}], model)
    billed_in += res["usage"].get("prompt_tokens", 0)
    billed_out += res["usage"].get("completion_tokens", 0)
    envelope = None
    import json
    import re
    m = re.search(r"\{.*\}", res["text"] or "", re.S)
    if m:
        try:
            envelope = json.loads(m.group(0))
        except json.JSONDecodeError:
            envelope = None
    if not isinstance(envelope, dict):
        envelope = {"severity": "caution",
                    "finding": f"the agent claimed '{unsupported[0]['claim']}' "
                               f"but the artifact shows otherwise: "
                               f"{unsupported[0]['why']}"}
    sev = str(envelope.get("severity") or "none").strip().lower()
    if sev not in ("none", "note", "caution", "blocker"):
        sev = "none"
    return {"severity": sev,
            "finding": str(envelope.get("finding") or "")[:2000],
            "claims": claims, "verdicts": verdicts,
            "input_tokens": billed_in, "output_tokens": billed_out}
