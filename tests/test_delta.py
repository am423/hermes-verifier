"""Delta extraction / rendering / redaction unit tests."""

from __future__ import annotations

import json

from verifier_pkg import delta


def test_strip_findings():
    text = (
        "user says go\n"
        '<finding verifier="default" severity="caution" guidance="x">\n'
        "watch the queue\n</finding>\n"
        "tail"
    )
    stripped = delta.strip_findings(text)
    assert "watch the queue" not in stripped
    assert "user says go" in stripped
    assert "tail" in stripped


def test_redact_secrets():
    text = (
        "key sk-abcdefghijklmnop12345678 leaked\n"
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abcdef.ghijklmnop\n"
        "aws AKIAABCDEFGHIJKLMNOP\n"
        "api_key=supersecretvalue99\n"
        "-----BEGIN RSA PRIVATE KEY-----\nAAAA\n-----END RSA PRIVATE KEY-----"
    )
    redacted = delta.redact(text)
    assert "sk-" not in redacted
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted
    assert "supersecretvalue99" not in redacted
    assert "RSA PRIVATE KEY" not in redacted
    assert "[REDACTED" in redacted


def test_redact_long_tokens():
    assert delta.redact("x" * 300) == "[long-token]"


def test_content_to_text_handles_shapes():
    assert delta._content_to_text(None) == ""
    assert delta._content_to_text("plain") == "plain"
    assert delta._content_to_text([{"type": "text", "text": "hi"},
                                   {"type": "image_url", "image_url": {}}]) == "hi [image]"
    assert delta._content_to_text({"a": 1}) == '{"a": 1}'


def test_extract_delta_first_review_replays_tail():
    history = [{"role": "user", "content": "hello"},
               {"role": "assistant", "content": "hi there"}]
    cursor, text = delta.extract_delta(
        cursor_len=None, conversation_history=history,
        assistant_response="done", max_chars=20000,
    )
    assert cursor == 2
    assert "[user] hello" in text
    assert "[assistant] hi there" in text
    assert "[assistant final] done" in text


def test_extract_delta_only_new_rows():
    history = [{"role": "user", "content": "one"},
               {"role": "assistant", "content": "two"}]
    cursor, text = delta.extract_delta(
        cursor_len=2, conversation_history=history, assistant_response="", max_chars=20000,
    )
    assert cursor == 2
    assert text is None

    history.append({"role": "user", "content": "three"})
    cursor, text = delta.extract_delta(
        cursor_len=2, conversation_history=history, assistant_response="four", max_chars=20000,
    )
    assert cursor == 3
    assert "[user] three" in text
    assert "[user] one" not in text
    assert "[assistant final] four" in text


def test_extract_delta_compaction_rewinds():
    long_history = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
    cursor, _ = delta.extract_delta(
        cursor_len=None, conversation_history=long_history,
        assistant_response="", max_chars=20000,
    )
    assert cursor == 30

    compacted = long_history[:8]  # transcript rewritten
    new_cursor, text = delta.extract_delta(
        cursor_len=cursor, conversation_history=compacted,
        assistant_response="after compaction", max_chars=20000,
    )
    assert new_cursor == 8
    assert "[user] msg 7" in text
    assert "[assistant final] after compaction" in text


def test_extract_delta_truncation_marker():
    history = [{"role": "user", "content": "word " * 500}]
    _, text = delta.extract_delta(
        cursor_len=None, conversation_history=history,
        assistant_response="", max_chars=100,
    )
    assert text is not None
    assert "… [delta truncated]" in text


def test_user_anchor_stable():
    assert delta.user_anchor("same message") == delta.user_anchor("same message")
    assert delta.user_anchor("same message") != delta.user_anchor("other")


def test_tool_call_args_are_truncated_tighter():
    """Tool-call args echo inflates the delta; the advisor needs intent, not payload.

    A write_file call carrying a large file body, or a terminal call carrying a
    long command, would otherwise dominate the delta and the token bill.
    """
    from verifier_pkg.delta import render_message
    big = json.dumps({"path": "/tmp/x.py", "content": "code " * 500})
    msg = {"role": "assistant", "content": "",
           "tool_calls": [{"function": {"name": "write_file", "arguments": big}}]}
    rendered = render_message(msg)
    assert "write_file" in rendered, "the tool name must survive"
    assert len(rendered) < 400, (
        f"tool args must be tightly truncated, got {len(rendered)} chars"
    )
    # The payload is cut off well before the full body would fit.
    assert rendered.count("code") < 40, (
        f"payload still dominates the row: {rendered.count('code')} repeats"
    )


def test_tool_result_rows_are_capped_in_the_delta():
    """Tool results dominate the delta; the advisor needs the gist, not the body.

    A read_file of a large file injects thousands of tokens into every review,
    and max_delta_chars does not catch it because it truncates the JOINED
    result, not each row.
    """
    hist = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "read_file",
                                      "arguments": json.dumps({"path": "runtime.py"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "line of code\n" * 400},
    ]
    _cursor, rendered = delta.extract_delta(
        cursor_len=None, conversation_history=hist,
        assistant_response="Reviewed the file.", max_chars=20000,
    )
    assert len(rendered) < 4000, (
        f"tool results must be capped per row, got {len(rendered)} chars"
    )


def test_delta_truncation_marks_and_bounds():
    """The cap must bound output and say so, so a cut review is visible.

    The marker sits MID-text on purpose: the newest content (including the
    agent's final answer) is retained, so a cut delta ends in real content.
    This assertion previously required the marker at the very end, which only
    held because the old truncation discarded the tail outright.
    """
    hist = [{"role": "tool", "tool_call_id": "c1",
             "content": "\n".join(f"line {i}" for i in range(3000))} for _ in range(12)]
    _cursor, rendered = delta.extract_delta(
        cursor_len=None, conversation_history=hist,
        assistant_response="done", max_chars=2000,
    )
    assert len(rendered) <= 2000, "the cap must bound the rendered delta"
    assert "[delta truncated]" in rendered, "a cut delta must be marked"
    assert not rendered.endswith("[delta truncated]"), (
        "the newest content must survive, so the marker cannot be the last thing"
    )


# ── Truncation must not delete the newest content ────────────────────────────

# NOTE: use realistic multi-line text, never a single long repeated token.
# redact() collapses a long token to "[long-token]", which silently shrinks the
# content below the cap and means the truncation under test never runs.


def _realistic_body(lines: int = 80) -> str:
    return "\n".join(
        f"2026-10-10 14:00:{i % 60:02d} INFO worker completed task {i}"
        for i in range(lines)
    )


def test_truncation_keeps_the_final_answer():
    """The delta is built oldest->newest with [assistant final] appended LAST.

    Cutting from the end therefore removes the primary agent's final answer --
    the single most important thing a reviewer needs, because it is what the
    agent is actually claiming. The reviewer would otherwise judge the start of
    a turn whose conclusion it cannot see.
    """
    from verifier_pkg import delta

    history = [{"role": "user", "content": "do the thing"},
               {"role": "assistant", "content": _realistic_body()}]
    _, text = delta.extract_delta(cursor_len=None, conversation_history=history,
                                  assistant_response="FINAL ANSWER MARKER",
                                  max_chars=2000)
    assert text is not None
    assert "truncated" in text, "precondition: this case must actually truncate"
    assert "FINAL ANSWER MARKER" in text, (
        "truncation dropped the final answer; the reviewer cannot review what "
        "the agent actually concluded"
    )


def test_truncation_keeps_both_ends():
    """Head carries the request, tail carries the outcome. Drop the middle."""
    from verifier_pkg import delta

    history = [{"role": "user", "content": "HEAD MARKER"},
               {"role": "assistant", "content": _realistic_body(120)}]
    _, text = delta.extract_delta(cursor_len=None, conversation_history=history,
                                  assistant_response="TAIL MARKER", max_chars=1500)
    assert "truncated" in text, "precondition: this case must actually truncate"
    assert "HEAD MARKER" in text
    assert "TAIL MARKER" in text


def test_truncation_respects_the_cap():
    from verifier_pkg import delta

    history = [{"role": "assistant", "content": _realistic_body(200)}]
    _, text = delta.extract_delta(cursor_len=None, conversation_history=history,
                                  assistant_response="done", max_chars=2000)
    assert len(text) <= 2000 + 40


def test_a_short_delta_is_untouched():
    from verifier_pkg import delta

    history = [{"role": "user", "content": "hi"}]
    _, text = delta.extract_delta(cursor_len=None, conversation_history=history,
                                  assistant_response="done", max_chars=20000)
    assert "truncated" not in text
