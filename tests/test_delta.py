"""Delta extraction / rendering / redaction unit tests."""

from __future__ import annotations

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
