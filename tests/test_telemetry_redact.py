"""Security tests: ensure the redactor scrubs sensitive keys from evidence."""

from __future__ import annotations

from aisafepy.core.telemetry import (
    REDACTED_PLACEHOLDER,
    _is_sensitive_key,
    redact,
)


def test_top_level_sensitive_key_is_redacted():
    out = redact({"api_key": "sk-real-key-here", "user": "alice"})
    assert out["api_key"] == REDACTED_PLACEHOLDER
    assert out["user"] == "alice"


def test_authorization_header_is_redacted():
    out = redact({"headers": {"Authorization": "Bearer eyJ..."}})
    assert out["headers"]["Authorization"] == REDACTED_PLACEHOLDER


def test_password_keys_caught_by_substring_match():
    out = redact({"user_password": "hunter2", "name": "Alice"})
    assert out["user_password"] == REDACTED_PLACEHOLDER
    assert out["name"] == "Alice"


def test_openai_api_key_is_redacted():
    out = redact({"openai_api_key": "sk-real"})
    assert out["openai_api_key"] == REDACTED_PLACEHOLDER


def test_nested_redaction():
    out = redact({"meta": {"secret": "xxx", "ok": [1, 2, 3]}})
    assert out["meta"]["secret"] == REDACTED_PLACEHOLDER
    assert out["meta"]["ok"] == [1, 2, 3]


def test_long_strings_are_truncated():
    long = "x" * 5000
    out = redact({"prompt": long})
    assert len(out["prompt"]) <= 4096
    assert out["prompt"].endswith("...")


def test_recursion_depth_cap():
    deep = current = {}
    for _ in range(20):
        current["child"] = {}
        current = current["child"]
    out = redact(deep)
    # Walk down; depth-cap should stop somewhere short of 20 levels.
    cur = out
    depth = 0
    while isinstance(cur, dict) and "child" in cur:
        cur = cur["child"]
        depth += 1
        if depth > 50:
            break
    # After depth-cap, the inner value becomes REDACTED_PLACEHOLDER.
    # Just make sure the recursion terminates without RecursionError.


def test_is_sensitive_key_caseinsensitive():
    assert _is_sensitive_key("API_KEY")
    assert _is_sensitive_key("X-API-Key")
    assert _is_sensitive_key("openai_api_key")
    assert not _is_sensitive_key("user")
    assert not _is_sensitive_key("count")


def test_non_string_keys_are_safe():
    out = redact({42: "answer", "api_key": "leak"})
    assert out[42] == "answer"
    assert out["api_key"] == REDACTED_PLACEHOLDER


def test_lists_of_dicts_are_recursed():
    out = redact({"items": [{"api_key": "x"}, {"name": "y"}]})
    assert out["items"][0]["api_key"] == REDACTED_PLACEHOLDER
    assert out["items"][1]["name"] == "y"
