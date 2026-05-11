from __future__ import annotations

import pytest

from aisafepy.stream.deterministic import (
    AhoCorasickGuard,
    BlocklistGuard,
    RegexGuard,
)
from aisafepy.stream.pipeline import Context


@pytest.mark.asyncio
async def test_regex_pii_blocks_email():
    guard = RegexGuard.compile_pii()
    ctx = Context(buffer="Reach me at alice@example.com any time.")
    d = await guard(ctx)
    assert d.is_blocked
    matches = [m[0] for m in d.evidence["matches"]]
    assert "email" in matches


@pytest.mark.asyncio
async def test_regex_pii_redacts_when_configured():
    guard = RegexGuard.compile_pii(redact=True)
    ctx = Context(buffer="Email me at alice@example.com.")
    d = await guard(ctx)
    assert d.action.value == "transform"
    assert "alice@example.com" not in d.transformed_content
    assert "[REDACTED]" in d.transformed_content


@pytest.mark.asyncio
async def test_blocklist_hits_terms():
    guard = AhoCorasickGuard(name="bl", terms=["api_key=", "BEGIN PRIVATE"])
    ctx = Context(buffer="here is your api_key=SECRET")
    d = await guard(ctx)
    assert d.is_blocked
    assert "api_key=" in d.evidence["matches"]


@pytest.mark.asyncio
async def test_blocklist_no_match_is_allow():
    guard = BlocklistGuard(name="bl", terms=["nope"])
    ctx = Context(buffer="all clear")
    d = await guard(ctx)
    assert d.action.value == "allow"


@pytest.mark.asyncio
async def test_regex_from_patterns_with_dict_strings():
    guard = RegexGuard.from_patterns(
        name="custom",
        patterns={"secret_key": r"sk-[A-Za-z0-9]{20,}"},
    )
    ctx = Context(buffer="my key is sk-AbCdEfGhIjKlMnOpQrSt")
    d = await guard(ctx)
    assert d.is_blocked


@pytest.mark.asyncio
async def test_input_length_cap_truncates():
    # Regex DoS protection: extremely long input is truncated before scanning.
    guard = RegexGuard.compile_pii()
    guard.max_input_chars = 100
    ctx = Context(buffer=" " * 100_000 + "alice@example.com")
    d = await guard(ctx)
    assert d.is_blocked  # the email is in the last 100 chars
