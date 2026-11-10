"""Extended tests for the Tier-1 deterministic guards.

Covers the behaviours added on top of the original guards: validators
that suppress false positives, all-occurrence redaction with overlap
merging, streaming scope selection, whole-word blocklists, and a real
partial-schema mode.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from aisafepy.stream.deterministic import (
    AhoCorasickGuard,
    RegexGuard,
    SchemaGuard,
    looks_like_key_material,
    luhn_check,
    shannon_entropy,
    valid_ipv4,
)
from aisafepy.stream.pipeline import Context

# ---- validators --------------------------------------------------------------


@pytest.mark.parametrize(
    "number, ok",
    [
        ("4111 1111 1111 1111", True),  # Visa test number
        ("5500-0000-0000-0004", True),  # Mastercard test number
        ("378282246310005", True),  # Amex test number
        ("4111 1111 1111 1112", False),  # checksum off by one
        ("1234567890123", False),
        ("12345", False),
    ],
)
def test_luhn_check(number, ok):
    assert luhn_check(number) is ok


@pytest.mark.parametrize(
    "ip, ok",
    [
        ("192.168.0.1", True),
        ("8.8.8.8", True),
        ("256.1.1.1", False),
        ("1.2.3", False),
        ("01.2.3.4", False),
        ("999.999.999.999", False),
    ],
)
def test_valid_ipv4(ip, ok):
    assert valid_ipv4(ip) is ok


def test_shannon_entropy_bounds():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaaaaaa") == 0.0
    assert shannon_entropy("abcdefgh") == pytest.approx(3.0)


@pytest.mark.parametrize(
    "token, ok",
    [
        ("internationalizationstandardsboard", False),  # long word, no digits
        ("00000000000000000000000000000000", False),  # digits only, zero entropy
        ("d41d8cd98f00b204e9800998ecf8427e", False),  # md5 digest
        ("sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc8Xy2", True),
        ("AbCd3fGh1jKlMn0pQrStUvWxYz123456", True),
    ],
)
def test_looks_like_key_material(token, ok):
    assert looks_like_key_material(token) is ok


# ---- RegexGuard: false-positive discipline -------------------------------------------


@pytest.mark.asyncio
async def test_pii_guard_ignores_luhn_invalid_digit_runs():
    guard = RegexGuard.compile_pii(only=["credit_card"])
    d = await guard(Context(buffer="order 1234 5678 9012 3456 shipped"))
    assert d.action.value == "allow"


@pytest.mark.asyncio
async def test_pii_guard_blocks_valid_card_number():
    guard = RegexGuard.compile_pii(only=["credit_card"])
    d = await guard(Context(buffer="card 4111 1111 1111 1111 exp 12/29"))
    assert d.is_blocked
    assert d.evidence["labels"] == ["credit_card"]


@pytest.mark.asyncio
async def test_pii_guard_ignores_version_strings_that_look_like_ips():
    guard = RegexGuard.compile_pii(only=["ipv4"])
    d = await guard(Context(buffer="upgraded from 10.2.300.1 to 1.2.3.4"))
    assert d.is_blocked
    assert [m[1] for m in d.evidence["matches"]] == ["1.2.3.4"]


@pytest.mark.asyncio
async def test_pii_guard_ignores_sha256_digest_but_flags_key():
    guard = RegexGuard.compile_pii(only=["api_key_generic"])
    digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    d = await guard(Context(buffer=f"checksum {digest}"))
    assert d.action.value == "allow"
    d2 = await guard(Context(buffer="token AbCd3fGh1jKlMn0pQrStUvWxYz123456 ok"))
    assert d2.is_blocked


def test_compile_pii_only_and_exclude_and_unknown():
    g = RegexGuard.compile_pii(only=["email", "ssn"])
    assert set(g.patterns) == {"email", "ssn"}
    g2 = RegexGuard.compile_pii(exclude=["api_key_generic", "phone_us"])
    assert "api_key_generic" not in g2.patterns
    assert "email" in g2.patterns
    with pytest.raises(KeyError):
        RegexGuard.compile_pii(only=["emial"])


# ---- RegexGuard: redaction -----------------------------------------------------------


@pytest.mark.asyncio
async def test_redaction_masks_every_occurrence():
    guard = RegexGuard.compile_pii(redact=True, only=["email"])
    d = await guard(Context(buffer="cc alice@a.com and bob@b.org and carol@c.net"))
    assert d.action.value == "transform"
    assert d.transformed_content == "cc [REDACTED] and [REDACTED] and [REDACTED]"
    assert len(d.evidence["spans"]) == 3


@pytest.mark.asyncio
async def test_redaction_merges_overlapping_spans():
    # The SSN pattern and a custom "digits" pattern overlap on the same text.
    guard = RegexGuard.from_patterns(
        "overlap",
        {"ssn": r"\d{3}-\d{2}-\d{4}", "digits": r"\d{2}-\d{4}"},
        action_on_match="transform",
    )
    d = await guard(Context(buffer="ssn 123-45-6789."))
    assert d.transformed_content == "ssn [REDACTED]."


@pytest.mark.asyncio
async def test_custom_redaction_token_and_spans_reference_scanned_text():
    guard = RegexGuard.from_patterns(
        "mask", {"email": r"[\w.]+@[\w.]+"}, action_on_match="transform", redact_with="<x>"
    )
    text = "mail a@b.c now"
    d = await guard(Context(buffer=text))
    assert d.transformed_content == "mail <x> now"
    span = d.evidence["spans"][0]
    assert text[span["start"] : span["end"]] == "a@b.c"


def test_redact_text_helper_without_hits_returns_input():
    guard = RegexGuard.compile_pii(redact=True)
    assert guard.redact_text("nothing here") == "nothing here"


# ---- RegexGuard: actions, scope, caps --------------------------------------------------


@pytest.mark.asyncio
async def test_escalate_action():
    guard = RegexGuard.from_patterns("esc", {"w": r"wire transfer"}, action_on_match="escalate")
    d = await guard(Context(buffer="please wire transfer now"))
    assert d.action.value == "escalate"
    assert d.tier == 1


def test_invalid_action_rejected():
    with pytest.raises(ValueError):
        RegexGuard.from_patterns("bad", {"a": "a"}, action_on_match="explode")


@pytest.mark.asyncio
async def test_scope_buffer_catches_secret_split_across_chunks():
    auto = RegexGuard.compile_pii(only=["email"])
    buffered = RegexGuard.compile_pii(only=["email"])
    buffered.scope = "buffer"
    ctx = Context(chunk="example.com", buffer="write to alice@example.com")
    assert (await auto(ctx)).action.value == "allow"
    assert (await buffered(ctx)).is_blocked


@pytest.mark.asyncio
async def test_scope_chunk_ignores_buffer():
    guard = RegexGuard.compile_pii(only=["email"])
    guard.scope = "chunk"
    d = await guard(Context(chunk="", buffer="alice@example.com"))
    assert d.action.value == "allow"


@pytest.mark.asyncio
async def test_truncation_is_reported_in_evidence():
    guard = RegexGuard.compile_pii(only=["email"])
    guard.max_input_chars = 50
    d = await guard(Context(buffer="x" * 500))
    assert d.action.value == "allow"
    assert d.evidence == {"truncated": True}


@pytest.mark.asyncio
async def test_match_cap_per_pattern():
    guard = RegexGuard.from_patterns("cap", {"a": r"a"})
    guard.max_matches_per_pattern = 5
    d = await guard(Context(buffer="a" * 1000))
    assert len(d.evidence["matches"]) == 5


def test_regex_to_spec_round_trips_patterns():
    guard = RegexGuard.from_patterns("spec", {"k": r"sk-\w+"}, action_on_match="transform")
    spec = guard.to_spec()
    assert spec["kind"] == "regex"
    assert spec["patterns"] == {"k": r"sk-\w+"}
    assert spec["action_on_match"] == "transform"


# ---- AhoCorasickGuard -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocklist_reports_every_offset():
    guard = AhoCorasickGuard(name="bl", terms=["bad"])
    d = await guard(Context(buffer="bad things, bad times"))
    assert d.evidence["matches"] == ["bad", "bad"]
    assert d.evidence["offsets"] == [0, 12]


@pytest.mark.asyncio
async def test_blocklist_whole_words_suppresses_substring_hits():
    loose = AhoCorasickGuard(name="loose", terms=["ass"])
    strict = AhoCorasickGuard(name="strict", terms=["ass"], whole_words=True)
    assert (await loose(Context(buffer="the class began"))).is_blocked
    assert (await strict(Context(buffer="the class began"))).action.value == "allow"
    assert (await strict(Context(buffer="what an ass."))).is_blocked


@pytest.mark.asyncio
async def test_blocklist_case_sensitivity_toggle():
    ci = AhoCorasickGuard(name="ci", terms=["Secret"])
    cs = AhoCorasickGuard(name="cs", terms=["Secret"], case_sensitive=True)
    assert (await ci(Context(buffer="the secret"))).is_blocked
    assert (await cs(Context(buffer="the secret"))).action.value == "allow"


def test_blocklist_drops_empty_terms_and_serializes():
    guard = AhoCorasickGuard(name="bl", terms=["", "x"], whole_words=True)
    assert guard.terms == ["x"]
    assert guard.to_spec()["whole_words"] is True


# ---- SchemaGuard -------------------------------------------------------------------------


class TransferArgs(BaseModel):
    to_account: str
    amount: float
    memo: str | None = None


@pytest.mark.asyncio
async def test_schema_partial_accepts_subset_of_fields():
    guard = SchemaGuard(name="schema", model=TransferArgs)
    d = await guard(Context(buffer='{"to_account": "ACC-1"}'))
    assert d.action.value == "allow"


@pytest.mark.asyncio
async def test_schema_partial_rejects_wrong_type():
    guard = SchemaGuard(name="schema", model=TransferArgs)
    d = await guard(Context(buffer='{"amount": "lots"}'))
    assert d.is_blocked
    assert d.evidence["errors"][0].startswith("amount:")


@pytest.mark.asyncio
async def test_schema_partial_rejects_unknown_fields_by_default():
    guard = SchemaGuard(name="schema", model=TransferArgs)
    d = await guard(Context(buffer='{"to_account": "ACC-1", "override_limits": true}'))
    assert d.is_blocked
    assert "unknown field" in d.evidence["errors"][0]
    lenient = SchemaGuard(name="schema", model=TransferArgs, reject_unknown=False)
    d2 = await lenient(Context(buffer='{"to_account": "ACC-1", "override_limits": true}'))
    assert d2.action.value == "allow"


@pytest.mark.asyncio
async def test_schema_full_mode_requires_all_fields():
    guard = SchemaGuard(name="schema", model=TransferArgs, partial=False)
    d = await guard(Context(buffer='{"to_account": "ACC-1"}'))
    assert d.is_blocked
    assert any("amount" in e for e in d.evidence["errors"])
    ok = await guard(Context(buffer='{"to_account": "ACC-1", "amount": 10}'))
    assert ok.action.value == "allow"


@pytest.mark.asyncio
async def test_schema_incomplete_json_is_deferred_not_blocked():
    guard = SchemaGuard(name="schema", model=TransferArgs)
    d = await guard(Context(buffer='{"to_account": "ACC'))
    assert d.action.value == "allow"
    assert d.evidence["incomplete"] is True


@pytest.mark.asyncio
async def test_schema_non_object_json_is_blocked():
    guard = SchemaGuard(name="schema", model=TransferArgs)
    d = await guard(Context(buffer="[1, 2, 3]"))
    assert d.is_blocked


@pytest.mark.asyncio
async def test_schema_reads_decoded_arguments_from_metadata():
    guard = SchemaGuard(name="schema", model=TransferArgs)
    ctx = Context(buffer="not json", metadata={"tool_arguments": {"amount": -1}})
    d = await guard(ctx)
    assert d.action.value == "allow"  # -1 is a valid float; type-only in partial mode
    ctx2 = Context(buffer="not json", metadata={"tool_arguments": {"amount": "x"}})
    assert (await guard(ctx2)).is_blocked
