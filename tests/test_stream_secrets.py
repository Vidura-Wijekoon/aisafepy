from __future__ import annotations

import asyncio

import pytest

from aisafepy.stream.pipeline import Context
from aisafepy.stream.secrets import (
    SecretsGuard,
    card_brand,
    luhn_valid,
    mask_secret,
    shannon_entropy,
)


def run(guard, text: str):
    return asyncio.run(guard(Context(buffer=text)))


# ---- helpers ----------------------------------------------------------------


def test_shannon_entropy_bounds():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert abs(shannon_entropy("abcd") - 2.0) < 1e-9


@pytest.mark.parametrize(
    ("number", "valid"),
    [
        ("4111111111111111", True),
        ("4111111111111112", False),
        ("5555555555554444", True),
        ("378282246310005", True),
        ("6011111111111117", True),
        ("1234567812345678", False),
    ],
)
def test_luhn(number, valid):
    assert luhn_valid(number) is valid


@pytest.mark.parametrize(
    ("number", "brand"),
    [
        ("4111111111111111", "visa"),
        ("5555555555554444", "mastercard"),
        ("2221000000000009", "mastercard"),
        ("378282246310005", "amex"),
        ("6011111111111117", "discover"),
        ("3530111333300000", "jcb"),
        ("30569309025904", "diners"),
        ("9999999999999999", None),
    ],
)
def test_card_brand(number, brand):
    assert card_brand(number) == brand


def test_mask_secret_keeps_edges_only():
    assert mask_secret("sk-" + "abcdefghijklmnop") == "sk***op"
    assert mask_secret("short") == "***"


# ---- known patterns ---------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "sample"),
    [
        ("openai_api_key", "sk-" + "proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD"),
        ("anthropic_api_key", "sk-" + "ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"),
        ("github_token", "ghp_" + "A1b2C3d4" * 5),
        ("slack_token", "xoxb-" + "123456789012-ABCDEFGHIJKL"),
        ("slack_webhook", "https://hooks.slack.com/services/" + "T0000000/B0000000/XXXXXXXXXXXXXXXXXXXXXXXX"),
        ("aws_access_key_id", "AKIA" + "IOSFODNN7EXAMPLE"),
        ("google_api_key", "AIza" + "SyA1234567890abcdefghijklmnopqrstuv"),
        ("stripe_key", "sk_live_" + "a1B2c3D4" * 4),
        ("sendgrid_api_key", "SG." + "a" * 22 + "." + "b" * 43),
        ("jwt", "eyJhbGciOiJIUzI1NiJ9" + ".eyJzdWIiOiIxMjM0NTY3ODkwIn0" + ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"),
        ("private_key_block", "-----BEGIN RSA PRIVATE" + " KEY-----\nMIIE..."),
        ("huggingface_token", "hf_" + "abcDEF123" * 4),
        ("basic_auth_url", "postgres://admin:" + "s3cretPassw0rd@db.internal:5432/app"),
        ("credential_assignment", "password = 'Tr0ub4dor&3xyz'"),
        ("azure_storage_key", "AccountKey=" + "Q" * 86 + "=="),
    ],
)
def test_known_patterns_detected(kind, sample):
    matches = SecretsGuard(detect_entropy=False, detect_cards=False).scan(f"here: {sample} end")
    assert kind in {m.kind for m in matches}, [m.kind for m in matches]


def test_anthropic_key_is_not_double_reported_as_openai():
    kinds = [m.kind for m in SecretsGuard().scan("sk-" + "ant-api03-abcdefghijklmnopqrstuvwxyz0123456789")]
    assert kinds == ["anthropic_api_key"]


def test_basic_auth_url_redacts_only_the_password():
    guard = SecretsGuard(mode="redact", detect_entropy=False)
    text = "connect to postgres://admin:" + "hunter22hunter@db.internal/app"
    out = guard.redact(text)
    assert "hunter22hunter" not in out
    assert out.startswith("connect to postgres://admin:" + "[REDACTED:basic_auth_url]@db.internal")


@pytest.mark.parametrize(
    "benign",
    [
        "password must be at least 8 characters long",
        "password: <your-password-here>",
        "api_key = ${OPENAI_API_KEY}",
        "secret = changeme",
        "The skeleton key (sk) is in the drawer",
        "Order number 1234-5678-9012 shipped",
        "Call us at +1 (555) 123-4567 or 555-765-4321",
        "commit 3f8a9c2b1d4e3f8a9c2b1d4e3f8a9c2b1d4e3f8a fixes the bug",
        "request id 123e4567-e89b-12d3-a456-426614174000 failed",
        "Pneumonoultramicroscopicsilicovolcanoconiosis is a long word",
        "https://example.com/docs/getting-started/installation-guide",
        "tracking number 1Z999AA10123456784",
    ],
)
def test_benign_text_not_flagged(benign):
    assert SecretsGuard().scan(benign) == []


# ---- cards --------------------------------------------------------------------


def test_luhn_valid_card_with_separators_detected_and_masked():
    guard = SecretsGuard()
    matches = guard.scan("card 4111 1111 1111 1111 exp 12/27")
    assert len(matches) == 1
    m = matches[0]
    assert m.kind == "payment_card" and m.brand == "visa" and m.detector == "luhn"
    assert m.preview == "4111***1111"
    assert "4111 1111 1111 1111" not in str(m.to_dict())


def test_luhn_invalid_16_digits_not_flagged():
    assert SecretsGuard().scan("ref 4111111111111112") == []


def test_unknown_brand_requires_opt_in():
    # 16 digits, Luhn valid, prefix 9 is not a known brand.
    assert SecretsGuard().scan("9999999999999995") == []
    assert SecretsGuard(require_known_brand=False).scan("9999999999999995")[0].kind == "payment_card"


# ---- entropy ------------------------------------------------------------------


def test_high_entropy_base64_token_detected():
    token = "Zm9vYmFy" + "MTIzNDU2Nzg5MGFiY2RlZmdoaWprbG1ub3A"
    matches = SecretsGuard().scan(f"token={token}")
    kinds = {m.kind for m in matches}
    assert kinds & {"high_entropy_base64", "credential_assignment"}
    for m in matches:
        assert token not in m.preview


def test_high_entropy_hex_detected_but_git_sha_lengths_skipped():
    hex40 = "3f8a9c2b1d4e3f8a9c2b1d4e3f8a9c2b1d4e3f8a"
    hex32 = "3f8a9c2b1d4e7a6b5c4d3e2f1a0b9c8d"
    assert SecretsGuard().scan(hex40) == []
    assert SecretsGuard(allow_git_sha=False).scan(hex40)[0].kind == "high_entropy_hex"
    assert SecretsGuard().scan(hex32)[0].kind == "high_entropy_hex"


def test_skip_urls_ignores_tokens_inside_urls():
    url = "https://cdn.example.com/assets/Zm9vYmFy" + "MTIzNDU2Nzg5MGFiY2RlZmdoaWprbG1ub3A.js"
    assert SecretsGuard(skip_urls=True).scan(url) == []
    assert SecretsGuard(skip_urls=False).scan(url)


def test_entropy_only_hit_has_lower_confidence():
    d = run(SecretsGuard(), "3f8a9c2b1d4e7a6b5c4d3e2f1a0b9c8d")
    assert d.is_blocked and d.confidence == 0.8 and d.severity == "medium"


# ---- guard behaviour ------------------------------------------------------------


def test_block_decision_has_masked_evidence_and_severity():
    d = run(SecretsGuard(), "my key is sk-" + "proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD ok")
    assert d.is_blocked
    assert d.severity == "critical"
    assert d.evidence["kinds"] == ["openai_api_key"]
    assert "abcdefghijklmnop" not in str(d.evidence) and "abcdefghijklmnop" not in d.rationale
    assert d.evidence["matches"][0]["span"] == [10, 10 + len("sk-" + "proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD")]


def test_redact_mode_returns_transform():
    d = run(SecretsGuard(mode="redact"), "card 4111 1111 1111 1111 and AKIA" + "IOSFODNN7EXAMPLE")
    assert d.action.value == "transform"
    assert d.transformed_content == "card [REDACTED:payment_card] and [REDACTED:aws_access_key_id]"


def test_clean_text_allows():
    d = run(SecretsGuard(), "The quarterly report is attached. Regards, Ana.")
    assert d.action.value == "allow" and d.latency_ms >= 0.0


def test_chunk_target_default_scans_chunk_not_buffer():
    guard = SecretsGuard()
    ctx = Context(chunk="clean chunk", buffer="AKIA" + "IOSFODNN7EXAMPLE clean chunk")
    assert asyncio.run(guard(ctx)).action.value == "allow"
    ctx2 = Context(chunk="clean chunk", buffer="AKIA" + "IOSFODNN7EXAMPLE clean chunk")
    assert asyncio.run(SecretsGuard(scan_target="buffer")(ctx2)).is_blocked


def test_overlapping_matches_are_deduped():
    # The AWS key is also a high-entropy token; only the pattern match survives.
    matches = SecretsGuard().scan("AKIA" + "IOSFODNN7EXAMPLE")
    assert [m.kind for m in matches] == ["aws_access_key_id"]
