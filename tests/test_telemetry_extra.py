"""Value-based redaction and runtime configuration of the redactor."""

from __future__ import annotations

import json
import logging

import pytest

from aisafepy.core.decisions import GuardDecision
from aisafepy.core.telemetry import (
    REDACTED_PLACEHOLDER,
    attach_decision,
    configure_redaction,
    find_secret_values,
    redact,
    redact_text,
    redaction_config,
    structured_log,
)


@pytest.fixture(autouse=True)
def _reset_redaction():
    configure_redaction(reset=True)
    yield
    configure_redaction(reset=True)


@pytest.mark.parametrize(
    "text, label",
    [
        ("key sk-" + "ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 ok", "anthropic_key"),
        ("key sk-" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123 ok", "openai_key"),
        ("key sk-" + "proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123 ok", "openai_key"),
        ("ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789", "github_token"),
        ("xoxb-" + "123456789012-abcdefghijkl", "slack_token"),
        ("AKIA" + "IOSFODNN7EXAMPLE", "aws_access_key"),
        ("AIza" + "SyA1234567890abcdefghijklmnopqrstuv", "google_api_key"),
        ("hf_" + "abcdefghijklmnopqrstuvwxyz0123456789", "huggingface_token"),
        ("sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc", "stripe_key"),
        ("eyJhbGciOiJIUzI1NiJ9" + ".eyJzdWIiOiIxMjM0NTY3ODkwIn0" + ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", "jwt"),
        ("Authorization: Bearer " + "abcdefghijklmnopqrstuvwxyz", "bearer_token"),
        ("-----BEGIN RSA PRIVATE" + " KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----", "pem_private_key"),
    ],
)
def test_find_secret_values_detects_each_family(text, label):
    labels = [lab for lab, _, _ in find_secret_values(text)]
    assert label in labels


def test_redact_text_uses_typed_placeholder_and_keeps_context():
    out = redact_text("use key sk-" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123 for prod")
    assert out == "use key [REDACTED:openai_key] for prod"


def test_anthropic_key_wins_over_generic_openai_prefix():
    out = redact_text("sk-" + "ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789")
    assert out == "[REDACTED:anthropic_key]"


def test_basic_auth_url_masks_only_the_password():
    out = redact_text("fetch https://alice:" + "hunter2secret@db.example.com/x")
    assert out == "fetch https://alice:" + "[REDACTED:basic_auth_url]@db.example.com/x"


@pytest.mark.parametrize(
    "benign",
    [
        "The task-123 ticket mentions skills and tokens of appreciation.",
        "Bearer of bad news",
        "sk-short",
        "ghp_tooshort",
        "a normal sentence with an email alice@example.com",
        "AKIA is an abbreviation",
    ],
)
def test_benign_text_is_untouched(benign):
    assert redact_text(benign) == benign


def test_redact_scans_string_values_under_innocent_keys():
    out = redact({"snippet": "token is xoxb-" + "123456789012-abcdefghijkl here"})
    assert out["snippet"] == "token is [REDACTED:slack_token] here"


def test_redact_scans_strings_inside_lists():
    out = redact(["ok", "AKIA" + "IOSFODNN7EXAMPLE"])
    assert out == ["ok", "[REDACTED:aws_access_key]"]


def test_key_based_redaction_still_applies():
    out = redact({"api_key": "whatever", "note": "fine"})
    assert out["api_key"] == REDACTED_PLACEHOLDER
    assert out["note"] == "fine"


def test_truncation_happens_after_scrubbing():
    long = "x" * 5000 + " sk-" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123"
    out = redact({"prompt": long})
    assert len(out["prompt"]) <= 4096
    assert "sk-AbCd" not in out["prompt"]


def test_configure_extra_keys_and_substrings():
    configure_redaction(extra_keys=["TenantSecretRef"], extra_substrings=["pin"])
    out = redact({"tenantsecretref": "x", "card_pin": "1234", "spin": "ok?"})
    assert out["tenantsecretref"] == REDACTED_PLACEHOLDER
    assert out["card_pin"] == REDACTED_PLACEHOLDER
    # substring match is intentionally greedy; document the trade-off
    assert out["spin"] == REDACTED_PLACEHOLDER


def test_configure_extra_value_patterns():
    configure_redaction(extra_patterns={"acme_token": r"ACME-[0-9]{6}"})
    assert redact_text("ref ACME-123456 done") == "ref [REDACTED:acme_token] done"


def test_configure_disable_and_reset():
    configure_redaction(enabled=False)
    assert redact({"api_key": "leak"}) == {"api_key": "leak"}
    configure_redaction(reset=True)
    assert redact({"api_key": "leak"})["api_key"] == REDACTED_PLACEHOLDER
    assert redaction_config().enabled is True


def test_configure_limits_are_clamped():
    cfg = configure_redaction(max_string_chars=2, max_depth=0)
    assert cfg.max_string_chars == 16
    assert cfg.max_depth == 1


def test_structured_log_redacts_fields(caplog):
    with caplog.at_level(logging.INFO, logger="aisafepy"):
        structured_log("test.event", api_key="sk-" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123", note="ok")
    payload = json.loads(caplog.records[-1].getMessage())
    assert payload["event"] == "test.event"
    assert payload["api_key"] == REDACTED_PLACEHOLDER
    assert payload["note"] == "ok"


class _SpanRecorder:
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, k, v):
        self.attrs[k] = v


def test_attach_decision_redacts_rationale_and_evidence_values():
    span = _SpanRecorder()
    d = GuardDecision.block(
        tier=1,
        confidence=1.0,
        rationale="leaked AKIA" + "IOSFODNN7EXAMPLE in output",
        evidence={"excerpt": "Bearer " + "abcdefghijklmnopqrstuvwxyz", "headers": {"authorization": "x"}},
    )
    attach_decision(span, d)
    assert span.attrs["aisafepy.guard.rationale"] == "leaked [REDACTED:aws_access_key] in output"
    evidence = json.loads(span.attrs["aisafepy.guard.evidence"])
    assert evidence["excerpt"] == "[REDACTED:bearer_token]"
    assert evidence["headers"]["authorization"] == REDACTED_PLACEHOLDER
