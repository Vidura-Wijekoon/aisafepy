from __future__ import annotations

import pytest

from aisafepy.core.decisions import (
    Action,
    GuardDecision,
    IFCViolation,
    Tripwire,
)


def test_allow_constructor_defaults():
    d = GuardDecision.allow(guard_name="t1")
    assert d.action == Action.ALLOW
    assert d.severity == "info"
    assert d.guard_name == "t1"
    assert d.confidence == 1.0
    assert d.is_blocked is False
    assert d.is_terminal is False


def test_block_constructor_carries_fallback():
    d = GuardDecision.block(
        tier=2,
        confidence=0.97,
        rationale="jailbreak phrase",
        guard_name="prompt-guard-2",
    )
    assert d.is_blocked
    assert d.is_terminal
    assert d.fallback is not None
    assert "can't help" in d.fallback


def test_transform_carries_replacement():
    d = GuardDecision.transform(
        tier=1,
        confidence=0.99,
        rationale="redacted email",
        transformed_content="[REDACTED]@example.com",
    )
    assert d.action == Action.TRANSFORM
    assert d.transformed_content is not None


def test_decision_is_frozen():
    d = GuardDecision.allow()
    with pytest.raises(Exception):
        d.action = Action.BLOCK  # type: ignore[misc]


def test_ifc_violation_renders_to_decision():
    v = IFCViolation(
        reason="forbidden capability",
        tool="send_email",
        provenance=frozenset({"gmail.read"}),
        integrity="UNTRUSTED",
        capabilities=frozenset({"read.user", "read.secrets"}),
        required_capabilities=frozenset({"write.external"}),
    )
    d = v.to_guard_decision()
    assert d.action == Action.BLOCK
    assert d.severity == "critical"
    assert d.evidence["kind"] == "ifc_violation"
    assert "read.secrets" in d.evidence["capabilities"]


def test_tripwire_is_a_guard_decision():
    t = Tripwire.block(tier=2, confidence=0.9, rationale="content unsafe")
    assert isinstance(t, GuardDecision)
    assert t.action == Action.BLOCK
