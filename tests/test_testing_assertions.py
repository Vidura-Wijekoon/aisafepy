"""Tests for the assertion helpers and the unittest mixin."""

from __future__ import annotations

import asyncio
import unittest

import pytest

from aisafepy.flow import Capability, IFCContext, Policy
from aisafepy.flow.taint import lift
from aisafepy.stream.deterministic import AhoCorasickGuard, RegexGuard
from aisafepy.stream.pipeline import GuardPipeline
from aisafepy.testing.assertions import (
    GuardTestCase,
    assert_allows,
    assert_blocks,
    assert_decision,
    assert_escalates,
    assert_flags,
    assert_policy_allows,
    assert_policy_denies,
    assert_transforms,
    describe_decision,
    run_guard,
)

BLOCKER = AhoCorasickGuard(name="bl", terms=["FORBIDDEN"])
PIPE = GuardPipeline(tier1=[BLOCKER])
REDACT = RegexGuard.compile_pii(redact=True, only=["email"])
ESCALATOR = RegexGuard.from_patterns("esc", {"w": r"wire"}, action_on_match="escalate")


def test_run_guard_accepts_guard_or_pipeline_and_sets_context():
    seen = {}

    class Spy:
        name = "spy"
        tier = 1

        async def __call__(self, ctx):
            seen.update(role=ctx.role, prompt=ctx.prompt, meta=ctx.metadata, chunk=ctx.chunk)
            from aisafepy.core.decisions import GuardDecision

            return GuardDecision.allow(tier=1)

    run_guard(Spy(), "hi", role="user", prompt="p", metadata={"k": 1})
    assert seen == {"role": "user", "prompt": "p", "meta": {"k": 1}, "chunk": "hi"}
    assert run_guard(PIPE, "x FORBIDDEN").is_blocked


@pytest.mark.asyncio
async def test_run_guard_works_inside_a_running_loop():
    assert run_guard(PIPE, "FORBIDDEN").is_blocked
    await asyncio.sleep(0)


def test_assert_blocks_and_allows_pass_and_fail_with_rationale():
    assert_blocks(PIPE, "this is FORBIDDEN")
    assert_allows(PIPE, "fine")
    with pytest.raises(AssertionError) as exc:
        assert_blocks(PIPE, "fine")
    msg = str(exc.value)
    assert "expected action=block" in msg
    assert "rationale: no blocklist hits" in msg
    assert "evidence:" in msg


def test_assert_escalates_flags_and_transforms():
    assert_escalates(ESCALATOR, "please wire it")
    assert_flags(ESCALATOR, "please wire it")
    assert_flags(PIPE, "FORBIDDEN")
    with pytest.raises(AssertionError, match="terminal decision"):
        assert_flags(PIPE, "ok")
    d = assert_transforms(REDACT, "mail bob@example.com", expected_substring="[REDACTED]", forbidden_substring="bob@")
    assert d.action.value == "transform"
    with pytest.raises(AssertionError, match="containing"):
        assert_transforms(REDACT, "mail bob@example.com", expected_substring="nope")


def test_assert_decision_combinations():
    d = assert_decision(PIPE, "FORBIDDEN", action="block", min_confidence=0.9, guard_name="bl", severity_at_least="high")
    assert d.guard_name == "bl"
    with pytest.raises(AssertionError, match="confidence"):
        assert_decision(PIPE, "fine", min_confidence=1.1)
    with pytest.raises(AssertionError, match="guard_name"):
        assert_decision(PIPE, "FORBIDDEN", guard_name="other")
    with pytest.raises(AssertionError, match="severity"):
        assert_decision(PIPE, "fine", severity_at_least="high")


def test_describe_decision_redacts_and_truncates():
    from aisafepy.core.decisions import GuardDecision

    d = GuardDecision.block(tier=1, confidence=1.0, rationale="r", evidence={"api_key": "x", "big": "y" * 2000})
    text = describe_decision(d)
    assert "[REDACTED]" in text and "..." in text and len(text) < 900


def _policy() -> Policy:
    return (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source("gmail.read", integrity="UNTRUSTED", caps={Capability.READ_USER})
        .require("send_email", control_flow_integrity="TRUSTED", arg_max_integrity="TRUSTED")
    )


def test_policy_assertions():
    policy = _policy()
    tainted = lift("x", source="gmail.read", integrity="UNTRUSTED")
    trusted = lift("x", source="user_prompt", integrity="TRUSTED")
    v = assert_policy_denies(policy, "send_email", tainted, reason_contains="argument integrity")
    assert v.tool == "send_email"
    assert_policy_allows(policy, "send_email", trusted)
    with pytest.raises(AssertionError, match="expected policy to deny"):
        assert_policy_denies(policy, "send_email", trusted)
    with pytest.raises(AssertionError, match="expected policy to allow"):
        assert_policy_allows(policy, "send_email", tainted)
    with pytest.raises(AssertionError, match="does not contain"):
        assert_policy_denies(policy, "send_email", tainted, reason_contains="zzz")
    assert_policy_denies(policy, "send_email", trusted, context=IFCContext(control_flow_integrity="UNTRUSTED"))


def test_policy_assertions_force_strict_mode_without_mutating():
    policy = _policy().with_mode("mediated")
    tainted = lift("x", source="gmail.read", integrity="UNTRUSTED")
    assert_policy_denies(policy, "send_email", tainted)
    assert policy.mode == "mediated"


class TestMixin(GuardTestCase, unittest.TestCase):
    def test_methods_delegate(self):
        self.assert_blocks(PIPE, "FORBIDDEN")
        self.assertAllows(PIPE, "ok")
        self.assert_transforms(REDACT, "a@b.co", expected_substring="[REDACTED]")
        self.assertPolicyAllows(_policy(), "send_email", lift("x", source="user_prompt", integrity="TRUSTED"))
        with self.assertRaises(AssertionError):
            self.assertBlocks(PIPE, "ok")
