from __future__ import annotations

import pytest

from aisafepy.flow import Capability, Policy
from aisafepy.flow.interpreter import (
    IFCContext,
    IFCViolationError,
    evaluate_call,
)
from aisafepy.flow.taint import lift


def _policy_secret_email() -> Policy:
    return (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source("gmail.read", integrity="UNTRUSTED", caps={Capability.READ_USER})
        .label_source("secrets", integrity="UNTRUSTED", caps={Capability.READ_SECRETS})
        .require(
            "send_email",
            control_flow_integrity="TRUSTED",
            caps={Capability.WRITE_EXTERNAL},
        )
    )


def test_allow_when_args_meet_requirements():
    policy = _policy_secret_email()
    to = lift("alice@example.com", source="user_prompt", integrity="TRUSTED")
    body = lift("hi", source="user_prompt", integrity="TRUSTED",
                capabilities=("write.external",))
    decision = evaluate_call(
        tool="send_email",
        args=(to, body),
        kwargs={},
        policy=policy,
        context=IFCContext(control_flow_integrity="TRUSTED"),
    )
    assert decision.action.value == "allow"


def test_block_when_args_contain_secrets():
    policy = _policy_secret_email()
    to = lift("alice@example.com", source="user_prompt", integrity="TRUSTED")
    body = lift("supersecret", source="secrets", integrity="UNTRUSTED",
                capabilities=("read.secrets",))
    with pytest.raises(IFCViolationError) as exc_info:
        evaluate_call(
            tool="send_email",
            args=(to, body),
            kwargs={},
            policy=policy,
            context=IFCContext(control_flow_integrity="TRUSTED"),
        )
    v = exc_info.value.violation
    assert v.tool == "send_email"
    assert "read.secrets" in v.capabilities


def test_block_when_control_flow_untrusted():
    policy = _policy_secret_email()
    to = lift("alice@example.com", source="user_prompt", integrity="TRUSTED")
    body = lift("hi", source="user_prompt", integrity="TRUSTED",
                capabilities=("write.external",))
    with pytest.raises(IFCViolationError) as exc_info:
        evaluate_call(
            tool="send_email",
            args=(to, body),
            kwargs={},
            policy=policy,
            context=IFCContext(control_flow_integrity="UNTRUSTED"),
        )
    assert "control flow integrity" in exc_info.value.violation.reason


def test_deny_if_rule_fires():
    policy = (
        _policy_secret_email()
        .deny_if(
            "send_email",
            when=lambda *args, **kw: any(
                "@evil.com" in (a.value if hasattr(a, "value") else str(a))
                for a in (*args, *kw.values())
            ),
            reason="evil recipient",
        )
    )
    to = lift("attacker@evil.com", source="user_prompt", integrity="TRUSTED")
    body = lift("hi", source="user_prompt", integrity="TRUSTED",
                capabilities=("write.external",))
    with pytest.raises(IFCViolationError) as exc_info:
        evaluate_call(
            tool="send_email",
            args=(to, body),
            kwargs={},
            policy=policy,
            context=IFCContext(control_flow_integrity="TRUSTED"),
        )
    assert "evil recipient" in exc_info.value.violation.reason


def test_mediated_mode_invokes_mediator():
    policy = _policy_secret_email().with_mode("mediated")
    body = lift("supersecret", source="secrets", integrity="UNTRUSTED",
                capabilities=("read.secrets",))
    to = lift("alice@example.com", source="user_prompt", integrity="TRUSTED")
    decisions = []

    def mediator(_v):
        decisions.append(_v)
        return True  # human approves

    decision = evaluate_call(
        tool="send_email",
        args=(to, body),
        kwargs={},
        policy=policy,
        context=IFCContext(control_flow_integrity="TRUSTED"),
        mediator=mediator,
    )
    assert decision.action.value == "allow"
    assert len(decisions) == 1
