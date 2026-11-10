from __future__ import annotations

import json

import pytest

from aisafepy.flow.audit import DecisionRecord, DecisionRecorder
from aisafepy.flow.interpreter import (
    IFCContext,
    IFCViolationError,
    evaluate_call,
    ifc_run,
)
from aisafepy.flow.policy import Capability, Policy
from aisafepy.flow.taint import lift

USER = lift("alice@corp.example", source="user_prompt", integrity="TRUSTED")
SECRET = lift("hunter2", source="vault", capabilities=("read.secrets",))
MAIL = lift("hello", source="gmail.read", capabilities=("read.user",))


def _policy() -> Policy:
    return (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source("vault", caps={Capability.READ_SECRETS})
        .label_source("gmail.read", caps={Capability.READ_USER})
        .require("send_email", control_flow_integrity="TRUSTED", caps={Capability.READ_USER})
    )


def _allow(policy, ctx):
    return evaluate_call(tool="send_email", args=(), kwargs={"to": USER, "body": MAIL},
                         policy=policy, context=ctx)


def _deny(policy, ctx, **kw):
    return evaluate_call(tool="send_email", args=(), kwargs={"to": USER, "body": SECRET},
                         policy=policy, context=ctx, **kw)


def test_records_allow_and_violation_with_labels():
    policy = _policy()
    rec = DecisionRecorder()
    ctx = IFCContext(on_decision=rec)
    _allow(policy, ctx)
    with pytest.raises(IFCViolationError):
        _deny(policy, ctx)
    assert len(rec) == 2
    first, second = rec.entries
    assert first.kind == "decision" and first.action == "allow" and first.tool == "send_email"
    assert first.provenance == ("gmail.read", "user_prompt")
    assert first.cfi == "TRUSTED"
    assert second.kind == "violation" and second.action == "block"
    assert "forbidden capabilities" in second.reason
    assert "read.secrets" in second.capabilities
    assert second.required_capabilities == ("read.user",)
    assert second.ref_id and first.ref_id and second.seq == first.seq + 1


def test_mediated_approval_emits_violation_then_decision():
    policy = _policy().with_mode("mediated")
    rec = DecisionRecorder()
    ctx = IFCContext(on_decision=rec)
    decision = _deny(policy, ctx, mediator=lambda v: True)
    assert decision.action.value == "allow"
    kinds = [r.kind for r in rec.entries]
    assert kinds == ["violation", "decision"]
    assert rec.entries[1].mediated is True
    assert rec.summary()["mediated_approvals"] == 1


def test_hook_exceptions_never_change_enforcement(caplog):
    policy = _policy()

    def boom(outcome, tool):
        raise RuntimeError("observer failed")

    ctx = IFCContext(on_decision=boom)
    with caplog.at_level("WARNING", logger="aisafepy.flow"):
        decision = _allow(policy, ctx)
    assert decision.action.value == "allow"
    assert "on_decision hook raised" in caplog.text


def test_callback_backend_and_errors_are_isolated(caplog):
    seen: list[DecisionRecord] = []

    def cb(record):
        seen.append(record)
        raise ValueError("cb failure")

    rec = DecisionRecorder(callback=cb)
    ctx = IFCContext(on_decision=rec)
    with caplog.at_level("WARNING", logger="aisafepy.flow.audit"):
        _allow(_policy(), ctx)
    assert len(seen) == 1 and len(rec) == 1
    assert "callback raised" in caplog.text


def test_ring_buffer_evicts_but_lifetime_total_counts():
    rec = DecisionRecorder(max_entries=3)
    ctx = IFCContext(on_decision=rec)
    for _ in range(5):
        _allow(_policy(), ctx)
    assert len(rec) == 3
    assert rec.total == 5
    assert [r.seq for r in rec.entries] == [3, 4, 5]
    assert rec.summary()["lifetime_total"] == 5
    rec.clear()
    assert len(rec) == 0


def test_jsonl_backend_flushes_on_ifc_run_exit_and_redacts(tmp_path):
    path = tmp_path / "logs" / "audit.jsonl"
    rec = DecisionRecorder(path=path)
    policy = _policy()
    with ifc_run(policy, recorder=rec) as ctx:
        assert ctx.on_decision is rec
        _allow(policy, ctx)
        assert not path.exists()  # buffered until flush
        with pytest.raises(IFCViolationError):
            _deny(policy, ctx)
    rows = DecisionRecorder.read_jsonl(path)
    assert [r["kind"] for r in rows] == ["decision", "violation"]
    assert rows[0]["provenance"] == ["gmail.read", "user_prompt"]
    # Appends on the next flush; does not rewrite.
    with ifc_run(policy, recorder=rec) as ctx:
        _allow(policy, ctx)
    assert len(DecisionRecorder.read_jsonl(path)) == 3
    assert rec.flush() == 0
    assert DecisionRecorder.read_jsonl(tmp_path / "missing.jsonl") == []


def test_evidence_redaction_can_be_disabled():
    from aisafepy.core.decisions import GuardDecision

    secret_decision = GuardDecision.allow(evidence={"api_key": "sk-123", "ok": 1})
    redacted = DecisionRecorder()
    redacted(secret_decision, "t")
    assert redacted.entries[0].evidence["api_key"] == "[REDACTED]"
    raw = DecisionRecorder(redact_evidence=False)
    raw(secret_decision, "t")
    assert raw.entries[0].evidence["api_key"] == "sk-123"


def test_autoflush_writes_immediately(tmp_path):
    path = tmp_path / "a.jsonl"
    rec = DecisionRecorder(path=path, autoflush=True)
    _allow(_policy(), IFCContext(on_decision=rec))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_ifc_run_rejects_non_callable_recorder():
    with pytest.raises(TypeError), ifc_run(_policy(), recorder=object()):
        pass


def test_recording_context_manager_installs_and_restores(tmp_path):
    rec = DecisionRecorder(path=tmp_path / "r.jsonl")
    previous = lambda o, t: None  # noqa: E731
    ctx = IFCContext(on_decision=previous)
    with rec.recording(ctx) as inner:
        assert inner is ctx and ctx.on_decision is rec
        _allow(_policy(), ctx)
    assert ctx.on_decision is previous
    assert (tmp_path / "r.jsonl").exists()
    with rec.recording(_policy()) as fresh:
        assert isinstance(fresh, IFCContext) and fresh.on_decision is rec
    with rec.recording() as fresh2:
        assert fresh2.on_decision is rec


def test_summary_and_to_dicts():
    policy = _policy()
    rec = DecisionRecorder()
    ctx = IFCContext(on_decision=rec)
    _allow(policy, ctx)
    _allow(policy, ctx)
    for _ in range(2):
        with pytest.raises(IFCViolationError):
            _deny(policy, ctx)
    s = rec.summary()
    assert s["total"] == 4
    assert s["by_action"] == {"allow": 2, "violation": 2}
    assert s["by_tool"] == {"send_email": {"allow": 2, "violation": 2}}
    assert s["top_deny_reasons"][0][1] == 2
    assert len(rec.violations()) == 2
    dicts = rec.to_dicts()
    assert isinstance(dicts[0]["provenance"], list)
    json.dumps(dicts)  # serializable


def test_invalid_max_entries():
    with pytest.raises(ValueError):
        DecisionRecorder(max_entries=0)
