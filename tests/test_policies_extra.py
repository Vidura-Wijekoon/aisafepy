from __future__ import annotations

from pathlib import Path

import pytest

from aisafepy.core.policies import (
    PolicyDocument,
    PolicyRule,
    compile_selector,
    rules_from_dicts,
)
from aisafepy.stream.pipeline import Context, GuardPipeline

yaml = pytest.importorskip("yaml")


def _doc() -> PolicyDocument:
    return PolicyDocument.from_dict(
        {
            "name": "extra",
            "version": "2",
            "description": "vocabulary coverage",
            "rules": [
                {"id": "any-regex", "regex_any": [r"\bdrop\s+table\b", r"\brm\s+-rf\b"],
                 "decision": "block", "rationale": "destructive command", "severity": "critical"},
                {"id": "role-scoped", "contains": ["as an ai"], "when_role": "assistant",
                 "decision": "escalate", "rationale": "refusal leak"},
                {"id": "in-set", "field": "metadata.tenant", "field_in": ["free", "trial"],
                 "decision": "transform", "rationale": "redact for free tier",
                 "metadata": {"replacement": "[hidden]"}},
                {"id": "too-long", "length_gt": 20, "decision": "block", "rationale": "long"},
                {"id": "combo", "all_of": [{"contains": ["wire"]},
                                          {"not": {"contains": ["invoice"]}}],
                 "decision": "block", "rationale": "wire without invoice"},
                {"id": "either", "any_of": [{"equals": "ping"}, {"field": "prompt",
                                                                  "equals": "pong"}],
                 "decision": "allow", "rationale": "handshake"},
                {"id": "short", "length_lt": 3, "decision": "escalate", "rationale": "tiny"},
            ],
        }
    )


def test_regex_any_and_severity():
    doc = _doc()
    m = doc.evaluate({"content": "please DROP   TABLE users"})
    assert m is not None and m.id == "any-regex" and m.severity == "critical"
    assert doc.evaluate({"content": "rm -rf /"}).id == "any-regex"


def test_when_role_scopes_rule():
    doc = _doc()
    assert doc.evaluate({"content": "As an AI I cannot", "role": "user"}) is None
    assert doc.evaluate({"content": "As an AI I cannot", "role": "assistant"}).id == "role-scoped"


def test_field_in_with_dotted_field_and_nested_context():
    doc = _doc()
    assert doc.evaluate({"content": "hello", "metadata": {"tenant": "free"}}).id == "in-set"
    assert doc.evaluate({"content": "hello", "metadata.tenant": "trial"}).id == "in-set"
    assert doc.evaluate({"content": "hello", "metadata": {"tenant": "paid"}}) is None


def test_length_rules():
    doc = _doc()
    assert doc.evaluate({"content": "x" * 21}).id == "too-long"
    assert doc.evaluate({"content": "ab"}).id == "short"
    assert doc.evaluate({"content": 123}) is None  # len() not applicable


def test_all_of_any_of_not():
    doc = _doc()
    assert doc.evaluate({"content": "wire the funds"}).id == "combo"
    assert doc.evaluate({"content": "wire per invoice"}) is None
    assert doc.evaluate({"content": "ping"}).id == "either"
    assert doc.evaluate({"content": "zzzz", "prompt": "pong"}).id == "either"


def test_evaluate_all_and_explain():
    doc = _doc()
    ctx = {"content": "wire " + "x" * 20}
    ids = [r.id for r in doc.evaluate_all(ctx)]
    assert ids == ["too-long", "combo"]
    exp = doc.explain(ctx)
    assert [e["id"] for e in exp if e["matched"]] == ["too-long", "combo"]
    assert [e["id"] for e in exp if e["selected"]] == ["too-long"]
    assert doc.rule("combo") is not None and doc.rule("nope") is None
    assert len(doc) == 7


def test_invalid_specs_raise():
    with pytest.raises(ValueError):
        compile_selector({"id": "x", "pattern": "a", "contains": ["b"]})
    with pytest.raises(ValueError):
        compile_selector({"regex_any": "not-a-list"})
    with pytest.raises(ValueError):
        compile_selector({"all_of": []})
    with pytest.raises(ValueError):
        compile_selector({"not": "nope"})
    with pytest.raises(ValueError):
        rules_from_dicts([{"id": "x", "decision": "maybe"}])
    assert compile_selector({"id": "placeholder"})({"content": "anything"}) is False


def test_to_guard_decision_for_each_decision_kind():
    doc = _doc()
    assert doc.to_guard_decision(None).action.value == "allow"
    block = doc.to_guard_decision(doc.rule("any-regex"))
    assert block.action.value == "block" and block.severity == "critical"
    assert block.evidence["rule_id"] == "any-regex" and block.guard_name == "policy:extra"
    esc = doc.to_guard_decision(doc.rule("role-scoped"))
    assert esc.action.value == "escalate"
    tr = doc.to_guard_decision(doc.rule("in-set"))
    assert tr.action.value == "transform" and tr.transformed_content == "[hidden]"
    allow = doc.to_guard_decision(doc.rule("either"))
    assert allow.action.value == "allow" and allow.evidence["rule_id"] == "either"


@pytest.mark.asyncio
async def test_as_guard_runs_inside_pipeline():
    doc = _doc()
    guard = doc.as_guard()
    assert guard.name == "policy:extra" and guard.tier == 1
    pipe = GuardPipeline(tier1=[guard])
    d = await pipe.evaluate_chunk(Context(buffer="drop table x"))
    assert d.is_blocked and d.evidence["rule_id"] == "any-regex"
    d2 = await pipe.evaluate_chunk(Context(buffer="As an AI", role="assistant"))
    assert d2.action.value == "escalate"
    d3 = await pipe.evaluate_chunk(Context(buffer="hello", metadata={"tenant": "free"}))
    assert d3.action.value == "transform"
    d4 = await pipe.evaluate_chunk(Context(buffer="hello there"))
    assert d4.action.value == "allow" and d4.latency_ms >= 0


def test_yaml_round_trip_preserves_declarative_rules(tmp_path: Path):
    doc = _doc()
    doc.add(PolicyRule(id="py", selector=lambda c: True, decision="allow", rationale="callable"))
    path = tmp_path / "policy.yaml"
    text = doc.to_yaml(path)
    assert path.read_text(encoding="utf-8") == text
    data = yaml.safe_load(text)
    assert data["rules"][-1]["opaque"] is True
    assert data["rules"][0]["regex_any"][0] == r"\bdrop\s+table\b"
    assert data["rules"][2]["field"] == "metadata.tenant"
    again = PolicyDocument.from_yaml(path)
    assert [r.id for r in again.rules] == [r.id for r in doc.rules]
    assert again.description == "vocabulary coverage"
    for ctx in (
        {"content": "drop table t"},
        {"content": "wire money"},
        {"content": "hi", "metadata": {"tenant": "trial"}},
        {"content": "As an AI", "role": "assistant"},
    ):
        assert again.evaluate(ctx).id == doc.evaluate(ctx).id
    # The opaque rule never matches after reload but keeps its identity.
    opaque = again.rule("py")
    assert opaque is not None and opaque.decision == "allow"
    assert opaque.matches({"content": "x"}) is False


def test_to_dict_round_trip_without_yaml():
    doc = _doc()
    again = PolicyDocument.from_dict(doc.to_dict())
    assert again.to_dict() == doc.to_dict()
