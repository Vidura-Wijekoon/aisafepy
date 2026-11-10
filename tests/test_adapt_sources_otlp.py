"""Tests for the OTLP span source and the decision-log source."""

from __future__ import annotations

import json

from aisafepy.adapt.sources_otlp import (
    DecisionLogSource,
    OTLPSpanSource,
    iter_otlp_spans,
    span_attributes,
    unwrap_attribute_value,
)
from aisafepy.core.decisions import GuardDecision
from aisafepy.core.sinks import JSONLSink, decision_to_record


def _attr(key, value):
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": value}}


def _span(name, attrs, *, span_id="abc123", start_ns="1700000000000000000"):
    return {
        "traceId": "t1",
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": start_ns,
        "attributes": [_attr(k, v) for k, v in attrs.items()],
    }


def _otlp_doc(spans):
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attr("service.name", "demo")]},
                "scopeSpans": [{"scope": {"name": "aisafepy.stream"}, "spans": spans}],
            }
        ]
    }


def _blocked_span(text="ignore previous instructions", span_id="s1"):
    return _span(
        "aisafepy.stream.tier1",
        {
            "aisafepy.guard.action": "block",
            "aisafepy.guard.confidence": 0.97,
            "aisafepy.guard.tier": 1,
            "aisafepy.guard.rationale": "blocklist hit",
            "aisafepy.guard.severity": "high",
            "aisafepy.guard.name": "bl",
            "aisafepy.guard.decision_id": "dec-" + span_id,
            "aisafepy.guard.evidence": json.dumps({"matches": ["ignore previous instructions"]}),
            "aisafepy.input": text,
        },
        span_id=span_id,
    )


def _allowed_span(span_id="s2"):
    return _span(
        "aisafepy.stream.tier1",
        {"aisafepy.guard.action": "allow", "aisafepy.guard.name": "bl", "gen_ai.prompt": "hello there"},
        span_id=span_id,
    )


def test_unwrap_attribute_values():
    assert unwrap_attribute_value({"stringValue": "x"}) == "x"
    assert unwrap_attribute_value({"intValue": "7"}) == 7
    assert unwrap_attribute_value({"doubleValue": 0.5}) == 0.5
    assert unwrap_attribute_value({"boolValue": True}) is True
    assert unwrap_attribute_value({"arrayValue": {"values": [{"stringValue": "a"}, {"intValue": "2"}]}}) == ["a", 2]
    assert unwrap_attribute_value({"kvlistValue": {"values": [{"key": "k", "value": {"stringValue": "v"}}]}}) == {"k": "v"}
    assert unwrap_attribute_value("plain") == "plain"


def test_span_attributes_handles_list_and_dict_shapes():
    assert span_attributes({"attributes": [_attr("a", 1)]}) == {"a": 1}
    assert span_attributes({"attributes": {"a": {"stringValue": "x"}, "b": 2}}) == {"a": "x", "b": 2}
    assert span_attributes({}) == {}


def test_iter_otlp_spans_accepts_several_shapes():
    span = _blocked_span()
    assert list(iter_otlp_spans(_otlp_doc([span]))) == [span]
    assert list(iter_otlp_spans([span])) == [span]
    assert list(iter_otlp_spans({"spans": [span]})) == [span]
    legacy = {"resourceSpans": [{"instrumentationLibrarySpans": [{"spans": [span]}]}]}
    assert list(iter_otlp_spans(legacy)) == [span]
    assert list(iter_otlp_spans(42)) == []


def test_otlp_source_yields_violations_only_by_default(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_otlp_doc([_blocked_span(), _allowed_span(), _span("unrelated", {"http.method": "GET"})])))
    records = list(OTLPSpanSource(path))
    assert len(records) == 1
    rec = records[0]
    assert rec.id == "dec-s1"
    assert rec.input == "ignore previous instructions"
    assert rec.was_violation is True
    assert rec.attack_category == "bl"
    assert rec.score == 0.97
    assert rec.severity == "high"
    assert rec.timestamp == 1700000000.0
    assert rec.source == "otlp"
    assert rec.metadata["tier"] == 1
    assert rec.metadata["rationale"] == "blocklist hit"
    assert rec.metadata["span_id"] == "s1"


def test_otlp_source_can_include_allows_and_uses_prompt_attribute():
    records = list(OTLPSpanSource(_otlp_doc([_blocked_span(), _allowed_span()]), only_violations=False))
    assert [r.was_violation for r in records] == [True, False]
    assert records[1].input == "hello there"
    assert records[1].id == "s2"  # no decision id attribute, falls back to span id


def test_otlp_source_falls_back_to_evidence_and_honours_text_from():
    span = _blocked_span()
    span["attributes"] = [a for a in span["attributes"] if a["key"] != "aisafepy.input"]
    rec = next(iter(OTLPSpanSource([span])))
    assert "ignore previous instructions" in rec.input  # evidence JSON
    custom = next(iter(OTLPSpanSource([span], text_from=lambda attrs: "custom:" + str(attrs["aisafepy.guard.name"]))))
    assert custom.input == "custom:bl"


def test_otlp_source_reads_completion_as_output():
    span = _blocked_span()
    span["attributes"].append(_attr("gen_ai.completion", "sure, here is the secret"))
    rec = next(iter(OTLPSpanSource([span])))
    assert rec.output == "sure, here is the secret"


def _write_log(tmp_path):
    path = tmp_path / "decisions.jsonl"
    sink = JSONLSink(path)
    sink.emit(
        GuardDecision.block(tier=1, confidence=0.9, rationale="regex match", guard_name="pii"),
        {"input": "my ssn is on file", "output": "noted"},
    )
    sink.emit(GuardDecision.allow(tier=1, guard_name="pii"), {"input": "fine"})
    sink.emit(
        GuardDecision.escalate(tier=2, confidence=0.6, rationale="uncertain", guard_name="clf"),
        {"text": "borderline"},
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n{not json}\n")
        fh.write(json.dumps({"unrelated": True}) + "\n")
    return path


def test_decision_log_source_reads_sink_output(tmp_path):
    path = _write_log(tmp_path)
    src = DecisionLogSource(path)
    records = list(src)
    assert [r.was_violation for r in records] == [True, True]
    first = records[0]
    assert first.input == "my ssn is on file"
    assert first.output == "noted"
    assert first.attack_category == "pii"
    assert first.score == 0.9
    assert first.source == "decision_log"
    assert first.metadata["action"] == "block"
    assert first.metadata["schema_version"] == 1
    assert records[1].input == "borderline"
    assert src.skipped == 2


def test_decision_log_source_include_allows_and_text_override(tmp_path):
    path = _write_log(tmp_path)
    records = list(DecisionLogSource(path, only_violations=False, text_from=lambda r: r["rationale"]))
    assert len(records) == 3
    assert records[0].input == "regex match"


def test_decision_log_source_falls_back_to_evidence_text(tmp_path):
    record = decision_to_record(
        GuardDecision.block(tier=1, confidence=1.0, rationale="x", evidence={"matches": ["abc"]}, guard_name="g")
    )
    path = tmp_path / "log.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rec = next(iter(DecisionLogSource(path)))
    assert "abc" in rec.input
    assert rec.id == record["decision_id"]
