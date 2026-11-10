from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from aisafepy.core.decisions import Action, GuardDecision
from aisafepy.core.sinks import (
    RECORD_SCHEMA_VERSION,
    CallbackSink,
    DecisionSink,
    FilterSink,
    InMemorySink,
    JSONLSink,
    MultiSink,
    decision_to_record,
    record_to_decision,
    severity_rank,
)


def _block(sev="high", name="g"):
    return GuardDecision.block(
        tier=2,
        confidence=0.9,
        rationale="token=abc leaked",
        severity=sev,
        guard_name=name,
        evidence={"api_key": "sk-1", "matches": ["x"]},
    )


def test_severity_rank_orders_and_defaults_unknown_to_medium():
    assert [severity_rank(s) for s in ("info", "low", "medium", "high", "critical")] == [0, 1, 2, 3, 4]
    assert severity_rank("bogus") == severity_rank("medium")


def test_decision_to_record_schema_and_redaction():
    d = _block()
    rec = decision_to_record(d, extra={"password": "p", "user": "u"}, timestamp=5.0)
    assert rec["schema_version"] == RECORD_SCHEMA_VERSION
    assert rec["timestamp"] == 5.0
    assert rec["action"] == "block" and rec["decision_id"] == d.decision_id
    assert rec["evidence"] == {"api_key": "[REDACTED]", "matches": ["x"]}
    assert rec["extra"] == {"password": "[REDACTED]", "user": "u"}
    assert rec["transformed"] is False
    raw = decision_to_record(d, redact_fields=False)
    assert raw["evidence"]["api_key"] == "sk-1"
    json.dumps(rec)  # must be JSON-safe


def test_record_round_trip():
    d = GuardDecision.transform(tier=1, confidence=1.0, rationale="r",
                                transformed_content="[X]", guard_name="t")
    rec = decision_to_record(d)
    assert rec["transformed"] is True
    back = record_to_decision(rec)
    assert back.action == Action.TRANSFORM and back.decision_id == d.decision_id
    assert back.transformed_content is None  # content is never persisted
    assert back.guard_name == "t" and back.tier == 1
    with pytest.raises(ValueError):
        record_to_decision({"schema_version": 99, "action": "allow"})
    odd = record_to_decision({"action": "allow", "evidence": "plain"})
    assert odd.evidence == {"value": "plain"}


def test_in_memory_sink_is_bounded_and_counts_actions():
    sink = InMemorySink(maxlen=3)
    assert isinstance(sink, DecisionSink)
    for _ in range(5):
        sink.emit(_block())
    sink.emit(GuardDecision.allow())
    assert len(sink) == 3
    assert sink.actions() == {"block": 2, "allow": 1}
    assert list(sink)[-1]["action"] == "allow"
    sink.clear()
    assert len(sink) == 0


def test_jsonl_sink_appends_and_reads(tmp_path: Path):
    sink = JSONLSink(tmp_path / "logs" / "decisions.jsonl")
    sink.emit(_block(), context={"request_id": "r1"})
    sink.emit(GuardDecision.allow(guard_name="a"))
    rows = sink.read()
    assert len(rows) == 2
    assert rows[0]["extra"] == {"request_id": "r1"}
    assert rows[0]["evidence"]["api_key"] == "[REDACTED]"
    assert rows[1]["guard_name"] == "a"
    assert JSONLSink(tmp_path / "missing.jsonl").read() == []


def test_jsonl_sink_is_thread_safe(tmp_path: Path):
    sink = JSONLSink(tmp_path / "d.jsonl")

    def worker():
        for _ in range(50):
            sink.emit(GuardDecision.allow())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = sink.read()
    assert len(rows) == 200
    assert all(r["action"] == "allow" for r in rows)


def test_callback_sink_swallows_errors():
    seen: list[dict] = []
    ok = CallbackSink(seen.append)
    ok.emit(_block())
    assert seen and seen[0]["action"] == "block"
    bad = CallbackSink(lambda r: 1 / 0)
    bad.emit(_block())
    bad.emit(_block())
    assert bad.errors == 2


def test_multi_sink_isolates_failures():
    mem = InMemorySink()
    bad = CallbackSink(lambda r: None)
    bad.emit = lambda d, c=None: (_ for _ in ()).throw(RuntimeError("x"))  # type: ignore[assignment]
    multi = MultiSink([bad, mem]).add(InMemorySink())
    multi.emit(_block())
    assert len(mem) == 1 and multi.errors == {0: 1}
    assert len(multi.sinks) == 3


def test_filter_sink_by_severity_action_and_guard():
    mem = InMemorySink()
    flt = FilterSink(mem, min_severity="high", actions=[Action.BLOCK, "escalate"],
                     guard_names=["g"])
    flt.emit(_block("high", "g"))  # passes
    flt.emit(_block("medium", "g"))  # severity too low
    flt.emit(_block("critical", "other"))  # wrong guard
    flt.emit(GuardDecision.escalate(tier=1, confidence=1.0, rationale="x", severity="high",
                                    guard_name="g"))  # passes
    flt.emit(GuardDecision.allow(guard_name="g"))  # action filtered, severity info
    assert len(mem) == 2 and flt.dropped == 3
    assert mem.actions() == {"block": 1, "escalate": 1}
    loose = FilterSink(InMemorySink())
    assert loose.accepts(GuardDecision.allow())
