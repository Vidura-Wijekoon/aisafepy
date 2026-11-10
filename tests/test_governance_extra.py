from __future__ import annotations

import csv
import json
from pathlib import Path

from aisafepy.adapt.governance import AuditLog, VerifyResult
from aisafepy.core.decisions import GuardDecision


def test_verify_detailed_reports_index_and_reason(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path)
    for i in range(4):
        log.append("e", actor="ci", payload={"i": i})
    ok = log.verify_detailed()
    assert ok.ok and ok.entries_checked == 4 and bool(ok)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[2] = lines[2].replace('"i": 2', '"i": 99')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad = AuditLog(path=path).verify_detailed()
    assert bad.ok is False and bad.first_bad_index == 2
    assert "hash mismatch" in bad.reason
    assert isinstance(bad, VerifyResult) and not bad


def test_verify_detects_deleted_and_inserted_lines(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path)
    for i in range(3):
        log.append("e", actor="ci", payload={"i": i})
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    res = AuditLog(path=path).verify_detailed()
    assert not res.ok and res.first_bad_index == 1 and "prev_hash" in res.reason
    path.write_text("\n".join([lines[0], "{not json", lines[1]]) + "\n", encoding="utf-8")
    res = AuditLog(path=path).verify_detailed()
    assert not res.ok and res.first_bad_index == 1 and "malformed" in res.reason


def test_empty_or_missing_log_verifies(tmp_path: Path):
    log = AuditLog(path=tmp_path / "none.jsonl")
    assert log.verify_detailed() == VerifyResult(ok=True, entries_checked=0)
    assert log.entries() == [] and len(log) == 0 and log.tail() == []


def test_hmac_signing_and_tamper_detection(tmp_path: Path):
    path = tmp_path / "signed.jsonl"
    log = AuditLog(path=path, hmac_key=b"k" * 32)
    log.append("a", actor="ci", payload={})
    log.append("b", actor="ci", payload={})
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert all("__sig__" in r for r in raw)
    assert log.verify()
    assert AuditLog(path=path, hmac_key="k" * 32).verify()  # str keys are accepted
    wrong = AuditLog(path=path, hmac_key=b"other")
    res = wrong.verify_detailed()
    assert not res.ok and "HMAC" in res.reason and res.first_bad_index == 0
    # An unsigned log fails verification once a key is required.
    unsigned = AuditLog(path=tmp_path / "plain.jsonl")
    unsigned.append("a", actor="ci", payload={})
    res2 = AuditLog(path=tmp_path / "plain.jsonl", hmac_key=b"k").verify_detailed()
    assert not res2.ok and "missing HMAC" in res2.reason
    # Rewriting an entry and recomputing its hash without the key still fails.
    obj = raw[0]
    obj.pop("__hash__")
    obj.pop("__sig__")
    obj["payload"] = {"forged": True}
    from aisafepy.adapt.governance import AuditEntry

    entry = AuditEntry(**obj)
    forged = {**obj, "__hash__": entry.compute_hash(), "__sig__": "00"}
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(json.dumps(forged) + "\n" + lines[1] + "\n", encoding="utf-8")
    assert not AuditLog(path=path, hmac_key=b"k" * 32).verify()


def test_chain_continues_across_processes_simulated(tmp_path: Path):
    """Two AuditLog objects on the same file must chain onto each other."""
    path = tmp_path / "shared.jsonl"
    a = AuditLog(path=path)
    b = AuditLog(path=path)
    a.append("a1", actor="a", payload={})
    b.append("b1", actor="b", payload={})  # b was opened on an empty file
    a.append("a2", actor="a", payload={})  # a's cached last hash is stale
    res = AuditLog(path=path).verify_detailed()
    assert res.ok and res.entries_checked == 3
    events = [e.event for e in AuditLog(path=path).iter_entries()]
    assert events == ["a1", "b1", "a2"]


def test_tail_and_iter_entries(tmp_path: Path):
    log = AuditLog(path=tmp_path / "t.jsonl")
    for i in range(7):
        log.append(f"e{i}", actor="ci", payload={"i": i})
    assert [e.event for e in log.tail(3)] == ["e4", "e5", "e6"]
    assert [e.event for e in log.tail(100)] == [f"e{i}" for i in range(7)]
    assert log.tail(0) == []
    assert len(log) == 7
    assert next(log.iter_entries()).payload == {"i": 0}


def test_export_csv(tmp_path: Path):
    log = AuditLog(path=tmp_path / "t.jsonl")
    log.append("compile", actor="ci", payload={"clusters": 2, "nested": {"a": 1}})
    log.append("promote", actor="ops", payload={})
    out = log.export_csv(tmp_path / "out" / "audit.csv")
    with out.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["event"] for r in rows] == ["compile", "promote"]
    assert json.loads(rows[0]["payload"]) == {"clusters": 2, "nested": {"a": 1}}
    assert rows[1]["prev_hash"] != "" and rows[0]["prev_hash"] == ""
    assert rows[1]["actor"] == "ops"


def test_append_decision_redacts_secrets(tmp_path: Path):
    log = AuditLog(path=tmp_path / "d.jsonl")
    d = GuardDecision.block(
        tier=2,
        confidence=0.9,
        rationale="leak",
        guard_name="g",
        evidence={"api_key": "sk-live-123", "matches": ["x"]},
    )
    entry = log.append_decision(d, actor="pipeline", extra={"password": "p", "user": "u"})
    assert entry.event == "guard.decision"
    assert entry.payload["evidence"]["api_key"] == "[REDACTED]"
    assert entry.payload["evidence"]["matches"] == ["x"]
    assert entry.payload["extra"] == {"password": "[REDACTED]", "user": "u"}
    assert entry.payload["decision_id"] == d.decision_id
    assert log.verify()
