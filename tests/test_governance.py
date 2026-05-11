from __future__ import annotations

from pathlib import Path

from aisafepy.adapt.governance import AuditLog


def test_audit_log_appends_and_verifies(tmp_path: Path):
    log = AuditLog(path=tmp_path / "audit.jsonl")
    log.append("compile", actor="ci", payload={"clusters": 4})
    log.append("promote", actor="ci", payload={"artifact": "regex-0"})
    assert log.verify()
    entries = log.entries()
    assert len(entries) == 2
    assert entries[0].event == "compile"
    assert entries[1].event == "promote"


def test_audit_log_detects_tampering(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path)
    log.append("compile", actor="ci", payload={"x": 1})
    log.append("promote", actor="ci", payload={"x": 2})
    raw = path.read_text(encoding="utf-8").splitlines()
    tampered = raw[0].replace('"x": 1', '"x": 999')
    path.write_text("\n".join([tampered] + raw[1:]) + "\n")
    log2 = AuditLog(path=path)
    assert log2.verify() is False
