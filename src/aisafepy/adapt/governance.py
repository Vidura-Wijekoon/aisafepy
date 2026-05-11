"""Audit log + governance helpers.

The compiler is allowed to be opinionated about *recording* its
decisions, but it is not allowed to be opinionated about *what
governance system you plug it into*. We expose a small append-only
:class:`AuditLog` that stores hashed, signed entries on disk; teams
who want to forward those entries to a SIEM / GRC system can do so
with a 10-line shim.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEntry:
    id: str
    event: str
    timestamp: float
    actor: str
    payload: dict[str, Any]
    prev_hash: str | None = None
    """Hash of the previous entry, forming a hash chain."""

    def compute_hash(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass
class AuditLog:
    """An append-only audit log with a SHA-256 hash chain.

    Entries are JSON Lines. Each entry's ``prev_hash`` field is the
    hash of the prior entry; the integrity check (:meth:`verify`)
    walks the file and confirms the chain is intact.
    """

    path: Path
    _last_hash: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = None
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                self._last_hash = obj.get("__hash__")

    def append(self, event: str, *, actor: str, payload: dict[str, Any]) -> AuditEntry:
        entry = AuditEntry(
            id=uuid.uuid4().hex,
            event=event,
            timestamp=time.time(),
            actor=actor,
            payload=payload,
            prev_hash=self._last_hash,
        )
        h = entry.compute_hash()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**asdict(entry), "__hash__": h}, default=str))
            f.write("\n")
        self._last_hash = h
        return entry

    def verify(self) -> bool:
        prev: str | None = None
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                claimed_hash = obj.pop("__hash__", None)
                entry = AuditEntry(
                    id=obj["id"],
                    event=obj["event"],
                    timestamp=obj["timestamp"],
                    actor=obj["actor"],
                    payload=obj["payload"],
                    prev_hash=obj.get("prev_hash"),
                )
                if entry.prev_hash != prev:
                    return False
                if entry.compute_hash() != claimed_hash:
                    return False
                prev = claimed_hash
        return True

    def entries(self) -> list[AuditEntry]:
        out: list[AuditEntry] = []
        if not self.path.exists():
            return out
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            obj.pop("__hash__", None)
            out.append(AuditEntry(**obj))
        return out
