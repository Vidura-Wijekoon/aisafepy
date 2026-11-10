"""Audit log + governance helpers.

The compiler is allowed to be opinionated about *recording* its
decisions, but it is not allowed to be opinionated about *what
governance system you plug it into*. We expose a small append-only
:class:`AuditLog` that stores hash-chained (and optionally
HMAC-signed) entries on disk; teams who want to forward those entries
to a SIEM / GRC system can do so with a 10-line shim.

File format
-----------
One JSON object per line. Every object carries the
:class:`AuditEntry` fields plus ``__hash__`` (SHA-256 over the entry,
including ``prev_hash``) and, when the log was opened with an
``hmac_key``, ``__sig__`` (HMAC-SHA256 of ``__hash__``). The hash
chain detects any edit, insertion or deletion; the HMAC additionally
detects wholesale rewrites by someone who does not hold the key.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import hmac
import json
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from aisafepy.core.decisions import GuardDecision

# ---- cross-platform file locking ----------------------------------------

#: Byte offset used as the Windows lock sentinel. Windows locks are
#: mandatory, so locking a byte inside the data region would block the
#: readers that verify the chain. Locking one byte far beyond any
#: realistic log size gives an exclusive mutex without touching data.
_LOCK_OFFSET = 0x7FFF_0000

if sys.platform == "win32":
    import msvcrt

    def _exclusive_lock(f: IO) -> None:
        # Every writer locks the same sentinel byte, which serializes
        # appends across processes on Windows.
        f.seek(_LOCK_OFFSET)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

    def _exclusive_unlock(f: IO) -> None:
        with contextlib.suppress(OSError):
            f.seek(_LOCK_OFFSET)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _exclusive_lock(f: IO) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _exclusive_unlock(f: IO) -> None:
        with contextlib.suppress(OSError):
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


_TAIL_BYTES = 65_536


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


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of :meth:`AuditLog.verify_detailed`."""

    ok: bool
    entries_checked: int
    first_bad_index: int | None = None
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class AuditLog:
    """An append-only audit log with a SHA-256 hash chain.

    Entries are JSON Lines. Each entry's ``prev_hash`` field is the
    hash of the prior entry; the integrity check (:meth:`verify`)
    walks the file and confirms the chain is intact. Pass ``hmac_key``
    to additionally sign every line; verification then requires the
    same key.
    """

    path: Path
    hmac_key: bytes | None = None
    _last_hash: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(self.hmac_key, str):
            self.hmac_key = self.hmac_key.encode("utf-8")
        self._last_hash = None
        if self.path.exists():
            self._last_hash = self._read_last_hash_from_tail()

    # ---- writing -------------------------------------------------------

    def append(self, event: str, *, actor: str, payload: dict[str, Any]) -> AuditEntry:
        """Append a new entry to the log.

        The write is protected by an OS file lock so concurrent
        appends from multiple processes do not corrupt the hash chain.
        Uses ``fcntl.flock`` on POSIX and ``msvcrt.locking`` on Windows.
        Inside the lock the tail of the file is re-read so that, if
        another process appended since this object last wrote, the
        new entry chains onto the real last hash rather than a stale one.
        """
        with self.path.open("a+b") as f:
            _exclusive_lock(f)
            try:
                prev = self._last_hash_from_binary(f)
                entry = AuditEntry(
                    id=uuid.uuid4().hex,
                    event=event,
                    timestamp=time.time(),
                    actor=actor,
                    payload=payload,
                    prev_hash=prev,
                )
                h = entry.compute_hash()
                obj: dict[str, Any] = {**asdict(entry), "__hash__": h}
                if self.hmac_key is not None:
                    obj["__sig__"] = self._sign(h)
                f.seek(0, 2)  # to end
                f.write((json.dumps(obj, default=str) + "\n").encode("utf-8"))
                f.flush()
            finally:
                _exclusive_unlock(f)
        self._last_hash = h
        return entry

    def append_decision(
        self,
        decision: GuardDecision,
        *,
        actor: str,
        event: str = "guard.decision",
        extra: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record a :class:`~aisafepy.core.decisions.GuardDecision`.

        Evidence and rationale are passed through
        :func:`aisafepy.core.telemetry.redact` so secrets captured by a
        guard never land in the audit trail.
        """
        from aisafepy.core.telemetry import redact

        payload: dict[str, Any] = {
            "decision_id": decision.decision_id,
            "action": decision.action.value,
            "confidence": decision.confidence,
            "tier": decision.tier,
            "severity": decision.severity,
            "guard_name": decision.guard_name,
            "rationale": redact(decision.rationale),
            "evidence": redact(decision.evidence),
            "latency_ms": decision.latency_ms,
        }
        if extra:
            payload["extra"] = redact(extra)
        return self.append(event, actor=actor, payload=payload)

    # ---- reading -------------------------------------------------------

    def iter_entries(self) -> Iterator[AuditEntry]:
        """Stream entries without loading the whole file."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                yield self._entry_from_obj(obj)

    def entries(self) -> list[AuditEntry]:
        return list(self.iter_entries())

    def tail(self, n: int = 10) -> list[AuditEntry]:
        """Return the last ``n`` entries (oldest first)."""
        if n <= 0:
            return []
        out: list[AuditEntry] = []
        for e in self.iter_entries():
            out.append(e)
            if len(out) > n:
                del out[0]
        return out

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_entries())

    # ---- verification ---------------------------------------------------

    def verify(self) -> bool:
        return self.verify_detailed().ok

    def verify_detailed(self) -> VerifyResult:
        """Walk the file and report the first break in the chain, if any."""
        prev: str | None = None
        idx = -1
        if not self.path.exists():
            return VerifyResult(ok=True, entries_checked=0)
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                idx += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    return VerifyResult(False, idx, idx, "malformed JSON line")
                claimed_hash = obj.pop("__hash__", None)
                claimed_sig = obj.pop("__sig__", None)
                try:
                    entry = AuditEntry(
                        id=obj["id"],
                        event=obj["event"],
                        timestamp=obj["timestamp"],
                        actor=obj["actor"],
                        payload=obj["payload"],
                        prev_hash=obj.get("prev_hash"),
                    )
                except (KeyError, TypeError) as exc:
                    return VerifyResult(False, idx, idx, f"missing field: {exc}")
                if entry.prev_hash != prev:
                    return VerifyResult(False, idx, idx, "prev_hash does not match previous entry")
                if entry.compute_hash() != claimed_hash:
                    return VerifyResult(False, idx, idx, "entry hash mismatch (content altered)")
                if self.hmac_key is not None:
                    if claimed_sig is None:
                        return VerifyResult(False, idx, idx, "missing HMAC signature")
                    if not hmac.compare_digest(self._sign(claimed_hash), str(claimed_sig)):
                        return VerifyResult(False, idx, idx, "HMAC signature mismatch")
                prev = claimed_hash
        return VerifyResult(ok=True, entries_checked=idx + 1)

    # ---- export --------------------------------------------------------

    def export_csv(self, path: str | Path) -> Path:
        """Write ``id,event,timestamp,actor,prev_hash,payload`` rows (payload as JSON)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "event", "timestamp", "actor", "prev_hash", "payload"])
            for e in self.iter_entries():
                w.writerow(
                    [
                        e.id,
                        e.event,
                        repr(e.timestamp),
                        e.actor,
                        e.prev_hash or "",
                        json.dumps(e.payload, sort_keys=True, default=str),
                    ]
                )
        return path

    # ---- internals -----------------------------------------------------

    def _sign(self, h: str) -> str:
        assert self.hmac_key is not None
        return hmac.new(self.hmac_key, h.encode("utf-8"), hashlib.sha256).hexdigest()

    @staticmethod
    def _entry_from_obj(obj: dict[str, Any]) -> AuditEntry:
        obj = dict(obj)
        obj.pop("__hash__", None)
        obj.pop("__sig__", None)
        return AuditEntry(**obj)

    def _read_last_hash_from_tail(self) -> str | None:
        """Return the ``__hash__`` of the last non-empty line.

        Reads only the last :data:`_TAIL_BYTES` of the file; audit
        lines are small so this always contains at least one full line
        unless the file is empty.
        """
        if not self.path.exists():
            return None
        with self.path.open("rb") as fb:
            return self._last_hash_from_binary(fb)

    @staticmethod
    def _last_hash_from_binary(fb: IO[bytes]) -> str | None:
        fb.seek(0, 2)
        size = fb.tell()
        if size == 0:
            return None
        fb.seek(max(0, size - _TAIL_BYTES))
        chunk = fb.read().decode("utf-8", errors="replace")
        for line in reversed(chunk.splitlines()):
            if line.strip():
                try:
                    return json.loads(line).get("__hash__")
                except json.JSONDecodeError:
                    return None
        return None
