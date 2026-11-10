"""Decision sinks: where ``GuardDecision`` objects go after they are made.

OpenTelemetry spans (``aisafepy.core.telemetry``) are the primary
observability channel, but many deployments also want a plain record
stream: a JSONL file for offline analysis, an in-memory ring buffer
for tests, or a callback into an existing event bus. A
:class:`DecisionSink` is that stream.

All sinks go through :func:`decision_to_record`, which produces a
stable, versioned, redacted dict. ``schema_version`` is bumped only
when a field changes meaning; new optional fields do not bump it.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from aisafepy.core.decisions import Action, GuardDecision
from aisafepy.core.telemetry import redact

RECORD_SCHEMA_VERSION = 1

_SEVERITY_ORDER: tuple[str, ...] = ("info", "low", "medium", "high", "critical")


def severity_rank(severity: str) -> int:
    """Numeric rank for a severity label (``info`` = 0 .. ``critical`` = 4).

    Unknown labels rank as ``medium`` so a typo never silently drops
    an event below a filter threshold.
    """
    try:
        return _SEVERITY_ORDER.index(severity)
    except ValueError:
        return _SEVERITY_ORDER.index("medium")


def decision_to_record(
    decision: GuardDecision,
    *,
    redact_fields: bool = True,
    extra: dict[str, Any] | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Render a decision as a JSON-safe dict with a stable schema.

    ``evidence``, ``rationale`` and ``extra`` are passed through
    :func:`aisafepy.core.telemetry.redact` unless ``redact_fields`` is
    False (only do that for in-process sinks that never persist).
    """
    rationale: Any = decision.rationale
    evidence: Any = dict(decision.evidence)
    extra_out: Any = dict(extra) if extra else {}
    if redact_fields:
        rationale = redact(rationale)
        evidence = redact(evidence)
        extra_out = redact(extra_out)
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "timestamp": time.time() if timestamp is None else timestamp,
        "decision_id": decision.decision_id,
        "action": decision.action.value,
        "confidence": decision.confidence,
        "tier": decision.tier,
        "severity": decision.severity,
        "guard_name": decision.guard_name,
        "rationale": rationale,
        "evidence": evidence,
        "latency_ms": decision.latency_ms,
        "fallback": decision.fallback,
        "transformed": decision.transformed_content is not None,
        "extra": extra_out,
    }


def record_to_decision(record: dict[str, Any]) -> GuardDecision:
    """Inverse of :func:`decision_to_record` (lossy: redacted values stay redacted).

    ``transformed_content`` is never stored in a record, so a
    ``transform`` decision round-trips with ``transformed_content=None``.
    """
    version = int(record.get("schema_version", 1))
    if version > RECORD_SCHEMA_VERSION:
        raise ValueError(f"unsupported decision record schema_version {version}")
    evidence = record.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {"value": evidence}
    return GuardDecision(
        action=Action(record["action"]),
        confidence=float(record.get("confidence", 1.0)),
        tier=int(record.get("tier", 0)),
        rationale=str(record.get("rationale", "")),
        evidence=evidence,
        latency_ms=float(record.get("latency_ms", 0.0)),
        severity=record.get("severity", "medium"),
        fallback=record.get("fallback"),
        decision_id=str(record.get("decision_id") or GuardDecision.allow().decision_id),
        guard_name=record.get("guard_name"),
    )


@runtime_checkable
class DecisionSink(Protocol):
    """Anything that accepts decisions."""

    def emit(self, decision: GuardDecision, context: dict[str, Any] | None = None) -> None:
        """Deliver one decision. Must not raise for ordinary inputs."""
        ...  # pragma: no cover - protocol


class InMemorySink:
    """Bounded ring buffer of records. Handy for tests and dashboards."""

    def __init__(self, maxlen: int = 1_000, *, redact_fields: bool = False):
        self.records: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.redact_fields = redact_fields
        self._lock = threading.Lock()

    def emit(self, decision: GuardDecision, context: dict[str, Any] | None = None) -> None:
        rec = decision_to_record(decision, redact_fields=self.redact_fields, extra=context)
        with self._lock:
            self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(list(self.records))

    def actions(self) -> dict[str, int]:
        """Count of records per action value."""
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r["action"]] = counts.get(r["action"], 0) + 1
        return counts

    def clear(self) -> None:
        with self._lock:
            self.records.clear()


class JSONLSink:
    """Append one JSON object per decision to a file.

    Writes are serialized with a lock and flushed per record so a
    crash loses at most the record being written. The file is opened
    per emit; for very high rates wrap this sink in your own buffer.
    """

    def __init__(self, path: str | Path, *, redact_fields: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.redact_fields = redact_fields
        self._lock = threading.Lock()

    def emit(self, decision: GuardDecision, context: dict[str, Any] | None = None) -> None:
        rec = decision_to_record(decision, redact_fields=self.redact_fields, extra=context)
        line = json.dumps(rec, default=str) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def read(self) -> list[dict[str, Any]]:
        """Load every record written so far."""
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out


class CallbackSink:
    """Forward each record to a user callable. Exceptions are swallowed
    and counted on ``errors`` so a broken consumer cannot break a guard."""

    def __init__(
        self,
        fn: Callable[[dict[str, Any]], Any],
        *,
        redact_fields: bool = True,
    ):
        self.fn = fn
        self.redact_fields = redact_fields
        self.errors = 0

    def emit(self, decision: GuardDecision, context: dict[str, Any] | None = None) -> None:
        rec = decision_to_record(decision, redact_fields=self.redact_fields, extra=context)
        try:
            self.fn(rec)
        except Exception:
            self.errors += 1


class MultiSink:
    """Fan out to several sinks. One failing sink does not stop the others."""

    def __init__(self, sinks: Iterable[DecisionSink]):
        self.sinks: list[DecisionSink] = list(sinks)
        self.errors: dict[int, int] = {}

    def emit(self, decision: GuardDecision, context: dict[str, Any] | None = None) -> None:
        for i, s in enumerate(self.sinks):
            try:
                s.emit(decision, context)
            except Exception:
                self.errors[i] = self.errors.get(i, 0) + 1

    def add(self, sink: DecisionSink) -> MultiSink:
        self.sinks.append(sink)
        return self


class FilterSink:
    """Forward only decisions matching a severity floor and/or action set."""

    def __init__(
        self,
        inner: DecisionSink,
        *,
        min_severity: str = "info",
        actions: Sequence[str | Action] | None = None,
        guard_names: Sequence[str] | None = None,
    ):
        self.inner = inner
        self.min_rank = severity_rank(min_severity)
        self.actions = (
            {a.value if isinstance(a, Action) else str(a) for a in actions}
            if actions is not None
            else None
        )
        self.guard_names = set(guard_names) if guard_names is not None else None
        self.dropped = 0

    def accepts(self, decision: GuardDecision) -> bool:
        if severity_rank(decision.severity) < self.min_rank:
            return False
        if self.actions is not None and decision.action.value not in self.actions:
            return False
        return not (self.guard_names is not None and decision.guard_name not in self.guard_names)

    def emit(self, decision: GuardDecision, context: dict[str, Any] | None = None) -> None:
        if self.accepts(decision):
            self.inner.emit(decision, context)
        else:
            self.dropped += 1
