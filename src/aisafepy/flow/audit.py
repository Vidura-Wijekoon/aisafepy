"""Decision recording for the IFC interpreter.

:func:`aisafepy.flow.interpreter.evaluate_call` reports every outcome
to the ``IFCContext.on_decision`` hook. :class:`DecisionRecorder` is
the reference implementation of that hook. It keeps a bounded
in-memory ring buffer for introspection, optionally appends JSON Lines
to a file, and optionally forwards each record to a callback (a SIEM
shipper, a metrics counter, a test spy).

Evidence dictionaries are passed through
:func:`aisafepy.core.telemetry.redact` before they are stored so API
keys and similar secrets that flowed through tool arguments do not end
up in an audit file.

Usage::

    recorder = DecisionRecorder(path="flow-audit.jsonl")
    with ifc_run(policy, recorder=recorder) as ctx:
        evaluate_call(tool="send_email", args=(...), kwargs={}, policy=policy, context=ctx)
    print(recorder.summary())
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aisafepy.core.decisions import GuardDecision, IFCViolation
from aisafepy.core.telemetry import redact
from aisafepy.flow.interpreter import IFCContext
from aisafepy.flow.policy import Policy

_logger = logging.getLogger("aisafepy.flow.audit")


@dataclass(frozen=True)
class DecisionRecord:
    """One recorded interpreter outcome, flattened for logs and dashboards."""

    seq: int
    timestamp: float
    tool: str
    kind: str
    """``decision`` for a :class:`GuardDecision`, ``violation`` for an
    :class:`IFCViolation`."""
    action: str
    """``allow`` / ``block`` / ``transform`` / ``escalate`` for decisions;
    always ``block`` for violations."""
    reason: str
    provenance: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    integrity: str | None = None
    required_integrity: str | None = None
    required_capabilities: tuple[str, ...] = ()
    cfi: str | None = None
    mediated: bool = False
    ref_id: str = ""
    """``decision_id`` or ``violation_id`` of the underlying object."""
    taint_chain: tuple[tuple[str, str, str], ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["provenance"] = list(self.provenance)
        d["capabilities"] = list(self.capabilities)
        d["required_capabilities"] = list(self.required_capabilities)
        d["taint_chain"] = [list(t) for t in self.taint_chain]
        return d


class DecisionRecorder:
    """Capture interpreter outcomes to memory, a JSONL file, and/or a callback.

    Parameters
    ----------
    max_entries:
        Size of the in-memory ring buffer. Older records are evicted
        from memory (never from the file).
    path:
        Optional JSON Lines file. Records are buffered and written on
        :meth:`flush` (which :func:`aisafepy.flow.interpreter.ifc_run`
        and :meth:`recording` call automatically) or immediately when
        ``autoflush=True``.
    callback:
        Optional ``callable(record)``. Exceptions are logged, not raised.
    redact_evidence:
        Run evidence through :func:`aisafepy.core.telemetry.redact`.
    """

    def __init__(
        self,
        *,
        max_entries: int = 1000,
        path: str | Path | None = None,
        callback: Callable[[DecisionRecord], None] | None = None,
        redact_evidence: bool = True,
        autoflush: bool = False,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.path = Path(path) if path is not None else None
        self.callback = callback
        self.redact_evidence = redact_evidence
        self.autoflush = autoflush
        self._entries: deque[DecisionRecord] = deque(maxlen=max_entries)
        self._pending: list[DecisionRecord] = []
        self._seq = itertools.count(1)
        self._lock = threading.Lock()
        self._total = 0

    # ---- hook ------------------------------------------------------------

    def __call__(self, outcome: GuardDecision | IFCViolation, tool: str) -> None:
        """The ``IFCContext.on_decision`` entry point."""
        record = self._to_record(outcome, tool)
        with self._lock:
            self._entries.append(record)
            self._total += 1
            if self.path is not None:
                self._pending.append(record)
        if self.callback is not None:
            try:
                self.callback(record)
            except Exception as exc:
                _logger.warning("DecisionRecorder callback raised: %r", exc)
        if self.autoflush:
            self.flush()

    record = __call__

    def _to_record(self, outcome: GuardDecision | IFCViolation, tool: str) -> DecisionRecord:
        seq = next(self._seq)
        now = time.time()
        if isinstance(outcome, IFCViolation):
            return DecisionRecord(
                seq=seq,
                timestamp=now,
                tool=tool or outcome.tool,
                kind="violation",
                action="block",
                reason=outcome.reason,
                provenance=tuple(sorted(outcome.provenance)),
                capabilities=tuple(sorted(outcome.capabilities)),
                integrity=outcome.integrity,
                required_integrity=outcome.required_integrity,
                required_capabilities=tuple(sorted(outcome.required_capabilities)),
                ref_id=outcome.violation_id,
                taint_chain=tuple(outcome.taint_chain),
            )
        evidence = outcome.evidence or {}
        if self.redact_evidence:
            evidence = redact(evidence)
        return DecisionRecord(
            seq=seq,
            timestamp=now,
            tool=tool,
            kind="decision",
            action=outcome.action.value,
            reason=outcome.rationale,
            provenance=tuple(evidence.get("provenance", ()) or ()),
            capabilities=tuple(evidence.get("capabilities", ()) or ()),
            integrity=evidence.get("integrity"),
            cfi=evidence.get("cfi"),
            mediated=evidence.get("mediation") == "approved",
            ref_id=outcome.decision_id,
            evidence=dict(evidence),
        )

    # ---- persistence -----------------------------------------------------

    def flush(self) -> int:
        """Write pending records to ``path``. Returns the number written."""
        with self._lock:
            pending, self._pending = self._pending, []
        if not pending or self.path is None:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for rec in pending:
                f.write(json.dumps(rec.to_dict(), default=str) + "\n")
        return len(pending)

    @staticmethod
    def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
        """Load records previously written by :meth:`flush`."""
        out: list[dict[str, Any]] = []
        p = Path(path)
        if not p.exists():
            return out
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    # ---- introspection ---------------------------------------------------

    @property
    def entries(self) -> list[DecisionRecord]:
        with self._lock:
            return list(self._entries)

    @property
    def total(self) -> int:
        """Records seen over the recorder's lifetime, including evicted ones."""
        return self._total

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._pending.clear()

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.entries]

    def violations(self) -> list[DecisionRecord]:
        return [r for r in self.entries if r.kind == "violation"]

    def summary(self) -> dict[str, Any]:
        """Aggregate counts over the in-memory buffer.

        Returns ``{"total", "by_action", "by_tool", "top_deny_reasons",
        "mediated_approvals"}`` where ``by_tool`` maps tool name to an
        action-count mapping and ``top_deny_reasons`` is a list of
        ``[reason, count]`` pairs, most frequent first.
        """
        entries = self.entries
        by_action: Counter[str] = Counter()
        by_tool: dict[str, Counter[str]] = {}
        deny_reasons: Counter[str] = Counter()
        mediated = 0
        for r in entries:
            key = "violation" if r.kind == "violation" else r.action
            by_action[key] += 1
            by_tool.setdefault(r.tool, Counter())[key] += 1
            if r.kind == "violation":
                deny_reasons[r.reason] += 1
            if r.mediated:
                mediated += 1
        return {
            "total": len(entries),
            "lifetime_total": self._total,
            "by_action": dict(by_action),
            "by_tool": {t: dict(c) for t, c in by_tool.items()},
            "top_deny_reasons": [[reason, n] for reason, n in deny_reasons.most_common(10)],
            "mediated_approvals": mediated,
        }

    # ---- installation ----------------------------------------------------

    @contextlib.contextmanager
    def recording(self, target: IFCContext | Policy | None = None) -> Iterator[IFCContext]:
        """Install this recorder for the duration of a block.

        Pass an existing :class:`IFCContext` to attach to it (the
        previous ``on_decision`` hook is restored on exit), a
        :class:`Policy` or ``None`` to get a fresh context. The block
        receives the context; pending file writes are flushed on exit.
        """
        if isinstance(target, IFCContext):
            ctx = target
            previous = ctx.on_decision
            ctx.on_decision = self
            try:
                yield ctx
            finally:
                ctx.on_decision = previous
                self.flush()
            return
        ctx = IFCContext(on_decision=self)
        try:
            yield ctx
        finally:
            self.flush()
