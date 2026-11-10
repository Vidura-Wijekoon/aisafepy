"""Offline evaluation of guards and pipelines against labeled records.

Red-team sources give us labeled data for free: every
:class:`~aisafepy.adapt.sources.FailureRecord` carries
``was_violation``. This module runs a guard (or a whole
:class:`~aisafepy.stream.GuardPipeline`) over those records and
reports the usual detection metrics, a per-attack-category breakdown,
latency statistics, and the ids of misclassified records so an
operator can look at exactly what slipped through or was wrongly
blocked.

The same machinery supports threshold selection: sweep a guard
factory over candidate thresholds and pick the most permissive
threshold that still meets a false-positive ceiling.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from aisafepy.adapt.sources import FailureRecord
from aisafepy.core.decisions import Action, GuardDecision

Field = Literal["input", "output", "both"]
_DEFAULT_FLAG_ACTIONS: tuple[str, ...] = ("block", "escalate")


@dataclass
class Metrics:
    """A binary confusion matrix with derived rates.

    The positive class is "violation". ``tp`` means the guard flagged
    a real violation; ``fp`` means it flagged a benign record.
    """

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def support(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        return self.tn + self.fp

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def fnr(self) -> float:
        denom = self.fn + self.tp
        return self.fn / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.support if self.support else 0.0

    def add(self, *, label: bool, flagged: bool) -> None:
        if label and flagged:
            self.tp += 1
        elif label and not flagged:
            self.fn += 1
        elif not label and flagged:
            self.fp += 1
        else:
            self.tn += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "support": self.support,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "accuracy": self.accuracy,
        }


@dataclass
class LatencyStats:
    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> LatencyStats:
        if not samples:
            return cls()
        ordered = sorted(samples)
        return cls(
            count=len(ordered),
            mean_ms=statistics.fmean(ordered),
            p50_ms=_percentile(ordered, 0.50),
            p95_ms=_percentile(ordered, 0.95),
            max_ms=ordered[-1],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
        }


@dataclass
class EvaluationReport:
    name: str
    metrics: Metrics
    per_category: dict[str, Metrics] = field(default_factory=dict)
    latency: LatencyStats = field(default_factory=LatencyStats)
    false_positives: list[str] = field(default_factory=list)
    false_negatives: list[str] = field(default_factory=list)
    n_records: int = 0
    field: str = "input"
    threshold: float | None = None
    errors: int = 0
    """Number of records on which the guard raised. They count as not flagged."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_records": self.n_records,
            "field": self.field,
            "threshold": self.threshold,
            "errors": self.errors,
            "metrics": self.metrics.to_dict(),
            "per_category": {k: v.to_dict() for k, v in self.per_category.items()},
            "latency": self.latency.to_dict(),
            "false_positives": list(self.false_positives),
            "false_negatives": list(self.false_negatives),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        m = self.metrics
        lines = [
            f"# Evaluation: {self.name}",
            "",
            f"- Records: {self.n_records} ({m.positives} violations, {m.negatives} benign)",
            f"- Field: `{self.field}`" + (f", threshold: {self.threshold}" if self.threshold is not None else ""),
            f"- Precision: {m.precision:.3f}  Recall: {m.recall:.3f}  F1: {m.f1:.3f}",
            f"- FPR: {m.fpr:.4f}  FNR: {m.fnr:.4f}  Accuracy: {m.accuracy:.3f}",
            f"- Latency: p50 {self.latency.p50_ms:.2f} ms, p95 {self.latency.p95_ms:.2f} ms, "
            f"max {self.latency.max_ms:.2f} ms",
        ]
        if self.errors:
            lines.append(f"- Guard errors: {self.errors} (counted as not flagged)")
        lines += [
            "",
            "| | flagged | not flagged |",
            "|---|---:|---:|",
            f"| violation | {m.tp} | {m.fn} |",
            f"| benign | {m.fp} | {m.tn} |",
        ]
        if self.per_category:
            lines += [
                "",
                "## By attack category",
                "",
                "| category | n | precision | recall | fpr |",
                "|---|---:|---:|---:|---:|",
            ]
            for cat, cm in sorted(self.per_category.items()):
                lines.append(
                    f"| {cat} | {cm.support} | {cm.precision:.3f} | {cm.recall:.3f} | {cm.fpr:.4f} |"
                )
        if self.false_negatives:
            lines += ["", f"Missed violations (first {len(self.false_negatives)}): "
                      + ", ".join(f"`{i}`" for i in self.false_negatives)]
        if self.false_positives:
            lines += ["", f"Wrongly flagged benign (first {len(self.false_positives)}): "
                      + ", ".join(f"`{i}`" for i in self.false_positives)]
        return "\n".join(lines) + "\n"


def unsafe_score(decision: GuardDecision) -> float:
    """Map a decision onto a 0..1 "how unsafe" scale.

    Terminal decisions report ``confidence`` as their certainty that
    the content is unsafe; allow decisions report certainty that it is
    safe, so the unsafe score is the complement.
    """
    if decision.action in (Action.BLOCK, Action.ESCALATE, Action.TRANSFORM):
        return float(decision.confidence)
    return 1.0 - float(decision.confidence)


def _record_text(rec: FailureRecord, field_: Field) -> str:
    if field_ == "input":
        return rec.input
    if field_ == "output":
        return rec.output
    return rec.input + "\n" + rec.output


def _run_coroutine(coro: Any) -> Any:
    """Drive a coroutine to completion from sync code, inside or outside a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _invoke(guard: Any, ctx: Any) -> GuardDecision:
    if hasattr(guard, "evaluate_chunk"):
        return _run_coroutine(guard.evaluate_chunk(ctx))
    res = guard(ctx)
    if asyncio.iscoroutine(res):
        return _run_coroutine(res)
    return res


def evaluate_guard(
    guard: Any,
    records: Iterable[FailureRecord],
    *,
    field: Field = "input",
    threshold: float | None = None,
    flag_actions: Sequence[str] = _DEFAULT_FLAG_ACTIONS,
    max_misclassified: int = 50,
    name: str | None = None,
    role: str = "user",
) -> EvaluationReport:
    """Evaluate ``guard`` (a Guard or a GuardPipeline) over labeled records.

    Parameters
    ----------
    field:
        Which text the guard sees: the attack ``input``, the model
        ``output``, or ``both`` concatenated.
    threshold:
        When given, a record is flagged if :func:`unsafe_score` of the
        decision is at least ``threshold``; otherwise a record is
        flagged when the decision's action is in ``flag_actions``.
    flag_actions:
        Actions that count as "flagged" (default block and escalate).
    max_misclassified:
        Cap on the number of FP / FN ids retained in the report.
    """
    from aisafepy.stream.pipeline import Context

    metrics = Metrics()
    per_cat: dict[str, Metrics] = {}
    latencies: list[float] = []
    fps: list[str] = []
    fns: list[str] = []
    n = 0
    errors = 0
    flag_set = {a.lower() for a in flag_actions}

    for rec in records:
        n += 1
        ctx = Context(buffer=_record_text(rec, field), prompt=rec.input, role=role)  # type: ignore[arg-type]
        start = time.perf_counter()
        try:
            decision = _invoke(guard, ctx)
            flagged = (
                unsafe_score(decision) >= threshold
                if threshold is not None
                else decision.action.value in flag_set
            )
        except Exception:
            errors += 1
            flagged = False
        latencies.append((time.perf_counter() - start) * 1000.0)
        label = bool(rec.was_violation)
        metrics.add(label=label, flagged=flagged)
        cat = rec.attack_category or "uncategorized"
        per_cat.setdefault(cat, Metrics()).add(label=label, flagged=flagged)
        if label and not flagged and len(fns) < max_misclassified:
            fns.append(rec.id)
        if not label and flagged and len(fps) < max_misclassified:
            fps.append(rec.id)

    return EvaluationReport(
        name=name or getattr(guard, "name", type(guard).__name__),
        metrics=metrics,
        per_category=per_cat,
        latency=LatencyStats.from_samples(latencies),
        false_positives=fps,
        false_negatives=fns,
        n_records=n,
        field=field,
        threshold=threshold,
        errors=errors,
    )


def threshold_sweep(
    guard_factory: Callable[[float], Any],
    records: Iterable[FailureRecord],
    thresholds: Iterable[float],
    *,
    field: Field = "input",
    use_score: bool = True,
    **kwargs: Any,
) -> list[tuple[float, Metrics]]:
    """Evaluate ``guard_factory(t)`` for every threshold ``t``.

    With ``use_score=True`` the decision's :func:`unsafe_score` is
    compared against ``t`` as well, which makes the sweep meaningful
    even for guards that ignore their threshold argument.
    """
    recs = list(records)
    out: list[tuple[float, Metrics]] = []
    for t in thresholds:
        guard = guard_factory(t)
        rep = evaluate_guard(
            guard, recs, field=field, threshold=(t if use_score else None), **kwargs
        )
        out.append((float(t), rep.metrics))
    return out


def best_threshold(
    sweep: Sequence[tuple[float, Metrics]],
    *,
    max_fpr: float = 0.01,
) -> tuple[float, Metrics] | None:
    """Pick the threshold with the highest recall whose FPR is within ``max_fpr``.

    Ties are broken by F1 and then by the lower threshold (more
    permissive, fewer false blocks on unseen data). Returns ``None``
    when no threshold meets the FPR ceiling.
    """
    eligible = [(t, m) for t, m in sweep if m.fpr <= max_fpr]
    if not eligible:
        return None
    eligible.sort(key=lambda tm: (tm[1].recall, tm[1].f1, -tm[0]), reverse=True)
    return eligible[0]


def _percentile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)
