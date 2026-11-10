"""Shared helpers for the offline AIsafePy benchmarks.

These benchmarks are plain scripts, not pytest tests. They load the
labelled dataset, time guards, compute detection metrics, and print an
ASCII report. Every script accepts ``--quick`` (tiny sizes, used by the
smoke test) and ``--json`` (machine-readable output).

Nothing here imports torch, transformers, or an agent framework: the
offline suite runs on a laptop with only the core install.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATASET = Path(__file__).resolve().parent / "datasets" / "injection_eval.jsonl"

_VALID_LABELS = {"attack", "benign"}


@dataclass(frozen=True)
class Record:
    """One labelled evaluation record."""

    id: str
    text: str
    label: str
    family: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_attack(self) -> bool:
        return self.label == "attack"


def load_dataset(path: str | Path = DATASET) -> list[Record]:
    """Load and validate ``injection_eval.jsonl``.

    Raises ``ValueError`` on a missing field, an unknown label, or a
    duplicate id so a corrupted dataset fails loudly rather than
    skewing a benchmark.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {p}")
    records: list[Record] = []
    seen: set[str] = set()
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        for key in ("id", "text", "label", "family"):
            if key not in obj:
                raise ValueError(f"{p}:{lineno}: missing required key {key!r}")
        if obj["label"] not in _VALID_LABELS:
            raise ValueError(f"{p}:{lineno}: label must be in {_VALID_LABELS}")
        if obj["id"] in seen:
            raise ValueError(f"{p}:{lineno}: duplicate id {obj['id']!r}")
        seen.add(obj["id"])
        known = {"id", "text", "label", "family", "notes"}
        records.append(
            Record(
                id=obj["id"],
                text=obj["text"],
                label=obj["label"],
                family=obj["family"],
                notes=obj.get("notes", ""),
                extra={k: v for k, v in obj.items() if k not in known},
            )
        )
    return records


# ---- timing ----------------------------------------------------------------


@dataclass
class Timing:
    label: str
    count: int
    repeats: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_per_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "repeats": self.repeats,
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "mean_ms": round(self.mean_ms, 4),
            "throughput_per_s": round(self.throughput_per_s, 1),
        }


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = round((pct / 100.0) * (len(ordered) - 1))
    return ordered[max(0, min(k, len(ordered) - 1))]


def run_guard(guard: Any, text: str) -> Any:
    """Call an async guard once from synchronous code (one event loop per call)."""
    from aisafepy.stream.pipeline import Context

    return asyncio.run(guard(Context(buffer=text)))


def time_guard(guard: Any, texts: Sequence[str], *, repeats: int = 1, label: str | None = None) -> Timing:
    """Measure per-call latency of ``guard`` over ``texts``.

    One asyncio loop is created per call so the numbers reflect guard
    work, not loop reuse tricks a real caller would not get.
    """
    from aisafepy.stream.pipeline import Context

    name = label or getattr(guard, "name", type(guard).__name__)
    samples: list[float] = []
    started = time.perf_counter()
    for _ in range(repeats):
        for text in texts:
            ctx = Context(buffer=text)
            t0 = time.perf_counter()
            asyncio.run(guard(ctx))
            samples.append((time.perf_counter() - t0) * 1000.0)
    elapsed = time.perf_counter() - started
    n = len(samples)
    return Timing(
        label=name,
        count=len(texts),
        repeats=repeats,
        p50_ms=_percentile(samples, 50),
        p95_ms=_percentile(samples, 95),
        p99_ms=_percentile(samples, 99),
        mean_ms=statistics.fmean(samples) if samples else 0.0,
        throughput_per_s=(n / elapsed) if elapsed > 0 else 0.0,
    )


# ---- detection metrics -----------------------------------------------------


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fpr(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 4),
            "accuracy": round(self.accuracy, 4),
        }


def confusion(preds: Iterable[bool], labels: Iterable[bool]) -> Confusion:
    """Build a confusion matrix from predicted / true "is attack" booleans."""
    c = Confusion()
    for pred, truth in zip(preds, labels, strict=True):
        if pred and truth:
            c.tp += 1
        elif pred and not truth:
            c.fp += 1
        elif not pred and truth:
            c.fn += 1
        else:
            c.tn += 1
    return c


# ---- rendering -------------------------------------------------------------


def render_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Minimal ASCII table (kept separate from the CLI so benchmarks stand alone)."""
    materialised = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in materialised:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  "
    out = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    out.append(sep.join("-" * w for w in widths))
    out.extend(sep.join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in materialised)
    return "\n".join(out)


def print_report(title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"title": title, "columns": list(headers), "rows": [list(r) for r in rows]}, default=str))
    else:
        print(title)
        print(render_table(headers, rows))


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the two flags every benchmark shares."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p.add_argument("--quick", action="store_true", help="tiny sizes for smoke testing")
    return p
