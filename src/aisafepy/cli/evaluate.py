"""``aisafepy eval``: measure a guard or pipeline against a labelled dataset.

The dataset is JSON Lines with one record per line. Field names are
configurable; the defaults match ``benchmarks/datasets/injection_eval.jsonl``
(``text``, ``label`` with ``attack`` as the positive label, ``family``).
Records are converted to :class:`FailureRecord` objects and scored with
:func:`aisafepy.adapt.evaluate.evaluate_guard`, so the numbers here are
the same ones the library computes programmatically.

The guard comes from ``--guard module:factory[:kwargs-json]`` (a class,
factory function, or ready-made guard instance) or from
``--pipeline CONFIG.yaml``. ``--threshold-sweep`` re-instantiates the
factory with each ``threshold`` and reports the operating curve.
``--min-recall`` and ``--max-fpr`` turn the command into a CI gate.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from aisafepy.adapt.sources import FailureRecord
from aisafepy.cli import _io
from aisafepy.cli._io import EXIT_FLAGGED, EXIT_OK, CLIError


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "eval",
        help="evaluate a guard or pipeline against a labelled JSONL dataset",
        description=(
            "Compute precision, recall, F1 and false-positive rate for a guard over a "
            "labelled dataset, with a per-family breakdown and optional threshold sweep."
        ),
    )
    p.add_argument("dataset", metavar="DATASET.jsonl")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--guard",
        metavar="module:factory[:kwargs-json]",
        help="guard class, factory, or instance to evaluate",
    )
    target.add_argument("--pipeline", metavar="CONFIG.yaml", help="evaluate a YAML pipeline")
    p.add_argument("--text-field", default="text", help="record field holding the text")
    p.add_argument("--label-field", default="label", help="record field holding the label")
    p.add_argument("--positive-label", default="attack", help="label value that means unsafe")
    p.add_argument("--family-field", default="family", help="record field for the breakdown")
    p.add_argument("--id-field", default="id", help="record field used as the record id")
    p.add_argument(
        "--role",
        choices=["user", "assistant", "tool"],
        default="user",
        help="Context.role used for each evaluation (default: user)",
    )
    p.add_argument(
        "--flag-actions",
        default="block,escalate",
        help="decision actions counted as a positive prediction (default: block,escalate)",
    )
    p.add_argument(
        "--threshold-sweep",
        metavar="T[,T...]",
        help="re-instantiate the factory with each threshold and report the curve",
    )
    p.add_argument("--min-recall", type=float, help="exit 2 if recall is below this")
    p.add_argument("--max-fpr", type=float, help="exit 2 if the false-positive rate exceeds this")
    p.add_argument("--max-misclassified", type=int, default=20, help="ids to list per class")
    p.set_defaults(handler=run_eval)


# ---- loading ----------------------------------------------------------------


def load_records(args: argparse.Namespace) -> list[FailureRecord]:
    records: list[FailureRecord] = []
    for i, raw in enumerate(_io.iter_jsonl(args.dataset, stdin=getattr(args, "stdin", None))):
        text = raw.get(args.text_field)
        if not isinstance(text, str):
            raise CLIError(
                f"{args.dataset}: record {i} has no string field {args.text_field!r}"
            )
        if args.label_field not in raw:
            raise CLIError(f"{args.dataset}: record {i} has no field {args.label_field!r}")
        label = raw[args.label_field]
        positive = str(label).lower() == str(args.positive_label).lower() or label is True
        family = raw.get(args.family_field)
        records.append(
            FailureRecord(
                id=str(raw.get(args.id_field, f"rec-{i}")),
                input=text,
                output="",
                was_violation=positive,
                attack_category=str(family) if family is not None else None,
                source=Path(args.dataset).name,
                metadata={k: v for k, v in raw.items() if k != args.text_field},
            )
        )
    if not records:
        raise CLIError(f"{args.dataset}: no records")
    return records


def resolve_factory(spec: str) -> tuple[Callable[..., Any], dict[str, Any], str]:
    """Return ``(callable_or_instance, kwargs, display_name)`` for a guard spec."""
    parts = spec.split(":", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise CLIError(f"--guard must look like module:factory[:kwargs-json], got {spec!r}")
    module_path, attr = parts[0], parts[1]
    kwargs: dict[str, Any] = {}
    if len(parts) == 3 and parts[2].strip():
        try:
            kwargs = json.loads(parts[2])
        except json.JSONDecodeError as exc:
            raise CLIError(f"--guard kwargs are not valid JSON: {exc.msg}") from exc
        if not isinstance(kwargs, dict):
            raise CLIError("--guard kwargs must be a JSON object")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise CLIError(f"cannot import {module_path!r}: {exc}") from exc
    obj: Any = module
    for part in attr.split("."):
        if not hasattr(obj, part):
            raise CLIError(f"module {module_path!r} has no attribute {attr!r}")
        obj = getattr(obj, part)
    return obj, kwargs, f"{module_path}:{attr}"


def build_guard(obj: Any, kwargs: dict[str, Any], **extra: Any) -> Any:
    """Instantiate ``obj`` if it is a class or factory; pass instances through."""
    is_instance = hasattr(obj, "tier") and not isinstance(obj, type)
    if is_instance and not extra:
        return obj
    if not callable(obj):
        raise CLIError(f"{obj!r} is neither callable nor a guard instance")
    try:
        return obj(**{**kwargs, **extra})
    except TypeError as exc:
        raise CLIError(f"could not instantiate guard: {exc}") from exc


def _parse_thresholds(value: str | None) -> list[float]:
    if not value:
        return []
    out: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(float(item))
        except ValueError as exc:
            raise CLIError(f"bad threshold {item!r}") from exc
    return out


# ---- command ------------------------------------------------------------------


def run_eval(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    from aisafepy.adapt.evaluate import evaluate_guard, threshold_sweep

    records = load_records(args)
    flag_actions = tuple(a.strip() for a in args.flag_actions.split(",") if a.strip())

    if args.pipeline:
        from aisafepy.cli.pipeline import load_pipeline_or_die

        guard: Any = load_pipeline_or_die(args.pipeline)
        factory: Callable[..., Any] | None = None
        kwargs: dict[str, Any] = {}
        name = f"pipeline:{Path(args.pipeline).name}"
    else:
        factory, kwargs, name = resolve_factory(args.guard)
        guard = build_guard(factory, kwargs)

    report = evaluate_guard(
        guard,
        records,
        field="input",
        flag_actions=flag_actions,
        max_misclassified=max(0, args.max_misclassified),
        name=name,
        role=args.role,
    )

    sweep: list[tuple[float, Any]] = []
    thresholds = _parse_thresholds(args.threshold_sweep)
    if thresholds:
        if factory is None:
            raise CLIError("--threshold-sweep needs --guard (a factory that takes threshold=)")

        def make(t: float) -> Any:
            return build_guard(factory, kwargs, threshold=t)

        sweep = threshold_sweep(make, records, thresholds, field="input", use_score=False)

    m = report.metrics
    gate_failures: list[str] = []
    if args.min_recall is not None and m.recall < args.min_recall:
        gate_failures.append(f"recall {m.recall:.3f} < min {args.min_recall:.3f}")
    if args.max_fpr is not None and m.fpr > args.max_fpr:
        gate_failures.append(f"fpr {m.fpr:.3f} > max {args.max_fpr:.3f}")

    if args.json:
        payload = report.to_dict()
        payload["dataset"] = args.dataset
        payload["gates"] = {"ok": not gate_failures, "failures": gate_failures}
        if sweep:
            payload["sweep"] = [{"threshold": t, **mm.to_dict()} for t, mm in sweep]
        _io.dump_json(out, payload)
        return EXIT_FLAGGED if gate_failures else EXIT_OK

    _io.emit(out, f"{name} over {_io.human_count('record', report.n_records)} from {args.dataset}")
    _io.emit(
        out,
        f"precision={m.precision:.3f} recall={m.recall:.3f} f1={m.f1:.3f} "
        f"fpr={m.fpr:.3f} accuracy={m.accuracy:.3f} "
        f"(tp={m.tp} fp={m.fp} tn={m.tn} fn={m.fn}, errors={report.errors})",
    )
    lat = report.latency
    _io.emit(out, f"latency ms: mean={lat.mean_ms:.2f} max={lat.max_ms:.2f} over {lat.count}")
    if report.per_category:
        _io.emit(out)
        rows = []
        for cat, cm in sorted(report.per_category.items()):
            rows.append([cat, cm.support, f"{cm.recall:.2f}", f"{cm.fpr:.2f}", cm.tp, cm.fp, cm.fn])
        _io.emit(out, _io.render_table(["family", "n", "recall", "fpr", "tp", "fp", "fn"], rows))
    if sweep:
        _io.emit(out)
        rows = [
            [f"{t:g}", f"{mm.precision:.3f}", f"{mm.recall:.3f}", f"{mm.f1:.3f}", f"{mm.fpr:.3f}"]
            for t, mm in sweep
        ]
        _io.emit(out, _io.render_table(["threshold", "precision", "recall", "f1", "fpr"], rows))
    if not args.quiet and (report.false_positives or report.false_negatives):
        _io.emit(out)
        if report.false_positives:
            _io.emit(out, "false positives: " + ", ".join(report.false_positives))
        if report.false_negatives:
            _io.emit(out, "false negatives: " + ", ".join(report.false_negatives))
    if gate_failures:
        _io.emit(out)
        for failure in gate_failures:
            _io.emit(out, f"gate failed: {failure}")
    return EXIT_FLAGGED if gate_failures else EXIT_OK
