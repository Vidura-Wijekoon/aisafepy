"""``aisafepy pipeline``: work with YAML-declared guard cascades from the shell.

``validate``
    Static checks only (unknown kinds, missing required parameters,
    schema errors). Nothing heavy is instantiated. Exit 2 on problems so
    a CI step can gate on the config.
``describe``
    Builds the pipeline and prints :meth:`GuardPipeline.describe`.
``run``
    Builds the pipeline, evaluates one input, prints the decision with
    its rationale and evidence, and exits 2 when the decision is a
    block or an escalation.
``registry``
    Lists every guard kind the default registry knows, with the
    parameters it requires.
``bench``
    Replays an input file through the pipeline ``--repeat`` times and
    reports p50 / p95 / p99 latency per evaluation plus the pipeline's
    own :class:`PipelineMetrics` snapshot.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import IO, Any

from aisafepy.cli import _io
from aisafepy.cli._io import EXIT_FLAGGED, EXIT_OK, CLIError
from aisafepy.core.telemetry import redact


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "pipeline",
        help="validate, describe, run, or benchmark a YAML guard pipeline",
        description="Work with aisafepy.stream pipelines declared in YAML or JSON.",
    )
    sub = p.add_subparsers(dest="pipeline_command", metavar="SUBCOMMAND")
    sub.required = True

    validate = sub.add_parser("validate", help="static checks on a pipeline config")
    validate.add_argument("config", metavar="CONFIG.yaml")
    validate.set_defaults(handler=run_validate)

    describe = sub.add_parser("describe", help="build the pipeline and print its structure")
    describe.add_argument("config", metavar="CONFIG.yaml")
    describe.set_defaults(handler=run_describe)

    run = sub.add_parser("run", help="evaluate one input through the pipeline")
    run.add_argument("config", metavar="CONFIG.yaml")
    src = run.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", metavar="TEXT", help="content to evaluate")
    src.add_argument("--input-file", metavar="FILE", help="read content from FILE ('-' for stdin)")
    run.add_argument(
        "--role",
        choices=["user", "assistant", "tool"],
        default="assistant",
        help="Context.role for the evaluation (default: assistant)",
    )
    run.add_argument("--prompt", metavar="TEXT", help="the user prompt, for guards that use it")
    run.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra Context.metadata entries (repeatable)",
    )
    run.set_defaults(handler=run_run)

    registry = sub.add_parser("registry", help="list the guard kinds the config loader accepts")
    registry.set_defaults(handler=run_registry)

    bench = sub.add_parser("bench", help="measure evaluation latency over an input file")
    bench.add_argument("config", metavar="CONFIG.yaml")
    bench.add_argument(
        "--input-file",
        required=True,
        metavar="FILE",
        help="one sample per line ('-' for stdin); blank lines are skipped",
    )
    bench.add_argument("--repeat", type=int, default=5, help="passes over the file (default: 5)")
    bench.add_argument(
        "--role", choices=["user", "assistant", "tool"], default="assistant"
    )
    bench.set_defaults(handler=run_bench)


# ---- shared ---------------------------------------------------------------


def load_pipeline_or_die(path: str) -> Any:
    """Build a pipeline from ``path``, turning loader failures into one-line errors."""
    from aisafepy.stream.config import ConfigError, load_pipeline

    p = Path(path)
    if not p.exists():
        raise CLIError(f"no such config file: {p}")
    try:
        return load_pipeline(p)
    except ConfigError as exc:
        raise CLIError(f"invalid pipeline config {p}: {exc}") from exc
    except ImportError as exc:
        raise CLIError(f"a guard in {p} needs a dependency that is not installed: {exc}") from exc
    except Exception as exc:
        raise CLIError(f"could not build pipeline from {p}: {type(exc).__name__}: {exc}") from exc


def decision_payload(decision: Any) -> dict[str, Any]:
    """JSON-friendly, redacted view of a ``GuardDecision``."""
    data = decision.model_dump(mode="json")
    data["evidence"] = redact(data.get("evidence") or {})
    data["rationale"] = redact(data.get("rationale") or "")
    return data


def _flagged(decision: Any) -> bool:
    return bool(decision.is_terminal)


def _emit_decision(out: IO[str], decision: Any, *, quiet: bool) -> None:
    _io.emit(
        out,
        f"{decision.action.value} [{decision.severity}] tier={decision.tier} "
        f"guard={decision.guard_name or '-'} confidence={decision.confidence:.2f} "
        f"latency={decision.latency_ms:.1f}ms",
    )
    _io.emit(out, f"  rationale: {redact(decision.rationale)}")
    if decision.fallback and decision.is_terminal:
        _io.emit(out, f"  fallback: {decision.fallback}")
    if decision.transformed_content is not None:
        _io.emit(out, f"  transformed: {_io.truncate(decision.transformed_content, 200)}")
    if decision.evidence and not quiet:
        for key, value in sorted(redact(decision.evidence).items()):
            _io.emit(out, f"  evidence.{key}: {_io.truncate(value, 160)}")


# ---- validate ---------------------------------------------------------------


def run_validate(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    from aisafepy.stream.config import validate_config

    p = Path(args.config)
    if not p.exists():
        raise CLIError(f"no such config file: {p}")
    try:
        problems = validate_config(p)
    except Exception as exc:
        problems = [f"{type(exc).__name__}: {exc}"]

    if args.json:
        _io.dump_json(out, {"config": str(p), "ok": not problems, "problems": problems})
    elif problems:
        _io.emit(out, f"{p}: {_io.human_count('problem', len(problems))}")
        for item in problems:
            _io.emit(out, f"  - {item}")
    elif not args.quiet:
        _io.emit(out, f"{p}: ok")
    return EXIT_FLAGGED if problems else EXIT_OK


# ---- describe ---------------------------------------------------------------


def run_describe(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    pipeline = load_pipeline_or_die(args.config)
    if args.json:
        _io.dump_json(out, {"config": args.config, "pipeline": pipeline.to_dict()})
    else:
        _io.emit(out, pipeline.describe())
    return EXIT_OK


# ---- run --------------------------------------------------------------------


def run_run(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    pipeline = load_pipeline_or_die(args.config)
    text = (
        args.input
        if args.input is not None
        else _io.read_text(args.input_file, stdin=getattr(args, "stdin", None))
    )
    metadata: dict[str, Any] = dict(_io.parse_key_values(args.metadata))
    decision = pipeline.evaluate_text(text, prompt=args.prompt, role=args.role, metadata=metadata)

    if args.json:
        _io.dump_json(
            out,
            {
                "config": args.config,
                "flagged": _flagged(decision),
                "decision": decision_payload(decision),
            },
        )
    else:
        _emit_decision(out, decision, quiet=args.quiet)
    return EXIT_FLAGGED if _flagged(decision) else EXIT_OK


# ---- registry ---------------------------------------------------------------


def run_registry(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    from aisafepy.stream.config import default_registry

    reg = default_registry()
    rows = [[kind, ", ".join(reg.required_params(kind)) or "-"] for kind in sorted(reg.names())]
    if args.json:
        _io.dump_json(
            out,
            {"kinds": {kind: list(reg.required_params(kind)) for kind in sorted(reg.names())}},
        )
    else:
        _io.emit(out, _io.render_table(["kind", "required parameters"], rows))
        if not args.quiet:
            _io.emit(out, f"{_io.human_count('kind', len(rows))} registered")
    return EXIT_OK


# ---- bench ------------------------------------------------------------------


def run_bench(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    from aisafepy.stream.metrics import PipelineMetrics

    if args.repeat < 1:
        raise CLIError("--repeat must be >= 1")
    pipeline = load_pipeline_or_die(args.config)
    samples = [
        line
        for line in _io.read_text(args.input_file, stdin=getattr(args, "stdin", None)).splitlines()
        if line.strip()
    ]
    if not samples:
        raise CLIError(f"no non-empty lines in {args.input_file}")

    metrics = PipelineMetrics()
    pipeline.metrics = metrics
    latencies: list[float] = []
    actions: dict[str, int] = {}
    started = time.perf_counter()
    for _ in range(args.repeat):
        for text in samples:
            t0 = time.perf_counter()
            decision = pipeline.evaluate_text(text, role=args.role)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            actions[decision.action.value] = actions.get(decision.action.value, 0) + 1
    wall_s = time.perf_counter() - started

    summary = {
        "config": args.config,
        "samples": len(samples),
        "repeat": args.repeat,
        "evaluations": len(latencies),
        "wall_s": round(wall_s, 4),
        "throughput_per_s": round(len(latencies) / wall_s, 2) if wall_s > 0 else None,
        "latency_ms": {
            "p50": round(_io.percentile(latencies, 50), 3),
            "p95": round(_io.percentile(latencies, 95), 3),
            "p99": round(_io.percentile(latencies, 99), 3),
            "max": round(max(latencies), 3),
            "mean": round(sum(latencies) / len(latencies), 3),
        },
        "actions": actions,
        "metrics": metrics.snapshot(),
    }
    if args.json:
        _io.dump_json(out, summary)
        return EXIT_OK

    lat = summary["latency_ms"]
    _io.emit(
        out,
        f"{len(latencies)} evaluations ({len(samples)} samples x {args.repeat}) in {wall_s:.2f}s",
    )
    _io.emit(
        out,
        f"latency ms: p50={lat['p50']} p95={lat['p95']} p99={lat['p99']} "
        f"max={lat['max']} mean={lat['mean']}",
    )
    _io.emit(out, "actions: " + ", ".join(f"{k}={v}" for k, v in sorted(actions.items())))
    if not args.quiet:
        rows = []
        for name, stats in sorted(summary["metrics"].get("guards", {}).items()):
            latency = stats.get("latency", {})
            rows.append(
                [
                    name,
                    stats.get("tier"),
                    stats.get("count"),
                    stats.get("block_rate"),
                    latency.get("p95", latency.get("max", "-")),
                ]
            )
        if rows:
            _io.emit(out)
            _io.emit(out, _io.render_table(["guard", "tier", "count", "block_rate", "p95_ms"], rows))
    return EXIT_OK
