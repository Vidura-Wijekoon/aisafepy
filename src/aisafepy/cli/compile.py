"""``aisafepy compile``: turn red-team failures into guard artifacts.

A shell front end for :mod:`aisafepy.adapt`. It reads a source
(PyRIT JSONL, a Garak report, an Inspect log, or plain JSONL of
``FailureRecord`` fields), clusters the records, runs the selected
compiler targets, and writes every artifact payload to an output
directory next to a ``report.txt``.

Clustering defaults to an *offline* TF-IDF path. The library's
:func:`cluster_failures` prefers sentence-transformers and will happily
download a 400 MB embedding model on first use, which is the wrong
default for a command that may run in CI. Pass ``--online`` to opt into
the full :class:`GuardCompiler` pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any

from aisafepy.adapt.cluster import Cluster, _summarize, _tfidf_cluster_fallback
from aisafepy.adapt.compile import CompilationReport, CompiledArtifact, GuardCompiler, Target
from aisafepy.adapt.sources import FailureRecord, GarakReport, InspectLog, PyRITSource
from aisafepy.cli import _io
from aisafepy.cli._io import EXIT_OK, CLIError

SOURCES = ("pyrit-jsonl", "garak", "inspect", "jsonl")
TARGETS = ("regex", "policy", "deliberative", "classifier")

_ARTIFACT_EXT = {
    "regex": ".json",
    "policy": ".txt",
    "deliberative": ".md",
    "classifier": ".json",
    "steering": ".json",
}


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "compile",
        help="compile red-team failures (PyRIT / Garak / Inspect / JSONL) into guard artifacts",
        description=(
            "Cluster failure records and compile each cluster into regexes, policy rules, "
            "and deliberative cases. Offline by default."
        ),
    )
    p.add_argument(
        "--source",
        required=True,
        choices=SOURCES,
        help="input format",
    )
    p.add_argument("path", metavar="PATH", help="source file")
    p.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="TARGET",
        help=(
            "regex | policy[:cedar|rego] | deliberative:POLICY.md | classifier[:BASE] "
            "(repeatable; default: regex and policy:cedar)"
        ),
    )
    p.add_argument(
        "--min-asr",
        type=float,
        default=0.10,
        help="drop clusters whose attack success rate is below this (default 0.10)",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=4,
        help="minimum records per cluster (default 4)",
    )
    p.add_argument(
        "--all-records",
        action="store_true",
        help="include non-violation records from the source (default: violations only)",
    )
    p.add_argument(
        "--online",
        action="store_true",
        help="use the full GuardCompiler (sentence-transformers + HDBSCAN when installed)",
    )
    p.add_argument("-o", "--out", metavar="DIR", help="write report.txt and artifacts here")
    p.set_defaults(handler=run)


# ---- sources ----------------------------------------------------------------


def open_source(kind: str, path: str, *, only_violations: bool) -> Iterable[FailureRecord]:
    p = Path(path)
    if not p.exists():
        raise CLIError(f"no such source file: {p}")
    if kind == "pyrit-jsonl":
        return PyRITSource(jsonl_path=p, only_violations=only_violations)
    if kind == "garak":
        return GarakReport(p, only_violations=only_violations)
    if kind == "inspect":
        return InspectLog(p, only_violations=only_violations)
    if kind == "jsonl":
        return _plain_jsonl(p, only_violations=only_violations)
    raise CLIError(f"unknown source kind {kind!r}")


def _plain_jsonl(path: Path, *, only_violations: bool) -> list[FailureRecord]:
    """Plain JSONL where each line already uses ``FailureRecord`` field names."""
    records: list[FailureRecord] = []
    for i, obj in enumerate(_io.iter_jsonl(path)):
        rec = FailureRecord(
            id=str(obj.get("id", f"rec-{i}")),
            input=str(obj.get("input", "")),
            output=str(obj.get("output", "")),
            was_violation=bool(obj.get("was_violation", True)),
            attack_category=obj.get("attack_category"),
            severity=str(obj.get("severity", "medium")),
            scorer=obj.get("scorer"),
            score=obj.get("score"),
            metadata=obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {},
            timestamp=obj.get("timestamp"),
            source=str(obj.get("source", "jsonl")),
        )
        if only_violations and not rec.was_violation:
            continue
        records.append(rec)
    return records


# ---- targets ----------------------------------------------------------------


def parse_targets(specs: list[str]) -> list[Any]:
    """Turn ``["regex", "policy:rego", "deliberative:p.md"]`` into target objects."""
    if not specs:
        specs = ["regex", "policy:cedar"]
    targets: list[Any] = []
    for spec in specs:
        kind, _, arg = spec.partition(":")
        if kind == "regex":
            targets.append(Target.synthesize_regex())
        elif kind == "policy":
            dsl = arg or "cedar"
            if dsl not in ("cedar", "rego"):
                raise CLIError(f"policy DSL must be cedar or rego, got {dsl!r}")
            targets.append(Target.policy_rule(dsl=dsl))
        elif kind == "deliberative":
            if not arg:
                raise CLIError("deliberative target needs a policy file: deliberative:POLICY.md")
            targets.append(Target.deliberative_case(policy=arg))
        elif kind == "classifier":
            targets.append(Target.distill_classifier(base=arg or "distilbert-base-uncased"))
        else:
            raise CLIError(f"unknown target {kind!r}; expected one of {', '.join(TARGETS)}")
    return targets


# ---- offline clustering -----------------------------------------------------


def cluster_records_offline(
    records: list[FailureRecord], *, min_cluster_size: int
) -> list[Cluster]:
    """Deterministic clustering that never touches the network.

    Uses the TF-IDF + agglomerative fallback from
    :mod:`aisafepy.adapt.cluster` directly, skipping the
    sentence-transformers attempt. Summaries use the phrase-frequency
    heuristic, which is also offline.
    """
    if not records:
        return []
    texts = [r.input + "\n" + r.output for r in records]
    labels = _tfidf_cluster_fallback(texts, min_cluster_size=min_cluster_size)
    by_label: dict[int, list[FailureRecord]] = {}
    for rec, lab in zip(records, labels, strict=True):
        by_label.setdefault(int(lab), []).append(rec)
    clusters = [
        Cluster(label=lab, records=recs, summary=_summarize(recs, None))
        for lab, recs in by_label.items()
        if lab != -1
    ]
    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


def compile_offline(
    records: list[FailureRecord],
    targets: list[Any],
    *,
    min_asr: float,
    min_cluster_size: int,
) -> CompilationReport:
    """The offline equivalent of :meth:`GuardCompiler.compile`."""
    clusters = cluster_records_offline(records, min_cluster_size=min_cluster_size)
    kept = [c for c in clusters if c.attack_success_rate >= min_asr]
    rejected = [c for c in clusters if c.attack_success_rate < min_asr]
    artifacts: list[CompiledArtifact] = []
    for cluster in kept:
        for tgt in targets:
            try:
                art = tgt.compile_for_cluster(cluster)
            except Exception as exc:
                art = CompiledArtifact(
                    kind=getattr(tgt, "kind", "unknown"),
                    name=f"failed-{cluster.label}",
                    payload=None,
                    cluster_label=cluster.label,
                    attack_success_rate=cluster.attack_success_rate,
                    n_records=cluster.size,
                    metadata={"error": repr(exc)},
                )
            if art is not None:
                artifacts.append(art)
    return CompilationReport(artifacts=artifacts, clusters=kept, rejected_clusters=rejected)


# ---- output -----------------------------------------------------------------


def write_report(report: CompilationReport, out_dir: Path) -> list[Path]:
    """Write ``report.txt``, ``report.json`` and one file per artifact payload."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    text_path = out_dir / "report.txt"
    text_path.write_text(report.summary() + "\n", encoding="utf-8")
    written.append(text_path)

    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report_to_dict(report), indent=2, default=str), encoding="utf-8")
    written.append(json_path)

    for art in report.artifacts:
        if art.payload is None:
            continue
        ext = _ARTIFACT_EXT.get(art.kind, ".txt")
        path = out_dir / f"{_safe_name(art.name)}{ext}"
        if isinstance(art.payload, dict | list):
            path.write_text(json.dumps(art.payload, indent=2, default=str), encoding="utf-8")
        else:
            path.write_text(str(art.payload), encoding="utf-8")
        written.append(path)
    return written


def report_to_dict(report: CompilationReport) -> dict[str, Any]:
    return {
        "artifacts": [
            {
                "kind": a.kind,
                "name": a.name,
                "cluster_label": a.cluster_label,
                "attack_success_rate": a.attack_success_rate,
                "n_records": a.n_records,
                "summary": a.summary,
                "metadata": a.metadata,
                "has_payload": a.payload is not None,
            }
            for a in report.artifacts
        ],
        "clusters": [
            {
                "label": c.label,
                "size": c.size,
                "attack_success_rate": c.attack_success_rate,
                "summary": c.summary,
                "record_ids": [r.id for r in c.records],
            }
            for c in report.clusters
        ],
        "rejected_clusters": [
            {"label": c.label, "size": c.size, "attack_success_rate": c.attack_success_rate}
            for c in report.rejected_clusters
        ],
    }


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "artifact"


# ---- command ----------------------------------------------------------------


def run(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    source = open_source(args.source, args.path, only_violations=not args.all_records)
    targets = parse_targets(args.target)
    try:
        records = list(source)
    except ModuleNotFoundError as exc:
        raise CLIError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise CLIError(f"{args.path}: invalid JSON on line {exc.lineno}: {exc.msg}") from exc
    if not records:
        raise CLIError(f"{args.path}: no records found (try --all-records)")

    if args.online:
        compiler = GuardCompiler(
            source=records,  # a list satisfies the iterable protocol
            targets=targets,
            min_attack_success_rate=args.min_asr,
        )
        report = compiler.compile()
    else:
        report = compile_offline(
            records,
            targets,
            min_asr=args.min_asr,
            min_cluster_size=args.min_cluster_size,
        )

    written: list[Path] = []
    if args.out:
        written = write_report(report, Path(args.out))

    if args.json:
        _io.dump_json(
            out,
            {
                "records": len(records),
                "report": report_to_dict(report),
                "written": [str(p) for p in written],
            },
        )
    else:
        _io.emit(out, report.summary())
        if not args.quiet:
            _io.emit(out, f"  records read: {len(records)}")
            for p in written:
                _io.emit(out, f"  wrote {p}")
    return EXIT_OK
