"""Compiler targets and the top-level :class:`GuardCompiler`.

A "target" is one output modality of the compiler. Each target lives
in its own submodule (``classifier``, ``regex``, ``policy``,
``steering``, ``deliberative``) and is constructed via the
:class:`Target` factory.

The :class:`CompilationReport` is the compiler's durable output. It
serializes to a directory bundle (``report.json`` plus one file per
artifact payload) so a CI job can compile on one machine and a
deployment job can :func:`~aisafepy.adapt.canary.promote` on another.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aisafepy.adapt.cluster import Cluster, cluster_failures
from aisafepy.adapt.sources import FailureRecord, RedTeamSource

REPORT_SCHEMA_VERSION = 1

#: File suffix used when an artifact payload is written to a bundle.
_PAYLOAD_SUFFIX: dict[str, str] = {
    "regex": ".regex.json",
    "deliberative": ".md",
    "classifier": ".ref.txt",
    "steering": ".ref.txt",
}


@dataclass
class CompiledArtifact:
    """One concrete output of the compiler.

    ``kind`` identifies the artifact category (``classifier``,
    ``regex``, ``policy``, ``steering``, ``deliberative``). ``payload``
    is the runtime form. A model path, a list of regexes, a
    serialized Cedar policy, etc. ``metadata`` carries the cluster's
    provenance so the canary deploy can attribute its FP rate back
    to a specific red-team finding.
    """

    kind: str
    name: str
    payload: Any
    cluster_label: int
    attack_success_rate: float
    n_records: int
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def deferred(self) -> bool:
        """True when the target could not produce a runtime payload."""
        return self.payload is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "payload": self.payload,
            "cluster_label": self.cluster_label,
            "attack_success_rate": self.attack_success_rate,
            "n_records": self.n_records,
            "summary": self.summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompiledArtifact:
        return cls(
            kind=str(data["kind"]),
            name=str(data["name"]),
            payload=data.get("payload"),
            cluster_label=int(data.get("cluster_label", -1)),
            attack_success_rate=float(data.get("attack_success_rate", 0.0)),
            n_records=int(data.get("n_records", 0)),
            summary=str(data.get("summary", "")),
            metadata=dict(data.get("metadata") or {}),
        )

    def payload_filename(self) -> str:
        """Name of the file this artifact's payload is written to in a bundle."""
        if self.kind == "policy":
            dsl = str(self.metadata.get("dsl", "cedar"))
            return f"{_safe_name(self.name)}.{dsl}"
        return f"{_safe_name(self.name)}{_PAYLOAD_SUFFIX.get(self.kind, '.payload.json')}"


def _cluster_to_dict(c: Cluster, *, include_records: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "label": c.label,
        "size": c.size,
        "attack_success_rate": c.attack_success_rate,
        "summary": c.summary,
        "record_ids": [r.id for r in c.records],
    }
    if include_records:
        out["records"] = [_record_to_dict(r) for r in c.records]
    return out


def _record_to_dict(r: FailureRecord) -> dict[str, Any]:
    return {
        "id": r.id,
        "input": r.input,
        "output": r.output,
        "was_violation": r.was_violation,
        "attack_category": r.attack_category,
        "severity": r.severity,
        "scorer": r.scorer,
        "score": r.score,
        "timestamp": r.timestamp,
        "source": r.source,
    }


def _cluster_from_dict(d: dict[str, Any]) -> Cluster:
    records = [
        FailureRecord(
            id=str(x["id"]),
            input=str(x.get("input", "")),
            output=str(x.get("output", "")),
            was_violation=bool(x.get("was_violation", False)),
            attack_category=x.get("attack_category"),
            severity=str(x.get("severity", "medium")),
            scorer=x.get("scorer"),
            score=x.get("score"),
            timestamp=x.get("timestamp"),
            source=str(x.get("source", "unknown")),
        )
        for x in d.get("records", [])
    ]
    c = Cluster(label=int(d["label"]), records=records, summary=str(d.get("summary", "")))
    if not records:
        # Preserve the summary statistics when records were not serialized.
        c.centroid = {
            "size": int(d.get("size", 0)),
            "attack_success_rate": float(d.get("attack_success_rate", 0.0)),
            "record_ids": list(d.get("record_ids", [])),
        }
    return c


@dataclass
class CompilationReport:
    artifacts: list[CompiledArtifact] = field(default_factory=list)
    clusters: list[Cluster] = field(default_factory=list)
    rejected_clusters: list[Cluster] = field(default_factory=list)
    """Clusters dropped because their attack-success-rate fell below
    ``min_attack_success_rate``."""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- introspection -------------------------------------------------

    def __len__(self) -> int:
        return len(self.artifacts)

    def __iter__(self) -> Iterator[CompiledArtifact]:
        return iter(self.artifacts)

    def filter(self, *, kind: str | None = None, deferred: bool | None = None) -> list[CompiledArtifact]:
        """Select artifacts by kind and/or deferred status."""
        out = []
        for a in self.artifacts:
            if kind is not None and a.kind != kind:
                continue
            if deferred is not None and a.deferred != deferred:
                continue
            out.append(a)
        return out

    def kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for a in self.artifacts:
            counts[a.kind] = counts.get(a.kind, 0) + 1
        return counts

    def summary(self) -> str:
        lines = [
            "AIsafePy compilation report",
            f"  total clusters seen: {len(self.clusters) + len(self.rejected_clusters)}",
            f"  promoted clusters: {len(self.clusters)}",
            f"  rejected (below ASR threshold): {len(self.rejected_clusters)}",
            f"  artifacts: {len(self.artifacts)}",
        ]
        for a in self.artifacts:
            lines.append(
                f"  - [{a.kind}] {a.name} "
                f"(cluster {a.cluster_label}, n={a.n_records}, asr={a.attack_success_rate:.2%})"
            )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """A human-readable report suitable for a PR comment or CI summary."""
        n_def = len(self.filter(deferred=True))
        lines = [
            "# AIsafePy compilation report",
            "",
            f"- Clusters seen: {len(self.clusters) + len(self.rejected_clusters)}",
            f"- Promoted clusters: {len(self.clusters)}",
            f"- Rejected clusters (below ASR threshold): {len(self.rejected_clusters)}",
            f"- Artifacts: {len(self.artifacts)} ({n_def} deferred)",
            "",
            "## Artifacts",
            "",
            "| kind | name | cluster | records | ASR | status |",
            "|---|---|---:|---:|---:|---|",
        ]
        for a in self.artifacts:
            status = "deferred" if a.deferred else "ready"
            if a.metadata.get("error"):
                status = "failed"
            lines.append(
                f"| {a.kind} | `{a.name}` | {a.cluster_label} | {a.n_records} | "
                f"{a.attack_success_rate:.1%} | {status} |"
            )
        if self.clusters:
            lines += ["", "## Clusters", ""]
            for c in self.clusters:
                lines.append(
                    f"- **{c.label}** (n={c.size}, asr={c.attack_success_rate:.1%}): {c.summary}"
                )
        if self.rejected_clusters:
            lines += ["", "## Rejected clusters", ""]
            for c in self.rejected_clusters:
                lines.append(
                    f"- {c.label} (n={c.size}, asr={c.attack_success_rate:.1%}): {c.summary}"
                )
        return "\n".join(lines) + "\n"

    # ---- serialization -------------------------------------------------

    def to_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "clusters": [_cluster_to_dict(c, include_records=include_records) for c in self.clusters],
            "rejected_clusters": [
                _cluster_to_dict(c, include_records=include_records) for c in self.rejected_clusters
            ],
        }

    def to_json(self, *, include_records: bool = False, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(include_records=include_records), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompilationReport:
        version = int(data.get("schema_version", 1))
        if version > REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"report schema_version {version} is newer than supported {REPORT_SCHEMA_VERSION}"
            )
        return cls(
            artifacts=[CompiledArtifact.from_dict(a) for a in data.get("artifacts", [])],
            clusters=[_cluster_from_dict(c) for c in data.get("clusters", [])],
            rejected_clusters=[_cluster_from_dict(c) for c in data.get("rejected_clusters", [])],
            created_at=float(data.get("created_at", time.time())),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> CompilationReport:
        return cls.from_dict(json.loads(text))

    def save(self, directory: str | Path, *, include_records: bool = False) -> Path:
        """Write a bundle: ``report.json`` plus one payload file per artifact.

        ``report.json`` keeps the payloads inline as well, so it is
        self-contained; the sidecar files exist for humans and for
        tools that consume a single Cedar / Rego / Markdown file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for art in self.artifacts:
            if art.payload is None:
                continue
            target = directory / art.payload_filename()
            if isinstance(art.payload, str):
                target.write_text(art.payload, encoding="utf-8")
            else:
                target.write_text(json.dumps(art.payload, indent=2, default=str), encoding="utf-8")
        (directory / "report.json").write_text(
            self.to_json(include_records=include_records), encoding="utf-8"
        )
        (directory / "report.md").write_text(self.to_markdown(), encoding="utf-8")
        return directory / "report.json"

    @classmethod
    def load(cls, directory: str | Path) -> CompilationReport:
        directory = Path(directory)
        path = directory / "report.json" if directory.is_dir() else directory
        return cls.from_json(path.read_text(encoding="utf-8"))


# ---- Target factory --------------------------------------------------


class Target:
    """Declarative target descriptors for :class:`GuardCompiler`."""

    @staticmethod
    def distill_classifier(
        *,
        base: str = "distilbert-base-uncased",
        out: str | None = None,
        augment_with: str | None = None,
        epochs: int = 3,
    ) -> _ClassifierTarget:
        from aisafepy.adapt.compile.classifier import _ClassifierTarget

        return _ClassifierTarget(base=base, out=out, augment_with=augment_with, epochs=epochs)

    @staticmethod
    def synthesize_regex(
        *,
        min_precision: float = 0.99,
        max_patterns: int = 20,
        min_support: int = 2,
        negatives: Iterable[str] | None = None,
    ) -> _RegexTarget:
        from aisafepy.adapt.compile.regex import _RegexTarget

        return _RegexTarget(
            min_precision=min_precision,
            max_patterns=max_patterns,
            min_support=min_support,
            negatives=list(negatives) if negatives is not None else None,
        )

    @staticmethod
    def policy_rule(*, dsl: str = "cedar") -> _PolicyTarget:
        from aisafepy.adapt.compile.policy import _PolicyTarget

        return _PolicyTarget(dsl=dsl)

    @staticmethod
    def steering_vector(
        *,
        model: str,
        layers: Iterable[int] = (16, 18),
        method: str = "conditional_activation_steering",
    ) -> _SteeringTarget:
        from aisafepy.adapt.compile.steering import _SteeringTarget

        return _SteeringTarget(model=model, layers=tuple(layers), method=method)

    @staticmethod
    def deliberative_case(*, policy: str, k_shot: int = 8) -> _DeliberativeTarget:
        from aisafepy.adapt.compile.deliberative import _DeliberativeTarget

        return _DeliberativeTarget(policy=policy, k_shot=k_shot)

    @staticmethod
    def hashed_classifier(**kwargs: Any) -> Any:
        """A pure-Python hashed n-gram classifier that trains without torch.

        Unlike :meth:`distill_classifier`, this target always produces a
        deployable Tier-2 guard on a laptop. See ``docs/hashed-classifier.md``.
        """
        from aisafepy.adapt.compile.hashed import hashed_classifier_target

        return hashed_classifier_target(**kwargs)


# ---- GuardCompiler ---------------------------------------------------


@dataclass
class CompilerHooks:
    """Optional observer callbacks invoked during :meth:`GuardCompiler.compile`.

    ``on_cluster(cluster, kept)`` fires once per cluster after the ASR
    gate; ``on_artifact(artifact)`` fires once per produced artifact.
    Exceptions raised by hooks are swallowed and recorded in the
    report's ``metadata["hook_errors"]`` so observability can never
    break compilation.
    """

    on_cluster: Callable[[Cluster, bool], None] | None = None
    on_artifact: Callable[[CompiledArtifact], None] | None = None


@dataclass
class GuardCompiler:
    source: RedTeamSource
    targets: list[Any]
    min_attack_success_rate: float = 0.10
    cluster_method: str = "hdbscan"
    deliberative_spec_path: str | None = None
    summarizer: Callable[[list[str]], str] | None = None
    negatives: Any = None
    """Benign pool used to measure precision of synthesized guards.
    Either a ``RedTeamSource`` / iterable of ``FailureRecord`` (their
    ``input`` + ``output`` text is used) or an iterable of plain strings.
    Non-violation records found in ``source`` are always added."""
    max_clusters: int | None = None
    """Keep only the ``max_clusters`` largest clusters that pass the ASR gate."""
    hooks: CompilerHooks | None = None
    dry_run: bool = False
    """Cluster and gate, but do not run any target."""
    min_cluster_size: int = 5

    def compile(self) -> CompilationReport:
        started = time.time()
        records: list[FailureRecord] = list(self.source)
        negatives = self._collect_negatives(records)
        n_positives = sum(1 for r in records if r.was_violation)

        # Cluster violations and non-violations together: a cluster's
        # attack-success rate is only meaningful when benign lookalikes
        # can land in the same cluster.
        clusters = cluster_failures(
            records,
            method=self.cluster_method,
            min_cluster_size=self.min_cluster_size,
            summarizer=self.summarizer,
        )

        hook_errors: list[str] = []
        kept: list[Cluster] = []
        rejected: list[Cluster] = []
        for c in clusters:
            keep = c.attack_success_rate >= self.min_attack_success_rate
            (kept if keep else rejected).append(c)
            self._fire(hook_errors, "on_cluster", c, keep)

        kept.sort(key=lambda c: c.size, reverse=True)
        if self.max_clusters is not None and len(kept) > self.max_clusters:
            rejected.extend(kept[self.max_clusters :])
            kept = kept[: self.max_clusters]

        artifacts: list[CompiledArtifact] = []
        if not self.dry_run:
            for cluster in kept:
                for tgt in self.targets:
                    art = self._run_target(tgt, cluster, negatives)
                    if art is not None:
                        artifacts.append(art)
                        self._fire(hook_errors, "on_artifact", art)

        return CompilationReport(
            artifacts=artifacts,
            clusters=kept,
            rejected_clusters=rejected,
            metadata={
                "n_records": len(records),
                "n_positives": n_positives,
                "n_negatives": len(negatives),
                "min_attack_success_rate": self.min_attack_success_rate,
                "cluster_method": self.cluster_method,
                "dry_run": self.dry_run,
                "targets": [getattr(t, "kind", type(t).__name__) for t in self.targets],
                "duration_s": time.time() - started,
                **({"hook_errors": hook_errors} if hook_errors else {}),
            },
        )

    # ---- internals -----------------------------------------------------

    def _collect_negatives(self, records: list[FailureRecord]) -> list[str]:
        pool: list[str] = [r.input + " " + r.output for r in records if not r.was_violation]
        if self.negatives is None:
            return pool
        for item in self.negatives:
            if isinstance(item, FailureRecord):
                if not item.was_violation:
                    pool.append(item.input + " " + item.output)
            elif isinstance(item, str):
                pool.append(item)
        return pool

    @staticmethod
    def _accepts_negatives(target: Any) -> bool:
        fn = getattr(target, "compile_for_cluster", None)
        if fn is None:
            return False
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False
        return "negatives" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    def _run_target(self, tgt: Any, cluster: Cluster, negatives: list[str]) -> CompiledArtifact | None:
        try:
            if negatives and self._accepts_negatives(tgt):
                return tgt.compile_for_cluster(cluster, negatives=negatives)
            return tgt.compile_for_cluster(cluster)
        except Exception as exc:
            return CompiledArtifact(
                kind=getattr(tgt, "kind", "unknown"),
                name=f"failed-{cluster.label}",
                payload=None,
                cluster_label=cluster.label,
                attack_success_rate=cluster.attack_success_rate,
                n_records=cluster.size,
                metadata={"error": repr(exc)},
            )

    def _fire(self, errors: list[str], name: str, *args: Any) -> None:
        if self.hooks is None:
            return
        cb = getattr(self.hooks, name, None)
        if cb is None:
            return
        try:
            cb(*args)
        except Exception as exc:
            errors.append(f"{name}: {exc!r}")


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name) or "artifact"
