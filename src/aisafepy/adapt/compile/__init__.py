"""Compiler targets and the top-level :class:`GuardCompiler`.

A "target" is one output modality of the compiler. Each target lives
in its own submodule (``classifier``, ``regex``, ``policy``,
``steering``, ``deliberative``) and is constructed via the
:class:`Target` factory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from aisafepy.adapt.cluster import Cluster, cluster_failures
from aisafepy.adapt.sources import FailureRecord, RedTeamSource


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


@dataclass
class CompilationReport:
    artifacts: list[CompiledArtifact] = field(default_factory=list)
    clusters: list[Cluster] = field(default_factory=list)
    rejected_clusters: list[Cluster] = field(default_factory=list)
    """Clusters dropped because their attack-success-rate fell below
    ``min_attack_success_rate``."""

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
    def synthesize_regex(*, min_precision: float = 0.99, max_patterns: int = 20) -> _RegexTarget:
        from aisafepy.adapt.compile.regex import _RegexTarget

        return _RegexTarget(min_precision=min_precision, max_patterns=max_patterns)

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


# ---- GuardCompiler ---------------------------------------------------


@dataclass
class GuardCompiler:
    source: RedTeamSource
    targets: list[Any]
    min_attack_success_rate: float = 0.10
    cluster_method: str = "hdbscan"
    deliberative_spec_path: str | None = None
    summarizer: Callable[[list[str]], str] | None = None

    def compile(self) -> CompilationReport:
        records: list[FailureRecord] = list(self.source)
        clusters = cluster_failures(
            records,
            method=self.cluster_method,
            summarizer=self.summarizer,
        )

        kept: list[Cluster] = []
        rejected: list[Cluster] = []
        for c in clusters:
            if c.attack_success_rate < self.min_attack_success_rate:
                rejected.append(c)
            else:
                kept.append(c)

        artifacts: list[CompiledArtifact] = []
        for cluster in kept:
            for tgt in self.targets:
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
        return CompilationReport(
            artifacts=artifacts,
            clusters=kept,
            rejected_clusters=rejected,
        )
