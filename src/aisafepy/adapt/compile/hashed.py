"""Hashed n-gram classifier target.

The transformer-distillation target in :mod:`aisafepy.adapt.compile.classifier`
defers to an out-of-band GPU job whenever ``torch`` is missing. This
target always produces a deployable artifact: it trains a
:class:`aisafepy.stream.hashed_classifier.HashedNGramClassifierGuard`
on the cluster's violations against a benign pool, saves the model as
JSON, and reports validation metrics in the payload so the canary can
refuse to promote a model whose false-positive rate is already too high
before it sees traffic.

Negative pool resolution, in order:

1. ``negatives`` passed to :meth:`HashedClassifierTarget.compile_for_cluster`
   (strings or ``FailureRecord`` objects; the compiler forwards its
   benign source here).
2. ``negatives`` given at construction.
3. The cluster's own non-violation records.

With fewer than ``min_negatives`` usable negatives, or fewer than
``min_positives`` violations, the target returns a deferred artifact
whose metadata says exactly what was missing. A classifier trained
against no negatives would learn "any text is unsafe", which is worse
than no classifier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aisafepy.adapt.cluster import Cluster
from aisafepy.adapt.compile import CompiledArtifact
from aisafepy.stream.hashed_classifier import HashedNGramClassifierGuard

KIND = "hashed_classifier"


def _texts(items: Sequence[Any] | None) -> list[str]:
    """Accept strings or FailureRecord-like objects and return their text."""
    out: list[str] = []
    for item in items or ():
        if isinstance(item, str):
            if item.strip():
                out.append(item)
            continue
        text = getattr(item, "input", None)
        if isinstance(text, str) and text.strip():
            out.append(text)
    return out


@dataclass
class HashedClassifierTarget:
    """Compile a cluster into a pure-Python hashed n-gram classifier."""

    out_dir: str | Path = "guards/hashed"
    min_positives: int = 8
    min_negatives: int = 8
    epochs: int = 10
    threshold: float = 0.5
    n_buckets: int = 1 << 18
    seed: int = 0
    lr: float = 0.1
    l2: float = 1e-5
    negatives: Sequence[Any] | None = None
    """Default benign pool, overridden by the per-call ``negatives``."""
    kind: str = field(default=KIND, init=False)

    def compile_for_cluster(
        self,
        cluster: Cluster,
        negatives: Sequence[Any] | None = None,
    ) -> CompiledArtifact | None:
        positives = _texts([r for r in cluster.records if r.was_violation])
        neg_pool = _texts(negatives if negatives is not None else self.negatives)
        neg_source = "negatives" if neg_pool else "cluster"
        if not neg_pool:
            neg_pool = _texts([r for r in cluster.records if not r.was_violation])
        # Never let a positive leak into the negative pool.
        pos_set = set(positives)
        neg_pool = [n for n in neg_pool if n not in pos_set]

        missing: list[str] = []
        if len(positives) < self.min_positives:
            missing.append(f"positives {len(positives)} < {self.min_positives}")
        if len(neg_pool) < self.min_negatives:
            missing.append(f"negatives {len(neg_pool)} < {self.min_negatives}")
        if missing:
            return CompiledArtifact(
                kind=self.kind,
                name=f"hashed-deferred-{cluster.label}",
                payload=None,
                cluster_label=cluster.label,
                attack_success_rate=cluster.attack_success_rate,
                n_records=cluster.size,
                summary=cluster.summary,
                metadata={
                    "deferred": True,
                    "reason": "; ".join(missing),
                    "n_positives": len(positives),
                    "n_negatives": len(neg_pool),
                },
            )

        guard = HashedNGramClassifierGuard(
            name=f"hashed-{cluster.label}",
            threshold=self.threshold,
            n_buckets=self.n_buckets,
        )
        report = guard.fit(
            positives,
            neg_pool,
            epochs=self.epochs,
            lr=self.lr,
            l2=self.l2,
            seed=self.seed,
        )
        out_path = Path(self.out_dir) / f"hashed_cluster_{cluster.label}.json"
        guard.save(out_path)
        metrics = {
            "precision": round(report.precision, 4),
            "recall": round(report.recall, 4),
            "f1": round(report.f1, 4),
            "false_positive_rate": round(report.false_positive_rate, 4),
            "n_validation": report.n_validation,
            "n_features": report.n_features,
            "epochs_run": report.epochs_run,
            "best_epoch": report.best_epoch,
            "train_seconds": round(report.train_seconds, 4),
        }
        return CompiledArtifact(
            kind=self.kind,
            name=f"hashed-{cluster.label}",
            payload={"path": str(out_path), "metrics": metrics, "threshold": self.threshold},
            cluster_label=cluster.label,
            attack_success_rate=cluster.attack_success_rate,
            n_records=cluster.size,
            summary=cluster.summary,
            metadata={
                "n_positives": len(positives),
                "n_negatives": len(neg_pool),
                "negatives_source": neg_source,
                "seed": self.seed,
            },
        )


def materialize_hashed_classifier(artifact: CompiledArtifact) -> HashedNGramClassifierGuard:
    """Load the guard an artifact points at; used by the canary controller."""
    if artifact.kind != KIND:
        raise ValueError(f"expected a {KIND} artifact, got {artifact.kind!r}")
    payload = artifact.payload
    if not isinstance(payload, dict) or "path" not in payload:
        raise RuntimeError(f"{artifact.name}: deferred artifact has no model path")
    guard = HashedNGramClassifierGuard.load(payload["path"])
    guard.name = artifact.name
    if "threshold" in payload:
        guard.threshold = float(payload["threshold"])
    return guard


def hashed_classifier_target(**kwargs: Any) -> HashedClassifierTarget:
    """Factory mounted as ``Target.hashed_classifier`` by the package."""
    return HashedClassifierTarget(**kwargs)
