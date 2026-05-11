"""Failure clustering.

Encode each failure with sentence-transformers, run HDBSCAN over UMAP,
and label each cluster with an LLM summarizer. The summarizer is
duck-typed. Any callable ``summarize(samples: list[str]) -> str``
works, so you can pass an OpenAI / Anthropic / vLLM endpoint.

If the optional deps are missing the function falls back to a
deterministic baseline: TF-IDF vectorization + scikit-learn's
``AgglomerativeClustering``. Less accurate but always runs.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from aisafepy.adapt.sources import FailureRecord

_logger = logging.getLogger("aisafepy.adapt.cluster")


@dataclass
class Cluster:
    label: int
    records: list[FailureRecord] = field(default_factory=list)
    summary: str = ""
    centroid: Any | None = None

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def attack_success_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.was_violation) / self.size


def cluster_failures(
    records: Iterable[FailureRecord],
    *,
    method: str = "hdbscan",
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    min_cluster_size: int = 5,
    summarizer: Callable[[list[str]], str] | None = None,
) -> list[Cluster]:
    """Group ``records`` into semantically coherent clusters."""
    records = list(records)
    if not records:
        return []

    texts = [r.input + "\n" + r.output for r in records]

    labels = _embed_and_cluster(texts, method=method,
                                embedding_model=embedding_model,
                                min_cluster_size=min_cluster_size)
    by_label: dict[int, list[FailureRecord]] = {}
    for rec, lab in zip(records, labels, strict=True):
        by_label.setdefault(int(lab), []).append(rec)

    clusters: list[Cluster] = []
    for lab, recs in by_label.items():
        if lab == -1:  # HDBSCAN's "noise" bin
            continue
        summary = _summarize(recs, summarizer)
        clusters.append(Cluster(label=lab, records=recs, summary=summary))

    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


def _embed_and_cluster(
    texts: list[str],
    *,
    method: str,
    embedding_model: str,
    min_cluster_size: int,
) -> list[int]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        embedder = SentenceTransformer(embedding_model)
        embeddings = embedder.encode(texts, show_progress_bar=False)
    except Exception as exc:
        _logger.warning("sentence-transformers unavailable; falling back to TF-IDF (%s)", exc)
        return _tfidf_cluster_fallback(texts, min_cluster_size=min_cluster_size)

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore[import-not-found]
            import umap  # type: ignore[import-not-found]
            reducer = umap.UMAP(n_components=min(15, len(texts) - 1), random_state=0)
            reduced = reducer.fit_transform(embeddings)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels = clusterer.fit_predict(reduced)
            return [int(l) for l in labels]
        except Exception as exc:
            _logger.warning("hdbscan/umap unavailable; falling back to k-means (%s)", exc)

    # K-means fallback.
    from sklearn.cluster import KMeans  # type: ignore[import-not-found]
    k = max(2, min(len(texts) // min_cluster_size, 20))
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    return [int(l) for l in km.fit_predict(np.asarray(embeddings))]


def _tfidf_cluster_fallback(texts: list[str], min_cluster_size: int) -> list[int]:
    """Deterministic, dependency-light fallback. Lower fidelity than the
    sentence-transformer + HDBSCAN path but always works."""
    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-not-found]
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
    except ImportError:
        # Last-resort: bucket by first significant word.
        labels: list[int] = []
        bucket: dict[str, int] = {}
        for t in texts:
            words = re.findall(r"\w+", t.lower())
            key = (words[0] if words else "_")
            labels.append(bucket.setdefault(key, len(bucket)))
        return labels

    vec = TfidfVectorizer(max_features=2048, stop_words="english")
    X = vec.fit_transform(texts)
    n_clusters = max(2, min(len(texts) // max(min_cluster_size, 1), 20))
    clf = AgglomerativeClustering(n_clusters=n_clusters)
    return [int(l) for l in clf.fit_predict(X.toarray())]


def _summarize(
    records: list[FailureRecord],
    summarizer: Callable[[list[str]], str] | None,
) -> str:
    samples = [r.input[:300] for r in records[:8]]
    if summarizer is None:
        # Heuristic: most common 2-3-word phrase across the sample.
        words: Counter[str] = Counter()
        for s in samples:
            tokens = re.findall(r"\w+", s.lower())
            for n in (2, 3):
                for i in range(len(tokens) - n + 1):
                    words[" ".join(tokens[i : i + n])] += 1
        top = ", ".join(p for p, _ in words.most_common(3))
        return f"cluster of {len(records)} failures; common phrases: {top}"
    try:
        return summarizer(samples)
    except Exception as exc:  # pragma: no cover - depends on summarizer
        return f"summarizer failed: {exc!r}"
