"""Failure clustering.

Encode each failure with sentence-transformers, run HDBSCAN over UMAP,
and label each cluster with an LLM summarizer. The summarizer is
duck-typed. Any callable ``summarize(samples: list[str]) -> str``
works, so you can pass an OpenAI / Anthropic / vLLM endpoint.

Three tiers of fallback keep the compiler usable everywhere:

1. ``method="hdbscan"`` (default): sentence-transformers embeddings,
   UMAP, HDBSCAN. Best quality; needs the ``adapt`` extra and a model
   download.
2. ``method="kmeans"``: sentence-transformers embeddings plus
   scikit-learn k-means.
3. ``method="tfidf"`` (also ``"lexical"``): no embeddings at all.
   scikit-learn's TF-IDF plus agglomerative clustering when available,
   otherwise :func:`lexical_cluster`, a pure-Python TF-IDF cosine
   clustering that ships with the core install. This is the path the
   CLI uses, and the one that runs offline in CI.

Passing ``method="tfidf"`` skips the embedding step entirely rather
than attempting a model download first.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from aisafepy.adapt.sources import FailureRecord

_logger = logging.getLogger("aisafepy.adapt.cluster")

LEXICAL_METHODS = frozenset({"tfidf", "lexical", "pure"})


class Cluster:
    """A group of failure records with a label and a human-readable summary."""

    def __init__(
        self,
        label: int,
        records: list[FailureRecord] | None = None,
        summary: str = "",
        centroid: Any | None = None,
    ) -> None:
        self.label = label
        self.records = list(records or [])
        self.summary = summary
        self.centroid = centroid

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def attack_success_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.was_violation) / self.size

    def __repr__(self) -> str:  # pragma: no cover - debug
        return f"Cluster(label={self.label}, size={self.size}, asr={self.attack_success_rate:.2f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cluster):
            return NotImplemented
        return (
            self.label == other.label
            and self.records == other.records
            and self.summary == other.summary
        )

    __hash__ = None  # type: ignore[assignment]


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

    if method in LEXICAL_METHODS:
        labels = _tfidf_cluster_fallback(texts, min_cluster_size=min_cluster_size)
    else:
        labels = _embed_and_cluster(
            texts,
            method=method,
            embedding_model=embedding_model,
            min_cluster_size=min_cluster_size,
        )
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
    try:
        from sklearn.cluster import KMeans  # type: ignore[import-not-found]
    except ImportError:
        _logger.warning("scikit-learn unavailable; falling back to lexical clustering")
        return lexical_cluster(texts, min_cluster_size=min_cluster_size)
    k = max(2, min(len(texts) // min_cluster_size, 20))
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    return [int(l) for l in km.fit_predict(np.asarray(embeddings))]


def _tfidf_cluster_fallback(texts: list[str], min_cluster_size: int) -> list[int]:
    """Dependency-light clustering. scikit-learn when present, else pure Python.

    Lower fidelity than the sentence-transformer + HDBSCAN path but always
    works, which is what the CLI and CI need.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-not-found]
        from sklearn.feature_extraction.text import (
            TfidfVectorizer,  # type: ignore[import-not-found]
        )
    except ImportError:
        return lexical_cluster(texts, min_cluster_size=min_cluster_size)

    if len(texts) < 2:
        return [0] * len(texts)
    vec = TfidfVectorizer(max_features=2048, stop_words="english")
    X = vec.fit_transform(texts)
    n_clusters = max(2, min(len(texts) // max(min_cluster_size, 1), 20))
    n_clusters = min(n_clusters, len(texts))
    clf = AgglomerativeClustering(n_clusters=n_clusters)
    return [int(l) for l in clf.fit_predict(X.toarray())]


# ---- pure-Python lexical clustering ------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# A compact English stop list. Enough to stop function words dominating
# cosine similarity without pulling in a corpus package.
STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "has", "have", "he", "her", "his", "i", "if", "in", "into",
        "is", "it", "its", "me", "my", "no", "not", "of", "on", "or",
        "our", "she", "so", "that", "the", "their", "them", "then", "there", "these",
        "they", "this", "to", "was", "we", "were", "what", "when", "which", "who",
        "will", "with", "would", "you", "your", "yours", "about", "above", "after", "again",
        "all", "am", "any", "been", "before", "being", "below", "between", "both", "can",
        "did", "do", "does", "doing", "down", "during", "each", "few", "further", "had",
        "having", "here", "how", "more", "most", "off", "once", "only", "other", "out",
        "over", "own", "same", "should", "some", "such", "than", "through", "too", "under",
        "until", "up", "very", "why",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stop words and single characters removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOP_WORDS]


def tfidf_vectors(
    texts: Sequence[str],
    *,
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 4096,
) -> list[dict[str, float]]:
    """Return L2-normalised sparse TF-IDF vectors as ``{term: weight}`` dicts.

    Terms are word unigrams and bigrams. Document frequency is smoothed
    (``log((1 + N) / (1 + df)) + 1``) so terms present in every document
    keep a small positive weight rather than vanishing. Only the
    ``max_features`` most frequent terms are kept.
    """
    lo, hi = ngram_range
    docs: list[Counter[str]] = []
    df: Counter[str] = Counter()
    for text in texts:
        toks = tokenize(text)
        terms: Counter[str] = Counter()
        for n in range(lo, hi + 1):
            for i in range(len(toks) - n + 1):
                terms[" ".join(toks[i : i + n])] += 1
        docs.append(terms)
        df.update(terms.keys())
    vocab = {t for t, _ in df.most_common(max_features)}
    n_docs = len(texts)
    vectors: list[dict[str, float]] = []
    for terms in docs:
        vec: dict[str, float] = {}
        for term, tf in terms.items():
            if term not in vocab:
                continue
            idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0
            vec[term] = (1.0 + math.log(tf)) * idf
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        vectors.append({t: w / norm for t, w in vec.items()})
    return vectors


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity of two sparse unit vectors (dot product)."""
    if len(b) < len(a):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def _centroid(vectors: list[dict[str, float]]) -> dict[str, float]:
    acc: dict[str, float] = {}
    for vec in vectors:
        for t, w in vec.items():
            acc[t] = acc.get(t, 0.0) + w
    norm = math.sqrt(sum(w * w for w in acc.values())) or 1.0
    return {t: w / norm for t, w in acc.items()}


def lexical_cluster(
    texts: Sequence[str],
    *,
    min_cluster_size: int = 2,
    threshold: float = 0.1,
    merge_threshold: float | None = None,
    max_clusters: int = 50,
) -> list[int]:
    """Pure-Python TF-IDF cosine clustering.

    Algorithm (deterministic, O(n * k) with k the number of clusters):

    1. Vectorise with :func:`tfidf_vectors`.
    2. Leader pass: visit documents in order; join the cluster whose
       centroid has the highest cosine similarity if that similarity is
       at least ``threshold``, otherwise start a new cluster. Centroids
       are updated incrementally.
    3. Merge pass: repeatedly merge the two clusters whose centroids are
       most similar while that similarity is at least
       ``merge_threshold`` (defaults to ``threshold``) or while there are
       more than ``max_clusters`` clusters.
    4. Reassign pass: move every document to its nearest centroid so
       early assignments made with immature centroids are corrected.
    5. Clusters smaller than ``min_cluster_size`` become noise (-1).

    The default ``threshold`` is deliberately low: red-team records are
    short, and two phrasings of the same attack often share only a
    handful of content words, so their bigram-inclusive cosine sits
    around 0.1 to 0.3 while unrelated records score 0.0.

    Labels are renumbered densely by descending cluster size so the
    largest cluster is 0. Returns one label per input text.
    """
    n = len(texts)
    if n == 0:
        return []
    if n == 1:
        return [0] if min_cluster_size <= 1 else [-1]
    vectors = tfidf_vectors(texts)

    members: list[list[int]] = []
    centroids: list[dict[str, float]] = []
    for idx, vec in enumerate(vectors):
        best, best_sim = -1, threshold
        for cid, cen in enumerate(centroids):
            sim = cosine(vec, cen)
            if sim >= best_sim:
                best, best_sim = cid, sim
        if best == -1:
            members.append([idx])
            centroids.append(dict(vec))
        else:
            members[best].append(idx)
            centroids[best] = _centroid([vectors[i] for i in members[best]])

    merge_at = threshold if merge_threshold is None else merge_threshold
    while len(members) > 1:
        best_pair, best_sim = None, -1.0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                sim = cosine(centroids[i], centroids[j])
                if sim > best_sim:
                    best_pair, best_sim = (i, j), sim
        if best_pair is None:
            break
        if best_sim < merge_at and len(members) <= max_clusters:
            break
        i, j = best_pair
        members[i].extend(members[j])
        centroids[i] = _centroid([vectors[k] for k in members[i]])
        del members[j]
        del centroids[j]

    if len(members) > 1:
        reassigned: list[list[int]] = [[] for _ in members]
        for idx, vec in enumerate(vectors):
            best = max(range(len(centroids)), key=lambda c: cosine(vec, centroids[c]))
            reassigned[best].append(idx)
        members = [m for m in reassigned if m]

    members.sort(key=lambda m: (-len(m), m[0]))
    labels = [-1] * n
    next_label = 0
    for group in members:
        if len(group) < min_cluster_size:
            continue
        for idx in group:
            labels[idx] = next_label
        next_label += 1
    return labels


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
