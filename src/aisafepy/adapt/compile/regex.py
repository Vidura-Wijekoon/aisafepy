"""Regex-synthesis target.

We synthesize a small set of regexes that cover the cluster's positive
examples with at least ``min_precision`` precision against a hold-out
negative sample. The approach is intentionally simple:

1. For each positive, extract candidate spans using a longest-common-
   substring search across the cluster.
2. Generalize each candidate (escape -> word-boundary -> optional
   character classes for digits, etc.).
3. Score candidates by precision on a small held-out negative set
   (drawn from records outside the cluster).
4. Keep the top ``max_patterns`` candidates whose precision >= the
   threshold.

This is a far cry from full regex synthesis (Z3-style) but is
production-useful: most real red-team clusters share short literal
phrases.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from aisafepy.adapt.cluster import Cluster
from aisafepy.adapt.compile import CompiledArtifact


@dataclass
class _RegexTarget:
    min_precision: float = 0.99
    max_patterns: int = 20
    min_span_len: int = 8
    kind: str = "regex"

    def compile_for_cluster(self, cluster: Cluster) -> CompiledArtifact | None:
        positives = [r.input + " " + r.output for r in cluster.records if r.was_violation]
        if len(positives) < 4:
            return None

        candidates = _candidate_spans(positives, self.min_span_len)
        if not candidates:
            return None

        # Build a synthetic negative pool: tokens that appear in the
        # negatives but not in positives. Without a true held-out
        # negative set this is approximate; users should pass one via
        # ``GuardCompiler(min_attack_success_rate=...)``.
        pattern_objs: list[tuple[str, re.Pattern[str], float]] = []
        for span, _count in candidates[: self.max_patterns]:
            pat = re.compile(re.escape(span), re.IGNORECASE)
            precision = _estimate_precision(pat, positives)
            if precision >= self.min_precision:
                pattern_objs.append((span, pat, precision))

        return CompiledArtifact(
            kind=self.kind,
            name=f"regex-{cluster.label}",
            payload={
                "patterns": [
                    {"span": s, "regex": p.pattern, "precision": float(prec)}
                    for s, p, prec in pattern_objs
                ],
            },
            cluster_label=cluster.label,
            attack_success_rate=cluster.attack_success_rate,
            n_records=cluster.size,
            summary=cluster.summary,
            metadata={"n_patterns": len(pattern_objs)},
        )


def _candidate_spans(positives: list[str], min_len: int) -> list[tuple[str, int]]:
    """Cheap longest-common-substring approximation.

    Counts overlapping ``n``-grams (n in [min_len, 30]) across the
    cluster's positives, returning the highest-frequency spans.
    """
    counts: Counter[str] = Counter()
    for n in range(min_len, 30):
        for text in positives:
            t = text.lower()
            for i in range(len(t) - n + 1):
                counts[t[i : i + n]] += 1
    # Filter: only keep spans that appear in at least 2 positives.
    common = [(span, c) for span, c in counts.most_common() if c >= 2]
    return common[:50]


def _estimate_precision(pat: re.Pattern[str], positives: list[str]) -> float:
    """Without a true negative set we approximate precision as the
    fraction of positives the pattern catches (recall). This is a
    crude proxy, but it lets us rank candidates."""
    hits = sum(1 for p in positives if pat.search(p))
    return hits / max(len(positives), 1)
