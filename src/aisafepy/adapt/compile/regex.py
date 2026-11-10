"""Regex-synthesis target.

We synthesize a small set of regexes that cover the cluster's positive
examples with at least ``min_precision`` precision against a negative
pool. The approach is intentionally simple:

1. Extract candidate spans as word-aligned n-grams that recur across
   the cluster's positives (a cheap longest-common-substring
   approximation that never splits a word).
2. Generalize each candidate: trim to word boundaries, collapse digit
   runs to ``\\d+`` and whitespace runs to ``\\s+``, escape the rest.
3. Score candidates by precision (``TP / (TP + FP)``) against the
   negative pool and recall against the positives.
4. Drop spans that are substrings of a longer kept span, and keep the
   top ``max_patterns`` candidates whose precision meets the threshold.

Every emitted pattern passes :func:`safe_regex`, a conservative
static check that rejects nested quantifiers and other shapes known
to backtrack catastrophically, so a compiled Tier-1 guard can never
become a regex-DoS vector (arXiv 2410.02916).

Negatives
---------
Without negatives there is no way to measure precision, so the
target falls back to the old behaviour (recall used as a proxy) and
flags it in the artifact metadata (``precision_source="recall_proxy"``).
Pass ``negatives=`` either at construction time or per call; the
:class:`~aisafepy.adapt.compile.GuardCompiler` forwards its benign
pool automatically.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from aisafepy.adapt.cluster import Cluster
from aisafepy.adapt.compile import CompiledArtifact

# Shapes that are a reliable signal of super-linear backtracking:
# nested quantifiers ((a+)+, (a*)*, (a|aa)+), and unbounded repeats of
# a group that itself contains an unbounded repeat.
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[*+][^()]*\)\s*[*+{]")
_ADJACENT_QUANTIFIERS = re.compile(r"[*+?}][*+]")
_OVERLAPPING_ALT = re.compile(r"\((?:[^()|\\]|\\.)*\|(?:[^()|\\]|\\.)*\)[*+]")

#: Maximum number of characters a synthesized pattern is applied to.
MAX_SEARCH_CHARS = 50_000

#: Token that stands in for any run of digits inside a candidate span.
NUMBER_PLACEHOLDER = "#"

_WORD = re.compile(r"\w+(?:['-]\w+)*")


def safe_regex(pattern: str) -> bool:
    """Return True when ``pattern`` compiles and has no known-dangerous shape.

    This is a static approximation: it rejects nested unbounded
    quantifiers, stacked quantifiers and unbounded repeats of
    overlapping alternations. It does not prove linearity, but it
    removes every construct the synthesizer can emit that would be a
    problem in practice.
    """
    if not pattern or len(pattern) > 2_000:
        return False
    try:
        re.compile(pattern)
    except re.error:
        return False
    if _NESTED_QUANTIFIER.search(pattern):
        return False
    if _ADJACENT_QUANTIFIERS.search(pattern):
        return False
    return not _OVERLAPPING_ALT.search(pattern)


def timed_search(pattern: re.Pattern[str], text: str, *, max_chars: int = MAX_SEARCH_CHARS) -> bool:
    """Apply ``pattern`` to at most ``max_chars`` of ``text``.

    The cap bounds the worst case for any pattern that passed
    :func:`safe_regex`: linear patterns stay linear, and a pattern that
    slipped through with mild polynomial behaviour is still bounded by
    the input size.
    """
    if len(text) > max_chars:
        text = text[-max_chars:]
    return pattern.search(text) is not None


def generalize_span(span: str) -> str:
    r"""Turn a literal span into a slightly generalized, escaped regex.

    * runs of digits become ``\d+``
    * runs of whitespace become ``\s+``
    * everything else is escaped literally
    * word boundaries are added when the span starts/ends with a word char

    Returns an empty string when nothing usable remains.
    """
    s = " ".join(span.split())
    if not s:
        return ""
    parts: list[str] = []
    for tok in re.split(r"(\d+|\s+|#)", s):
        if not tok:
            continue
        if tok.isdigit() or tok == NUMBER_PLACEHOLDER:
            parts.append(r"\d+")
        elif tok.isspace():
            parts.append(r"\s+")
        else:
            parts.append(re.escape(tok))
    out = "".join(parts)
    if re.match(r"[\w#]", s):
        out = r"\b" + out
    if re.search(r"[\w#]$", s):
        out = out + r"\b"
    return out


@dataclass
class _RegexTarget:
    min_precision: float = 0.99
    max_patterns: int = 20
    min_span_len: int = 8
    min_support: int = 2
    """Minimum number of distinct positives a span must occur in."""
    generalize: bool = True
    negatives: Sequence[str] | None = None
    """Default benign pool. Overridden by the ``negatives`` argument of
    :meth:`compile_for_cluster` when the compiler supplies one."""
    kind: str = "regex"
    max_candidates: int = 50
    _compiled_cache: dict[str, re.Pattern[str]] = field(default_factory=dict, repr=False)

    def compile_for_cluster(
        self,
        cluster: Cluster,
        negatives: Sequence[str] | None = None,
    ) -> CompiledArtifact | None:
        positives = [r.input + " " + r.output for r in cluster.records if r.was_violation]
        if len(positives) < 4:
            return None

        neg_pool: list[str] = list(negatives if negatives is not None else (self.negatives or []))
        # Non-violation records inside the cluster are negatives too.
        neg_pool.extend(r.input + " " + r.output for r in cluster.records if not r.was_violation)

        candidates = _candidate_spans(
            positives,
            self.min_span_len,
            min_support=self.min_support,
            max_candidates=self.max_candidates,
            fold_numbers=self.generalize,
        )
        if not candidates:
            return None

        scored: list[dict[str, Any]] = []
        for span, support in candidates:
            regex_src = generalize_span(span) if self.generalize else re.escape(span)
            if not regex_src or not safe_regex(regex_src):
                continue
            pat = self._compile(regex_src)
            tp = sum(1 for p in positives if timed_search(pat, p))
            if tp == 0:
                continue
            fp = sum(1 for n in neg_pool if timed_search(pat, n))
            recall = tp / len(positives)
            if neg_pool:
                precision = tp / (tp + fp)
            else:
                precision = recall  # proxy; flagged in metadata
            if precision < self.min_precision:
                continue
            scored.append(
                {
                    "span": span,
                    "regex": regex_src,
                    "precision": float(precision),
                    "recall": float(recall),
                    "support": int(support),
                    "tp": tp,
                    "fp": fp,
                }
            )

        scored = _collapse_substrings(scored)
        scored.sort(key=lambda d: (d["precision"], d["recall"], len(d["span"])), reverse=True)
        scored = scored[: self.max_patterns]

        covered = sum(
            1
            for p in positives
            if any(timed_search(self._compile(d["regex"]), p) for d in scored)
        )
        return CompiledArtifact(
            kind=self.kind,
            name=f"regex-{cluster.label}",
            payload={
                "patterns": [
                    {
                        "span": d["span"],
                        "regex": d["regex"],
                        "precision": d["precision"],
                        "recall": d["recall"],
                        "support": d["support"],
                    }
                    for d in scored
                ],
            },
            cluster_label=cluster.label,
            attack_success_rate=cluster.attack_success_rate,
            n_records=cluster.size,
            summary=cluster.summary,
            metadata={
                "n_patterns": len(scored),
                "n_positives": len(positives),
                "n_negatives": len(neg_pool),
                "precision_source": "negatives" if neg_pool else "recall_proxy",
                "cluster_recall": covered / len(positives) if positives else 0.0,
                "min_precision": self.min_precision,
            },
        )

    def _compile(self, src: str) -> re.Pattern[str]:
        pat = self._compiled_cache.get(src)
        if pat is None:
            pat = re.compile(src, re.IGNORECASE)
            self._compiled_cache[src] = pat
        return pat


def _candidate_spans(
    positives: list[str],
    min_len: int,
    *,
    min_support: int = 2,
    max_candidates: int = 50,
    max_words: int = 6,
    fold_numbers: bool = True,
) -> list[tuple[str, int]]:
    """Word-aligned common-substring search.

    Text is tokenized into words (punctuation dropped). Every window
    of 1..``max_words`` consecutive words whose character length is at
    least ``min_len`` is a candidate. We count the number of *distinct*
    positives each candidate occurs in (document frequency rather than
    raw occurrences, so one repetitive sample cannot dominate) and
    return the best-supported spans, longer first among ties.

    With ``fold_numbers`` every all-digit token is replaced by
    :data:`NUMBER_PLACEHOLDER` before counting, so ``"transfer 100 to"``
    and ``"transfer 2500 to"`` support the same span, which
    :func:`generalize_span` then renders as ``transfer\\s+\\d+\\s+to``.
    """
    support: Counter[str] = Counter()
    for text in positives:
        words = _WORD.findall(text.lower())
        if fold_numbers:
            words = [NUMBER_PLACEHOLDER if w.isdigit() else w for w in words]
        seen: set[str] = set()
        for n in range(1, max_words + 1):
            for i in range(len(words) - n + 1):
                span = " ".join(words[i : i + n])
                if len(span) >= min_len:
                    seen.add(span)
        support.update(seen)
    common = [(span, c) for span, c in support.items() if c >= min_support]
    common.sort(key=lambda sc: (sc[1], len(sc[0])), reverse=True)
    return common[:max_candidates]


def _collapse_substrings(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop spans dominated by a longer kept span.

    A shorter span is redundant when a longer span containing it has at
    least the same precision *and* the same recall: the longer span then
    catches everything the shorter one does while being more specific.
    When the shorter span has strictly higher recall it is kept because
    it covers positives the longer one misses.
    """
    by_len = sorted(scored, key=lambda d: len(d["span"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in by_len:
        redundant = any(
            cand["span"] in k["span"]
            and k["precision"] >= cand["precision"]
            and k["recall"] >= cand["recall"]
            for k in kept
        )
        if not redundant:
            kept.append(cand)
    return kept


def _estimate_precision(pat: re.Pattern[str], positives: list[str]) -> float:
    """Legacy proxy retained for backwards compatibility: the fraction of
    positives the pattern catches. Prefer the negative-pool precision
    computed inside :meth:`_RegexTarget.compile_for_cluster`."""
    hits = sum(1 for p in positives if pat.search(p))
    return hits / max(len(positives), 1)
