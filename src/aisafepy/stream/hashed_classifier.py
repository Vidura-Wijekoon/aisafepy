"""A pure-Python hashed n-gram text classifier that works as a Tier-2 guard.

Why this exists
---------------

The ``adapt`` compiler's classifier target distils a transformer, which
needs ``torch`` and a GPU-hour to produce anything deployable. Most
red-team clusters do not need that: the failures share surface
vocabulary ("ignore previous instructions", "developer mode", a
particular tool name) and a linear model over hashed n-grams catches
them with sub-millisecond latency and no dependencies. This module is
that model, packaged as a Guard so the canary controller can deploy it
the same way it deploys a regex.

How it works
------------

* **Features.** Text goes through
  :func:`aisafepy.stream.normalize.normalize_for_matching` (NFKC,
  zero-width stripping, confusable folding, casefold) so homoglyph and
  spacing tricks collapse onto the same tokens. Word unigrams, word
  bigrams and character 3- to 5-grams are hashed into ``n_buckets``
  slots with CRC32. A second hash picks a sign for each feature so
  collisions cancel in expectation instead of piling up (the "signed
  hashing trick"). Vectors are L2-normalised.
* **Model.** Logistic regression trained with mini-batch SGD and
  per-feature AdaGrad step sizes, L2 regularisation applied lazily to
  touched weights, inverse-frequency class weights for imbalance, and
  early stopping on a held-out split. Everything is deterministic given
  ``seed``; CRC32 is used instead of ``hash()`` precisely because the
  latter is salted per process.
* **Explanations.** During training the model remembers one example
  n-gram per bucket (bounded), so :meth:`explain` can say *which*
  n-grams pushed a text over the threshold.

Limits, stated plainly
----------------------

This is a bag-of-n-grams model. It does not understand paraphrase,
negation, or intent; "do not ignore the instructions" and "ignore the
instructions" share almost every feature. It generalises only to texts
that reuse the training vocabulary. Hash collisions can attribute weight
to unrelated n-grams, so :meth:`explain` is indicative, not causal.
Treat it as a cheap, fast, auditable Tier-2 signal for a *specific*
failure cluster, and pair it with Tier-1 structure checks and a Tier-3
judge for anything open-ended.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
import time
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.normalize import normalize_for_matching
from aisafepy.stream.pipeline import Context

SCHEMA_VERSION = 1

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


@dataclass
class TrainingReport:
    """What :meth:`HashedNGramClassifierGuard.fit` learned and how well."""

    n_positives: int
    n_negatives: int
    n_validation: int
    epochs_run: int
    best_epoch: int
    train_loss: list[float]
    validation_loss: list[float]
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    threshold: float
    n_features: int
    train_seconds: float
    early_stopped: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class HashedNGramClassifierGuard:
    """Linear classifier over hashed word and character n-grams.

    Parameters
    ----------
    name, tier, threshold:
        Guard protocol fields. ``threshold`` is the unsafe probability at
        or above which the guard blocks.
    n_buckets:
        Size of the hashed feature space. 2**18 keeps collisions rare for
        vocabularies in the tens of thousands while the JSON model file
        stays small because only non-zero weights are stored.
    char_ngram_range:
        Inclusive range of character n-gram lengths.
    use_word_bigrams:
        Add adjacent word pairs as features.
    max_chars:
        Inputs longer than this are truncated (head) before featurising
        so a hostile input cannot force quadratic work.
    max_feature_examples:
        Cap on the bucket -> example n-gram map kept for explanations.
    """

    name: str = "hashed-ngram"
    tier: int = 2
    threshold: float = 0.5
    n_buckets: int = 1 << 18
    char_ngram_range: tuple[int, int] = (3, 5)
    use_word_bigrams: bool = True
    max_chars: int = 20_000
    max_feature_examples: int = 50_000
    weights: dict[int, float] = field(default_factory=dict, repr=False)
    bias: float = 0.0
    fitted: bool = False
    feature_examples: dict[int, str] = field(default_factory=dict, repr=False)
    report: TrainingReport | None = field(default=None, repr=False)
    source_path: str | None = None

    # ---- features -------------------------------------------------------

    def _bucket(self, token: str) -> tuple[int, float]:
        data = token.encode("utf-8", errors="replace")
        idx = zlib.crc32(data) % self.n_buckets
        sign = 1.0 if (zlib.crc32(b"\x01" + data) & 1) else -1.0
        return idx, sign

    def tokens(self, text: str) -> list[str]:
        """Return the raw (unhashed) feature strings for ``text``."""
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        norm = normalize_for_matching(text)
        words = _WORD_RE.findall(norm)
        out: list[str] = [f"w:{w}" for w in words]
        if self.use_word_bigrams and len(words) > 1:
            out.extend(f"wb:{a} {b}" for a, b in itertools.pairwise(words))
        compact = _SPACE_RE.sub(" ", norm).strip()
        lo, hi = self.char_ngram_range
        for n in range(lo, hi + 1):
            if len(compact) < n:
                break
            prefix = f"c{n}:"
            out.extend(prefix + compact[i : i + n] for i in range(len(compact) - n + 1))
        return out

    def featurize(self, text: str, *, remember: bool = False) -> dict[int, float]:
        """Hash ``text`` into a sparse, L2-normalised feature vector."""
        counts: dict[int, float] = {}
        for tok in self.tokens(text):
            idx, sign = self._bucket(tok)
            counts[idx] = counts.get(idx, 0.0) + sign
            if remember and idx not in self.feature_examples and len(
                self.feature_examples
            ) < self.max_feature_examples:
                self.feature_examples[idx] = tok
        norm = math.sqrt(sum(v * v for v in counts.values()))
        if norm == 0.0:
            return {}
        return {k: v / norm for k, v in counts.items() if v != 0.0}

    # ---- inference ------------------------------------------------------

    def _logit(self, feats: dict[int, float]) -> float:
        w = self.weights
        return self.bias + sum(w.get(k, 0.0) * v for k, v in feats.items())

    def predict_proba(self, text: str) -> float:
        """Probability that ``text`` belongs to the positive (unsafe) class."""
        return _sigmoid(self._logit(self.featurize(text)))

    def predict_batch(self, texts: Iterable[str]) -> list[float]:
        return [self.predict_proba(t) for t in texts]

    def explain(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Top contributing n-grams for ``text``, most influential first.

        Each entry has ``feature`` (an example n-gram for the bucket, or
        ``bucket:<id>`` when none was recorded), ``weight``, ``value``
        and ``contribution`` (their product, signed).
        """
        feats = self.featurize(text)
        contribs = []
        for k, v in feats.items():
            w = self.weights.get(k)
            if w is None or w == 0.0:
                continue
            contribs.append((abs(w * v), k, w, v))
        contribs.sort(reverse=True)
        return [
            {
                "feature": self.feature_examples.get(k, f"bucket:{k}"),
                "bucket": k,
                "weight": round(w, 6),
                "value": round(v, 6),
                "contribution": round(w * v, 6),
            }
            for _, k, w, v in contribs[:top_k]
        ]

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        if not self.fitted:
            return GuardDecision.allow(
                tier=self.tier,
                rationale=f"{self.name}: model not fitted; failing open",
                guard_name=self.name,
                evidence={"unfitted": True},
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        text = ctx.buffer or ctx.chunk or ""
        if not text.strip():
            return GuardDecision.allow(tier=self.tier, guard_name=self.name, rationale="empty input")
        try:
            prob = self.predict_proba(text)
            top = self.explain(text, top_k=5)
        except Exception as exc:  # pragma: no cover - defensive
            return GuardDecision.allow(
                tier=self.tier,
                rationale=f"{self.name}: inference raised; failing open",
                guard_name=self.name,
                evidence={"error": repr(exc)},
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        evidence = {"prob": round(prob, 6), "threshold": self.threshold, "top_features": top}
        if prob >= self.threshold:
            return GuardDecision.block(
                tier=self.tier,
                confidence=prob,
                rationale=f"{self.name}: P(unsafe)={prob:.3f} >= {self.threshold}",
                guard_name=self.name,
                evidence=evidence,
                severity="high" if prob >= 0.9 else "medium",
                latency_ms=latency_ms,
            )
        return GuardDecision.allow(
            tier=self.tier,
            confidence=1.0 - prob,
            rationale=f"{self.name}: P(unsafe)={prob:.3f} < {self.threshold}",
            guard_name=self.name,
            evidence=evidence,
            latency_ms=latency_ms,
        )

    # ---- training -------------------------------------------------------

    def fit(
        self,
        positives: Sequence[str],
        negatives: Sequence[str],
        *,
        epochs: int = 10,
        lr: float = 0.1,
        l2: float = 1e-5,
        seed: int = 0,
        validation_fraction: float = 0.1,
        batch_size: int = 32,
        patience: int = 2,
    ) -> TrainingReport:
        """Train from scratch on ``positives`` (unsafe) and ``negatives`` (safe).

        A ``validation_fraction`` of each class is held out (at least one
        example per class when the class has two or more); training
        stops when validation loss has not improved for ``patience``
        epochs and the best epoch's weights are kept.
        """
        if not positives or not negatives:
            raise ValueError("fit() needs at least one positive and one negative example")
        t0 = time.perf_counter()
        rng = random.Random(seed)
        self.weights = {}
        self.bias = 0.0
        self.feature_examples = {}

        pos = list(positives)
        neg = list(negatives)
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_val_pos = _holdout(len(pos), validation_fraction)
        n_val_neg = _holdout(len(neg), validation_fraction)
        val = [(self.featurize(t), 1.0) for t in pos[:n_val_pos]]
        val += [(self.featurize(t), 0.0) for t in neg[:n_val_neg]]
        train = [(self.featurize(t, remember=True), 1.0) for t in pos[n_val_pos:]]
        train += [(self.featurize(t, remember=True), 0.0) for t in neg[n_val_neg:]]

        n_pos = len(pos) - n_val_pos
        n_neg = len(neg) - n_val_neg
        total = n_pos + n_neg
        w_pos = total / (2.0 * max(n_pos, 1))
        w_neg = total / (2.0 * max(n_neg, 1))

        accum: dict[int, float] = {}
        accum_bias = 0.0
        eps = 1e-8
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val = math.inf
        best_state: tuple[dict[int, float], float] = ({}, 0.0)
        best_epoch = 0
        stale = 0
        epochs_run = 0
        early = False

        for epoch in range(1, epochs + 1):
            rng.shuffle(train)
            epoch_loss = 0.0
            for start in range(0, len(train), batch_size):
                batch = train[start : start + batch_size]
                grad: dict[int, float] = {}
                grad_bias = 0.0
                for feats, y in batch:
                    p = _sigmoid(self._logit(feats))
                    cw = w_pos if y == 1.0 else w_neg
                    epoch_loss += cw * _log_loss(p, y)
                    err = (p - y) * cw / len(batch)
                    grad_bias += err
                    for k, v in feats.items():
                        grad[k] = grad.get(k, 0.0) + err * v
                for k, g in grad.items():
                    wk = self.weights.get(k, 0.0)
                    g += l2 * wk
                    a = accum.get(k, 0.0) + g * g
                    accum[k] = a
                    self.weights[k] = wk - lr * g / math.sqrt(a + eps)
                accum_bias += grad_bias * grad_bias
                self.bias -= lr * grad_bias / math.sqrt(accum_bias + eps)
            train_losses.append(epoch_loss / max(len(train), 1))
            epochs_run = epoch

            if val:
                v_loss = sum(_log_loss(_sigmoid(self._logit(f)), y) for f, y in val) / len(val)
            else:
                v_loss = train_losses[-1]
            val_losses.append(v_loss)
            if v_loss < best_val - 1e-9:
                best_val = v_loss
                best_state = (dict(self.weights), self.bias)
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    early = True
                    break

        self.weights, self.bias = best_state
        self.weights = {k: w for k, w in self.weights.items() if w != 0.0}
        self.fitted = True

        eval_set = val or train
        tp = fp = tn = fn = 0
        for feats, y in eval_set:
            pred = _sigmoid(self._logit(feats)) >= self.threshold
            if pred and y == 1.0:
                tp += 1
            elif pred:
                fp += 1
            elif y == 1.0:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        self.report = TrainingReport(
            n_positives=len(pos),
            n_negatives=len(neg),
            n_validation=len(val),
            epochs_run=epochs_run,
            best_epoch=best_epoch,
            train_loss=[round(x, 6) for x in train_losses],
            validation_loss=[round(x, 6) for x in val_losses],
            precision=precision,
            recall=recall,
            f1=f1,
            false_positive_rate=fpr,
            threshold=self.threshold,
            n_features=len(self.weights),
            train_seconds=time.perf_counter() - t0,
            early_stopped=early,
        )
        return self.report

    # ---- persistence ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "tier": self.tier,
            "threshold": self.threshold,
            "n_buckets": self.n_buckets,
            "char_ngram_range": list(self.char_ngram_range),
            "use_word_bigrams": self.use_word_bigrams,
            "max_chars": self.max_chars,
            "bias": self.bias,
            "weights": {str(k): v for k, v in sorted(self.weights.items())},
            "feature_examples": {str(k): v for k, v in self.feature_examples.items()},
            "fitted": self.fitted,
            "report": self.report.to_dict() if self.report else None,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8")
        self.source_path = str(p)
        return p

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HashedNGramClassifierGuard:
        version = int(data.get("schema_version", 1))
        if version > SCHEMA_VERSION:
            raise ValueError(f"unsupported hashed classifier schema_version {version}")
        rng = data.get("char_ngram_range", [3, 5])
        guard = cls(
            name=str(data.get("name", "hashed-ngram")),
            tier=int(data.get("tier", 2)),
            threshold=float(data.get("threshold", 0.5)),
            n_buckets=int(data.get("n_buckets", 1 << 18)),
            char_ngram_range=(int(rng[0]), int(rng[1])),
            use_word_bigrams=bool(data.get("use_word_bigrams", True)),
            max_chars=int(data.get("max_chars", 20_000)),
            bias=float(data.get("bias", 0.0)),
            weights={int(k): float(v) for k, v in (data.get("weights") or {}).items()},
            feature_examples={int(k): str(v) for k, v in (data.get("feature_examples") or {}).items()},
            fitted=bool(data.get("fitted", False)),
        )
        rep = data.get("report")
        if isinstance(rep, dict):
            try:
                guard.report = TrainingReport(**rep)
            except TypeError:
                guard.report = None
        return guard

    @classmethod
    def load(cls, path: str | Path) -> HashedNGramClassifierGuard:
        p = Path(path)
        guard = cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        guard.source_path = str(p)
        return guard

    def to_spec(self) -> dict[str, Any]:
        """Serializable description for the pipeline config loader."""
        spec: dict[str, Any] = {
            "kind": "hashed_classifier",
            "name": self.name,
            "tier": self.tier,
            "threshold": self.threshold,
        }
        if self.source_path:
            spec["path"] = self.source_path
        return spec


# ---- helpers ----------------------------------------------------------------


def _holdout(n: int, fraction: float) -> int:
    if n < 2 or fraction <= 0.0:
        return 0
    return max(1, min(n - 1, round(n * fraction)))


def _log_loss(p: float, y: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
