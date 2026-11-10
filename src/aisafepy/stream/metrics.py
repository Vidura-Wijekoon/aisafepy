"""Pipeline observability: latency histograms, outcome counters, FP monitoring.

Three building blocks:

* :class:`LatencyHistogram`. A bounded reservoir of latency samples
  that reports exact percentiles over the retained sample. Reservoir
  sampling (Vitter's Algorithm R) keeps memory constant under
  unbounded traffic while remaining an unbiased sample.
* :class:`PipelineMetrics`. Per-guard, per-tier and final-decision
  counters plus histograms. Attach one to a
  :class:`aisafepy.stream.GuardPipeline` via ``metrics=``; the pipeline
  records every guard call and every final decision. ``snapshot()``
  gives a JSON-shaped dict and ``to_prometheus()`` renders the text
  exposition format for scraping.
* :class:`FalsePositiveMonitor`. A sliding-window block-rate tracker
  keyed by tenant / client / IP. Adversarial false-positive attacks
  (arXiv 2410.02916) try to drive a guard into blocking benign traffic;
  per-key block-rate alerts make that visible before it becomes an
  outage. See ``docs/CAVEATS.md``.

Everything here is thread-safe and dependency-free.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from aisafepy.core.decisions import Action, GuardDecision

_ACTIONS: tuple[str, ...] = tuple(a.value for a in Action)


# ---- latency histogram ---------------------------------------------------


class LatencyHistogram:
    """Bounded reservoir of latency samples with exact percentiles.

    Parameters
    ----------
    capacity:
        Maximum samples retained. Once full, new samples replace
        existing ones with probability ``capacity / n`` (Algorithm R),
        so the reservoir stays a uniform sample of everything seen.
    seed:
        Optional RNG seed for deterministic replacement in tests.
    """

    def __init__(self, capacity: int = 2048, *, seed: int | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._samples: list[float] = []
        self._count = 0
        self._sum = 0.0
        self._max = 0.0
        self._min = math.inf
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def record(self, value_ms: float) -> None:
        if value_ms < 0:
            raise ValueError("latency must be non-negative")
        with self._lock:
            self._count += 1
            self._sum += value_ms
            self._max = max(self._max, value_ms)
            self._min = min(self._min, value_ms)
            if len(self._samples) < self.capacity:
                self._samples.append(value_ms)
            else:
                j = self._rng.randrange(self._count)
                if j < self.capacity:
                    self._samples[j] = value_ms

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        with self._lock:
            return self._sum / self._count if self._count else 0.0

    @property
    def max(self) -> float:
        return self._max if self._count else 0.0

    @property
    def min(self) -> float:
        return self._min if self._count else 0.0

    def percentile(self, q: float) -> float:
        """Return the ``q``-th percentile (``q`` in [0, 100]) of the reservoir.

        Uses nearest-rank on the sorted sample. Returns ``0.0`` when
        empty.
        """
        if not 0.0 <= q <= 100.0:
            raise ValueError("q must be in [0, 100]")
        with self._lock:
            data = sorted(self._samples)
        if not data:
            return 0.0
        rank = max(1, math.ceil(q / 100.0 * len(data)))
        return data[rank - 1]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean, 4),
            "min_ms": round(self.min, 4),
            "max_ms": round(self.max, 4),
            "p50_ms": round(self.p50, 4),
            "p95_ms": round(self.p95, 4),
            "p99_ms": round(self.p99, 4),
        }

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._count = 0
            self._sum = 0.0
            self._max = 0.0
            self._min = math.inf


# ---- per-guard stats -----------------------------------------------------


@dataclass
class GuardStats:
    """Counters and latency for a single guard (or a tier)."""

    name: str
    tier: int = 0
    count: int = 0
    actions: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_ACTIONS, 0))
    errors: int = 0
    timeouts: int = 0
    latency: LatencyHistogram = field(default_factory=LatencyHistogram)

    def record(
        self,
        decision: GuardDecision,
        latency_ms: float,
        *,
        error: bool = False,
        timeout: bool = False,
    ) -> None:
        self.count += 1
        self.actions[decision.action.value] = self.actions.get(decision.action.value, 0) + 1
        if error:
            self.errors += 1
        if timeout:
            self.timeouts += 1
        self.latency.record(max(0.0, latency_ms))

    @property
    def block_rate(self) -> float:
        return self.actions.get("block", 0) / self.count if self.count else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "count": self.count,
            "actions": dict(self.actions),
            "errors": self.errors,
            "timeouts": self.timeouts,
            "block_rate": round(self.block_rate, 6),
            "latency": self.latency.snapshot(),
        }


# ---- pipeline metrics ----------------------------------------------------


class PipelineMetrics:
    """Aggregated metrics for one :class:`GuardPipeline`.

    All ``record_*`` methods are safe to call from multiple threads
    and from concurrently running coroutines.
    """

    def __init__(self, *, namespace: str = "aisafepy", histogram_capacity: int = 2048) -> None:
        self.namespace = namespace
        self.histogram_capacity = histogram_capacity
        self._lock = threading.Lock()
        self._guards: dict[str, GuardStats] = {}
        self._tiers: dict[int, GuardStats] = {}
        self._final = GuardStats(name="final")
        self._shadow: dict[str, GuardStats] = {}
        self._started_at = time.time()

    # ---- recording -------------------------------------------------------

    def record_guard(
        self,
        name: str,
        *,
        tier: int,
        decision: GuardDecision,
        latency_ms: float,
        error: bool = False,
        timeout: bool = False,
    ) -> None:
        with self._lock:
            stats = self._guards.get(name)
            if stats is None:
                stats = GuardStats(
                    name=name, tier=tier, latency=LatencyHistogram(self.histogram_capacity)
                )
                self._guards[name] = stats
            stats.record(decision, latency_ms, error=error, timeout=timeout)

    def record_tier(self, tier: int, decision: GuardDecision) -> None:
        with self._lock:
            stats = self._tiers.get(tier)
            if stats is None:
                stats = GuardStats(
                    name=f"tier{tier}", tier=tier, latency=LatencyHistogram(self.histogram_capacity)
                )
                self._tiers[tier] = stats
            stats.record(decision, decision.latency_ms)

    def record_final(self, decision: GuardDecision, *, latency_ms: float) -> None:
        with self._lock:
            self._final.record(decision, latency_ms)

    def record_shadow(self, decision: GuardDecision) -> None:
        name = decision.guard_name or "shadow"
        with self._lock:
            stats = self._shadow.get(name)
            if stats is None:
                stats = GuardStats(
                    name=name,
                    tier=decision.tier,
                    latency=LatencyHistogram(self.histogram_capacity),
                )
                self._shadow[name] = stats
            stats.record(decision, decision.latency_ms)

    # ---- reporting -------------------------------------------------------

    def guard(self, name: str) -> GuardStats | None:
        with self._lock:
            return self._guards.get(name)

    @property
    def final(self) -> GuardStats:
        return self._final

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "namespace": self.namespace,
                "uptime_s": round(time.time() - self._started_at, 3),
                "final": self._final.snapshot(),
                "tiers": {str(t): s.snapshot() for t, s in sorted(self._tiers.items())},
                "guards": {n: s.snapshot() for n, s in sorted(self._guards.items())},
                "shadow": {n: s.snapshot() for n, s in sorted(self._shadow.items())},
            }

    def reset(self) -> None:
        with self._lock:
            self._guards.clear()
            self._tiers.clear()
            self._shadow.clear()
            self._final = GuardStats(name="final")
            self._started_at = time.time()

    def to_prometheus(self) -> str:
        """Render the Prometheus text exposition format (version 0.0.4)."""
        ns = self.namespace
        out: list[str] = []

        def emit(
            metric: str, mtype: str, help_text: str, rows: Iterable[tuple[dict[str, str], float]]
        ) -> None:
            rows = list(rows)
            if not rows:
                return
            out.append(f"# HELP {ns}_{metric} {help_text}")
            out.append(f"# TYPE {ns}_{metric} {mtype}")
            for labels, value in rows:
                out.append(f"{ns}_{metric}{_fmt_labels(labels)} {_fmt_value(value)}")

        with self._lock:
            guards = list(self._guards.values())
            tiers = list(self._tiers.values())
            shadow = list(self._shadow.values())
            final = self._final

        def stat_rows(
            stats_list: list[GuardStats], scope: str
        ) -> tuple[list[tuple[dict[str, str], float]], ...]:
            evals: list[tuple[dict[str, str], float]] = []
            decisions: list[tuple[dict[str, str], float]] = []
            errors: list[tuple[dict[str, str], float]] = []
            timeouts: list[tuple[dict[str, str], float]] = []
            latency: list[tuple[dict[str, str], float]] = []
            for s in stats_list:
                base = {"scope": scope, "name": s.name, "tier": str(s.tier)}
                evals.append((base, s.count))
                for action, n in s.actions.items():
                    decisions.append(({**base, "action": action}, n))
                errors.append((base, s.errors))
                timeouts.append((base, s.timeouts))
                for q, v in (("0.5", s.latency.p50), ("0.95", s.latency.p95), ("0.99", s.latency.p99)):
                    latency.append(({**base, "quantile": q}, v))
            return evals, decisions, errors, timeouts, latency

        g = stat_rows(guards, "guard")
        t = stat_rows(tiers, "tier")
        sh = stat_rows(shadow, "shadow")
        f = stat_rows([final] if final.count else [], "final")

        emit(
            "guard_evaluations_total",
            "counter",
            "Number of guard evaluations.",
            g[0] + t[0] + sh[0] + f[0],
        )
        emit(
            "guard_decisions_total",
            "counter",
            "Guard decisions by action.",
            g[1] + t[1] + sh[1] + f[1],
        )
        emit("guard_errors_total", "counter", "Guard calls that raised.", g[2] + sh[2])
        emit("guard_timeouts_total", "counter", "Guard calls that timed out.", g[3] + sh[3])
        emit(
            "guard_latency_ms",
            "summary",
            "Guard latency quantiles in milliseconds.",
            g[4] + t[4] + sh[4] + f[4],
        )
        return "\n".join(out) + ("\n" if out else "")


def _fmt_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{_escape_label(v)}"' for k, v in labels.items()]
    return "{" + ",".join(parts) + "}"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt_value(value: float) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


# ---- false-positive monitor ---------------------------------------------


@dataclass(frozen=True)
class FPAlert:
    """Raised (as data) when a key's block rate exceeds the threshold."""

    key: str
    block_rate: float
    blocked: int
    total: int
    window_s: float


class FalsePositiveMonitor:
    """Sliding-window block-rate tracker per key.

    Parameters
    ----------
    window_s:
        Length of the sliding window in seconds.
    max_keys:
        Upper bound on tracked keys. The least recently seen key is
        evicted when the bound is hit, which also caps memory under a
        key-flooding attack.
    clock:
        Injectable time source (defaults to :func:`time.monotonic`).
    """

    def __init__(
        self,
        *,
        window_s: float = 300.0,
        max_keys: int = 10_000,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        if max_keys < 1:
            raise ValueError("max_keys must be >= 1")
        self.window_s = float(window_s)
        self.max_keys = max_keys
        self._clock = clock or time.monotonic
        self._events: dict[str, deque[tuple[float, bool]]] = {}
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def record(self, key: str, blocked: bool, *, now: float | None = None) -> None:
        ts = self._clock() if now is None else now
        with self._lock:
            q = self._events.get(key)
            if q is None:
                if len(self._events) >= self.max_keys:
                    self._evict_lru()
                q = deque()
                self._events[key] = q
            q.append((ts, bool(blocked)))
            self._last_seen[key] = ts
            self._trim(q, ts)

    def record_decision(self, key: str, decision: GuardDecision, *, now: float | None = None) -> None:
        self.record(key, decision.action == Action.BLOCK, now=now)

    def counts(self, key: str, *, now: float | None = None) -> tuple[int, int]:
        """Return ``(blocked, total)`` within the window for ``key``."""
        ts = self._clock() if now is None else now
        with self._lock:
            q = self._events.get(key)
            if q is None:
                return 0, 0
            self._trim(q, ts)
            total = len(q)
            blocked = sum(1 for _, b in q if b)
            return blocked, total

    def block_rate(self, key: str, *, now: float | None = None) -> float:
        blocked, total = self.counts(key, now=now)
        return blocked / total if total else 0.0

    def alerts(
        self,
        *,
        threshold: float = 0.5,
        min_samples: int = 20,
        now: float | None = None,
    ) -> list[FPAlert]:
        """Return keys whose block rate meets ``threshold`` over at least
        ``min_samples`` events in the window, highest rate first."""
        ts = self._clock() if now is None else now
        out: list[FPAlert] = []
        with self._lock:
            for key, q in self._events.items():
                self._trim(q, ts)
                total = len(q)
                if total < min_samples:
                    continue
                blocked = sum(1 for _, b in q if b)
                rate = blocked / total
                if rate >= threshold:
                    out.append(FPAlert(key, rate, blocked, total, self.window_s))
        out.sort(key=lambda a: (a.block_rate, a.total), reverse=True)
        return out

    def prune(self, *, now: float | None = None) -> int:
        """Drop keys with no events inside the window. Returns the count dropped."""
        ts = self._clock() if now is None else now
        dropped = 0
        with self._lock:
            for key in list(self._events):
                q = self._events[key]
                self._trim(q, ts)
                if not q:
                    del self._events[key]
                    self._last_seen.pop(key, None)
                    dropped += 1
        return dropped

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._events)

    def _trim(self, q: deque[tuple[float, bool]], now: float) -> None:
        cutoff = now - self.window_s
        while q and q[0][0] <= cutoff:
            q.popleft()

    def _evict_lru(self) -> None:
        if not self._last_seen:
            return
        oldest = min(self._last_seen, key=self._last_seen.__getitem__)
        self._events.pop(oldest, None)
        self._last_seen.pop(oldest, None)


# ---- helpers --------------------------------------------------------------


def decision_rate_summary(decisions: Iterable[GuardDecision]) -> dict[str, Any]:
    """Summarize a batch of decisions: action counts/rates and latency stats."""
    hist = LatencyHistogram(capacity=1 << 16)
    counts = dict.fromkeys(_ACTIONS, 0)
    by_guard: dict[str, int] = {}
    total = 0
    for d in decisions:
        total += 1
        counts[d.action.value] = counts.get(d.action.value, 0) + 1
        hist.record(max(0.0, d.latency_ms))
        if d.action != Action.ALLOW:
            g = d.guard_name or "unknown"
            by_guard[g] = by_guard.get(g, 0) + 1
    rates = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return {
        "total": total,
        "counts": counts,
        "rates": {k: round(v, 6) for k, v in rates.items()},
        "non_allow_by_guard": dict(sorted(by_guard.items(), key=lambda kv: -kv[1])),
        "latency": hist.snapshot(),
    }
