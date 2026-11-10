from __future__ import annotations

import math
import threading

import pytest

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.metrics import (
    FalsePositiveMonitor,
    GuardStats,
    LatencyHistogram,
    PipelineMetrics,
    decision_rate_summary,
)


def _nearest_rank(data: list[float], q: float) -> float:
    s = sorted(data)
    rank = max(1, math.ceil(q / 100 * len(s)))
    return s[rank - 1]


# ---- histogram ----------------------------------------------------------------


def test_histogram_percentiles_match_sorted_truth():
    h = LatencyHistogram(capacity=10_000)
    values = [((i * 7919) % 1000) / 3.0 for i in range(1000)]
    for v in values:
        h.record(v)
    assert h.count == 1000
    assert h.p50 == _nearest_rank(values, 50)
    assert h.p95 == _nearest_rank(values, 95)
    assert h.p99 == _nearest_rank(values, 99)
    assert h.max == max(values)
    assert h.min == min(values)
    assert h.mean == pytest.approx(sum(values) / len(values))


def test_histogram_empty_and_validation():
    h = LatencyHistogram()
    assert h.p50 == 0.0 and h.max == 0.0 and h.mean == 0.0 and h.min == 0.0
    with pytest.raises(ValueError):
        h.record(-1)
    with pytest.raises(ValueError):
        h.percentile(101)
    with pytest.raises(ValueError):
        LatencyHistogram(capacity=0)


def test_histogram_reservoir_is_bounded_but_tracks_extremes():
    h = LatencyHistogram(capacity=64, seed=1)
    for i in range(5000):
        h.record(float(i))
    assert h.count == 5000
    assert len(h._samples) == 64
    assert h.max == 4999.0 and h.min == 0.0
    # The reservoir is a uniform sample, so the median lands near the middle.
    assert 1500 < h.p50 < 3500
    h.reset()
    assert h.count == 0 and h.p99 == 0.0


# ---- guard stats / pipeline metrics ------------------------------------------------


def test_guard_stats_counts_actions_errors_timeouts():
    s = GuardStats(name="g", tier=1)
    s.record(GuardDecision.allow(tier=1), 1.0)
    s.record(GuardDecision.block(tier=1, confidence=1.0, rationale="x"), 2.0)
    s.record(GuardDecision.allow(tier=1), 3.0, error=True)
    s.record(GuardDecision.allow(tier=1), 4.0, timeout=True)
    snap = s.snapshot()
    assert snap["count"] == 4
    assert snap["actions"]["allow"] == 3 and snap["actions"]["block"] == 1
    assert snap["errors"] == 1 and snap["timeouts"] == 1
    assert s.block_rate == 0.25
    assert snap["latency"]["max_ms"] == 4.0


def test_pipeline_metrics_snapshot_and_reset():
    m = PipelineMetrics()
    block = GuardDecision.block(tier=1, confidence=0.9, rationale="r", guard_name="bl")
    m.record_guard("bl", tier=1, decision=block, latency_ms=0.5)
    m.record_guard("bl", tier=1, decision=GuardDecision.allow(tier=1), latency_ms=0.7)
    m.record_tier(1, block)
    m.record_final(block, latency_ms=1.2)
    m.record_shadow(GuardDecision.allow(tier=2, guard_name="shadow-x").model_copy(
        update={"evidence": {"shadow": True}}
    ))
    snap = m.snapshot()
    assert snap["guards"]["bl"]["count"] == 2
    assert snap["guards"]["bl"]["actions"]["block"] == 1
    assert snap["tiers"]["1"]["count"] == 1
    assert snap["final"]["actions"]["block"] == 1
    assert snap["shadow"]["shadow-x"]["count"] == 1
    assert m.guard("bl") is not None and m.guard("nope") is None
    m.reset()
    assert m.snapshot()["guards"] == {} and m.final.count == 0


def test_pipeline_metrics_prometheus_format_and_escaping():
    m = PipelineMetrics(namespace="ns")
    weird = 'guard "q"\\n'
    m.record_guard(weird, tier=2, decision=GuardDecision.allow(tier=2), latency_ms=2.5)
    m.record_final(GuardDecision.allow(tier=2), latency_ms=3.0)
    text = m.to_prometheus()
    assert text.endswith("\n")
    assert "# TYPE ns_guard_evaluations_total counter" in text
    assert "# TYPE ns_guard_latency_ms summary" in text
    assert 'name="guard \\"q\\"\\\\n"' in text
    assert 'ns_guard_evaluations_total{scope="guard",name="guard \\"q\\"\\\\n",tier="2"} 1' in text
    assert 'ns_guard_decisions_total{scope="final",name="final",tier="0",action="allow"} 1' in text
    assert 'quantile="0.95"} 2.5' in text
    # Empty metrics render nothing rather than headers with no samples.
    assert PipelineMetrics().to_prometheus() == ""


def test_pipeline_metrics_thread_safe():
    m = PipelineMetrics()
    d = GuardDecision.allow(tier=1)

    def work():
        for _ in range(500):
            m.record_guard("g", tier=1, decision=d, latency_ms=0.1)
            m.record_final(d, latency_ms=0.2)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.guard("g").count == 4000
    assert m.final.count == 4000


# ---- FP monitor -------------------------------------------------------------------


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_fp_monitor_sliding_window_and_alerts():
    clock = Clock()
    mon = FalsePositiveMonitor(window_s=60, clock=clock)
    for i in range(10):
        mon.record("tenant-a", blocked=i < 8)
        mon.record("tenant-b", blocked=False)
    assert mon.counts("tenant-a") == (8, 10)
    assert mon.block_rate("tenant-a") == 0.8
    assert mon.block_rate("tenant-b") == 0.0
    assert mon.block_rate("unknown") == 0.0

    alerts = mon.alerts(threshold=0.5, min_samples=10)
    assert [a.key for a in alerts] == ["tenant-a"]
    assert alerts[0].blocked == 8 and alerts[0].total == 10 and alerts[0].window_s == 60
    assert mon.alerts(threshold=0.5, min_samples=11) == []

    # Advance past the window: old events drop out.
    clock.t += 61
    assert mon.counts("tenant-a") == (0, 0)
    mon.record("tenant-a", blocked=True)
    assert mon.counts("tenant-a") == (1, 1)
    assert mon.prune() == 1  # tenant-b had nothing left
    assert mon.keys() == ["tenant-a"]


def test_fp_monitor_record_decision_and_explicit_now():
    mon = FalsePositiveMonitor(window_s=10, clock=Clock(0))
    mon.record_decision("k", GuardDecision.block(tier=1, confidence=1.0, rationale="x"), now=5)
    mon.record_decision("k", GuardDecision.allow(tier=1), now=6)
    assert mon.counts("k", now=7) == (1, 2)
    assert mon.counts("k", now=16) == (0, 0)


def test_fp_monitor_lru_eviction_bounds_keys():
    clock = Clock()
    mon = FalsePositiveMonitor(window_s=100, max_keys=3, clock=clock)
    for i, k in enumerate(["a", "b", "c"]):
        clock.t = 1000 + i
        mon.record(k, blocked=False)
    clock.t = 1010
    mon.record("a", blocked=False)  # refresh a; b is now least recent
    clock.t = 1011
    mon.record("d", blocked=False)
    assert sorted(mon.keys()) == ["a", "c", "d"]
    with pytest.raises(ValueError):
        FalsePositiveMonitor(window_s=0)
    with pytest.raises(ValueError):
        FalsePositiveMonitor(max_keys=0)


# ---- summary ----------------------------------------------------------------------


def test_decision_rate_summary():
    decisions = [
        GuardDecision.allow(tier=1, latency_ms=1.0),
        GuardDecision.block(tier=1, confidence=1.0, rationale="x", guard_name="bl", latency_ms=2.0),
        GuardDecision.escalate(tier=2, confidence=0.5, rationale="y", guard_name="j", latency_ms=3.0),
        GuardDecision.block(tier=1, confidence=1.0, rationale="x", guard_name="bl", latency_ms=4.0),
    ]
    s = decision_rate_summary(decisions)
    assert s["total"] == 4
    assert s["counts"]["block"] == 2 and s["rates"]["block"] == 0.5
    assert s["non_allow_by_guard"] == {"bl": 2, "j": 1}
    assert s["latency"]["max_ms"] == 4.0
    assert decision_rate_summary([])["total"] == 0
