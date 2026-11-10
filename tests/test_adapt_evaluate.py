from __future__ import annotations

import json

import pytest

from aisafepy.adapt.evaluate import (
    EvaluationReport,
    LatencyStats,
    Metrics,
    best_threshold,
    evaluate_guard,
    threshold_sweep,
    unsafe_score,
)
from aisafepy.adapt.sources import FailureRecord
from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.classifiers import StubClassifierGuard
from aisafepy.stream.deterministic import RegexGuard
from aisafepy.stream.pipeline import GuardPipeline


def _records():
    return [
        FailureRecord(id="v1", input="HARMFUL request one", output="", was_violation=True,
                      attack_category="jailbreak"),
        FailureRecord(id="v2", input="another HARMFUL one", output="", was_violation=True,
                      attack_category="jailbreak"),
        FailureRecord(id="v3", input="subtle attack without the trigger", output="",
                      was_violation=True, attack_category="encoding"),
        FailureRecord(id="b1", input="benign hello", output="", was_violation=False),
        FailureRecord(id="b2", input="benign HARMFUL-looking but safe", output="",
                      was_violation=False, attack_category="lookalike"),
        FailureRecord(id="b3", input="plain weather question", output="", was_violation=False),
    ]


def test_metrics_math():
    m = Metrics(tp=8, fp=2, tn=88, fn=2)
    assert m.support == 100
    assert m.precision == pytest.approx(0.8)
    assert m.recall == pytest.approx(0.8)
    assert m.f1 == pytest.approx(0.8)
    assert m.fpr == pytest.approx(2 / 90)
    assert m.fnr == pytest.approx(0.2)
    assert m.accuracy == pytest.approx(0.96)
    empty = Metrics()
    assert empty.precision == 0.0 and empty.recall == 0.0 and empty.f1 == 0.0
    assert set(m.to_dict()) >= {"tp", "fp", "tn", "fn", "precision", "recall", "f1", "fpr"}


def test_latency_stats_percentiles():
    s = LatencyStats.from_samples([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s.count == 5 and s.p50_ms == 3.0 and s.max_ms == 100.0
    assert 4.0 < s.p95_ms < 100.0
    assert LatencyStats.from_samples([]).count == 0
    assert LatencyStats.from_samples([7.0]).p95_ms == 7.0


def test_unsafe_score_complements_allow_confidence():
    assert unsafe_score(GuardDecision.allow(confidence=0.9)) == pytest.approx(0.1)
    assert unsafe_score(GuardDecision.block(tier=1, confidence=0.7, rationale="x")) == 0.7


def test_evaluate_stub_guard_reports_confusion_and_categories():
    rep = evaluate_guard(StubClassifierGuard(triggers=("HARMFUL",)), _records())
    m = rep.metrics
    assert (m.tp, m.fn, m.fp, m.tn) == (2, 1, 1, 2)
    assert rep.false_negatives == ["v3"]
    assert rep.false_positives == ["b2"]
    assert rep.per_category["jailbreak"].recall == 1.0
    assert rep.per_category["encoding"].recall == 0.0
    assert rep.per_category["lookalike"].fpr == 1.0
    assert rep.n_records == 6 and rep.errors == 0 and rep.latency.count == 6
    assert rep.name == "stub"
    md = rep.to_markdown()
    assert "| violation | 2 | 1 |" in md and "encoding" in md and "`v3`" in md
    assert json.loads(rep.to_json())["metrics"]["tp"] == 2


def test_evaluate_pipeline_and_output_field():
    recs = [
        FailureRecord(id="o1", input="q", output="here is my api key: AKIA" + "ABCDEFGHIJKLMNOP",
                      was_violation=True),
        FailureRecord(id="o2", input="q", output="nothing sensitive", was_violation=False),
    ]
    pipe = GuardPipeline(tier1=[RegexGuard.compile_pii()])
    rep = evaluate_guard(pipe, recs, field="output", name="pii-pipeline")
    assert rep.metrics.tp == 1 and rep.metrics.tn == 1
    assert rep.name == "pii-pipeline" and rep.field == "output"
    both = evaluate_guard(pipe, recs, field="both")
    assert both.metrics.tp == 1


def test_evaluate_with_threshold_uses_unsafe_score():
    class Scored:
        name = "scored"
        tier = 2

        async def __call__(self, ctx):
            unsafe = 0.8 if "HARMFUL" in ctx.buffer else 0.2
            return GuardDecision.allow(tier=2, confidence=1.0 - unsafe, guard_name="scored")

    # Without a threshold the guard never blocks, so recall is zero.
    assert evaluate_guard(Scored(), _records()).metrics.recall == 0.0
    rep = evaluate_guard(Scored(), _records(), threshold=0.5)
    assert rep.metrics.tp == 2 and rep.metrics.fp == 1
    assert rep.threshold == 0.5


def test_guard_errors_count_as_not_flagged():
    class Boom:
        name = "boom"
        tier = 1

        async def __call__(self, ctx):
            raise RuntimeError("x")

    rep = evaluate_guard(Boom(), _records())
    assert rep.errors == 6 and rep.metrics.tp == 0 and rep.metrics.tn == 3


def test_sync_guard_is_supported():
    def guard(ctx):
        return GuardDecision.block(tier=1, confidence=1.0, rationale="always")

    rep = evaluate_guard(guard, _records(), name="always-block")
    assert rep.metrics.recall == 1.0 and rep.metrics.fpr == 1.0


@pytest.mark.asyncio
async def test_evaluate_works_inside_running_loop():
    rep = evaluate_guard(StubClassifierGuard(triggers=("HARMFUL",)), _records())
    assert rep.metrics.tp == 2


def test_threshold_sweep_and_best_threshold():
    class Scored:
        name = "scored"
        tier = 2

        def __init__(self, threshold):
            self.threshold = threshold

        async def __call__(self, ctx):
            text = ctx.buffer
            unsafe = 0.9 if "HARMFUL request" in text else 0.6 if "HARMFUL" in text else 0.1
            return GuardDecision.allow(tier=2, confidence=1.0 - unsafe, guard_name="scored")

    sweep = threshold_sweep(Scored, _records(), [0.05, 0.5, 0.8, 0.95])
    assert [t for t, _ in sweep] == [0.05, 0.5, 0.8, 0.95]
    by_t = dict(sweep)
    assert by_t[0.05].recall == 1.0 and by_t[0.05].fpr == 1.0
    assert by_t[0.95].recall == 0.0
    best = best_threshold(sweep, max_fpr=0.0)
    assert best is not None and best[0] == 0.8
    assert best_threshold(sweep, max_fpr=0.5)[0] == 0.5
    assert best_threshold([(0.1, Metrics(fp=5, tn=5))], max_fpr=0.1) is None


def test_report_max_misclassified_bounds_lists():
    recs = [FailureRecord(id=f"v{i}", input="x", output="", was_violation=True) for i in range(10)]
    rep = evaluate_guard(StubClassifierGuard(triggers=("NEVER",)), recs, max_misclassified=3)
    assert rep.false_negatives == ["v0", "v1", "v2"]
    assert rep.metrics.fn == 10
    assert isinstance(rep, EvaluationReport)
