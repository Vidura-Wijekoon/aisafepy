from __future__ import annotations

from aisafepy.core.progress import ProgressTracker, RepetitionReason, Step


def test_duplicate_tool_call_detected():
    tracker = ProgressTracker(duplicate_threshold=3, window=10)
    for _ in range(3):
        result = tracker.observe(Step(tool="search", args={"q": "weather"}))
    assert result is not None
    assert result.evidence["reason"] == RepetitionReason.DUPLICATE_TOOL_CALL.value


def test_no_loop_for_distinct_calls():
    tracker = ProgressTracker(duplicate_threshold=3, window=10)
    for q in ("weather", "news", "stocks", "calendar"):
        out = tracker.observe(Step(tool="search", args={"q": q}))
        assert out is None


def test_cycle_detected_for_ab_pattern():
    tracker = ProgressTracker(duplicate_threshold=99, window=12, cycle_lengths=(2,))
    pattern = [
        Step(tool="search", args={"q": "x"}),
        Step(tool="read", args={"id": 1}),
        Step(tool="search", args={"q": "x"}),
        Step(tool="read", args={"id": 1}),
    ]
    for s in pattern[:-1]:
        assert tracker.observe(s) is None
    out = tracker.observe(pattern[-1])
    assert out is not None
    assert out.evidence["reason"] == RepetitionReason.CYCLIC_TOOL_CALLS.value


def test_stagnation_on_repeated_observation():
    tracker = ProgressTracker(stagnation_window=3, duplicate_threshold=99, window=8)
    for _ in range(3):
        result = tracker.observe(Step(tool="search", args={"q": "x"}, observation="same"))
    assert result is not None
    assert result.evidence["reason"] == RepetitionReason.STAGNATION.value


def test_semantic_no_progress():
    tracker = ProgressTracker(stagnation_window=3, semantic_threshold=0.9)
    for _ in range(3):
        result = tracker.observe_similarity(0.96)
    assert result is not None
    assert result.evidence["reason"] == RepetitionReason.SEMANTIC_NO_PROGRESS.value


def test_fingerprint_stable_across_arg_order():
    a = Step(tool="t", args={"x": 1, "y": 2})
    b = Step(tool="t", args={"y": 2, "x": 1})
    assert a.fingerprint() == b.fingerprint()
