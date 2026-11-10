from __future__ import annotations

import pytest

from aisafepy.core.progress import (
    ProgressTracker,
    RepetitionReason,
    Step,
    jaccard,
    shingles,
)


def test_max_steps_escalates_at_cap():
    t = ProgressTracker(max_steps=3, duplicate_threshold=99)
    assert t.observe(Step("a", {"i": 1})) is None
    assert t.observe(Step("b", {"i": 2})) is None
    out = t.observe(Step("c", {"i": 3}))
    assert out is not None
    assert out.evidence["reason"] == RepetitionReason.MAX_STEPS.value
    assert out.evidence["total_steps"] == 3
    assert t.total_steps == 3
    with pytest.raises(ValueError):
        ProgressTracker(max_steps=0)


def test_ignore_args_makes_volatile_keys_invisible():
    t = ProgressTracker(duplicate_threshold=3, window=10, ignore_args={"request_id"})
    out = None
    for i in range(3):
        out = t.observe(Step("search", {"q": "x", "request_id": i}))
    assert out is not None and out.evidence["reason"] == "duplicate_tool_call"
    strict = ProgressTracker(duplicate_threshold=3, window=10)
    for i in range(3):
        assert strict.observe(Step("search", {"q": "x", "request_id": i})) is None


def test_normalize_strings_collapses_case_and_whitespace():
    t = ProgressTracker(duplicate_threshold=3, window=10, normalize_strings=True)
    variants = ["Weather", "weather  ", " WEATHER"]
    out = None
    for v in variants:
        out = t.observe(Step("search", {"q": v, "tags": ["A ", "b"]}))
    assert out is not None and out.evidence["reason"] == "duplicate_tool_call"


def test_fingerprint_options_are_stable():
    a = Step("t", {"x": "A", "ts": 1, "nested": {"k": " V "}})
    b = Step("t", {"x": "a", "ts": 2, "nested": {"k": "v"}})
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint(ignore_args={"ts"}, normalize_strings=True) == b.fingerprint(
        ignore_args={"ts"}, normalize_strings=True
    )


def test_shingles_and_jaccard():
    assert shingles("ab") == {"ab"}
    assert shingles("") == set()
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard(shingles("hello world"), shingles("hello world")) == 1.0
    assert jaccard(shingles("hello world"), shingles("goodbye moon")) < 0.2


def test_observe_text_detects_near_identical_observations():
    t = ProgressTracker(stagnation_window=3, semantic_threshold=0.8)
    base = "Page shows 42 items in the cart, total 103.50 USD, last updated "
    assert t.observe_text(base + "10:00:01") is None  # nothing to compare yet
    assert t.observe_text(base + "10:00:02") is None
    assert t.observe_text(base + "10:00:03") is None
    out = t.observe_text(base + "10:00:04")
    assert out is not None
    assert out.evidence["reason"] == RepetitionReason.SEMANTIC_NO_PROGRESS.value
    assert all(s >= 0.8 for s in out.evidence["scores"])


def test_observe_text_resets_run_on_real_change():
    t = ProgressTracker(stagnation_window=2, semantic_threshold=0.9)
    t.observe_text("the same thing again and again")
    t.observe_text("the same thing again and again!")
    assert t.observe_text("completely different content here") is None


def test_explain_and_reset():
    t = ProgressTracker(duplicate_threshold=3, window=8, max_steps=50)
    t.observe(Step("a", {"i": 1}, observation="o"))
    t.observe(Step("a", {"i": 1}, observation="o"))
    info = t.explain()
    assert info["total_steps"] == 2 and info["max_steps"] == 50
    assert info["duplicate"]["max_count"] == 2
    assert info["stagnation"]["identical_run"] == 2
    assert info["last_reason"] is None
    assert len(info["cycle"]["recent_fingerprints"]) == 2
    t.observe(Step("a", {"i": 1}, observation="o"))
    assert t.explain()["last_reason"] == "duplicate_tool_call"
    t.reset()
    assert len(t) == 0 and t.total_steps == 0 and t.explain()["last_reason"] is None
    assert t.explain()["stagnation"]["identical_run"] == 0
