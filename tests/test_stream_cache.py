"""Tests for the decision cache and the CachedGuard wrapper."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.cache import (
    CachedGuard,
    DecisionCache,
    cache_guards,
    cached_pipeline,
    is_cacheable,
    make_key,
    normalise_for_key,
)
from aisafepy.stream.classifiers import StubClassifierGuard
from aisafepy.stream.deterministic import AhoCorasickGuard
from aisafepy.stream.pipeline import Context, GuardPipeline


class _Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class _Counting:
    """A guard that counts calls and can be told to fail."""

    name = "counting"
    tier = 2

    def __init__(self, *, fail: bool = False, block_on: str | None = None):
        self.calls = 0
        self.fail = fail
        self.block_on = block_on

    async def __call__(self, ctx: Context) -> GuardDecision:
        self.calls += 1
        if self.fail:
            return GuardDecision.allow(tier=2, guard_name=self.name, evidence={"error": "boom"})
        if self.block_on and self.block_on in (ctx.buffer or ""):
            return GuardDecision.block(tier=2, confidence=0.9, rationale="hit", guard_name=self.name)
        return GuardDecision.allow(tier=2, guard_name=self.name, rationale="clean")


# ---- keys ------------------------------------------------------------------------


def test_key_normalises_whitespace_and_nfkc_but_keeps_case():
    assert normalise_for_key("  Hello \n  world ") == "Hello world"
    assert make_key("Hello   world") == make_key("Hello world")
    assert make_key("\uff28ello") == make_key("Hello")  # fullwidth H folds under NFKC
    assert make_key("hello") != make_key("Hello")
    assert make_key("x", role="user") != make_key("x", role="assistant")
    assert make_key("x", guard_name="a") != make_key("x", guard_name="b")
    assert make_key("x", extra=(("tenant", 1),)) != make_key("x", extra=(("tenant", 2),))


# ---- cache -------------------------------------------------------------------------


def _allow(**ev) -> GuardDecision:
    return GuardDecision.allow(tier=2, guard_name="g", evidence=ev)


def test_hit_miss_and_stats():
    cache = DecisionCache(max_entries=10)
    assert cache.get("k") is None
    assert cache.set("k", _allow())
    assert cache.get("k") is not None
    s = cache.stats()
    assert (s.hits, s.misses, s.stores) == (1, 1, 1)
    assert s.hit_rate == pytest.approx(0.5)
    assert "k" in cache and "zz" not in cache
    assert len(cache) == 1
    assert s.to_dict()["hit_rate"] == 0.5


def test_ttl_expiry_with_injected_clock():
    clock = _Clock()
    cache = DecisionCache(default_ttl_s=10, clock=clock)
    cache.set("k", _allow())
    clock.t += 9.9
    assert cache.get("k") is not None
    clock.t += 0.2
    assert cache.get("k") is None
    assert cache.stats().expirations == 1
    cache.set("p", _allow(), ttl_s=1)
    cache.set("q", _allow(), ttl_s=None)  # never expires
    clock.t += 5
    assert cache.prune() == 1
    assert cache.get("q") is not None


def test_lru_eviction_order():
    cache = DecisionCache(max_entries=2, default_ttl_s=None)
    cache.set("a", _allow())
    cache.set("b", _allow())
    assert cache.get("a") is not None  # a becomes most recent
    cache.set("c", _allow())  # evicts b
    assert cache.get("b") is None
    assert cache.get("a") is not None and cache.get("c") is not None
    assert cache.stats().evictions == 1


def test_error_and_timeout_decisions_are_not_cached():
    cache = DecisionCache()
    assert not cache.set("e", _allow(error="x"))
    assert not cache.set("t", _allow(timeout=True))
    assert not cache.set("b", _allow(budget_exceeded=True))
    assert cache.set("ok", _allow())
    assert cache.stats().skipped == 3
    assert is_cacheable(_allow()) and not is_cacheable(_allow(error="x"))


def test_invalidate():
    cache = DecisionCache()
    cache.set("a", _allow())
    cache.set("b", _allow())
    assert cache.invalidate("a") == 1
    assert cache.invalidate("a") == 0
    assert cache.invalidate() == 1
    assert len(cache) == 0


def test_capacity_validation():
    with pytest.raises(ValueError):
        DecisionCache(max_entries=0)


# ---- CachedGuard -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cached_guard_hits_copy_decision_with_fresh_id():
    inner = _Counting(block_on="BAD")
    cache = DecisionCache()
    guard = CachedGuard(inner, cache)
    assert guard.name == "cached:counting" and guard.tier == 2
    d1 = await guard(Context(buffer="this is BAD"))
    d2 = await guard(Context(buffer="this is  BAD"))  # whitespace differs, same key
    assert inner.calls == 1
    assert d1.is_blocked and d2.is_blocked
    assert d2.evidence["cached"] is True and "cached" not in d1.evidence
    assert d2.decision_id != d1.decision_id
    assert d2.guard_name == "counting"


@pytest.mark.asyncio
async def test_cached_guard_reuse_id_and_role_separation():
    inner = _Counting()
    guard = CachedGuard(inner, DecisionCache(), reuse_decision_id=True)
    d1 = await guard(Context(buffer="x", role="user"))
    d2 = await guard(Context(buffer="x", role="user"))
    d3 = await guard(Context(buffer="x", role="assistant"))
    assert d2.decision_id == d1.decision_id
    assert inner.calls == 2
    assert d3.decision_id != d1.decision_id


@pytest.mark.asyncio
async def test_cached_guard_does_not_cache_failures():
    inner = _Counting(fail=True)
    guard = CachedGuard(inner, DecisionCache())
    await guard(Context(buffer="x"))
    await guard(Context(buffer="x"))
    assert inner.calls == 2
    assert guard.stats().skipped == 2


@pytest.mark.asyncio
async def test_cached_guard_key_fields_and_invalidate():
    inner = _Counting()
    guard = CachedGuard(inner, DecisionCache(), key_fields=("tenant",))
    await guard(Context(buffer="x", metadata={"tenant": "a"}))
    await guard(Context(buffer="x", metadata={"tenant": "b"}))
    await guard(Context(buffer="x", metadata={"tenant": "a"}))
    assert inner.calls == 2
    plain = CachedGuard(_Counting(), DecisionCache())
    await plain(Context(buffer="y"))
    assert plain.invalidate("y") == 1
    await plain(Context(buffer="y"))
    assert plain.inner.calls == 2
    assert plain.invalidate() == 1


def test_cache_guards_passes_through_already_cached():
    cache = DecisionCache()
    g = CachedGuard(_Counting(), cache)
    out = cache_guards([g, StubClassifierGuard()], cache)
    assert out[0] is g
    assert isinstance(out[1], CachedGuard) and out[1].name == "cached:stub"


@pytest.mark.asyncio
async def test_cached_pipeline_wraps_selected_tiers_and_leaves_original_untouched():
    t1 = AhoCorasickGuard(name="bl", terms=["STOP"])
    t2 = StubClassifierGuard(triggers=("UNSAFE",))
    original = GuardPipeline(tier1=[t1], tier2=[t2], budget_ms_p95=42)
    cache = DecisionCache()
    wrapped = cached_pipeline(original, cache)
    assert original.tier2 == [t2] and original.tier1 == [t1]
    assert isinstance(wrapped.tier2[0], CachedGuard) and wrapped.tier1[0] is t1
    assert wrapped.budget_ms_p95 == 42
    wrapped.tier1.append(AhoCorasickGuard(name="extra", terms=["never-present"]))
    assert len(original.tier1) == 1
    d = await wrapped.evaluate_chunk(Context(buffer="UNSAFE text"))
    assert d.is_blocked
    d2 = await wrapped.evaluate_chunk(Context(buffer="UNSAFE text"))
    assert d2.evidence.get("cached") is True
    assert cache.stats().hits == 1
    with pytest.raises(ValueError):
        cached_pipeline(original, cache, tiers=(4,))


def test_thread_safety_smoke():
    cache = DecisionCache(max_entries=50, default_ttl_s=None)

    def work(i: int) -> None:
        for j in range(200):
            k = f"k{(i * 7 + j) % 80}"
            if cache.get(k) is None:
                cache.set(k, _allow())

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(8)))
    assert len(cache) <= 50
    s = cache.stats()
    assert s.hits + s.misses == 1600


def test_async_guard_from_sync_context():
    inner = _Counting()
    guard = CachedGuard(inner, DecisionCache())

    async def run():
        await guard(Context(buffer="z"))
        return await guard(Context(buffer="z"))

    d = asyncio.run(run())
    assert d.evidence["cached"] is True and inner.calls == 1
