from __future__ import annotations

import pytest

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.classifiers import StubClassifierGuard
from aisafepy.stream.pipeline import Context, GuardPipeline
from aisafepy.stream.session import SessionGuard, SessionStore


class Clock:
    def __init__(self, t: float = 10_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def block(name: str = "g", conf: float = 1.0) -> GuardDecision:
    return GuardDecision.block(tier=1, confidence=conf, rationale="r", guard_name=name)


def escalate(name: str = "g", conf: float = 0.5) -> GuardDecision:
    return GuardDecision.escalate(tier=1, confidence=conf, rationale="r", guard_name=name)


def make_guard(clock: Clock, **kw) -> SessionGuard:
    store = SessionStore(ttl_s=kw.pop("ttl_s", 3600), max_sessions=kw.pop("max_sessions", 100), clock=clock)
    return SessionGuard(store=store, **kw)


# ---- store --------------------------------------------------------------------------


def test_store_ttl_and_lru():
    clock = Clock()
    store = SessionStore(ttl_s=100, max_sessions=2, clock=clock)
    a = store.get("a")
    b = store.get("b")
    assert a is not None and b is not None and len(store) == 2
    store.get("a")  # a most recent
    store.get("c")  # evicts b
    assert sorted(store.ids()) == ["a", "c"]
    clock.t += 101
    assert store.get("a", create=False) is None
    assert store.prune() == 1  # c expired too
    assert len(store) == 0
    with pytest.raises(ValueError):
        SessionStore(ttl_s=0)
    with pytest.raises(ValueError):
        SessionStore(max_sessions=0)


# ---- guard ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_session_id_is_skipped():
    d = await make_guard(Clock())(Context(buffer="x"))
    assert d.action.value == "allow" and d.evidence["skipped"] is True


@pytest.mark.asyncio
async def test_blocks_in_window_escalate_then_window_expires():
    clock = Clock()
    guard = make_guard(clock, max_blocks=2, window_s=60, risk_threshold=0)
    ctx = Context(buffer="x", metadata={"session_id": "s1"})
    assert (await guard(ctx)).action.value == "allow"
    guard.record("s1", block())
    assert (await guard(ctx)).action.value == "allow"
    guard.record("s1", block())
    d = await guard(ctx)
    assert d.action.value == "escalate"
    assert d.evidence["blocks_in_window"] == 2
    assert "2 blocks in 60s" in d.rationale
    clock.t += 61
    assert (await guard(ctx)).action.value == "allow"


@pytest.mark.asyncio
async def test_block_action_and_escalation_counter():
    clock = Clock()
    guard = make_guard(clock, max_blocks=0, max_escalations=2, risk_threshold=0, action="block")
    ctx = Context(buffer="x", metadata={"session_id": "s"})
    guard.record("s", escalate())
    guard.record("s", escalate())
    d = await guard(ctx)
    assert d.is_blocked and d.severity == "high"
    assert d.fallback


@pytest.mark.asyncio
async def test_risk_score_accumulates_and_decays():
    clock = Clock()
    guard = make_guard(clock, max_blocks=0, max_escalations=0, risk_threshold=1.5, risk_half_life_s=100)
    ctx = Context(buffer="x", metadata={"session_id": "s"})
    guard.record("s", block(conf=1.0))  # risk 1.0
    assert (await guard(ctx)).action.value == "allow"
    guard.record("s", escalate(conf=1.0))  # +0.6 -> 1.6
    d = await guard(ctx)
    assert d.action.value == "escalate" and d.evidence["risk"] == pytest.approx(1.6, abs=1e-3)
    clock.t += 100  # one half-life -> 0.8
    d = await guard(ctx)
    assert d.action.value == "allow" and d.evidence["risk"] == pytest.approx(0.8, abs=1e-3)


@pytest.mark.asyncio
async def test_rate_limit_per_session():
    clock = Clock()
    guard = make_guard(clock, max_blocks=0, max_escalations=0, risk_threshold=0, max_turns_per_minute=3)
    ctx = Context(buffer="x", metadata={"session_id": "fast"})
    for _ in range(3):
        assert (await guard(ctx)).action.value == "allow"
    d = await guard(ctx)
    assert d.action.value == "escalate" and "turns/min" in d.rationale
    clock.t += 61
    assert (await guard(ctx)).action.value == "allow"


@pytest.mark.asyncio
async def test_last_decision_metadata_feeds_history_and_own_decisions_are_ignored():
    clock = Clock()
    guard = make_guard(clock, max_blocks=1, risk_threshold=0)
    d = await guard(Context(buffer="x", metadata={"session_id": "s", "last_decision": block()}))
    assert d.action.value == "escalate"
    # Feeding the guard's own trip decision back must not change the state.
    guard.record("s", d)
    assert guard.explain("s")["counters"]["block"] == 1
    # Shadow decisions are ignored too.
    guard.record("s", block().model_copy(update={"evidence": {"shadow": True}}))
    assert guard.explain("s")["counters"]["block"] == 1


def test_explain_and_reset():
    clock = Clock()
    guard = make_guard(clock)
    assert guard.explain("nobody") == {"id": "nobody", "known": False}
    guard.record("s", block())
    info = guard.explain("s")
    assert info["known"] is True
    assert info["counters"]["block"] == 1
    assert info["thresholds"]["blocks_in_window"] == [1, 3]
    assert info["recent"][0]["action"] == "block"
    assert guard.reset("s") is True and guard.reset("s") is False


def test_constructor_validation():
    with pytest.raises(ValueError):
        SessionGuard(action="explode")
    with pytest.raises(ValueError):
        SessionGuard(window_s=0)


@pytest.mark.asyncio
async def test_wired_into_pipeline_via_on_decision():
    clock = Clock()
    guard = make_guard(clock, max_blocks=2, window_s=600, risk_threshold=0)
    pipe = GuardPipeline(
        tier1=[guard],
        tier2=[StubClassifierGuard(triggers=("BAD",))],
        on_decision=lambda d, ctx: guard.record(ctx.metadata["session_id"], d),
    )
    meta = {"session_id": "conv-1"}
    assert (await pipe.evaluate_chunk(Context(buffer="BAD", metadata=meta))).tier == 2
    assert (await pipe.evaluate_chunk(Context(buffer="BAD", metadata=meta))).tier == 2
    d = await pipe.evaluate_chunk(Context(buffer="totally fine", metadata=meta))
    assert d.action.value == "escalate" and d.guard_name == "session"
    assert guard.explain("conv-1")["counters"]["block"] == 2
