from __future__ import annotations

import asyncio

import pytest

from aisafepy.core.budgets import (
    BudgetExceeded,
    BudgetSet,
    DollarBudget,
    IterationBudget,
    ModelPrice,
    RateBudget,
    TokenBudget,
    TokenPricing,
    WallClockBudget,
    budgeted,
)

# ---- snapshots / checkpoint callbacks ---------------------------------------


def test_snapshot_remaining_exceeded_and_to_dict():
    b = TokenBudget("t", limit=100)
    snap = b.consume(40)
    assert snap.remaining == 60 and snap.exceeded is False
    d = snap.to_dict()
    assert d["remaining"] == 60 and d["kind"] == "tokens"
    assert b.would_exceed(61) and not b.would_exceed(60)


def test_on_checkpoint_callbacks_fire_once_per_mark_and_tolerate_errors():
    b = IterationBudget("i", limit=4)
    marks: list[int] = []
    b.on_checkpoint.append(lambda snap, pct: marks.append(pct))
    b.on_checkpoint.append(lambda snap, pct: 1 / 0)
    for _ in range(4):
        b.consume()
    assert marks == [25, 50, 75, 100]
    b.reset()
    b.consume()
    assert marks == [25, 50, 75, 100, 25]


# ---- pricing ----------------------------------------------------------------


def test_token_pricing_exact_prefix_and_unknown():
    pricing = TokenPricing()
    exact = pricing.estimate("gpt-4o", 1_000_000, 0)
    assert exact == pytest.approx(2.5)
    assert pricing.lookup("gpt-4o-2024-08-06") == pricing.lookup("gpt-4o")
    # Longest prefix wins: gpt-4o-mini snapshots must not resolve to gpt-4o.
    assert pricing.lookup("gpt-4o-mini-2024-07-18") == pricing.lookup("gpt-4o-mini")
    with pytest.raises(KeyError):
        pricing.estimate("unknown-model", 1, 1)
    pricing.register("my-model", 1.0, 2.0)
    assert pricing.estimate("my-model", 500_000, 500_000) == pytest.approx(1.5)
    assert "my-model" in pricing.models()
    custom = TokenPricing({"x": ModelPrice(10.0, 10.0)})
    assert custom.estimate("x", 100_000, 0) == pytest.approx(1.0)


def test_dollar_budget_charge_uses_pricing_and_records_metadata():
    b = DollarBudget("usd", limit=1.0)
    pricing = TokenPricing({"m": ModelPrice(1.0, 3.0)})
    snap = b.charge("m", prompt_tokens=200_000, completion_tokens=100_000, pricing=pricing)
    assert snap.used == pytest.approx(0.5)
    assert snap.metadata["model"] == "m" and snap.metadata["cost"] == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded):
        b.charge("m", 500_000, 500_000, pricing=pricing)


# ---- rate budget -------------------------------------------------------------


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_rate_budget_sliding_window():
    clock = Clock()
    rb = RateBudget("rps", max_events=3, per_seconds=10.0, clock=clock)
    assert rb.try_acquire() and rb.try_acquire() and rb.try_acquire()
    assert not rb.try_acquire()
    assert rb.retry_after() == pytest.approx(10.0)
    assert rb.remaining() == 0
    with pytest.raises(BudgetExceeded) as exc:
        rb.acquire()
    assert exc.value.budget is rb
    clock.t = 5.0
    assert rb.retry_after() == pytest.approx(5.0)
    clock.t = 10.01  # all three events leave the window together
    assert rb.try_acquire()
    assert rb.snapshot().used == 1
    clock.t = 30.0
    assert rb.snapshot().used == 0 and rb.retry_after() == 0.0
    assert rb.consume(2).used == 2
    rb.reset()
    assert rb.remaining() == 3
    with pytest.raises(ValueError):
        RateBudget("bad", max_events=1, per_seconds=0)


def test_rate_budget_batch_acquire_respects_capacity():
    rb = RateBudget("r", max_events=5, per_seconds=1.0, clock=Clock())
    assert rb.try_acquire(3)
    assert not rb.try_acquire(3)
    assert rb.try_acquire(2)


# ---- budget set ----------------------------------------------------------


def test_budget_set_lookup_consume_and_exceeded():
    bs = BudgetSet("run", [IterationBudget("steps", 3), TokenBudget("tokens", 100)])
    assert len(bs) == 2 and "steps" in bs and "tokens" in bs and "iteration" in bs
    bs.consume("iteration")
    bs.consume("tokens", 50)
    assert bs.snapshots()["steps"].used == 1
    assert bs.exceeded() == []
    with pytest.raises(KeyError):
        bs.get("dollars")
    with pytest.raises(ValueError):
        bs.add(IterationBudget("steps", 9))
    with pytest.raises(BudgetExceeded):
        for _ in range(3):
            bs.consume("steps")
    assert [b.name for b in bs.exceeded()] == ["steps"]
    with pytest.raises(BudgetExceeded):
        bs.check_all()
    bs.reset()
    assert bs.check_all()[0].used == 0


def test_budget_set_kind_lookup_is_ambiguous_with_two_of_a_kind():
    bs = BudgetSet("x", [TokenBudget("a", 1), TokenBudget("b", 1)])
    assert "tokens" not in bs
    assert bs.get("a").name == "a"


def test_budget_set_context_manager_starts_clocks_and_checks_on_exit():
    wall = WallClockBudget("wall", limit_seconds=60)
    it = IterationBudget("steps", 2)
    with BudgetSet("run", [wall, it]) as bs:
        assert wall._started_at is not None
        bs.consume("steps")
    with pytest.raises(BudgetExceeded), BudgetSet("run2", [IterationBudget("s", 1)]) as bs2:
        bs2.get("s")._used = 5  # simulate the application over-consuming


# ---- budgeted decorator ------------------------------------------------------


def test_budgeted_sync_consumes_iterations_and_checks_rate():
    steps = IterationBudget("steps", 2)
    rate = RateBudget("rate", max_events=10, per_seconds=60)

    @budgeted(steps, rate)
    def act(x):
        return x * 2

    assert act(1) == 2 and act(2) == 4
    assert rate.snapshot().used == 2
    with pytest.raises(BudgetExceeded):
        act(3)
    assert act.__aisafepy_budgets__ == [steps, rate]


def test_budgeted_accepts_budget_set_and_async():
    bs = BudgetSet("run", [IterationBudget("steps", 1), TokenBudget("tokens", 10)])

    @budgeted(bs)
    async def step():
        return "ok"

    assert asyncio.run(step()) == "ok"
    with pytest.raises(BudgetExceeded):
        asyncio.run(step())


def test_budgeted_refuses_call_when_token_budget_already_exceeded():
    tokens = TokenBudget("tokens", 10)
    tokens._used = 11  # over budget via application accounting

    @budgeted(tokens)
    def f():
        return 1

    with pytest.raises(BudgetExceeded):
        f()
