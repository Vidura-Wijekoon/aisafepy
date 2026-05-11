from __future__ import annotations

import time

import pytest

from aisafepy.core.budgets import (
    BudgetExceeded,
    DollarBudget,
    IterationBudget,
    TokenBudget,
    WallClockBudget,
    composite_check,
)


def test_iteration_budget_increments_and_caps():
    b = IterationBudget("steps", limit=3)
    for i in range(3):
        snap = b.consume()
        assert snap.used == i + 1
    with pytest.raises(BudgetExceeded):
        b.consume()


def test_token_budget_can_consume_floats():
    b = TokenBudget("tokens", limit=1000)
    b.consume(250.5)
    assert b.snapshot().used == pytest.approx(250.5)
    assert b.remaining() == pytest.approx(749.5)


def test_dollar_budget_negative_amount_rejected():
    b = DollarBudget("dollars", limit=10.0)
    with pytest.raises(ValueError):
        b.consume(-1.0)


def test_zero_or_negative_limit_rejected():
    with pytest.raises(ValueError):
        IterationBudget("bad", limit=0)
    with pytest.raises(ValueError):
        IterationBudget("bad", limit=-1)


def test_wall_clock_budget_raises_when_elapsed():
    b = WallClockBudget("wall", limit_seconds=0.05)
    b.start()
    time.sleep(0.08)
    with pytest.raises(BudgetExceeded):
        b.check()


def test_composite_check_returns_one_snapshot_per_budget():
    b1 = IterationBudget("i", limit=10)
    b2 = TokenBudget("t", limit=100)
    b1.consume()
    b2.consume(50)
    snaps = composite_check([b1, b2])
    assert len(snaps) == 2
    assert snaps[0].kind == "iteration"
    assert snaps[1].used == 50


def test_reset_clears_used():
    b = IterationBudget("i", limit=5)
    b.consume()
    b.consume()
    b.reset()
    assert b.snapshot().used == 0
