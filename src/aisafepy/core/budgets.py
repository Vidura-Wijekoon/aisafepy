"""Budgets. Iteration, token, wall-clock, and dollar.

Inspired by browser-use#191, AgentBudget.dev, and *Beyond Max Tokens*
(arXiv 2601.10955). Every long-running agent in production hits one of
these limits; rebuilding them in every codebase is wasted work.

The budget objects are thread-safe (anyio-compatible) and emit
structured log events on ~25%/50%/75%/100% checkpoints so an operator
can see runaway sessions in their tracing backend before they go over.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

from aisafepy.core.telemetry import structured_log


class BudgetExceeded(RuntimeError):
    """Raised when a budget's hard cap is exceeded.

    The ``budget`` attribute points back to the failing budget so
    callers can inspect it (e.g. to render a friendly message that
    cites the actual cap).
    """

    def __init__(self, message: str, budget: "Budget"):
        super().__init__(message)
        self.budget = budget


@dataclass
class BudgetSnapshot:
    name: str
    kind: str
    used: float
    limit: float
    fraction: float
    metadata: dict[str, object] = field(default_factory=dict)


class Budget:
    """Base class. Subclasses define what ``used`` means."""

    kind: str = "abstract"

    def __init__(self, name: str, limit: float):
        if limit <= 0:
            raise ValueError(f"Budget {name!r} limit must be > 0 (got {limit})")
        self.name = name
        self.limit = float(limit)
        self._lock = threading.Lock()
        self._used = 0.0
        self._checkpoints_emitted: set[int] = set()

    # ---- core operations ------------------------------------------------

    def consume(self, amount: float = 1.0) -> BudgetSnapshot:
        with self._lock:
            if amount < 0:
                raise ValueError(f"amount must be non-negative (got {amount})")
            self._used += amount
            used = self._used
            limit = self.limit
        fraction = used / limit
        self._maybe_emit_checkpoint(used, limit, fraction)
        if used > limit:
            raise BudgetExceeded(
                f"budget {self.name!r} ({self.kind}) exceeded: {used} > {limit}",
                budget=self,
            )
        return BudgetSnapshot(
            name=self.name,
            kind=self.kind,
            used=used,
            limit=limit,
            fraction=fraction,
        )

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.limit - self._used)

    def reset(self) -> None:
        with self._lock:
            self._used = 0.0
            self._checkpoints_emitted.clear()

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            used = self._used
            limit = self.limit
        return BudgetSnapshot(
            name=self.name,
            kind=self.kind,
            used=used,
            limit=limit,
            fraction=used / limit if limit else 1.0,
        )

    # ---- checkpointing --------------------------------------------------

    _CHECKPOINTS = (25, 50, 75, 100)

    def _maybe_emit_checkpoint(self, used: float, limit: float, fraction: float) -> None:
        if limit <= 0:
            return
        pct = int(fraction * 100)
        for ck in self._CHECKPOINTS:
            if pct >= ck and ck not in self._checkpoints_emitted:
                self._checkpoints_emitted.add(ck)
                structured_log(
                    "budget.checkpoint",
                    budget=self.name,
                    kind=self.kind,
                    used=used,
                    limit=limit,
                    pct=ck,
                )

    def __repr__(self) -> str:  # pragma: no cover - debug
        snap = self.snapshot()
        return f"<{type(self).__name__} {snap.name} {snap.used:.2f}/{snap.limit:.2f}>"


class IterationBudget(Budget):
    """Caps the number of agent steps / tool calls."""

    kind = "iteration"


class TokenBudget(Budget):
    """Caps the cumulative tokens used (prompt + completion)."""

    kind = "tokens"


class DollarBudget(Budget):
    """Caps the cumulative dollar cost (caller is responsible for pricing)."""

    kind = "dollars"


class WallClockBudget(Budget):
    """Caps elapsed wall-clock seconds since the budget was started."""

    kind = "wall_clock"

    def __init__(self, name: str, limit_seconds: float):
        super().__init__(name, limit_seconds)
        self._started_at: float | None = None

    def start(self) -> None:
        with self._lock:
            self._started_at = time.monotonic()
            self._used = 0.0
            self._checkpoints_emitted.clear()

    def check(self) -> BudgetSnapshot:
        """Inspect elapsed wall time without consuming. Raises if over."""
        with self._lock:
            if self._started_at is None:
                self._started_at = time.monotonic()
            elapsed = time.monotonic() - self._started_at
            self._used = elapsed
            limit = self.limit
        fraction = elapsed / limit
        self._maybe_emit_checkpoint(elapsed, limit, fraction)
        if elapsed > limit:
            raise BudgetExceeded(
                f"wall-clock budget {self.name!r} exceeded: {elapsed:.2f}s > {limit:.2f}s",
                budget=self,
            )
        return BudgetSnapshot(
            name=self.name,
            kind=self.kind,
            used=elapsed,
            limit=limit,
            fraction=fraction,
        )


def composite_check(budgets: Iterable[Budget]) -> list[BudgetSnapshot]:
    """Snapshot a collection of budgets. Useful for periodic logging."""
    snaps: list[BudgetSnapshot] = []
    for b in budgets:
        if isinstance(b, WallClockBudget):
            snaps.append(b.check())
        else:
            snaps.append(b.snapshot())
    return snaps
