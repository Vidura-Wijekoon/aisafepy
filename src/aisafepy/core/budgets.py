"""Budgets. Iteration, token, wall-clock, dollar, and rate.

Inspired by browser-use#191, AgentBudget.dev, and *Beyond Max Tokens*
(arXiv 2601.10955). Every long-running agent in production hits one of
these limits; rebuilding them in every codebase is wasted work.

The budget objects are thread-safe (anyio-compatible) and emit
structured log events on ~25%/50%/75%/100% checkpoints so an operator
can see runaway sessions in their tracing backend before they go over.
Checkpoint callbacks (``Budget.on_checkpoint``) let an application
react without parsing logs.

Grouping and enforcement helpers:

* :class:`BudgetSet` holds several budgets under one name and checks
  them together.
* :func:`budgeted` wraps a sync or async function so that each call
  consumes one iteration and re-checks every wall-clock budget first.
* :class:`TokenPricing` turns token counts into dollars so a
  :class:`DollarBudget` can be charged directly from a model response.
* :class:`RateBudget` is a sliding-window rate limit (events per
  period) with the same ``BudgetExceeded`` contract.
"""

from __future__ import annotations

import functools
import inspect
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

from aisafepy.core.telemetry import structured_log

F = TypeVar("F", bound=Callable[..., Any])


class BudgetExceeded(RuntimeError):
    """Raised when a budget's hard cap is exceeded.

    The ``budget`` attribute points back to the failing budget so
    callers can inspect it (e.g. to render a friendly message that
    cites the actual cap).
    """

    def __init__(self, message: str, budget: Budget):
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

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.used)

    @property
    def exceeded(self) -> bool:
        return self.used > self.limit

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "used": self.used,
            "limit": self.limit,
            "fraction": self.fraction,
            "remaining": self.remaining,
            "exceeded": self.exceeded,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


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
        self.on_checkpoint: list[Callable[[BudgetSnapshot, int], None]] = []
        """Callbacks invoked as ``cb(snapshot, pct)`` the first time usage
        crosses each of the 25/50/75/100 percent marks."""

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

    def would_exceed(self, amount: float = 1.0) -> bool:
        """Whether consuming ``amount`` would push the budget over its cap."""
        with self._lock:
            return self._used + amount > self.limit

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
                if self.on_checkpoint:
                    snap = BudgetSnapshot(
                        name=self.name, kind=self.kind, used=used, limit=limit, fraction=fraction
                    )
                    for cb in list(self.on_checkpoint):
                        try:
                            cb(snap, ck)
                        except Exception as exc:
                            structured_log("budget.checkpoint.callback_error", error=repr(exc))

    def __repr__(self) -> str:  # pragma: no cover - debug
        snap = self.snapshot()
        return f"<{type(self).__name__} {snap.name} {snap.used:.2f}/{snap.limit:.2f}>"


class IterationBudget(Budget):
    """Caps the number of agent steps / tool calls."""

    kind = "iteration"


class TokenBudget(Budget):
    """Caps the cumulative tokens used (prompt + completion)."""

    kind = "tokens"


# ---- pricing ------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrice:
    """USD per one million tokens."""

    prompt_per_million: float
    completion_per_million: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.prompt_per_million + completion_tokens * self.completion_per_million
        ) / 1_000_000.0


class TokenPricing:
    """A small, overridable price table.

    The shipped defaults are list prices as published by vendors at the
    time of writing and exist so examples run; production users should
    :meth:`register` their negotiated rates. Lookups are exact first,
    then by longest registered prefix (``"gpt-4o-2024-08-06"`` resolves
    to ``"gpt-4o"``), so dated snapshots inherit their family's price.
    """

    def __init__(self, prices: dict[str, ModelPrice] | None = None):
        self._prices: dict[str, ModelPrice] = dict(_DEFAULT_PRICES)
        if prices:
            self._prices.update(prices)

    def register(self, model: str, prompt_per_million: float, completion_per_million: float) -> None:
        self._prices[model] = ModelPrice(prompt_per_million, completion_per_million)

    def lookup(self, model: str) -> ModelPrice | None:
        if model in self._prices:
            return self._prices[model]
        best: str | None = None
        for key in self._prices:
            if model.startswith(key) and (best is None or len(key) > len(best)):
                best = key
        return self._prices[best] if best is not None else None

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """USD cost; raises ``KeyError`` for unknown models so silent zero-cost
        accounting never happens."""
        price = self.lookup(model)
        if price is None:
            raise KeyError(f"no price registered for model {model!r}")
        return price.cost(prompt_tokens, completion_tokens)

    def models(self) -> list[str]:
        return sorted(self._prices)


_DEFAULT_PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(15.0, 75.0),
    "claude-opus-5-5": ModelPrice(5.0, 25.0),
    "claude-sonnet-5-5": ModelPrice(3.0, 15.0),
    "claude-haiku-5-5": ModelPrice(1.0, 5.0),
    "gpt-4o": ModelPrice(2.5, 10.0),
    "gpt-4o-mini": ModelPrice(0.15, 0.6),
    "gemini-2.5-pro": ModelPrice(1.25, 10.0),
    "gemini-2.5-flash": ModelPrice(0.3, 2.5),
}

DEFAULT_PRICING = TokenPricing()


class DollarBudget(Budget):
    """Caps the cumulative dollar cost (caller is responsible for pricing)."""

    kind = "dollars"

    def charge(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        pricing: TokenPricing | None = None,
    ) -> BudgetSnapshot:
        """Consume the USD cost of a model call computed from a price table."""
        cost = (pricing or DEFAULT_PRICING).estimate(model, prompt_tokens, completion_tokens)
        snap = self.consume(cost)
        snap.metadata = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        }
        return snap


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


class RateBudget(Budget):
    """Sliding-window rate limit: at most ``max_events`` per ``per_seconds``.

    Unlike the cumulative budgets, usage decays as old events leave the
    window. ``limit`` is ``max_events``; ``used`` is the number of
    events inside the current window.
    """

    kind = "rate"

    def __init__(
        self,
        name: str,
        max_events: int,
        per_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        if per_seconds <= 0:
            raise ValueError("per_seconds must be > 0")
        super().__init__(name, max_events)
        self.per_seconds = float(per_seconds)
        self._clock = clock
        self._events: deque[float] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self.per_seconds
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()
        self._used = float(len(self._events))

    def try_acquire(self, n: int = 1) -> bool:
        """Record ``n`` events if they fit in the window; return whether they did."""
        if n < 0:
            raise ValueError("n must be non-negative")
        with self._lock:
            now = self._clock()
            self._prune(now)
            if len(self._events) + n > self.limit:
                return False
            for _ in range(n):
                self._events.append(now)
            self._used = float(len(self._events))
            used, limit = self._used, self.limit
        self._maybe_emit_checkpoint(used, limit, used / limit)
        return True

    def acquire(self, n: int = 1) -> BudgetSnapshot:
        """Like :meth:`try_acquire` but raises :class:`BudgetExceeded` when full."""
        if not self.try_acquire(n):
            raise BudgetExceeded(
                f"rate budget {self.name!r} exceeded: {int(self.limit)} per {self.per_seconds}s",
                budget=self,
            )
        return self.snapshot()

    def consume(self, amount: float = 1.0) -> BudgetSnapshot:
        return self.acquire(int(amount))

    def retry_after(self) -> float:
        """Seconds until at least one slot frees up (0 when a slot is free now)."""
        with self._lock:
            now = self._clock()
            self._prune(now)
            if len(self._events) < self.limit:
                return 0.0
            return max(0.0, self._events[0] + self.per_seconds - now)

    def remaining(self) -> float:
        with self._lock:
            self._prune(self._clock())
            return max(0.0, self.limit - self._used)

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            self._prune(self._clock())
        return super().snapshot()

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._used = 0.0
            self._checkpoints_emitted.clear()


def composite_check(budgets: Iterable[Budget]) -> list[BudgetSnapshot]:
    """Snapshot a collection of budgets. Useful for periodic logging."""
    snaps: list[BudgetSnapshot] = []
    for b in budgets:
        if isinstance(b, WallClockBudget):
            snaps.append(b.check())
        else:
            snaps.append(b.snapshot())
    return snaps


# ---- grouping ---------------------------------------------------------


class BudgetSet:
    """A named group of budgets checked and consumed together.

    Budgets are addressable by ``name`` or by ``kind`` (when exactly one
    budget of that kind is present). As a context manager the set
    starts every wall-clock budget on entry and runs a final
    :meth:`check_all` on a clean exit.
    """

    def __init__(self, name: str, budgets: Iterable[Budget] = ()):
        self.name = name
        self._budgets: dict[str, Budget] = {}
        for b in budgets:
            self.add(b)

    def add(self, budget: Budget) -> BudgetSet:
        if budget.name in self._budgets:
            raise ValueError(f"budget named {budget.name!r} already in set {self.name!r}")
        self._budgets[budget.name] = budget
        return self

    def __iter__(self) -> Iterator[Budget]:
        return iter(self._budgets.values())

    def __len__(self) -> int:
        return len(self._budgets)

    def __contains__(self, key: str) -> bool:
        return self._resolve(key) is not None

    def get(self, key: str) -> Budget:
        b = self._resolve(key)
        if b is None:
            raise KeyError(f"no budget named or of kind {key!r} in set {self.name!r}")
        return b

    def _resolve(self, key: str) -> Budget | None:
        if key in self._budgets:
            return self._budgets[key]
        matches = [b for b in self._budgets.values() if b.kind == key]
        return matches[0] if len(matches) == 1 else None

    def consume(self, key: str, amount: float = 1.0) -> BudgetSnapshot:
        """Consume from the budget named ``key`` (or of kind ``key``)."""
        return self.get(key).consume(amount)

    def check_all(self) -> list[BudgetSnapshot]:
        """Check wall-clock budgets and snapshot the rest; raises on any breach."""
        snaps = composite_check(self._budgets.values())
        for s in snaps:
            if s.exceeded:
                raise BudgetExceeded(
                    f"budget {s.name!r} ({s.kind}) exceeded: {s.used} > {s.limit}",
                    budget=self._budgets[s.name],
                )
        return snaps

    def snapshots(self) -> dict[str, BudgetSnapshot]:
        return {b.name: b.snapshot() for b in self._budgets.values()}

    def exceeded(self) -> list[Budget]:
        """Budgets currently over their cap (never raises)."""
        out: list[Budget] = []
        for b in self._budgets.values():
            try:
                snap = b.check() if isinstance(b, WallClockBudget) else b.snapshot()
            except BudgetExceeded:
                out.append(b)
                continue
            if snap.exceeded:
                out.append(b)
        return out

    def reset(self) -> None:
        for b in self._budgets.values():
            b.reset()

    def start_clocks(self) -> None:
        for b in self._budgets.values():
            if isinstance(b, WallClockBudget):
                b.start()

    def __enter__(self) -> BudgetSet:
        self.start_clocks()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.check_all()


def budgeted(
    *budgets: Budget | BudgetSet,
    iteration_cost: float = 1.0,
) -> Callable[[F], F]:
    """Enforce budgets around every call of the decorated function.

    Before each call every :class:`WallClockBudget` in ``budgets`` is
    checked and every :class:`IterationBudget` consumes
    ``iteration_cost``. :class:`RateBudget` instances acquire one slot.
    Other budget kinds are only snapshot-checked (they are consumed by
    the application, which knows the token / dollar amounts). Works for
    sync and async functions.
    """
    flat: list[Budget] = []
    for b in budgets:
        if isinstance(b, BudgetSet):
            flat.extend(b)
        else:
            flat.append(b)

    def _pre_call() -> None:
        for b in flat:
            if isinstance(b, WallClockBudget):
                b.check()
            elif isinstance(b, RateBudget):
                b.acquire()
            elif isinstance(b, IterationBudget):
                b.consume(iteration_cost)
            elif b.snapshot().exceeded:
                raise BudgetExceeded(
                    f"budget {b.name!r} ({b.kind}) exceeded before call", budget=b
                )

    def deco(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                _pre_call()
                return await fn(*args, **kwargs)

            async_wrapper.__aisafepy_budgets__ = flat  # type: ignore[attr-defined]
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _pre_call()
            return fn(*args, **kwargs)

        wrapper.__aisafepy_budgets__ = flat  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return deco
