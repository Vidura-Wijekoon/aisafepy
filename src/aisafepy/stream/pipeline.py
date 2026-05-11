"""The tiered guardrail pipeline.

Design constraints (from the proposal, section 3.2):

* Each tier has a latency band. Tier 1 <10 ms, Tier 2 <100 ms,
  Tier 3 <500 ms.
* Tiers run in parallel within a tier and short-circuit on a
  terminal decision (block / escalate).
* Tier 1 -> Tier 2 advances by default; Tier 2 -> Tier 3 gates on
  uncertainty so Tier 3 only runs on ~1-3% of traffic.
* A p95 latency budget is enforced. When exceeded the pipeline emits a
  ``budget_exceeded`` evidence flag and either fails open (default) or
  blocks (``on_budget_exceeded="block"``).
* Every decision becomes an OTel span with the canonical
  ``aisafepy.guard.*`` attributes.

The pipeline is async; for sync callers a ``run_sync`` convenience is
provided that drives the event loop internally.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Literal,
    Protocol,
    runtime_checkable,
)

from aisafepy.core.decisions import Action, GuardDecision
from aisafepy.core.telemetry import (
    attach_decision,
    get_tracer,
    span_for_decision,
    structured_log,
)

Tier = Literal[1, 2, 3]


@dataclass
class Context:
    """The unit of evaluation passed to each guard."""

    chunk: str | None = None
    buffer: str = ""
    prompt: str | None = None
    hidden_states: Any | None = None
    role: Literal["user", "assistant", "tool"] = "assistant"
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Guard(Protocol):
    """Protocol every guard implements."""

    name: str
    tier: Tier

    async def __call__(self, ctx: Context) -> GuardDecision:  # pragma: no cover
        ...


@dataclass
class GuardPipeline:
    """A three-tier guardrail pipeline.

    Construct with explicit tier lists. The pipeline runs Tier 1, then
    Tier 2 if Tier 1 was non-terminal, then Tier 3 only when Tier 2's
    decision falls inside the uncertainty band.
    """

    tier1: list[Guard] = field(default_factory=list)
    tier2: list[Guard] = field(default_factory=list)
    tier3: list[Guard] = field(default_factory=list)
    budget_ms_p95: float = 80.0
    uncertainty_band: tuple[float, float] = (0.30, 0.70)
    on_budget_exceeded: Literal["allow", "block"] = "allow"
    on_violation: Callable[[GuardDecision], Awaitable[None] | None] | None = None

    async def evaluate_chunk(self, ctx: Context) -> GuardDecision:
        """Evaluate one context through the cascade."""
        deadline = time.perf_counter() + (self.budget_ms_p95 / 1000.0)

        tier1_decision = await self._run_tier(1, self.tier1, ctx)
        if tier1_decision.is_terminal:
            await self._notify(tier1_decision)
            return tier1_decision
        if not self.tier2 and not self.tier3:
            return tier1_decision
        if time.perf_counter() > deadline:
            return self._budget_decision(ctx)

        tier2_decision = await self._run_tier(2, self.tier2, ctx)
        if tier2_decision.is_terminal:
            await self._notify(tier2_decision)
            return tier2_decision
        if not self.tier3:
            return tier2_decision
        if not _within(self.uncertainty_band, tier2_decision.confidence):
            return tier2_decision
        if time.perf_counter() > deadline:
            return self._budget_decision(ctx)

        tier3_decision = await self._run_tier(3, self.tier3, ctx)
        if tier3_decision.is_terminal:
            await self._notify(tier3_decision)
        return tier3_decision

    async def guard_stream(
        self,
        chunks: AsyncIterator[str] | Iterable[str],
        *,
        prompt: str | None = None,
    ) -> AsyncIterator[str | GuardDecision]:
        """Yield safe chunks; on a terminal decision yield it and stop."""
        buffer: list[str] = []
        async for chunk in _aiter(chunks):
            buffer.append(chunk)
            ctx = Context(chunk=chunk, buffer="".join(buffer), prompt=prompt)
            decision = await self.evaluate_chunk(ctx)
            if decision.action == Action.BLOCK:
                yield decision
                return
            if decision.action == Action.TRANSFORM and decision.transformed_content is not None:
                yield decision.transformed_content
                continue
            if decision.action == Action.ESCALATE:
                yield decision
                return
            yield chunk

    def run_sync(self, ctx: Context) -> GuardDecision:
        """Synchronous wrapper.

        Works both at module scope and from inside a running loop
        (pytest-asyncio, Jupyter). In the loop case we offload to a
        worker thread.
        """
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is None:
            return asyncio.run(self.evaluate_chunk(ctx))

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, self.evaluate_chunk(ctx))
            return future.result()

    async def _run_tier(self, tier: Tier, guards: list[Guard], ctx: Context) -> GuardDecision:
        if not guards:
            return GuardDecision.allow(tier=tier, rationale=f"tier {tier} empty")
        tracer = get_tracer("aisafepy.stream")
        with span_for_decision(f"aisafepy.stream.tier{tier}", tracer=tracer) as span:
            results = await asyncio.gather(*(self._safe_call(g, ctx) for g in guards))
            chosen = _aggregate(tier, results)
            attach_decision(span, chosen)
            return chosen

    @staticmethod
    async def _safe_call(guard: Guard, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        try:
            decision = await guard(ctx)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            structured_log(
                "guard.error",
                guard=getattr(guard, "name", repr(guard)),
                error=repr(exc),
                latency_ms=elapsed,
            )
            return GuardDecision.allow(
                tier=getattr(guard, "tier", 0),
                rationale=f"guard raised; failing open: {exc!r}",
                guard_name=getattr(guard, "name", repr(guard)),
                evidence={"error": repr(exc)},
                latency_ms=elapsed,
            )
        return decision

    def _budget_decision(self, _ctx: Context) -> GuardDecision:
        if self.on_budget_exceeded == "block":
            return GuardDecision.block(
                tier=0,
                confidence=0.5,
                rationale="latency budget exceeded; failing closed",
                evidence={"budget_exceeded": True},
                severity="medium",
            )
        return GuardDecision.allow(
            tier=0,
            rationale="latency budget exceeded; failing open with flag",
            evidence={"budget_exceeded": True},
        )

    async def _notify(self, decision: GuardDecision) -> None:
        if self.on_violation is None:
            return
        try:
            res = self.on_violation(decision)
            if asyncio.iscoroutine(res):
                await res
        except Exception as exc:
            structured_log("on_violation.error", error=repr(exc))


def _within(band: tuple[float, float], conf: float) -> bool:
    return band[0] <= conf <= band[1]


def _aggregate(tier: Tier, results: list[GuardDecision]) -> GuardDecision:
    blocks = [r for r in results if r.action == Action.BLOCK]
    if blocks:
        blocks.sort(key=lambda d: d.confidence, reverse=True)
        return blocks[0]
    escalations = [r for r in results if r.action == Action.ESCALATE]
    if escalations:
        escalations.sort(key=lambda d: d.confidence, reverse=True)
        return escalations[0]
    transforms = [r for r in results if r.action == Action.TRANSFORM]
    if transforms:
        return transforms[0]
    if not results:
        return GuardDecision.allow(tier=tier)
    results.sort(key=lambda d: d.confidence, reverse=True)
    return results[0]


async def _aiter(source: AsyncIterator[str] | Iterable[str]) -> AsyncIterator[str]:
    if hasattr(source, "__aiter__"):
        async for x in source:  # type: ignore[union-attr]
            yield x
    else:
        for x in source:  # type: ignore[union-attr]
            yield x
