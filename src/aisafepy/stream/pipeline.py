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

Beyond the cascade itself the pipeline offers:

* **Shadow guards.** ``shadow_guards`` run alongside Tier 1 on every
  evaluation but never influence the outcome. Their decisions are
  tagged ``evidence["shadow"] = True`` and surfaced through
  ``on_decision`` and ``metrics`` so a new guard can be observed on
  live traffic before it is promoted into a tier.
* **Decision hook.** ``on_decision(decision, ctx)`` fires for every
  final decision (allows included) and every shadow decision. It may
  be sync or async; exceptions are logged and swallowed.
* **Metrics.** Attach a :class:`aisafepy.stream.metrics.PipelineMetrics`
  and the pipeline records per-guard latency / outcome counters and
  per-tier / final-decision counters.
* **Per-tier timeouts.** ``tier_timeouts_ms={2: 50, 3: 400}`` bounds
  each guard call in that tier with ``asyncio.wait_for``. A guard that
  times out fails open with ``evidence["timeout"] = True``.
* **Configuration.** :meth:`GuardPipeline.from_yaml`,
  :meth:`GuardPipeline.from_dict` and :meth:`GuardPipeline.to_dict`
  round-trip the pipeline through the declarative format documented
  in :mod:`aisafepy.stream.config`.
* **Streaming cadence.** :meth:`GuardPipeline.guard_stream` accepts
  ``evaluate_every`` (evaluate every N-th chunk plus always the last
  one) and ``min_buffer_chars`` (skip evaluation until the buffer is at
  least that long) so a cheap pipeline can run per token while an
  expensive one batches. Defaults preserve per-chunk evaluation.

The pipeline is async; for sync callers ``run_sync`` and
``evaluate_text`` drive the event loop internally.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
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

if TYPE_CHECKING:  # pragma: no cover
    from aisafepy.stream.metrics import PipelineMetrics

_logger = logging.getLogger("aisafepy.stream")

Tier = Literal[1, 2, 3]

DecisionHook = Callable[[GuardDecision, "Context"], "Awaitable[None] | None"]


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
    shadow_guards: list[Guard] = field(default_factory=list)
    """Guards evaluated on every call whose decisions never affect the
    outcome. Reported through ``on_decision`` / ``metrics`` with
    ``evidence["shadow"] = True``."""
    on_decision: DecisionHook | None = None
    """Called with every final decision (allows included) and every
    shadow decision. May be sync or async."""
    metrics: PipelineMetrics | None = None
    """Optional :class:`aisafepy.stream.metrics.PipelineMetrics`."""
    tier_timeouts_ms: dict[int, float] | None = None
    """Per-tier wall-clock limit for each guard call. Timed-out guards
    fail open with ``evidence["timeout"] = True``."""
    name: str = "pipeline"

    # ---- construction helpers ------------------------------------------

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, registry: Any | None = None) -> GuardPipeline:
        """Build a pipeline from a configuration mapping.

        See :mod:`aisafepy.stream.config` for the schema.
        """
        from aisafepy.stream.config import load_pipeline

        return load_pipeline(cfg, registry=registry)

    @classmethod
    def from_yaml(cls, path_or_text: str | Path, *, registry: Any | None = None) -> GuardPipeline:
        """Build a pipeline from a YAML file path or a YAML document string."""
        from aisafepy.stream.config import load_pipeline

        return load_pipeline(path_or_text, registry=registry)

    def to_dict(self) -> dict[str, Any]:
        """Best-effort serialization back to the configuration format."""
        from aisafepy.stream.config import dump_pipeline

        return dump_pipeline(self)

    def describe(self) -> str:
        """Return a multi-line human-readable summary of the cascade."""
        lines = [
            f"GuardPipeline {self.name!r}",
            f"  budget_ms_p95={self.budget_ms_p95} "
            f"uncertainty_band={self.uncertainty_band} "
            f"on_budget_exceeded={self.on_budget_exceeded}",
        ]
        if self.tier_timeouts_ms:
            lines.append(f"  tier_timeouts_ms={dict(sorted(self.tier_timeouts_ms.items()))}")
        for tier, guards in ((1, self.tier1), (2, self.tier2), (3, self.tier3)):
            lines.append(f"  tier {tier}: {len(guards)} guard(s)")
            for g in guards:
                lines.append(f"    - {_guard_name(g)} ({type(g).__name__})")
        if self.shadow_guards:
            lines.append(f"  shadow: {len(self.shadow_guards)} guard(s)")
            for g in self.shadow_guards:
                lines.append(f"    - {_guard_name(g)} ({type(g).__name__})")
        lines.append(f"  metrics={'attached' if self.metrics is not None else 'none'}")
        return "\n".join(lines)

    # ---- evaluation ------------------------------------------------------

    async def evaluate_chunk(self, ctx: Context) -> GuardDecision:
        """Evaluate one context through the cascade."""
        started = time.perf_counter()
        deadline = started + (self.budget_ms_p95 / 1000.0)

        shadow_task: asyncio.Task[list[GuardDecision]] | None = None
        if self.shadow_guards:
            shadow_task = asyncio.ensure_future(self._run_shadow(ctx))

        try:
            decision = await self._cascade(ctx, deadline)
        finally:
            if shadow_task is not None:
                await self._finish_shadow(shadow_task, ctx)

        self._record_final(decision, started)
        await self._emit(decision, ctx)
        return decision

    async def _cascade(self, ctx: Context, deadline: float) -> GuardDecision:
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
        evaluate_every: int = 1,
        min_buffer_chars: int = 0,
    ) -> AsyncIterator[str | GuardDecision]:
        """Yield safe chunks; on a terminal decision yield it and stop.

        Parameters
        ----------
        evaluate_every:
            Evaluate the cascade on every N-th chunk. The final chunk is
            always evaluated so nothing slips through at the tail.
        min_buffer_chars:
            Skip evaluation while the accumulated buffer is shorter than
            this many characters (again, the final chunk is always
            evaluated). Chunks that skip evaluation are yielded as-is.
        """
        if evaluate_every < 1:
            raise ValueError("evaluate_every must be >= 1")
        buffer: list[str] = []
        pending: list[str] = []
        index = 0
        source = _aiter(chunks)
        lookahead = await _anext(source)
        while lookahead is not _SENTINEL:
            chunk = lookahead  # type: ignore[assignment]
            lookahead = await _anext(source)
            is_last = lookahead is _SENTINEL
            index += 1
            buffer.append(chunk)
            pending.append(chunk)
            joined = "".join(buffer)
            due = (index % evaluate_every == 0) and len(joined) >= min_buffer_chars
            if not (due or is_last):
                continue
            ctx = Context(chunk="".join(pending), buffer=joined, prompt=prompt)
            decision = await self.evaluate_chunk(ctx)
            if decision.action == Action.BLOCK:
                yield decision
                return
            if decision.action == Action.ESCALATE:
                yield decision
                return
            if decision.action == Action.TRANSFORM and decision.transformed_content is not None:
                yield decision.transformed_content
                pending.clear()
                continue
            for p in pending:
                yield p
            pending.clear()

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

    def evaluate_text(
        self,
        text: str,
        *,
        prompt: str | None = None,
        role: Literal["user", "assistant", "tool"] = "assistant",
        metadata: dict[str, Any] | None = None,
    ) -> GuardDecision:
        """Evaluate a complete string synchronously.

        Builds a :class:`Context` with ``chunk`` and ``buffer`` both set
        to ``text`` and delegates to :meth:`run_sync`.
        """
        ctx = Context(
            chunk=text,
            buffer=text,
            prompt=prompt,
            role=role,
            metadata=dict(metadata or {}),
        )
        return self.run_sync(ctx)

    # ---- internals -------------------------------------------------------

    async def _run_tier(self, tier: Tier, guards: list[Guard], ctx: Context) -> GuardDecision:
        if not guards:
            return GuardDecision.allow(tier=tier, rationale=f"tier {tier} empty")
        tracer = get_tracer("aisafepy.stream")
        timeout = self._timeout_for(tier)
        with span_for_decision(f"aisafepy.stream.tier{tier}", tracer=tracer) as span:
            results = await asyncio.gather(
                *(self._safe_call(g, ctx, timeout=timeout, tier=tier) for g in guards)
            )
            chosen = _aggregate(tier, results)
            attach_decision(span, chosen)
            if self.metrics is not None:
                self.metrics.record_tier(tier, chosen)
            return chosen

    def _timeout_for(self, tier: int) -> float | None:
        if not self.tier_timeouts_ms:
            return None
        ms = self.tier_timeouts_ms.get(tier)
        if ms is None or ms <= 0:
            return None
        return ms / 1000.0

    async def _safe_call(
        self,
        guard: Guard,
        ctx: Context,
        *,
        timeout: float | None = None,
        tier: int | None = None,
        shadow: bool = False,
    ) -> GuardDecision:
        start = time.perf_counter()
        name = _guard_name(guard)
        guard_tier = tier if tier is not None else getattr(guard, "tier", 0)
        try:
            if timeout is None:
                decision = await guard(ctx)
            else:
                decision = await asyncio.wait_for(guard(ctx), timeout=timeout)
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000.0
            structured_log("guard.timeout", guard=name, timeout_ms=timeout * 1000.0, tier=tier)
            decision = GuardDecision.allow(
                tier=guard_tier,
                rationale=f"guard timed out after {elapsed:.1f} ms; failing open",
                guard_name=name,
                evidence={"timeout": True, "timeout_ms": timeout * 1000.0},
                latency_ms=elapsed,
            )
            self._record_guard(name, guard_tier, decision, elapsed, error=False, timeout=True)
            return decision
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            structured_log("guard.error", guard=name, error=repr(exc), latency_ms=elapsed)
            decision = GuardDecision.allow(
                tier=guard_tier,
                rationale=f"guard raised; failing open: {exc!r}",
                guard_name=name,
                evidence={"error": repr(exc)},
                latency_ms=elapsed,
            )
            self._record_guard(name, guard_tier, decision, elapsed, error=True, timeout=False)
            return decision
        elapsed = (time.perf_counter() - start) * 1000.0
        if shadow:
            decision = _mark_shadow(decision)
        self._record_guard(name, guard_tier, decision, elapsed, error=False, timeout=False)
        return decision

    async def _run_shadow(self, ctx: Context) -> list[GuardDecision]:
        timeout = self._timeout_for(0)
        return list(
            await asyncio.gather(
                *(self._safe_call(g, ctx, timeout=timeout, shadow=True) for g in self.shadow_guards)
            )
        )

    async def _finish_shadow(self, task: asyncio.Task[list[GuardDecision]], ctx: Context) -> None:
        try:
            decisions = await task
        except Exception as exc:  # pragma: no cover - _safe_call already guards
            structured_log("shadow.error", error=repr(exc))
            return
        for d in decisions:
            d = _mark_shadow(d)
            if self.metrics is not None:
                self.metrics.record_shadow(d)
            await self._emit(d, ctx)

    def _record_guard(
        self,
        name: str,
        tier: int,
        decision: GuardDecision,
        latency_ms: float,
        *,
        error: bool,
        timeout: bool,
    ) -> None:
        if self.metrics is None:
            return
        try:
            self.metrics.record_guard(
                name,
                tier=tier,
                decision=decision,
                latency_ms=latency_ms,
                error=error,
                timeout=timeout,
            )
        except Exception as exc:  # pragma: no cover - defensive
            structured_log("metrics.error", error=repr(exc))

    def _record_final(self, decision: GuardDecision, started: float) -> None:
        if self.metrics is None:
            return
        try:
            self.metrics.record_final(decision, latency_ms=(time.perf_counter() - started) * 1000.0)
        except Exception as exc:  # pragma: no cover - defensive
            structured_log("metrics.error", error=repr(exc))

    async def _emit(self, decision: GuardDecision, ctx: Context) -> None:
        if self.on_decision is None:
            return
        try:
            res = self.on_decision(decision, ctx)
            if asyncio.iscoroutine(res):
                await res
        except Exception as exc:
            structured_log("on_decision.error", error=repr(exc))

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


def _guard_name(guard: Any) -> str:
    return str(getattr(guard, "name", None) or type(guard).__name__)


def _mark_shadow(decision: GuardDecision) -> GuardDecision:
    if decision.evidence.get("shadow") is True:
        return decision
    return decision.model_copy(update={"evidence": {**decision.evidence, "shadow": True}})


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


_SENTINEL: Any = object()


async def _anext(it: AsyncIterator[str]) -> Any:
    try:
        return await it.__anext__()
    except StopAsyncIteration:
        return _SENTINEL


async def _aiter(source: AsyncIterator[str] | Iterable[str]) -> AsyncIterator[str]:
    if hasattr(source, "__aiter__"):
        async for x in source:  # type: ignore[union-attr]
            yield x
    else:
        for x in source:  # type: ignore[union-attr]
            yield x
