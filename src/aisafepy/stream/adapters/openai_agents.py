"""OpenAI Agents SDK input / output guardrail adapter."""

from __future__ import annotations

from typing import Any

from aisafepy.core.decisions import Action, Tripwire
from aisafepy.stream.pipeline import Context, GuardPipeline


def as_output_guardrail(pipeline: GuardPipeline) -> Any:
    """Return an ``output_guardrail``-compatible callable.

    The OpenAI Agents SDK calls output guardrails with the full
    response text; we accumulate it into a single Context and run the
    cascade once. On non-allow decisions we raise a ``Tripwire``-bearing
    exception, which the SDK surfaces as
    ``output_guardrail_tripwire_triggered``.
    """

    async def guardrail(_agent: Any, _run_context: Any, output: Any) -> Tripwire | None:
        text = output if isinstance(output, str) else getattr(output, "output_text", str(output))
        ctx = Context(buffer=text, role="assistant")
        decision = await pipeline.evaluate_chunk(ctx)
        if decision.action == Action.ALLOW:
            return None
        return Tripwire(**decision.model_dump())

    return guardrail


def as_input_guardrail(pipeline: GuardPipeline) -> Any:
    """Return an ``input_guardrail``-compatible callable."""

    async def guardrail(_agent: Any, _run_context: Any, input_text: str) -> Tripwire | None:
        ctx = Context(buffer=str(input_text), role="user")
        decision = await pipeline.evaluate_chunk(ctx)
        if decision.action == Action.ALLOW:
            return None
        return Tripwire(**decision.model_dump())

    return guardrail
