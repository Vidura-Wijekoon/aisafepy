"""LangChain ``Runnable`` adapter.

Wraps a :class:`aisafepy.stream.GuardPipeline` into a Runnable that
LangChain users can compose into their chains. The wrapper accepts
either a string or a ``dict`` with an ``output`` field.
"""

from __future__ import annotations

from typing import Any

from aisafepy.core.decisions import Action
from aisafepy.stream.pipeline import Context, GuardPipeline


class GuardrailRunnable:
    """A duck-typed LangChain Runnable.

    LangChain's interface lives in ``langchain_core``; we don't import
    it here to keep the dependency optional. Any object that exposes
    ``invoke`` / ``ainvoke`` works in a LangChain chain.
    """

    def __init__(self, pipeline: GuardPipeline):
        self.pipeline = pipeline

    def invoke(self, input: Any) -> Any:
        text = input if isinstance(input, str) else input.get("output", "")
        ctx = Context(buffer=str(text))
        decision = self.pipeline.run_sync(ctx)
        if decision.action == Action.ALLOW:
            return text
        if decision.action == Action.TRANSFORM and decision.transformed_content is not None:
            return decision.transformed_content
        return decision.fallback or "[blocked]"

    async def ainvoke(self, input: Any) -> Any:
        text = input if isinstance(input, str) else input.get("output", "")
        ctx = Context(buffer=str(text))
        decision = await self.pipeline.evaluate_chunk(ctx)
        if decision.action == Action.ALLOW:
            return text
        if decision.action == Action.TRANSFORM and decision.transformed_content is not None:
            return decision.transformed_content
        return decision.fallback or "[blocked]"


def as_langchain_runnable(pipeline: GuardPipeline) -> GuardrailRunnable:
    return GuardrailRunnable(pipeline)
