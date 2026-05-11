"""LlamaIndex postprocessor adapter.

LlamaIndex post-processors operate on a list of ``NodeWithScore`` (or
on the synthesized response). We wrap the GuardPipeline as a
post-processor that filters or transforms the final response text.
"""

from __future__ import annotations

from typing import Any

from aisafepy.core.decisions import Action
from aisafepy.stream.pipeline import Context, GuardPipeline


class GuardrailPostprocessor:
    """Duck-typed LlamaIndex response post-processor."""

    def __init__(self, pipeline: GuardPipeline):
        self.pipeline = pipeline

    def postprocess_response(self, response: Any) -> Any:
        text = getattr(response, "response", str(response))
        ctx = Context(buffer=text, role="assistant")
        decision = self.pipeline.run_sync(ctx)
        if decision.action == Action.ALLOW:
            return response
        if decision.action == Action.TRANSFORM and decision.transformed_content is not None:
            if hasattr(response, "response"):
                response.response = decision.transformed_content
                return response
            return decision.transformed_content
        if hasattr(response, "response"):
            response.response = decision.fallback or "[blocked]"
            return response
        return decision.fallback or "[blocked]"


def as_llamaindex_postprocessor(pipeline: GuardPipeline) -> GuardrailPostprocessor:
    return GuardrailPostprocessor(pipeline)
