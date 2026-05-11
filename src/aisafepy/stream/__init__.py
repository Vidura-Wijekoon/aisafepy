"""Streaming-native cascaded guardrails.

The ``aisafepy.stream`` module implements the design from the
proposal's Gap 2: a tiered cascade where Tier 1 is deterministic
(regex / Aho-Corasick / schema, <10 ms), Tier 2 is small-classifier
(LlamaPromptGuard-2 22M, Qwen3Guard-Stream, Kelp, <100 ms), and Tier 3
is heavyweight (Llama Guard 4, ShieldGemma 9B, LLM-as-judge, or
white-box activation probes for self-hosted models).

Decisions are emitted as structured ``GuardDecision`` objects with
OpenTelemetry spans. The pipeline enforces a p95 latency budget and
supports fail-open-with-flag when the budget is exceeded.
"""

from aisafepy.stream.deterministic import (
    AhoCorasickGuard,
    BlocklistGuard,
    RegexGuard,
    SchemaGuard,
)
from aisafepy.stream.pipeline import (
    Context,
    Guard,
    GuardPipeline,
    Tier,
)

# Subpackages exported as namespaces.
from aisafepy.stream import classifiers, forecasters, probes  # noqa: F401

__all__ = [
    "AhoCorasickGuard",
    "BlocklistGuard",
    "Context",
    "Guard",
    "GuardPipeline",
    "RegexGuard",
    "SchemaGuard",
    "Tier",
    "as_openai_output_guardrail",
    "classifiers",
    "forecasters",
    "probes",
]


def as_openai_output_guardrail(pipeline: GuardPipeline):
    """Adapt a :class:`GuardPipeline` to an OpenAI Agents SDK output guardrail.

    The returned object is a callable that the SDK can register as
    ``output_guardrails=[...]``. On any non-allow decision the wrapper
    raises an exception that the SDK treats as
    ``output_guardrail_tripwire_triggered``.
    """
    from aisafepy.stream.adapters.openai_agents import as_output_guardrail

    return as_output_guardrail(pipeline)


# Convenient re-export for the most common HF classifier path. The
# import is wrapped to keep ``transformers`` optional.
class ClassifierGuard:
    """Stub indirection. Defer to :mod:`aisafepy.stream.classifiers`."""

    @staticmethod
    def from_hf(*args, **kwargs):
        from aisafepy.stream.classifiers import HFClassifierGuard

        return HFClassifierGuard.from_hf(*args, **kwargs)
