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

Tier 1 ships a family of dependency-free guards: regex and blocklist
matching, credential detection (:class:`SecretsGuard`), smuggled
payload decoding (:class:`EncodingGuard`), tool-call argument policy
(:class:`ToolCallGuard`), weighted prompt-injection heuristics
(:class:`InjectionHeuristicGuard`), and the smaller structural guards
in :mod:`aisafepy.stream.heuristics`. All of them run on text that has
been through the offset-preserving :class:`Normalizer`.

Pipelines can be declared in YAML (:func:`load_pipeline`,
:meth:`GuardPipeline.from_yaml`), observed through
:class:`PipelineMetrics` and :class:`FalsePositiveMonitor`, and
extended across turns with :class:`SessionGuard`.
"""

# Subpackages exported as namespaces.
from aisafepy.stream import (
    cache,
    classifiers,
    config,
    encoding,
    forecasters,
    hashed_classifier,
    heuristics,
    injection,
    judge,
    metrics,
    normalize,
    pii,
    probes,
    secrets,
    session,
    toolcalls,
)
from aisafepy.stream.cache import CachedGuard, DecisionCache, cache_guards, cached_pipeline
from aisafepy.stream.config import (
    ConfigError,
    GuardRegistry,
    GuardSpec,
    PipelineConfig,
    default_registry,
    dump_pipeline,
    load_pipeline,
    validate_config,
)
from aisafepy.stream.deterministic import (
    AhoCorasickGuard,
    BlocklistGuard,
    RegexGuard,
    SchemaGuard,
)
from aisafepy.stream.encoding import INJECTION_PHRASES, EncodingGuard
from aisafepy.stream.hashed_classifier import HashedNGramClassifierGuard, TrainingReport
from aisafepy.stream.heuristics import (
    CharsetGuard,
    JSONDepthGuard,
    LengthGuard,
    MarkdownImageExfilGuard,
    RepetitionGuard,
    ScriptMixGuard,
    URLGuard,
)
from aisafepy.stream.injection import (
    DEFAULT_FAMILIES,
    InjectionHeuristicGuard,
    PatternFamily,
)
from aisafepy.stream.judge import (
    JudgeTranscript,
    LLMJudgeGuard,
    MajorityJudge,
    StubJudge,
    parse_verdict,
)
from aisafepy.stream.metrics import (
    FalsePositiveMonitor,
    FPAlert,
    GuardStats,
    LatencyHistogram,
    PipelineMetrics,
    decision_rate_summary,
)
from aisafepy.stream.normalize import Normalized, Normalizer, normalize_for_matching
from aisafepy.stream.pii import PIIDetector, PIIGuard, PIIMatch
from aisafepy.stream.pipeline import (
    Context,
    Guard,
    GuardPipeline,
    Tier,
)
from aisafepy.stream.secrets import SecretMatch, SecretsGuard
from aisafepy.stream.session import SessionGuard, SessionState, SessionStore
from aisafepy.stream.toolcalls import (
    PathPolicy,
    ShellPolicy,
    ToolCall,
    ToolCallGuard,
    URLPolicy,
    extract_tool_calls,
)

__all__ = [
    # pipeline
    "Context",
    "Guard",
    "GuardPipeline",
    "Tier",
    # tier-1 deterministic guards
    "AhoCorasickGuard",
    "BlocklistGuard",
    "RegexGuard",
    "SchemaGuard",
    # tier-1 content guards
    "PIIDetector",
    "PIIGuard",
    "PIIMatch",
    "CharsetGuard",
    "EncodingGuard",
    "INJECTION_PHRASES",
    "InjectionHeuristicGuard",
    "DEFAULT_FAMILIES",
    "PatternFamily",
    "JSONDepthGuard",
    "LengthGuard",
    "MarkdownImageExfilGuard",
    "RepetitionGuard",
    "ScriptMixGuard",
    "SecretMatch",
    "SecretsGuard",
    "URLGuard",
    # tool-call boundary
    "PathPolicy",
    "ShellPolicy",
    "ToolCall",
    "ToolCallGuard",
    "URLPolicy",
    "extract_tool_calls",
    # normalisation
    "Normalized",
    "Normalizer",
    "normalize_for_matching",
    # tier-2 pure-Python classifier
    "HashedNGramClassifierGuard",
    "TrainingReport",
    # decision cache
    "CachedGuard",
    "DecisionCache",
    "cache_guards",
    "cached_pipeline",
    # configuration
    "ConfigError",
    "GuardRegistry",
    "GuardSpec",
    "PipelineConfig",
    "default_registry",
    "dump_pipeline",
    "load_pipeline",
    "validate_config",
    # observability
    "FPAlert",
    "FalsePositiveMonitor",
    "GuardStats",
    "LatencyHistogram",
    "PipelineMetrics",
    "decision_rate_summary",
    # tier-3 judge
    "JudgeTranscript",
    "LLMJudgeGuard",
    "MajorityJudge",
    "StubJudge",
    "parse_verdict",
    # conversation-level state
    "SessionGuard",
    "SessionState",
    "SessionStore",
    # adapters and namespaces
    "as_openai_output_guardrail",
    "cache",
    "classifiers",
    "config",
    "encoding",
    "forecasters",
    "hashed_classifier",
    "heuristics",
    "injection",
    "judge",
    "metrics",
    "normalize",
    "pii",
    "probes",
    "secrets",
    "session",
    "toolcalls",
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
