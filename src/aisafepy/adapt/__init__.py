"""The continuous eval-to-guardrail compiler.

``aisafepy.adapt`` closes the loop between red-team frameworks (PyRIT,
Garak, Inspect AI, Promptfoo, DeepEval), production traces (Langfuse,
Arize Phoenix), and the runtime guardrail pipeline. It takes a stream of
``FailureRecord``s, clusters them, and compiles each cluster into one
or more of:

* a distilled classifier (small encoder, drop-in for Tier 2)
* a synthesized regex (Tier 1), scored against a negative pool
* a Cedar / OPA policy rule
* a steering vector (self-hosted models only)
* a deliberative case (prompt-injected reasoning hint, CADA-style)

The compiled artifacts are bundled in a :class:`CompilationReport`
which can be saved to disk, evaluated offline against labelled records
(:func:`evaluate_guard`), and canary-deployed through a
:class:`CanaryController` that enforces an explicit false-positive
budget and rolls back automatically.
"""

from aisafepy.adapt.canary import CanaryController, CanaryResult, GuardStats, ShadowGuard, promote
from aisafepy.adapt.cluster import Cluster, cluster_failures
from aisafepy.adapt.compile import (
    CompilationReport,
    CompiledArtifact,
    CompilerHooks,
    GuardCompiler,
    Target,
)
from aisafepy.adapt.evaluate import (
    EvaluationReport,
    Metrics,
    best_threshold,
    evaluate_guard,
    threshold_sweep,
)
from aisafepy.adapt.governance import AuditEntry, AuditLog, VerifyResult
from aisafepy.adapt.sources import (
    CSVSource,
    DeepEvalSource,
    FailureRecord,
    GarakReport,
    InMemorySource,
    InspectLog,
    JSONLSource,
    LangfuseTraces,
    ProductionTraces,
    PromptfooSource,
    PyRITSource,
    RecordFilter,
    RedTeamSource,
    dedupe_records,
    merge_sources,
)
from aisafepy.adapt.sources_otlp import DecisionLogSource, OTLPSpanSource

__all__ = [
    # sources
    "CSVSource",
    "DecisionLogSource",
    "DeepEvalSource",
    "FailureRecord",
    "GarakReport",
    "InMemorySource",
    "InspectLog",
    "JSONLSource",
    "LangfuseTraces",
    "OTLPSpanSource",
    "ProductionTraces",
    "PromptfooSource",
    "PyRITSource",
    "RecordFilter",
    "RedTeamSource",
    "dedupe_records",
    "merge_sources",
    # clustering and compilation
    "Cluster",
    "CompilationReport",
    "CompiledArtifact",
    "CompilerHooks",
    "GuardCompiler",
    "Target",
    "cluster_failures",
    # evaluation
    "EvaluationReport",
    "Metrics",
    "best_threshold",
    "evaluate_guard",
    "threshold_sweep",
    # canary deployment
    "CanaryController",
    "CanaryResult",
    "GuardStats",
    "ShadowGuard",
    "promote",
    # governance
    "AuditEntry",
    "AuditLog",
    "VerifyResult",
]
