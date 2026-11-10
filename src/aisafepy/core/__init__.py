"""Shared primitives for AIsafePy.

The core module is dependency-light by design. Only ``pydantic``,
``typing-extensions``, ``opentelemetry-api``, ``anyio``, and ``tenacity``
are required. All ML / agent-framework integrations live in submodules
and only load their heavy dependencies on demand.

What lives here:

* :mod:`aisafepy.core.decisions`: ``GuardDecision`` and ``IFCViolation``,
  the typed outcomes every guard and enforcement point returns.
* :mod:`aisafepy.core.budgets`: iteration, token, dollar, wall-clock and
  rate budgets, ``BudgetSet`` and the ``budgeted`` decorator.
* :mod:`aisafepy.core.progress`: loop and no-progress detection.
* :mod:`aisafepy.core.policies`: the YAML policy-document DSL.
* :mod:`aisafepy.core.sinks`: where decisions go (JSONL, callbacks,
  in-memory buffers) with a stable record schema.
* :mod:`aisafepy.core.telemetry`: OpenTelemetry conventions and the
  redactor that scrubs credentials before anything leaves the process.
"""

from aisafepy.core.budgets import (
    Budget,
    BudgetExceeded,
    BudgetSet,
    BudgetSnapshot,
    DollarBudget,
    IterationBudget,
    ModelPrice,
    RateBudget,
    TokenBudget,
    TokenPricing,
    WallClockBudget,
    budgeted,
)
from aisafepy.core.decisions import (
    Action,
    GuardDecision,
    IFCViolation,
    Severity,
    Tripwire,
)
from aisafepy.core.policies import PolicyDocument, PolicyRule, compile_selector
from aisafepy.core.progress import (
    LoopDetected,
    ProgressTracker,
    RepetitionReason,
    Step,
    jaccard,
    shingles,
)
from aisafepy.core.sinks import (
    CallbackSink,
    DecisionSink,
    FilterSink,
    InMemorySink,
    JSONLSink,
    MultiSink,
    decision_to_record,
    record_to_decision,
    severity_rank,
)
from aisafepy.core.telemetry import (
    AISAFEPY_NS,
    GEN_AI_NS,
    configure_redaction,
    get_tracer,
    redact,
    redact_text,
    span_for_decision,
    structured_log,
)

__all__ = [
    # telemetry
    "AISAFEPY_NS",
    "GEN_AI_NS",
    "configure_redaction",
    "get_tracer",
    "redact",
    "redact_text",
    "span_for_decision",
    "structured_log",
    # decisions
    "Action",
    "GuardDecision",
    "IFCViolation",
    "Severity",
    "Tripwire",
    # budgets
    "Budget",
    "BudgetExceeded",
    "BudgetSet",
    "BudgetSnapshot",
    "DollarBudget",
    "IterationBudget",
    "ModelPrice",
    "RateBudget",
    "TokenBudget",
    "TokenPricing",
    "WallClockBudget",
    "budgeted",
    # progress
    "LoopDetected",
    "ProgressTracker",
    "RepetitionReason",
    "Step",
    "jaccard",
    "shingles",
    # policies
    "PolicyDocument",
    "PolicyRule",
    "compile_selector",
    # sinks
    "CallbackSink",
    "DecisionSink",
    "FilterSink",
    "InMemorySink",
    "JSONLSink",
    "MultiSink",
    "decision_to_record",
    "record_to_decision",
    "severity_rank",
]
