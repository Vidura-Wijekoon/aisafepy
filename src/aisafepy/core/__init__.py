"""Shared primitives for AIsafePy.

The core module is dependency-light by design. Only ``pydantic``,
``typing-extensions``, ``opentelemetry-api``, ``anyio``, and ``tenacity``
are required. All ML / agent-framework integrations live in submodules
and only load their heavy dependencies on demand.
"""

from aisafepy.core.budgets import (
    Budget,
    BudgetExceeded,
    BudgetSnapshot,
    DollarBudget,
    IterationBudget,
    TokenBudget,
    WallClockBudget,
)
from aisafepy.core.decisions import (
    Action,
    GuardDecision,
    IFCViolation,
    Severity,
    Tripwire,
)
from aisafepy.core.policies import PolicyDocument, PolicyRule
from aisafepy.core.progress import LoopDetected, ProgressTracker, RepetitionReason
from aisafepy.core.telemetry import (
    AISAFEPY_NS,
    GEN_AI_NS,
    get_tracer,
    span_for_decision,
    structured_log,
)

__all__ = [
    "AISAFEPY_NS",
    "GEN_AI_NS",
    "Action",
    "Budget",
    "BudgetExceeded",
    "BudgetSnapshot",
    "DollarBudget",
    "GuardDecision",
    "IFCViolation",
    "IterationBudget",
    "LoopDetected",
    "PolicyDocument",
    "PolicyRule",
    "ProgressTracker",
    "RepetitionReason",
    "Severity",
    "TokenBudget",
    "Tripwire",
    "WallClockBudget",
    "get_tracer",
    "span_for_decision",
    "structured_log",
]
