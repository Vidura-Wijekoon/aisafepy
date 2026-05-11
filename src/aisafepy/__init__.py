"""AIsafePy. Capability-based IFC, streaming guardrails, and an eval-to-guard compiler.

Top-level public API re-exports the most commonly used types from each module.
Submodules (flow, stream, adapt, contrib) are imported lazily on first use so
that the heavy optional dependencies (transformers, torch, hdbscan, openai-agents,
langgraph, ...) do not pay an import cost unless the user actually reaches for them.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from aisafepy.core.budgets import Budget, BudgetExceeded
from aisafepy.core.decisions import (
    Action,
    GuardDecision,
    IFCViolation,
    Severity,
    Tripwire,
)
from aisafepy.core.policies import PolicyDocument
from aisafepy.core.progress import LoopDetected, ProgressTracker
from aisafepy.core.telemetry import get_tracer, span_for_decision

__version__ = "0.1.0"

__all__ = [
    # version
    "__version__",
    # core re-exports
    "Action",
    "Budget",
    "BudgetExceeded",
    "GuardDecision",
    "IFCViolation",
    "LoopDetected",
    "PolicyDocument",
    "ProgressTracker",
    "Severity",
    "Tripwire",
    "get_tracer",
    "span_for_decision",
    # lazy submodules
    "flow",
    "stream",
    "adapt",
    "contrib",
]

_LAZY_SUBMODULES = {"flow", "stream", "adapt", "contrib"}

if TYPE_CHECKING:  # pragma: no cover
    from aisafepy import adapt, contrib, flow, stream


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = import_module(f"aisafepy.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'aisafepy' has no attribute {name!r}")
