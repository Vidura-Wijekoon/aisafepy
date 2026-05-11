"""Typed guard decisions.

A ``GuardDecision`` is the lingua franca of AIsafePy. Every guard in
``aisafepy.stream``, every IFC enforcement point in ``aisafepy.flow``,
and every compiled guard produced by ``aisafepy.adapt`` returns one of
these. Production observability rests on the structured ``evidence`` and
``rationale`` fields, which solve the "why was this blocked?" problem
flagged by Skywork, Arthur, and the *No Free Lunch with Guardrails*
paper (arXiv 2504.00441).
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["info", "low", "medium", "high", "critical"]


class Action(str, Enum):
    """Outcome of a guard evaluation.

    ``allow``. Pass through. The most common case for a healthy pipeline.
    ``block``. Refuse outright; emit ``fallback`` to the user.
    ``transform``. Return modified content (e.g. PII-redacted).
    ``escalate``. Route to human review or a stricter tier.
    """

    ALLOW = "allow"
    BLOCK = "block"
    TRANSFORM = "transform"
    ESCALATE = "escalate"


class GuardDecision(BaseModel):
    """A typed, structured outcome from any guard in AIsafePy.

    The fields mirror what production observability stacks (Langfuse,
    Arize Phoenix, Helicone, Datadog) need to debug a false positive at
    3 a.m. without re-running the pipeline. ``evidence`` is intentionally
    a free-form dict so guards can attach span offsets, probe logits,
    classifier label distributions, etc.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    tier: int = Field(ge=0)
    rationale: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0.0)
    severity: Severity = "medium"
    fallback: str | None = None
    decision_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    guard_name: str | None = None
    transformed_content: str | None = None
    """Replacement content emitted when action == TRANSFORM."""

    # ---- ergonomic constructors ----------------------------------------

    @classmethod
    def allow(
        cls,
        *,
        tier: int = 0,
        confidence: float = 1.0,
        rationale: str = "passed",
        guard_name: str | None = None,
        evidence: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
    ) -> GuardDecision:
        return cls(
            action=Action.ALLOW,
            confidence=confidence,
            tier=tier,
            rationale=rationale,
            evidence=evidence or {},
            latency_ms=latency_ms,
            severity="info",
            guard_name=guard_name,
        )

    @classmethod
    def block(
        cls,
        *,
        tier: int,
        confidence: float,
        rationale: str,
        fallback: str = "I can't help with that.",
        severity: Severity = "high",
        guard_name: str | None = None,
        evidence: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
    ) -> GuardDecision:
        return cls(
            action=Action.BLOCK,
            confidence=confidence,
            tier=tier,
            rationale=rationale,
            fallback=fallback,
            severity=severity,
            evidence=evidence or {},
            latency_ms=latency_ms,
            guard_name=guard_name,
        )

    @classmethod
    def transform(
        cls,
        *,
        tier: int,
        confidence: float,
        rationale: str,
        transformed_content: str,
        severity: Severity = "low",
        guard_name: str | None = None,
        evidence: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
    ) -> GuardDecision:
        return cls(
            action=Action.TRANSFORM,
            confidence=confidence,
            tier=tier,
            rationale=rationale,
            transformed_content=transformed_content,
            severity=severity,
            evidence=evidence or {},
            latency_ms=latency_ms,
            guard_name=guard_name,
        )

    @classmethod
    def escalate(
        cls,
        *,
        tier: int,
        confidence: float,
        rationale: str,
        severity: Severity = "medium",
        guard_name: str | None = None,
        evidence: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
    ) -> GuardDecision:
        return cls(
            action=Action.ESCALATE,
            confidence=confidence,
            tier=tier,
            rationale=rationale,
            severity=severity,
            evidence=evidence or {},
            latency_ms=latency_ms,
            guard_name=guard_name,
        )

    # ---- predicates ----------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """Whether this decision halts downstream tiers."""
        return self.action in (Action.BLOCK, Action.ESCALATE)

    @property
    def is_blocked(self) -> bool:
        return self.action == Action.BLOCK


class Tripwire(GuardDecision):
    """Backwards-compatible alias used by the OpenAI Agents SDK adapter.

    ``Tripwire`` is semantically a guard decision that has tripped. The
    adapter raises a ``Tripwire`` (or wraps a normal exception in one)
    when ``action != allow``. This matches the OpenAI Agents SDK's
    ``input_guardrail_tripwire_triggered`` / ``output_guardrail_tripwire_triggered``
    exception types but is framework-agnostic at this layer.
    """


class IFCViolation(BaseModel):
    """A capability-based information-flow control violation.

    Emitted by ``aisafepy.flow`` whenever a tool call is denied due to
    integrity, capability, or provenance constraints. The ``taint_chain``
    field is a list of (source, op, dest) tuples reconstructed by the
    flow interpreter so an operator can answer "*why* did the email
    arguments end up labelled UNTRUSTED?"
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str
    tool: str
    provenance: frozenset[str]
    integrity: Literal["TRUSTED", "UNTRUSTED", "QUARANTINED"]
    required_integrity: Literal["TRUSTED", "UNTRUSTED", "QUARANTINED"] | None = None
    capabilities: frozenset[str] = frozenset()
    required_capabilities: frozenset[str] = frozenset()
    taint_chain: tuple[tuple[str, str, str], ...] = ()
    violation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = Field(default_factory=time.time)

    def to_guard_decision(self) -> GuardDecision:
        """Render this violation as a generic ``GuardDecision`` for unified pipelines."""
        return GuardDecision.block(
            tier=0,
            confidence=1.0,
            rationale=f"IFC violation: {self.reason}",
            severity="critical",
            guard_name=f"flow:{self.tool}",
            evidence={
                "kind": "ifc_violation",
                "tool": self.tool,
                "provenance": sorted(self.provenance),
                "integrity": self.integrity,
                "required_integrity": self.required_integrity,
                "capabilities": sorted(self.capabilities),
                "required_capabilities": sorted(self.required_capabilities),
                "taint_chain": list(self.taint_chain),
                "violation_id": self.violation_id,
            },
            fallback="This action was blocked by the information-flow policy.",
        )
