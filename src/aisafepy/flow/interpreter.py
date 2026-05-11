"""The IFC enforcement engine.

The interpreter is invoked at every tool-call boundary. It:

1. Inspects each argument: if it is a ``Tainted`` value, the argument's
   labels are used directly; otherwise it is lifted with the source
   label of "trusted user input" (TRUSTED integrity, empty caps).
2. Joins all argument labels into the *call* labels.
3. Looks up the tool's policy ``requirement`` and the declared
   ``ToolMetadata`` (from ``@secure_tool``) and verifies:
   - Argument integrity ≥ required integrity.
   - Argument capability set is a subset of the policy's permitted
     capability ceiling for that tool.
   - Control-flow integrity (provided by the adapter) meets the floor.
4. Evaluates each ``deny_if`` rule registered against the tool.
5. Emits a ``GuardDecision`` (allow / block) and, on block, raises an
   ``IFCViolation`` exception in ``strict`` mode or escalates to the
   adapter's mediation handler in ``mediated`` mode.

The implementation is intentionally interpreter-light: rather than
ship a Python sandbox for the planner's emitted DSL (the full CaMeL
approach), we inspect the tool-call site itself. This is the same
shape as RTBAS and is significantly easier to integrate into existing
agent frameworks. A more aggressive ``aisafepy.flow.dsl`` sub-module
that ships a restricted-Python evaluator for planner-emitted action
sequences is on the v0.2 roadmap.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from aisafepy.core.decisions import GuardDecision, IFCViolation
from aisafepy.core.telemetry import (
    attach_decision,
    attach_violation,
    get_tracer,
    span_for_decision,
)
from aisafepy.flow.policy import Policy, ToolMetadata
from aisafepy.flow.taint import Integrity, Tainted, join_all

_logger = logging.getLogger("aisafepy.flow")

_INTEGRITY_ORDER: tuple[Integrity, ...] = ("TRUSTED", "UNTRUSTED", "QUARANTINED")


def _meets(actual: Integrity, required: Integrity) -> bool:
    """``actual`` is at least as trusted as ``required``."""
    return _INTEGRITY_ORDER.index(actual) <= _INTEGRITY_ORDER.index(required)


@dataclass
class IFCContext:
    """Per-run context passed by the adapter.

    ``control_flow_integrity`` is the integrity label of the planner's
    decision to issue the next tool call. For CaMeL-style designs the
    adapter computes this from the labels of every value the planner's
    reasoning depended on; for simpler integrations the adapter may
    pessimistically set it to ``UNTRUSTED`` when *any* untrusted data
    is present in the context window.
    """

    control_flow_integrity: Integrity = "TRUSTED"
    taint_chain: list[tuple[str, str, str]] = field(default_factory=list)
    """A (source, op, dest) trail used to render an auditable taint
    chain in violation evidence."""
    user_metadata: dict[str, Any] = field(default_factory=dict)

    def record(self, source: str, op: str, dest: str) -> None:
        self.taint_chain.append((source, op, dest))


def evaluate_call(
    *,
    tool: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    policy: Policy,
    metadata: ToolMetadata | None = None,
    context: IFCContext | None = None,
    mediator: Callable[[IFCViolation], bool] | None = None,
) -> GuardDecision:
    """Evaluate a tool call against the IFC policy.

    Returns ``GuardDecision.allow(...)`` if the call is permitted.

    In ``strict`` mode, raises ``IFCViolation`` (wrapped in the
    exception's ``decision`` attribute for unified handling) on deny.

    In ``mediated`` mode, calls ``mediator(violation)`` to ask for
    human approval. If the mediator returns ``True`` the decision is
    upgraded to allow with an audit-log annotation.
    """
    context = context or IFCContext()
    tracer = get_tracer("aisafepy.flow")

    # 1. Compute joined labels across all arguments.
    tainted_args = [a for a in (*args, *kwargs.values()) if isinstance(a, Tainted)]
    prov, caps, integrity = join_all(tainted_args)

    # 2. Tool-level checks against the policy.
    requirement = policy.requirement(tool)
    with span_for_decision(f"aisafepy.flow.{tool}", tracer=tracer) as span:
        span.set_attribute("aisafepy.flow.tool", tool)
        span.set_attribute("aisafepy.flow.cfi", context.control_flow_integrity)

        if requirement is not None:
            if not _meets(context.control_flow_integrity, requirement.control_flow_integrity):
                v = IFCViolation(
                    reason=f"control flow integrity {context.control_flow_integrity}"
                           f" < required {requirement.control_flow_integrity}",
                    tool=tool,
                    provenance=prov,
                    integrity=integrity,
                    required_integrity=requirement.arg_max_integrity,
                    capabilities=caps,
                    required_capabilities=requirement.capabilities,
                    taint_chain=tuple(context.taint_chain),
                )
                return _handle_violation(v, policy, span, mediator)

            if not _meets(integrity, requirement.arg_max_integrity):
                v = IFCViolation(
                    reason=f"argument integrity {integrity}"
                           f" < required {requirement.arg_max_integrity}",
                    tool=tool,
                    provenance=prov,
                    integrity=integrity,
                    required_integrity=requirement.arg_max_integrity,
                    capabilities=caps,
                    required_capabilities=requirement.capabilities,
                    taint_chain=tuple(context.taint_chain),
                )
                return _handle_violation(v, policy, span, mediator)

            forbidden = caps - requirement.capabilities
            if forbidden:
                v = IFCViolation(
                    reason=f"forbidden capabilities present in arguments: "
                           f"{sorted(forbidden)}",
                    tool=tool,
                    provenance=prov,
                    integrity=integrity,
                    capabilities=caps,
                    required_capabilities=requirement.capabilities,
                    taint_chain=tuple(context.taint_chain),
                )
                return _handle_violation(v, policy, span, mediator)

        # 3. Tool-declared metadata (from @secure_tool).
        if metadata is not None:
            if not _meets(integrity, metadata.required_integrity):
                v = IFCViolation(
                    reason=f"argument integrity {integrity}"
                           f" < tool-declared required {metadata.required_integrity}",
                    tool=tool,
                    provenance=prov,
                    integrity=integrity,
                    required_integrity=metadata.required_integrity,
                    capabilities=caps,
                    required_capabilities=metadata.capabilities,
                    taint_chain=tuple(context.taint_chain),
                )
                return _handle_violation(v, policy, span, mediator)

        # 4. Custom deny_if rules.
        for rule in policy.deny_rules_for(tool):
            try:
                hit = bool(rule.when(*args, **kwargs))
            except Exception as exc:
                _logger.warning(
                    "deny_if rule raised for tool=%r; treating as no-match: %s",
                    tool,
                    exc,
                )
                hit = False
            if hit:
                v = IFCViolation(
                    reason=f"deny_if matched: {rule.reason}",
                    tool=tool,
                    provenance=prov,
                    integrity=integrity,
                    capabilities=caps,
                    required_capabilities=requirement.capabilities if requirement else frozenset(),
                    taint_chain=tuple(context.taint_chain),
                )
                return _handle_violation(v, policy, span, mediator)

        # 5. Allowed.
        decision = GuardDecision.allow(
            tier=0,
            guard_name=f"flow:{tool}",
            rationale="ifc check passed",
            evidence={
                "tool": tool,
                "provenance": sorted(prov),
                "integrity": integrity,
                "capabilities": sorted(caps),
                "cfi": context.control_flow_integrity,
            },
        )
        attach_decision(span, decision)
        return decision


def _handle_violation(
    v: IFCViolation,
    policy: Policy,
    span: Any,
    mediator: Callable[[IFCViolation], bool] | None,
) -> GuardDecision:
    """Either raise (strict) or call the mediator (mediated)."""
    attach_violation(span, v)
    if policy.mode == "mediated" and mediator is not None:
        approved = False
        with contextlib.suppress(Exception):
            approved = bool(mediator(v))
        if approved:
            decision = GuardDecision.allow(
                tier=0,
                guard_name=f"flow:{v.tool}:mediated",
                rationale=f"human approved despite IFC concern: {v.reason}",
                evidence={"mediation": "approved", "violation_id": v.violation_id},
            )
            attach_decision(span, decision)
            return decision
    # Strict: raise. The ``decision`` attribute lets the adapter render
    # the violation as a structured GuardDecision if it prefers.
    decision = v.to_guard_decision()
    attach_decision(span, decision)
    exc = IFCViolationError(v)
    exc.decision = decision  # type: ignore[attr-defined]
    raise exc


class IFCViolationError(RuntimeError):
    """Raised in ``strict`` mode when ``evaluate_call`` denies a call."""

    def __init__(self, violation: IFCViolation):
        super().__init__(violation.reason)
        self.violation = violation
        self.decision: GuardDecision | None = None


@contextlib.contextmanager
def ifc_run(policy: Policy, **kwargs) -> Iterator[IFCContext]:
    """Context manager that yields a fresh ``IFCContext`` per agent run."""
    ctx = IFCContext(**kwargs)
    try:
        yield ctx
    finally:
        # Currently a no-op; future versions may flush a structured audit log here.
        pass
