"""Assertions for guard and policy tests.

Every helper raises ``AssertionError`` with the decision rendered into
the message (action, guard, tier, confidence, rationale, evidence), so a
failing test explains *why* the guard did what it did instead of
printing ``assert False``.

:func:`run_guard` is the synchronous entry point. It accepts either a
single guard (anything with ``async __call__(ctx)``) or a
``GuardPipeline`` and works both at module scope and from inside a
running event loop (pytest-asyncio, notebooks), mirroring
``GuardPipeline.run_sync``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from collections.abc import Mapping
from typing import Any

from aisafepy.core.decisions import Action, GuardDecision, IFCViolation
from aisafepy.core.telemetry import redact
from aisafepy.flow.interpreter import IFCContext, IFCViolationError, evaluate_call
from aisafepy.flow.policy import Policy, ToolMetadata
from aisafepy.stream.pipeline import Context

_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def run_guard(
    guard: Any,
    text: str,
    *,
    role: str = "assistant",
    prompt: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GuardDecision:
    """Evaluate ``text`` with a guard or pipeline and return the decision.

    The context carries the text as both ``chunk`` and ``buffer`` so
    guards that scan either scope see it.
    """
    ctx = Context(
        chunk=text,
        buffer=text,
        prompt=prompt,
        role=role,  # type: ignore[arg-type]
        metadata=dict(metadata or {}),
    )
    if hasattr(guard, "evaluate_chunk"):
        return _run(guard.evaluate_chunk(ctx))
    return _run(guard(ctx))


def describe_decision(decision: GuardDecision) -> str:
    """Render a decision for assertion messages (evidence is redacted)."""
    try:
        evidence = json.dumps(redact(decision.evidence), default=str, sort_keys=True)
    except Exception:  # pragma: no cover - defensive
        evidence = repr(decision.evidence)
    if len(evidence) > 600:
        evidence = evidence[:597] + "..."
    return (
        f"action={decision.action.value} guard={decision.guard_name!r} tier={decision.tier} "
        f"confidence={decision.confidence:.3f} severity={decision.severity}\n"
        f"  rationale: {decision.rationale}\n"
        f"  evidence: {evidence}"
    )


def _fail(expectation: str, decision: GuardDecision, text: str) -> None:
    preview = text if len(text) <= 160 else text[:157] + "..."
    raise AssertionError(
        f"expected {expectation} for {preview!r}\n  got {describe_decision(decision)}"
    )


def assert_decision(
    guard: Any,
    text: str,
    *,
    action: str | Action | None = None,
    min_confidence: float | None = None,
    guard_name: str | None = None,
    severity_at_least: str | None = None,
    role: str = "assistant",
    prompt: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GuardDecision:
    """Assert a combination of properties on the decision and return it."""
    decision = run_guard(guard, text, role=role, prompt=prompt, metadata=metadata)
    if action is not None:
        wanted = action.value if isinstance(action, Action) else str(action)
        if decision.action.value != wanted:
            _fail(f"action={wanted}", decision, text)
    if min_confidence is not None and decision.confidence < min_confidence:
        _fail(f"confidence >= {min_confidence}", decision, text)
    if guard_name is not None and decision.guard_name != guard_name:
        _fail(f"guard_name={guard_name!r}", decision, text)
    if severity_at_least is not None:
        if _SEVERITY_RANK.get(decision.severity, -1) < _SEVERITY_RANK[severity_at_least]:
            _fail(f"severity >= {severity_at_least}", decision, text)
    return decision


def assert_blocks(guard: Any, text: str, **kw: Any) -> GuardDecision:
    """Assert the guard returns BLOCK."""
    return assert_decision(guard, text, action=Action.BLOCK, **kw)


def assert_allows(guard: Any, text: str, **kw: Any) -> GuardDecision:
    """Assert the guard returns ALLOW."""
    return assert_decision(guard, text, action=Action.ALLOW, **kw)


def assert_escalates(guard: Any, text: str, **kw: Any) -> GuardDecision:
    """Assert the guard returns ESCALATE."""
    return assert_decision(guard, text, action=Action.ESCALATE, **kw)


def assert_flags(guard: Any, text: str, **kw: Any) -> GuardDecision:
    """Assert the guard returns a terminal decision (BLOCK or ESCALATE)."""
    decision = run_guard(guard, text, **{k: kw[k] for k in ("role", "prompt", "metadata") if k in kw})
    if not decision.is_terminal:
        _fail("a terminal decision (block or escalate)", decision, text)
    return decision


def assert_transforms(
    guard: Any,
    text: str,
    *,
    expected_substring: str | None = None,
    forbidden_substring: str | None = None,
    **kw: Any,
) -> GuardDecision:
    """Assert TRANSFORM, optionally checking the transformed content."""
    decision = assert_decision(guard, text, action=Action.TRANSFORM, **kw)
    content = decision.transformed_content or ""
    if expected_substring is not None and expected_substring not in content:
        _fail(f"transformed content containing {expected_substring!r}", decision, text)
    if forbidden_substring is not None and forbidden_substring in content:
        _fail(f"transformed content without {forbidden_substring!r}", decision, text)
    return decision


# --- flow -------------------------------------------------------------------


def _policy_call(
    policy: Policy,
    tool: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    context: IFCContext | None,
    metadata: ToolMetadata | None,
) -> GuardDecision:
    strict = policy if policy.mode == "strict" else _strict_copy(policy)
    return evaluate_call(
        tool=tool,
        args=args,
        kwargs=kwargs,
        policy=strict,
        metadata=metadata,
        context=context or IFCContext(),
    )


def _strict_copy(policy: Policy) -> Policy:
    import copy

    clone = copy.copy(policy)
    clone.mode = "strict"
    return clone


def assert_policy_denies(
    policy: Policy,
    tool: str,
    *args: Any,
    context: IFCContext | None = None,
    metadata: ToolMetadata | None = None,
    reason_contains: str | None = None,
    **kwargs: Any,
) -> IFCViolation:
    """Assert the policy denies ``tool(*args, **kwargs)``; return the violation."""
    try:
        decision = _policy_call(policy, tool, args, kwargs, context, metadata)
    except IFCViolationError as exc:
        violation = exc.violation
        if reason_contains is not None and reason_contains not in violation.reason:
            raise AssertionError(
                f"policy denied {tool!r} but reason {violation.reason!r} "
                f"does not contain {reason_contains!r}"
            ) from None
        return violation
    raise AssertionError(
        f"expected policy to deny {tool!r}\n  got {describe_decision(decision)}"
    )


def assert_policy_allows(
    policy: Policy,
    tool: str,
    *args: Any,
    context: IFCContext | None = None,
    metadata: ToolMetadata | None = None,
    **kwargs: Any,
) -> GuardDecision:
    """Assert the policy allows ``tool(*args, **kwargs)``; return the decision."""
    try:
        return _policy_call(policy, tool, args, kwargs, context, metadata)
    except IFCViolationError as exc:
        v = exc.violation
        raise AssertionError(
            f"expected policy to allow {tool!r}, denied: {v.reason}\n"
            f"  provenance={sorted(v.provenance)} integrity={v.integrity} "
            f"capabilities={sorted(v.capabilities)}"
        ) from None


# --- unittest mixin ---------------------------------------------------------------


class GuardTestCase:
    """Mixin for ``unittest.TestCase`` subclasses.

    ::

        class TestMyGuards(GuardTestCase, unittest.TestCase):
            def test_blocks_override(self):
                self.assert_blocks(my_pipeline, "Ignore all previous instructions")
    """

    def run_guard(self, guard: Any, text: str, **kw: Any) -> GuardDecision:
        return run_guard(guard, text, **kw)

    def assert_blocks(self, guard: Any, text: str, **kw: Any) -> GuardDecision:
        return assert_blocks(guard, text, **kw)

    def assert_allows(self, guard: Any, text: str, **kw: Any) -> GuardDecision:
        return assert_allows(guard, text, **kw)

    def assert_escalates(self, guard: Any, text: str, **kw: Any) -> GuardDecision:
        return assert_escalates(guard, text, **kw)

    def assert_flags(self, guard: Any, text: str, **kw: Any) -> GuardDecision:
        return assert_flags(guard, text, **kw)

    def assert_transforms(self, guard: Any, text: str, **kw: Any) -> GuardDecision:
        return assert_transforms(guard, text, **kw)

    def assert_decision(self, guard: Any, text: str, **kw: Any) -> GuardDecision:
        return assert_decision(guard, text, **kw)

    def assert_policy_denies(self, policy: Policy, tool: str, *a: Any, **kw: Any) -> IFCViolation:
        return assert_policy_denies(policy, tool, *a, **kw)

    def assert_policy_allows(self, policy: Policy, tool: str, *a: Any, **kw: Any) -> GuardDecision:
        return assert_policy_allows(policy, tool, *a, **kw)

    # camelCase aliases for teams that keep unittest naming.
    assertBlocks = assert_blocks
    assertAllows = assert_allows
    assertEscalates = assert_escalates
    assertTransforms = assert_transforms
    assertPolicyDenies = assert_policy_denies
    assertPolicyAllows = assert_policy_allows
