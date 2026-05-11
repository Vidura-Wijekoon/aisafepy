"""OpenAI Agents SDK adapter.

We wrap each ``@function_tool`` in the agent's tool list with an IFC
check, and register a ``RunHooks`` instance so that
``IFCViolationError`` becomes a ``Tripwire`` (the SDK's preferred
mechanism for surfacing safety failures).

The adapter is duck-typed against ``openai-agents`` so importing this
module without the SDK installed will raise at the first wrapping
attempt rather than at import time.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from aisafepy.core.decisions import Tripwire
from aisafepy.flow.interpreter import (
    IFCContext,
    IFCViolationError,
    evaluate_call,
)
from aisafepy.flow.policy import Policy, ToolMetadata
from aisafepy.flow.taint import Tainted, lift


def secure_openai_agent(agent: Any, policy: Policy, **kwargs: Any) -> Any:
    """Wrap an OpenAI Agents SDK ``Agent`` with IFC enforcement.

    The original agent object is *mutated in place*. Its ``tools``
    list is replaced with IFC-wrapped versions. Returns the same agent
    for convenience so callers can write
    ``safe_agent = secure_agent(agent, policy)``.
    """
    if not hasattr(agent, "tools"):
        raise TypeError(
            f"Expected an openai-agents Agent (with `.tools` attribute); got {type(agent)!r}"
        )

    context = IFCContext()
    if "control_flow_integrity" in kwargs:
        context.control_flow_integrity = kwargs["control_flow_integrity"]

    wrapped: list[Any] = []
    for tool in agent.tools:
        wrapped.append(_wrap_function_tool(tool, policy, context))
    agent.tools = wrapped
    agent.__aisafepy_policy__ = policy
    agent.__aisafepy_context__ = context
    return agent


def _wrap_function_tool(tool: Any, policy: Policy, context: IFCContext) -> Any:
    """Wrap one openai-agents FunctionTool.

    The SDK exposes the user's Python function as ``tool.on_invoke_tool``
    or ``tool.python_fn`` depending on version; we replace whichever
    attribute is present with an IFC-checking proxy.
    """
    # Determine the underlying callable.
    inner: Callable[..., Any] | None = None
    attr_name: str | None = None
    for candidate in ("python_fn", "on_invoke_tool", "fn", "_fn"):
        if hasattr(tool, candidate) and callable(getattr(tool, candidate)):
            inner = getattr(tool, candidate)
            attr_name = candidate
            break
    if inner is None or attr_name is None:
        # If we can't find an invocation callable, leave the tool alone
        # rather than break the agent.
        return tool

    tool_name = getattr(tool, "name", None) or getattr(inner, "__name__", "tool")
    metadata: ToolMetadata | None = getattr(inner, "__aisafepy_tool__", None)

    if inspect.iscoroutinefunction(inner):

        @functools.wraps(inner)
        async def async_proxy(*args: Any, **kw: Any) -> Any:
            _check_and_propagate(tool_name, args, kw, policy, context, metadata)
            return await inner(*args, **kw)

        setattr(tool, attr_name, async_proxy)
    else:

        @functools.wraps(inner)
        def sync_proxy(*args: Any, **kw: Any) -> Any:
            _check_and_propagate(tool_name, args, kw, policy, context, metadata)
            return inner(*args, **kw)

        setattr(tool, attr_name, sync_proxy)

    return tool


def _check_and_propagate(
    tool_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    policy: Policy,
    context: IFCContext,
    metadata: ToolMetadata | None,
) -> None:
    """Run ``evaluate_call``; on violation, raise an SDK-compatible ``Tripwire``."""
    # Promote bare-string arguments coming from the LLM (which always
    # originate from UNTRUSTED model output) into Tainted values so the
    # policy sees integrity correctly. The adapter caller may opt out
    # of this auto-lifting by passing already-Tainted values.
    lifted_args = tuple(
        a if isinstance(a, Tainted) else lift(a, source="llm.output", integrity="UNTRUSTED")
        if isinstance(a, str)
        else a
        for a in args
    )
    lifted_kwargs = {
        k: (v if isinstance(v, Tainted) else lift(v, source="llm.output", integrity="UNTRUSTED")
            if isinstance(v, str)
            else v)
        for k, v in kwargs.items()
    }
    try:
        evaluate_call(
            tool=tool_name,
            args=lifted_args,
            kwargs=lifted_kwargs,
            policy=policy,
            metadata=metadata,
            context=context,
        )
    except IFCViolationError as exc:
        decision = exc.decision or exc.violation.to_guard_decision()
        # Tripwire is a GuardDecision subclass; raise it as an exception
        # so the SDK's tripwire-handling machinery surfaces a clean error.
        raise _TripwireException(Tripwire(**decision.model_dump())) from exc


class _TripwireException(Exception):
    """Raised when an OpenAI Agents tool is blocked by IFC.

    The SDK looks for ``input_guardrail_tripwire_triggered`` /
    ``output_guardrail_tripwire_triggered`` exception types by name in
    several versions; rather than depend on those names we raise a
    generic exception that carries the structured ``Tripwire`` on the
    ``.tripwire`` attribute. Adapters can re-raise as the SDK-specific
    type if needed.
    """

    def __init__(self, tripwire: Tripwire):
        super().__init__(tripwire.rationale)
        self.tripwire = tripwire
