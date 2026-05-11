"""LlamaIndex agent / tool adapter.

We replace each tool's ``fn`` with an IFC-checking proxy. LlamaIndex's
``FunctionTool`` and ``BaseTool`` subclasses expose the callable as
``.fn`` (post 0.10); older variants used ``._fn``. We try both.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from aisafepy.flow.interpreter import (
    IFCContext,
    IFCViolationError,
    evaluate_call,
)
from aisafepy.flow.policy import Policy, ToolMetadata
from aisafepy.flow.taint import Tainted, lift


def secure_llamaindex_agent(agent: Any, policy: Policy, **kwargs: Any) -> Any:
    """Wrap a LlamaIndex agent (Worker / Runner) with IFC enforcement."""
    context = IFCContext()
    if "control_flow_integrity" in kwargs:
        context.control_flow_integrity = kwargs["control_flow_integrity"]

    tools = _resolve_tools(agent)
    if tools is None:
        raise TypeError(f"Could not locate `.tools` on {type(agent)!r}")

    for tool in tools:
        _wrap_tool_in_place(tool, policy, context)

    setattr(agent, "__aisafepy_policy__", policy)
    setattr(agent, "__aisafepy_context__", context)
    return agent


def _resolve_tools(agent: Any) -> list[Any] | None:
    for attr in ("tools", "_tools"):
        t = getattr(agent, attr, None)
        if isinstance(t, list):
            return t
        if hasattr(t, "values"):
            return list(t.values())
    return None


def _wrap_tool_in_place(tool: Any, policy: Policy, context: IFCContext) -> None:
    name = getattr(tool, "metadata", None)
    tool_name = getattr(name, "name", None) if name is not None else getattr(tool, "name", "tool")
    inner = getattr(tool, "fn", None) or getattr(tool, "_fn", None)
    if inner is None or not callable(inner):
        return

    metadata: ToolMetadata | None = getattr(inner, "__aisafepy_tool__", None)
    is_async = inspect.iscoroutinefunction(inner)

    if is_async:

        @functools.wraps(inner)
        async def async_proxy(*args: Any, **kw: Any) -> Any:
            _check(tool_name, args, kw, policy, context, metadata)
            return await inner(*args, **kw)

        if hasattr(tool, "fn"):
            tool.fn = async_proxy
        else:
            tool._fn = async_proxy
    else:

        @functools.wraps(inner)
        def sync_proxy(*args: Any, **kw: Any) -> Any:
            _check(tool_name, args, kw, policy, context, metadata)
            return inner(*args, **kw)

        if hasattr(tool, "fn"):
            tool.fn = sync_proxy
        else:
            tool._fn = sync_proxy


def _check(
    tool_name: str,
    args: tuple[Any, ...],
    kw: dict[str, Any],
    policy: Policy,
    context: IFCContext,
    metadata: ToolMetadata | None,
) -> None:
    lifted_args = tuple(
        a if isinstance(a, Tainted) else lift(a, source="llm.output", integrity="UNTRUSTED")
        if isinstance(a, str)
        else a
        for a in args
    )
    lifted_kw = {
        k: (v if isinstance(v, Tainted) else lift(v, source="llm.output", integrity="UNTRUSTED")
            if isinstance(v, str)
            else v)
        for k, v in kw.items()
    }
    try:
        evaluate_call(
            tool=tool_name,
            args=lifted_args,
            kwargs=lifted_kw,
            policy=policy,
            metadata=metadata,
            context=context,
        )
    except IFCViolationError as exc:
        raise LlamaIndexIFCViolation(exc.violation) from exc


class LlamaIndexIFCViolation(RuntimeError):
    def __init__(self, violation):  # type: ignore[no-untyped-def]
        super().__init__(violation.reason)
        self.violation = violation
