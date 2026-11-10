"""LangChain tool adapter for ``aisafepy.flow``.

LangChain exposes tools as ``BaseTool`` objects: a ``name``, a sync
``_run`` and/or an async ``_arun``, and the public ``invoke`` /
``ainvoke`` entry points that the agent executor calls. LangGraph's
``ToolNode`` holds them in a ``tools_by_name`` mapping. This adapter
wraps those objects *in place* so the agent keeps calling them exactly
as before, while every invocation first passes through
:func:`aisafepy.flow.interpreter.evaluate_call`.

Conventions match the other framework adapters:

* Plain string arguments are lifted as UNTRUSTED ``llm.output``, because
  the model chose them. Already-``Tainted`` arguments keep their labels.
* The wrapped tool is called with the *original* (unwrapped) arguments
  so tool bodies written for LangChain keep working.
* Return values are lifted with the policy's source label for the tool
  (``Policy.lift_output``) when ``lift_outputs`` is on. ``Tainted.__str__``
  lowers, so LangChain's ``ToolMessage`` rendering is unaffected; agents
  that thread the value into a later tool call benefit from the label.
* Metadata from ``@secure_tool`` is honoured whether it sits on the tool
  object itself or on the underlying ``func`` / ``coroutine`` attribute.
* A denial raises :class:`LangChainIFCViolation`, carrying the
  structured :class:`IFCViolation` on ``.violation``.

Nothing from ``langchain`` is imported; the objects are duck-typed.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from aisafepy.core.decisions import IFCViolation
from aisafepy.flow.interpreter import IFCContext, IFCViolationError, evaluate_call
from aisafepy.flow.policy import Policy, ToolMetadata
from aisafepy.flow.taint import Tainted, lift

_SKIP_KWARGS = frozenset({"run_manager", "config", "callbacks", "tags", "metadata", "run_id"})
"""Keyword arguments LangChain threads through tool calls that are
plumbing, not data the model chose; they are neither lifted nor checked."""

_SENTINEL = "__aisafepy_wrapped__"


class LangChainIFCViolation(RuntimeError):
    """Raised when IFC denies a LangChain tool invocation."""

    def __init__(self, violation: IFCViolation):
        super().__init__(violation.reason)
        self.violation = violation


def secure_langchain_tools(
    tools: Iterable[Any],
    policy: Policy,
    *,
    context: IFCContext | None = None,
    mediator: Callable[[IFCViolation], bool] | None = None,
    lift_outputs: bool = True,
) -> list[Any]:
    """Wrap each LangChain tool in place and return them as a list.

    ``context`` is shared by every wrapped tool so one agent run keeps a
    single taint chain; pass a fresh :class:`IFCContext` per run when
    you want isolation. ``mediator`` is only consulted when
    ``policy.mode == "mediated"``.
    """
    ctx = context if context is not None else IFCContext()
    wrapped: list[Any] = []
    for tool in tools:
        wrapped.append(_secure_one(tool, policy, ctx, mediator, lift_outputs))
    return wrapped


def secure_tool_node(
    node: Any,
    policy: Policy,
    *,
    context: IFCContext | None = None,
    mediator: Callable[[IFCViolation], bool] | None = None,
    lift_outputs: bool = True,
) -> Any:
    """Wrap every tool held by a LangGraph ``ToolNode``-shaped object.

    The node must expose ``tools_by_name`` (a mapping). Tools are wrapped
    in place, so the node needs no further changes. Returns the node.
    """
    registry = getattr(node, "tools_by_name", None)
    if not isinstance(registry, Mapping):
        raise TypeError(
            f"expected an object with a tools_by_name mapping (LangGraph ToolNode); got {type(node)!r}"
        )
    ctx = context if context is not None else IFCContext()
    for tool in registry.values():
        _secure_one(tool, policy, ctx, mediator, lift_outputs)
    node.__aisafepy_policy__ = policy
    node.__aisafepy_context__ = ctx
    return node


# ---- internals --------------------------------------------------------------


def _secure_one(
    tool: Any,
    policy: Policy,
    ctx: IFCContext,
    mediator: Callable[[IFCViolation], bool] | None,
    lift_outputs: bool,
) -> Any:
    if getattr(tool, _SENTINEL, False):
        return tool
    name = _tool_name(tool)
    metadata = _metadata_for(tool)
    patched = False
    # Prefer the private entry points: LangChain's public invoke() calls them.
    for attr in ("_run", "_arun"):
        if _patch(tool, attr, name, policy, ctx, mediator, metadata, lift_outputs):
            patched = True
    if not patched:
        for attr in ("invoke", "ainvoke", "run", "arun"):
            if _patch(tool, attr, name, policy, ctx, mediator, metadata, lift_outputs):
                patched = True
    if not patched:
        raise TypeError(f"tool {name!r} exposes none of _run/_arun/invoke/ainvoke/run/arun")
    _set(tool, _SENTINEL, True)
    _set(tool, "__aisafepy_policy__", policy)
    _set(tool, "__aisafepy_context__", ctx)
    return tool


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    for attr in ("func", "coroutine"):
        fn = getattr(tool, attr, None)
        if fn is not None and getattr(fn, "__name__", None):
            return fn.__name__
    return type(tool).__name__


def _metadata_for(tool: Any) -> ToolMetadata | None:
    for candidate in (tool, getattr(tool, "func", None), getattr(tool, "coroutine", None)):
        meta = getattr(candidate, "__aisafepy_tool__", None)
        if isinstance(meta, ToolMetadata):
            return meta
    return None


def _patch(
    tool: Any,
    attr: str,
    name: str,
    policy: Policy,
    ctx: IFCContext,
    mediator: Callable[[IFCViolation], bool] | None,
    metadata: ToolMetadata | None,
    lift_outputs: bool,
) -> bool:
    inner = getattr(tool, attr, None)
    if inner is None or not callable(inner):
        return False
    if getattr(inner, _SENTINEL, False):
        return True

    if inspect.iscoroutinefunction(inner):

        @functools.wraps(inner)
        async def aproxy(*args: Any, **kwargs: Any) -> Any:
            _check(name, args, kwargs, policy, ctx, mediator, metadata)
            result = await inner(*args, **kwargs)
            return policy.lift_output(name, result) if lift_outputs else result

        proxy: Callable[..., Any] = aproxy
    else:

        @functools.wraps(inner)
        def sproxy(*args: Any, **kwargs: Any) -> Any:
            _check(name, args, kwargs, policy, ctx, mediator, metadata)
            result = inner(*args, **kwargs)
            return policy.lift_output(name, result) if lift_outputs else result

        proxy = sproxy

    setattr(proxy, _SENTINEL, True)
    return _set(tool, attr, proxy)


def _set(obj: Any, attr: str, value: Any) -> bool:
    """Set an attribute on a possibly-pydantic object; return success."""
    try:
        setattr(obj, attr, value)
        return True
    except Exception:
        with contextlib.suppress(Exception):
            object.__setattr__(obj, attr, value)
            return True
    return False


def _lift_value(value: Any) -> Any:
    if isinstance(value, Tainted):
        return value
    if isinstance(value, str):
        return lift(value, source="llm.output", integrity="UNTRUSTED")
    if isinstance(value, Mapping):
        # LangChain passes structured tool input as one dict; label each
        # string field so per-argument deny rules can see them.
        return {k: _lift_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_lift_value(v) for v in value]
    return value


def _check(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    policy: Policy,
    ctx: IFCContext,
    mediator: Callable[[IFCViolation], bool] | None,
    metadata: ToolMetadata | None,
) -> None:
    lifted_args = tuple(_lift_value(a) for a in args)
    lifted_kwargs = {k: _lift_value(v) for k, v in kwargs.items() if k not in _SKIP_KWARGS}
    # A single dict positional (LangChain's structured input) is flattened
    # into keyword arguments so deny rules written as ``when=lambda to, body``
    # or ``{"arg": "to", ...}`` match.
    if len(lifted_args) == 1 and isinstance(lifted_args[0], dict) and not lifted_kwargs:
        lifted_kwargs = dict(lifted_args[0])
        lifted_args = ()
    try:
        evaluate_call(
            tool=name,
            args=lifted_args,
            kwargs=lifted_kwargs,
            policy=policy,
            metadata=metadata,
            context=ctx,
            mediator=mediator,
        )
    except IFCViolationError as exc:
        raise LangChainIFCViolation(exc.violation) from exc
