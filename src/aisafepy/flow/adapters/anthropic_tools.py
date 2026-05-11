"""Anthropic SDK tool-use adapter.

The Anthropic SDK exposes tools as JSON schemas plus a user-provided
``tool_handler`` function (or per-tool dispatch). We wrap the dispatch
function so each tool invocation runs through ``evaluate_call`` before
the actual handler executes.

Because the Anthropic SDK does not own the handler callable (the user
does), this adapter takes the handler as an argument and returns a
wrapped version.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Mapping

from aisafepy.flow.interpreter import (
    IFCContext,
    IFCViolationError,
    evaluate_call,
)
from aisafepy.flow.policy import Policy, ToolMetadata
from aisafepy.flow.taint import Tainted, lift


def secure_anthropic_tools(
    handler: Callable[..., Any] | Mapping[str, Callable[..., Any]],
    policy: Policy,
    **kwargs: Any,
) -> Callable[..., Any] | Mapping[str, Callable[..., Any]]:
    """Wrap a single dispatch handler (``handler(tool_name, **args)``) or a
    per-tool dict (``{"send_email": send_email, ...}``)."""
    context = IFCContext()
    if "control_flow_integrity" in kwargs:
        context.control_flow_integrity = kwargs["control_flow_integrity"]

    if isinstance(handler, Mapping):
        return {name: _wrap_single(name, fn, policy, context) for name, fn in handler.items()}

    @functools.wraps(handler)
    def dispatch_wrapper(tool_name: str, *a: Any, **kw: Any) -> Any:
        _check(tool_name, a, kw, policy, context, getattr(handler, "__aisafepy_tool__", None))
        return handler(tool_name, *a, **kw)

    if inspect.iscoroutinefunction(handler):

        @functools.wraps(handler)
        async def async_dispatch_wrapper(tool_name: str, *a: Any, **kw: Any) -> Any:
            _check(tool_name, a, kw, policy, context, getattr(handler, "__aisafepy_tool__", None))
            return await handler(tool_name, *a, **kw)

        return async_dispatch_wrapper

    return dispatch_wrapper


def _wrap_single(
    name: str,
    fn: Callable[..., Any],
    policy: Policy,
    context: IFCContext,
) -> Callable[..., Any]:
    metadata: ToolMetadata | None = getattr(fn, "__aisafepy_tool__", None)
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def aproxy(*a: Any, **kw: Any) -> Any:
            _check(name, a, kw, policy, context, metadata)
            return await fn(*a, **kw)

        return aproxy

    @functools.wraps(fn)
    def proxy(*a: Any, **kw: Any) -> Any:
        _check(name, a, kw, policy, context, metadata)
        return fn(*a, **kw)

    return proxy


def _check(
    name: str,
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
            tool=name,
            args=lifted_args,
            kwargs=lifted_kw,
            policy=policy,
            metadata=metadata,
            context=context,
        )
    except IFCViolationError as exc:
        raise AnthropicIFCViolation(exc.violation) from exc


class AnthropicIFCViolation(RuntimeError):
    def __init__(self, violation):  # type: ignore[no-untyped-def]
        super().__init__(violation.reason)
        self.violation = violation
