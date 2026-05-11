"""LangGraph adapter.

We register a node-level pre-execution hook that intercepts tool nodes
in the graph and runs ``evaluate_call`` before delegating. The taint
field is stored on the state under ``state["__aisafepy_taint__"]`` so
adjacent nodes can read / extend it.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from aisafepy.flow.interpreter import (
    IFCContext,
    IFCViolationError,
    evaluate_call,
)
from aisafepy.flow.policy import Policy, ToolMetadata
from aisafepy.flow.taint import Tainted, lift


def secure_langgraph(graph: Any, policy: Policy, **kwargs: Any) -> Any:
    """Wrap a LangGraph compiled graph or builder with IFC enforcement.

    LangGraph's API evolves; we use duck typing. The graph object is
    expected to expose either:

    * ``add_node`` / ``set_node`` (the pre-compile builder), in which
      case we re-wrap every tool node, or
    * a ``nodes`` mapping (post-compile), in which case we replace the
      callable on each tool node.

    A run-scoped ``IFCContext`` is attached to the graph at
    ``graph.__aisafepy_context__``.
    """
    context = IFCContext()
    if "control_flow_integrity" in kwargs:
        context.control_flow_integrity = kwargs["control_flow_integrity"]

    # Try the post-compile path first.
    nodes = getattr(graph, "nodes", None)
    if isinstance(nodes, dict):
        for name, node in list(nodes.items()):
            wrapped = _wrap_node(name, node, policy, context)
            nodes[name] = wrapped
        setattr(graph, "__aisafepy_policy__", policy)
        setattr(graph, "__aisafepy_context__", context)
        return graph

    # Fall back to the builder path.
    original_add_node = getattr(graph, "add_node", None)
    if callable(original_add_node):

        def add_node_wrapped(name: str, action: Callable[..., Any], *a: Any, **kw: Any) -> Any:
            return original_add_node(name, _wrap_callable(name, action, policy, context), *a, **kw)

        setattr(graph, "add_node", add_node_wrapped)
        setattr(graph, "__aisafepy_policy__", policy)
        setattr(graph, "__aisafepy_context__", context)
        return graph

    raise TypeError(
        f"Expected a LangGraph compiled graph or builder; got {type(graph)!r}"
    )


def _wrap_node(name: str, node: Any, policy: Policy, context: IFCContext) -> Any:
    """Wrap a LangGraph node. The node is usually a callable or an object
    with a ``__call__`` / ``invoke`` method."""
    inner = getattr(node, "invoke", None) or node
    if not callable(inner):
        return node
    return _wrap_callable(name, inner, policy, context)


def _wrap_callable(name: str, inner: Callable[..., Any], policy: Policy, context: IFCContext) -> Callable[..., Any]:
    metadata: ToolMetadata | None = getattr(inner, "__aisafepy_tool__", None)

    @functools.wraps(inner)
    def proxy(state: Any, *a: Any, **kw: Any) -> Any:
        # LangGraph nodes receive a state dict; we use it to ferry the
        # current taint context between nodes.
        if isinstance(state, dict):
            existing = state.get("__aisafepy_taint__")
            if isinstance(existing, IFCContext):
                local_ctx = existing
            else:
                local_ctx = context
            state["__aisafepy_taint__"] = local_ctx
        else:
            local_ctx = context

        # Convert any string fields in the state that look like
        # tool-output payloads into Tainted UNTRUSTED so the policy
        # engine sees them correctly.
        args_for_eval: tuple[Any, ...] = ()
        kwargs_for_eval: dict[str, Any] = {}
        if isinstance(state, dict):
            args_for_eval = tuple(
                lift(v, source="state", integrity="UNTRUSTED")
                if isinstance(v, str) and not isinstance(v, Tainted)
                else v
                for v in state.values()
            )

        try:
            evaluate_call(
                tool=name,
                args=args_for_eval,
                kwargs=kwargs_for_eval,
                policy=policy,
                metadata=metadata,
                context=local_ctx,
            )
        except IFCViolationError as exc:
            raise LangGraphIFCViolation(exc.violation) from exc
        return inner(state, *a, **kw)

    return proxy


class LangGraphIFCViolation(RuntimeError):
    """Raised inside a LangGraph node when IFC denies execution."""

    def __init__(self, violation):  # type: ignore[no-untyped-def]
        super().__init__(violation.reason)
        self.violation = violation
