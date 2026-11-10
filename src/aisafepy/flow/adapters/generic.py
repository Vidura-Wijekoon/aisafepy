"""Framework-free IFC adapter for plain Python callables.

The other adapters in this package wrap a specific agent framework's
tool object. Many agent loops are hand-rolled: a dict of callables, a
JSON tool schema sent to the model, a dispatch on the returned tool
name. :func:`secure_callables` and :class:`ToolRegistry` cover that
case with no extra dependencies.

Conventions (shared with the OpenAI Agents adapter):

* Plain ``str`` arguments are assumed to originate from model output
  and are lifted as ``UNTRUSTED`` with provenance ``llm.output`` for
  the policy check. Already-``Tainted`` arguments keep their labels.
* The inner function is called with the *original* arguments, so tools
  written against plain values keep working. Tools that want labels can
  declare ``Tainted`` parameters and pass ``Tainted`` values in.
* Return values are lifted using :meth:`Policy.output_label`, so the
  output of ``gmail.read`` carries whatever label the policy assigned
  to that source, a declassifier's output is ``TRUSTED``, and an
  unknown tool's output is ``UNTRUSTED`` under ``tool:<name>``.
"""

from __future__ import annotations

import functools
import inspect
import types
import typing
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from aisafepy.core.decisions import IFCViolation
from aisafepy.flow.interpreter import IFCContext, evaluate_call
from aisafepy.flow.policy import Policy, ToolMetadata, secure_tool
from aisafepy.flow.taint import Integrity, Tainted, lift

LLM_INPUT_SOURCE = "llm.output"


def secure_callables(
    tools: Mapping[str, Callable[..., Any]],
    policy: Policy,
    *,
    context: IFCContext | None = None,
    lift_outputs: bool = True,
    lift_inputs: bool = True,
    input_source: str = LLM_INPUT_SOURCE,
    mediator: Callable[[IFCViolation], bool] | None = None,
) -> dict[str, Callable[..., Any]]:
    """Wrap each callable in ``tools`` with an IFC check.

    Returns a new dict with the same keys. Sync and ``async def``
    callables are both supported; the wrapper preserves the inner
    function's name, docstring, signature, and any ``@secure_tool``
    metadata. A denied call raises
    :class:`aisafepy.flow.interpreter.IFCViolationError` (strict mode)
    or consults ``mediator`` (mediated mode).

    ``context`` is shared across every wrapped tool so the taint chain
    and ``on_decision`` hook span the whole run; pass a fresh one per
    agent run.
    """
    ctx = context if context is not None else IFCContext()
    return {
        name: secure_callable(
            fn,
            policy,
            name=name,
            context=ctx,
            lift_outputs=lift_outputs,
            lift_inputs=lift_inputs,
            input_source=input_source,
            mediator=mediator,
        )
        for name, fn in tools.items()
    }


def secure_callable(
    fn: Callable[..., Any],
    policy: Policy,
    *,
    name: str | None = None,
    context: IFCContext | None = None,
    lift_outputs: bool = True,
    lift_inputs: bool = True,
    input_source: str = LLM_INPUT_SOURCE,
    mediator: Callable[[IFCViolation], bool] | None = None,
) -> Callable[..., Any]:
    """Wrap a single callable. See :func:`secure_callables`."""
    tool_name = name or getattr(fn, "__name__", "tool")
    ctx = context if context is not None else IFCContext()
    metadata: ToolMetadata | None = getattr(fn, "__aisafepy_tool__", None)

    def _check(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        a, kw = (args, kwargs)
        if lift_inputs:
            a = tuple(_lift_input(x, input_source) for x in args)
            kw = {k: _lift_input(v, input_source) for k, v in kwargs.items()}
        evaluate_call(
            tool=tool_name,
            args=a,
            kwargs=kw,
            policy=policy,
            metadata=metadata,
            context=ctx,
            mediator=mediator,
        )

    def _finish(result: Any) -> Any:
        if not lift_outputs:
            return result
        return policy.lift_output(tool_name, result)

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_proxy(*args: Any, **kwargs: Any) -> Any:
            _check(args, kwargs)
            return _finish(await fn(*args, **kwargs))

        async_proxy.__aisafepy_wrapped__ = True  # type: ignore[attr-defined]
        return async_proxy

    @functools.wraps(fn)
    def proxy(*args: Any, **kwargs: Any) -> Any:
        _check(args, kwargs)
        return _finish(fn(*args, **kwargs))

    proxy.__aisafepy_wrapped__ = True  # type: ignore[attr-defined]
    return proxy


def lift_tool_output(
    policy: Policy,
    tool: str,
    value: Any,
    *,
    source: str | None = None,
) -> Tainted[Any]:
    """Lift a tool's return value with the label the policy assigns to it.

    Thin alias of :meth:`Policy.lift_output`, kept here because adapter
    code reads more naturally with a free function.
    """
    return policy.lift_output(tool, value, source=source)


def _lift_input(value: Any, source: str) -> Any:
    if isinstance(value, Tainted):
        return value
    if isinstance(value, str):
        return lift(value, source=source, integrity="UNTRUSTED")
    return value


# ---- registry -----------------------------------------------------------------


class ToolRegistry:
    """A named collection of tools with IFC metadata and schema export.

    ::

        registry = ToolRegistry()

        @registry.register(capabilities={"write.external"})
        def send_email(to: str, body: str) -> None:
            '''Send an email.'''

        tools = registry.secured(policy)      # dict[str, Callable]
        schemas = registry.schemas()          # for the model / MCP manifest
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    # ---- registration ----------------------------------------------------

    def register(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        capabilities: Iterable[Any] = (),
        required_integrity: Integrity = "UNTRUSTED",
    ) -> Any:
        """Register ``fn`` (usable bare or with keyword options).

        Applies :func:`aisafepy.flow.policy.secure_tool` so the
        interpreter sees the declared capabilities. Registering the same
        name twice raises ``ValueError``.
        """

        def _apply(f: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or f.__name__
            if tool_name in self._tools:
                raise ValueError(f"tool {tool_name!r} is already registered")
            decorated = secure_tool(
                capabilities=capabilities,
                required_integrity=required_integrity,
                name=tool_name,
            )(f)
            self._tools[tool_name] = decorated
            return decorated

        if fn is not None:
            return _apply(fn)
        return _apply

    def add(self, name: str, fn: Callable[..., Any]) -> None:
        """Register an existing callable under ``name`` without decorating it."""
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        self._tools[name] = fn

    # ---- queries -----------------------------------------------------------

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Callable[..., Any]:
        return self._tools[name]

    def metadata(self, name: str) -> ToolMetadata | None:
        return getattr(self._tools[name], "__aisafepy_tool__", None)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools)

    def as_dict(self) -> dict[str, Callable[..., Any]]:
        return dict(self._tools)

    # ---- IFC -----------------------------------------------------------------

    def secured(self, policy: Policy, **kwargs: Any) -> dict[str, Callable[..., Any]]:
        """Return every registered tool wrapped by :func:`secure_callables`."""
        return secure_callables(self._tools, policy, **kwargs)

    # ---- schema export -------------------------------------------------------

    def schemas(self) -> dict[str, dict[str, Any]]:
        """JSON-schema style descriptions of every tool, keyed by name.

        Each entry has ``name``, ``description`` (first docstring
        paragraph), ``parameters`` (an ``object`` schema derived from the
        signature and annotations), and an ``aisafepy`` block with the
        declared capabilities and required integrity. The dict is
        deterministic, so it can be hashed for MCP manifest pinning via
        :func:`aisafepy.flow.adapters.mcp.schema_hash`.
        """
        return {name: tool_schema(name, fn) for name, fn in self._tools.items()}


def tool_schema(name: str, fn: Callable[..., Any]) -> dict[str, Any]:
    """Build the schema for one callable. See :meth:`ToolRegistry.schemas`."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None

    properties: dict[str, Any] = {}
    required: list[str] = []
    if sig is not None:
        for pname, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            schema = _annotation_schema(hints.get(pname, param.annotation))
            if param.default is inspect.Parameter.empty:
                required.append(pname)
            else:
                default = param.default
                if _is_json_scalar(default):
                    schema["default"] = default
            properties[pname] = schema

    meta: ToolMetadata | None = getattr(fn, "__aisafepy_tool__", None)
    doc = inspect.getdoc(fn) or ""
    description = doc.split("\n\n", 1)[0].strip()
    out: dict[str, Any] = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "aisafepy": {
            "capabilities": sorted(meta.capabilities) if meta else [],
            "required_integrity": meta.required_integrity if meta else "UNTRUSTED",
        },
    }
    if sig is not None and sig.return_annotation is not inspect.Signature.empty:
        out["returns"] = _annotation_schema(hints.get("return", sig.return_annotation))
    return out


_SCALARS: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
    bytes: "string",
}


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if isinstance(annotation, str):
        # Unresolvable forward reference; record it verbatim (minus the
        # quotes a string literal annotation keeps under PEP 563).
        return {"description": f"type: {annotation.strip().strip(chr(39) + chr(34))}"}
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if annotation in _SCALARS:
        return {"type": _SCALARS[annotation]}
    if origin is Tainted:
        inner = _annotation_schema(args[0]) if args else {}
        inner["x-tainted"] = True
        return inner
    if annotation is Tainted:
        return {"x-tainted": True}
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        schema = _annotation_schema(non_none[0]) if len(non_none) == 1 else {
            "anyOf": [_annotation_schema(a) for a in non_none]
        }
        if len(non_none) != len(args):
            schema["nullable"] = True
        return schema
    if origin in (list, tuple, set, frozenset) or annotation in (list, tuple, set, frozenset):
        schema: dict[str, Any] = {"type": "array"}
        if args and args[0] is not Ellipsis:
            schema["items"] = _annotation_schema(args[0])
        return schema
    if origin is dict or annotation is dict:
        schema = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = _annotation_schema(args[1])
        return schema
    if origin is typing.Literal:
        return {"enum": list(args)}
    name = getattr(annotation, "__name__", None) or str(annotation)
    return {"description": f"type: {name}"}


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))
