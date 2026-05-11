"""MCP (Model Context Protocol) server middleware.

Implements the two recommendations from MCP-38 (arXiv 2603.18063) and
*MCP Tool Poisoning* (arXiv 2603.22489):

1. **Static metadata analysis at registration time.** Every tool
   description is screened for known prompt-injection patterns and
   "hidden instructions" before the tool is allowed to register with
   the server.

2. **Manifest pinning.** The adapter computes a hash of each tool's
   declared schema and compares it to a user-supplied manifest. Drift
   is denied at registration time.

The runtime path mirrors ``aisafepy.flow.interpreter.evaluate_call``
for every ``tools/call`` JSON-RPC frame.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from aisafepy.flow.interpreter import (
    IFCContext,
    IFCViolationError,
    evaluate_call,
)
from aisafepy.flow.policy import Policy, ToolMetadata
from aisafepy.flow.taint import Tainted, lift

# Patterns commonly used to smuggle instructions into tool descriptions.
# Curated from MCP-38 examples; tune via ``ifc_mcp_server(extra_patterns=...)``.
_SUSPICIOUS_DESCRIPTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in (
        r"ignore (all|previous|prior) instructions",
        r"\bsystem\s*:\s*",
        r"\bassistant\s*:\s*",
        r"</?\s*(system|assistant|tool)\s*>",
        r"hidden (instruction|rule|directive)",
        r"do not reveal",
        r"act as (an? )?(unrestricted|jailbroken)",
    )
)


@dataclass(frozen=True)
class ToolManifestEntry:
    name: str
    schema_hash: str
    """SHA-256 of canonical JSON of the tool's schema."""


def schema_hash(schema: dict[str, Any]) -> str:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ToolPoisoningError(RuntimeError):
    """Raised when a tool description fails static screening."""


class ToolDriftError(RuntimeError):
    """Raised when a tool schema drifts from the pinned manifest."""


def screen_description(name: str, description: str) -> list[str]:
    """Return a list of pattern names that matched the description."""
    hits: list[str] = []
    for pat in _SUSPICIOUS_DESCRIPTION_PATTERNS:
        if pat.search(description):
            hits.append(pat.pattern)
    return hits


def ifc_mcp_server(
    server: Any,
    policy: Policy,
    *,
    manifest: Iterable[ToolManifestEntry] = (),
    extra_patterns: Iterable[str] = (),
    **kwargs: Any,
) -> Any:
    """Wrap an MCP server with IFC + tool-poisoning defenses."""
    manifest_by_name = {m.name: m for m in manifest}
    custom_patterns = tuple(re.compile(p, re.IGNORECASE | re.MULTILINE) for p in extra_patterns)
    context = IFCContext()
    if "control_flow_integrity" in kwargs:
        context.control_flow_integrity = kwargs["control_flow_integrity"]

    # Patch tool registration. MCP servers expose either an
    # ``add_tool(name, schema, handler)`` callable or a decorator
    # (``@server.tool``); we try the imperative form first.
    add_tool = getattr(server, "add_tool", None)
    if callable(add_tool):
        original_add_tool = add_tool

        def patched_add_tool(name: str, schema: dict[str, Any], handler: Callable[..., Any]) -> Any:
            description = schema.get("description", "")
            hits = screen_description(name, description)
            hits += [pat.pattern for pat in custom_patterns if pat.search(description)]
            if hits:
                raise ToolPoisoningError(
                    f"tool {name!r} description contains suspicious patterns: {hits}"
                )
            if name in manifest_by_name:
                computed = schema_hash(schema)
                if computed != manifest_by_name[name].schema_hash:
                    raise ToolDriftError(
                        f"tool {name!r} schema hash {computed} != "
                        f"manifest {manifest_by_name[name].schema_hash}"
                    )
            return original_add_tool(name, schema, _wrap_handler(name, handler, policy, context))

        setattr(server, "add_tool", patched_add_tool)

    # Intercept the JSON-RPC dispatch. Most MCP servers expose
    # ``handle_call_tool(name, arguments) -> ...``.
    handle_call_tool = getattr(server, "handle_call_tool", None)
    if callable(handle_call_tool):
        original_handle = handle_call_tool

        @functools.wraps(original_handle)
        def patched_handle(name: str, arguments: dict[str, Any]) -> Any:
            _check(name, (), arguments, policy, context, None)
            return original_handle(name, arguments)

        setattr(server, "handle_call_tool", patched_handle)

    setattr(server, "__aisafepy_policy__", policy)
    setattr(server, "__aisafepy_context__", context)
    return server


def _wrap_handler(
    name: str,
    handler: Callable[..., Any],
    policy: Policy,
    context: IFCContext,
) -> Callable[..., Any]:
    metadata: ToolMetadata | None = getattr(handler, "__aisafepy_tool__", None)

    @functools.wraps(handler)
    def proxy(*a: Any, **kw: Any) -> Any:
        _check(name, a, kw, policy, context, metadata)
        return handler(*a, **kw)

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
        a if isinstance(a, Tainted) else lift(a, source="mcp.input", integrity="UNTRUSTED")
        if isinstance(a, str)
        else a
        for a in args
    )
    lifted_kw = {
        k: (v if isinstance(v, Tainted) else lift(v, source="mcp.input", integrity="UNTRUSTED")
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
        raise MCPIFCViolation(exc.violation) from exc


class MCPIFCViolation(RuntimeError):
    def __init__(self, violation):  # type: ignore[no-untyped-def]
        super().__init__(violation.reason)
        self.violation = violation
