"""MCP (Model Context Protocol) server middleware.

Implements the two recommendations from MCP-38 (arXiv 2603.18063) and
*MCP Tool Poisoning* (arXiv 2603.22489):

1. **Static metadata analysis at registration time.** Every tool
   description, and every parameter description nested inside its
   input schema, is screened for known prompt-injection patterns and
   "hidden instructions" before the tool is allowed to register with
   the server.

2. **Manifest pinning.** The adapter computes a hash of each tool's
   declared schema and compares it to a user-supplied manifest. Drift
   is denied at registration time. :func:`build_manifest`,
   :func:`diff_manifest` and the JSON helpers make it practical to pin
   a manifest in CI and verify it on every server start.

The runtime path mirrors ``aisafepy.flow.interpreter.evaluate_call``
for every ``tools/call`` JSON-RPC frame. Both synchronous and
coroutine handlers are supported.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

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
        r"(ignore|disregard|forget|override)\s+(all\s+|any\s+)?((the|your)\s+)?"
        r"(previous|prior|above|earlier|preceding|system)?\s*(instructions|rules|directives|guidelines)",
        r"\bsystem\s*:\s*",
        r"\bassistant\s*:\s*",
        r"</?\s*(system|assistant|tool)\s*>",
        r"hidden (instruction|rule|directive)",
        r"do not (reveal|tell|mention|show)",
        r"act as (an? )?(unrestricted|jailbroken)",
        r"<important>|<!--\s*(instruction|note to (the )?(ai|assistant|model))",
        r"before (using|calling|invoking) this tool,? (you must|always|first)",
        r"(read|send|forward|upload) (the )?(contents? of )?(~|/)?\.?(ssh|env|aws|bashrc|config)",
        r"(exfiltrate|leak|smuggle)",
    )
)

# Schema keys whose string values are prose shown to the model.
_PROSE_KEYS = ("description", "title", "summary", "examples", "x-instructions")


@dataclass(frozen=True)
class ToolManifestEntry:
    name: str
    schema_hash: str
    """SHA-256 of canonical JSON of the tool's schema."""


@dataclass
class ManifestDiff:
    """Result of :func:`diff_manifest`."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    drifted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.added or self.removed or self.drifted)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def schema_hash(schema: dict[str, Any]) -> str:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ToolPoisoningError(RuntimeError):
    """Raised when a tool description fails static screening."""


class ToolDriftError(RuntimeError):
    """Raised when a tool schema drifts from the pinned manifest."""


# ---- static screening ---------------------------------------------------


def screen_description(
    name: str, description: str, *, extra_patterns: Iterable[re.Pattern[str]] = ()
) -> list[str]:
    """Return a list of pattern sources that matched the description."""
    hits: list[str] = []
    if not description:
        return hits
    for pat in (*_SUSPICIOUS_DESCRIPTION_PATTERNS, *extra_patterns):
        if pat.search(description):
            hits.append(pat.pattern)
    return hits


def iter_schema_prose(schema: Any, _path: str = "") -> Iterable[tuple[str, str]]:
    """Yield ``(json_path, text)`` for every prose field nested in a schema.

    Tool-poisoning payloads are frequently hidden in a *parameter*
    description rather than the top-level tool description, because
    many clients only display the latter to humans.
    """
    if isinstance(schema, Mapping):
        for key, value in schema.items():
            path = f"{_path}.{key}" if _path else str(key)
            if key in _PROSE_KEYS and isinstance(value, str):
                yield path, value
            elif key in _PROSE_KEYS and isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        yield f"{path}[{i}]", item
            else:
                yield from iter_schema_prose(value, path)
    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            yield from iter_schema_prose(item, f"{_path}[{i}]")


def screen_schema(
    schema: Mapping[str, Any], *, extra_patterns: Iterable[re.Pattern[str]] = ()
) -> dict[str, list[str]]:
    """Screen every prose field in ``schema``; returns ``{json_path: [patterns]}``."""
    findings: dict[str, list[str]] = {}
    for path, text in iter_schema_prose(schema):
        hits = screen_description(path, text, extra_patterns=extra_patterns)
        if hits:
            findings[path] = hits
    return findings


def screen_tools(
    tools: Iterable[Mapping[str, Any]], *, extra_patterns: Iterable[str] = ()
) -> dict[str, dict[str, list[str]]]:
    """Screen a ``tools/list`` style collection. Returns ``{tool: {path: hits}}``.

    Only tools with at least one finding appear in the result.
    """
    compiled = tuple(re.compile(p, re.IGNORECASE | re.MULTILINE) for p in extra_patterns)
    out: dict[str, dict[str, list[str]]] = {}
    for tool in tools:
        name = str(tool.get("name", "<unnamed>"))
        findings = screen_schema(tool, extra_patterns=compiled)
        if findings:
            out[name] = findings
    return out


# ---- manifests ---------------------------------------------------------


def _schema_of(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Pick the schema object out of a tool record.

    MCP uses ``inputSchema``; OpenAI-style records use ``parameters``;
    some registries nest under ``schema``. When none is present the
    whole record minus volatile fields is hashed so renames still show
    up as drift.
    """
    for key in ("inputSchema", "input_schema", "parameters", "schema"):
        value = tool.get(key)
        if isinstance(value, Mapping):
            return {"name": tool.get("name"), "description": tool.get("description", ""), key: value}
    return {k: v for k, v in tool.items() if k not in ("annotations", "_meta")}


def build_manifest(tools: Iterable[Mapping[str, Any]]) -> list[ToolManifestEntry]:
    """Compute a pinned manifest from a ``tools/list`` result."""
    entries: list[ToolManifestEntry] = []
    seen: set[str] = set()
    for tool in tools:
        name = str(tool.get("name") or "")
        if not name:
            raise ValueError("every tool needs a name to be pinned")
        if name in seen:
            raise ValueError(f"duplicate tool name in manifest input: {name!r}")
        seen.add(name)
        entries.append(ToolManifestEntry(name=name, schema_hash=schema_hash(_schema_of(tool))))
    entries.sort(key=lambda e: e.name)
    return entries


def manifest_to_json(entries: Iterable[ToolManifestEntry]) -> str:
    return json.dumps(
        {"version": 1, "tools": [asdict(e) for e in sorted(entries, key=lambda e: e.name)]},
        indent=2,
        sort_keys=True,
    )


def manifest_from_json(text: str) -> list[ToolManifestEntry]:
    data = json.loads(text)
    items = data["tools"] if isinstance(data, dict) else data
    return [ToolManifestEntry(name=str(i["name"]), schema_hash=str(i["schema_hash"])) for i in items]


def diff_manifest(
    pinned: Iterable[ToolManifestEntry], current: Iterable[ToolManifestEntry]
) -> ManifestDiff:
    """Compare a pinned manifest against the tools a server offers now."""
    old = {e.name: e.schema_hash for e in pinned}
    new = {e.name: e.schema_hash for e in current}
    diff = ManifestDiff()
    for name in sorted(set(old) | set(new)):
        if name not in old:
            diff.added.append(name)
        elif name not in new:
            diff.removed.append(name)
        elif old[name] != new[name]:
            diff.drifted.append(name)
        else:
            diff.unchanged.append(name)
    return diff


# ---- server wrapping -----------------------------------------------------


def ifc_mcp_server(
    server: Any,
    policy: Policy,
    *,
    manifest: Iterable[ToolManifestEntry] = (),
    extra_patterns: Iterable[str] = (),
    screen_parameters: bool = True,
    **kwargs: Any,
) -> Any:
    """Wrap an MCP server with IFC + tool-poisoning defenses.

    ``screen_parameters`` extends static screening from the top-level
    description to every prose field inside the input schema.
    """
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
            hits = screen_description(name, description, extra_patterns=custom_patterns)
            if hits:
                raise ToolPoisoningError(
                    f"tool {name!r} description contains suspicious patterns: {hits}"
                )
            if screen_parameters:
                nested = screen_schema(schema, extra_patterns=custom_patterns)
                nested.pop("description", None)
                if nested:
                    raise ToolPoisoningError(
                        f"tool {name!r} schema contains suspicious patterns: {nested}"
                    )
            if name in manifest_by_name:
                computed = schema_hash(schema)
                if computed != manifest_by_name[name].schema_hash:
                    raise ToolDriftError(
                        f"tool {name!r} schema hash {computed} != "
                        f"manifest {manifest_by_name[name].schema_hash}"
                    )
            return original_add_tool(name, schema, _wrap_handler(name, handler, policy, context))

        server.add_tool = patched_add_tool

    # Intercept the JSON-RPC dispatch. Most MCP servers expose
    # ``handle_call_tool(name, arguments) -> ...``.
    handle_call_tool = getattr(server, "handle_call_tool", None)
    if callable(handle_call_tool):
        original_handle = handle_call_tool

        if inspect.iscoroutinefunction(original_handle):

            @functools.wraps(original_handle)
            async def patched_handle_async(name: str, arguments: dict[str, Any]) -> Any:
                _check(name, (), arguments, policy, context, None)
                return await original_handle(name, arguments)

            server.handle_call_tool = patched_handle_async
        else:

            @functools.wraps(original_handle)
            def patched_handle(name: str, arguments: dict[str, Any]) -> Any:
                _check(name, (), arguments, policy, context, None)
                return original_handle(name, arguments)

            server.handle_call_tool = patched_handle

    server.__aisafepy_policy__ = policy
    server.__aisafepy_context__ = context
    return server


def _wrap_handler(
    name: str,
    handler: Callable[..., Any],
    policy: Policy,
    context: IFCContext,
) -> Callable[..., Any]:
    metadata: ToolMetadata | None = getattr(handler, "__aisafepy_tool__", None)

    if inspect.iscoroutinefunction(handler):

        @functools.wraps(handler)
        async def aproxy(*a: Any, **kw: Any) -> Any:
            _check(name, a, kw, policy, context, metadata)
            return await handler(*a, **kw)

        return aproxy

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
