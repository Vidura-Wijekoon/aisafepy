"""``Policy`` — the developer-facing API for declaring IFC constraints.

A ``Policy`` is constructed fluently:

    policy = (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source("gmail.read", integrity="UNTRUSTED", caps={Capability.READ_USER})
        .require("send_email", control_flow_integrity="TRUSTED",
                 caps={Capability.WRITE_EXTERNAL})
        .deny_if("send_email",
                 when=lambda to, body: "read.secrets" in body.provenance,
                 reason="secret-to-external-sink")
        .declassify("summarize", by="P-LLM only, structured output")
    )

The interpreter (``aisafepy.flow.interpreter``) consults the policy at
every tool-call evaluation point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from aisafepy.flow.taint import Integrity, Tainted


class Capability(str, Enum):
    """Coarse capability tags. Extend by subclassing in user code if you
    need finer-grained controls (the policy code treats capabilities as
    opaque strings)."""

    READ_PUBLIC = "read.public"
    READ_USER = "read.user"
    READ_SECRETS = "read.secrets"
    WRITE_EXTERNAL = "write.external"
    EXECUTE_CODE = "exec.code"
    NETWORK = "network"
    FILESYSTEM = "filesystem"


@dataclass(frozen=True)
class SourceLabel:
    name: str
    integrity: Integrity
    capabilities: frozenset[str]


@dataclass(frozen=True)
class ToolRequirement:
    name: str
    control_flow_integrity: Integrity = "UNTRUSTED"
    """Minimum integrity that the *control flow leading to this call* must have.

    "TRUSTED" means the call may only be issued when the planner's
    decision to issue it was derived from TRUSTED inputs. UNTRUSTED is
    permissive."""

    capabilities: frozenset[str] = frozenset()
    """Capabilities the tool's arguments are permitted to carry. Argument
    capability sets must be a subset of this set, unless the tool itself
    requires those capabilities to operate."""

    arg_max_integrity: Integrity = "UNTRUSTED"
    """The lowest integrity any argument may have. "TRUSTED" forbids
    UNTRUSTED arguments entirely."""


@dataclass(frozen=True)
class DenyRule:
    tool: str
    when: Callable[..., bool]
    reason: str


@dataclass(frozen=True)
class DeclassifyRule:
    tool: str
    by: str  # human-readable description of how the declassification is justified


@dataclass
class Policy:
    """Fluent IFC policy builder.

    All ``with_*`` / ``label_*`` / ``require`` methods return ``self`` so
    you can chain them. The resulting policy is a value object — clone
    via :func:`copy.deepcopy` if you need divergent variants.
    """

    sources: dict[str, SourceLabel] = field(default_factory=dict)
    requirements: dict[str, ToolRequirement] = field(default_factory=dict)
    deny_rules: list[DenyRule] = field(default_factory=list)
    declassifications: dict[str, DeclassifyRule] = field(default_factory=dict)
    mode: str = "strict"  # 'strict' | 'mediated'
    """Enforcement mode. 'strict' raises ``IFCViolation`` on deny;
    'mediated' surfaces a human-approval prompt (RTBAS-style) via the
    adapter."""

    # ---- builder API ---------------------------------------------------

    def label_source(
        self,
        name: str,
        *,
        integrity: Integrity = "UNTRUSTED",
        caps: Iterable[Capability | str] = (),
    ) -> "Policy":
        self.sources[name] = SourceLabel(
            name=name,
            integrity=integrity,
            capabilities=frozenset(_as_str(c) for c in caps),
        )
        return self

    def require(
        self,
        tool: str,
        *,
        control_flow_integrity: Integrity = "UNTRUSTED",
        caps: Iterable[Capability | str] = (),
        arg_max_integrity: Integrity = "UNTRUSTED",
    ) -> "Policy":
        self.requirements[tool] = ToolRequirement(
            name=tool,
            control_flow_integrity=control_flow_integrity,
            capabilities=frozenset(_as_str(c) for c in caps),
            arg_max_integrity=arg_max_integrity,
        )
        return self

    def deny_if(
        self,
        tool: str,
        *,
        when: Callable[..., bool],
        reason: str,
    ) -> "Policy":
        self.deny_rules.append(DenyRule(tool=tool, when=when, reason=reason))
        return self

    def declassify(self, tool: str, *, by: str) -> "Policy":
        self.declassifications[tool] = DeclassifyRule(tool=tool, by=by)
        return self

    def with_mode(self, mode: str) -> "Policy":
        if mode not in ("strict", "mediated"):
            raise ValueError(f"mode must be 'strict' or 'mediated' (got {mode!r})")
        self.mode = mode
        return self

    # ---- query API -----------------------------------------------------

    def source(self, name: str) -> SourceLabel | None:
        return self.sources.get(name)

    def requirement(self, tool: str) -> ToolRequirement | None:
        return self.requirements.get(tool)

    def deny_rules_for(self, tool: str) -> list[DenyRule]:
        return [r for r in self.deny_rules if r.tool == tool]

    def is_declassifier(self, tool: str) -> bool:
        return tool in self.declassifications

    def to_dict(self) -> dict[str, Any]:
        """Render the policy as a dict for logging / serialization.

        ``DenyRule.when`` callables are not serializable; we record their
        ``reason`` only.
        """
        return {
            "mode": self.mode,
            "sources": {
                k: {
                    "integrity": v.integrity,
                    "capabilities": sorted(v.capabilities),
                }
                for k, v in self.sources.items()
            },
            "requirements": {
                k: {
                    "control_flow_integrity": v.control_flow_integrity,
                    "capabilities": sorted(v.capabilities),
                    "arg_max_integrity": v.arg_max_integrity,
                }
                for k, v in self.requirements.items()
            },
            "deny_rules": [
                {"tool": r.tool, "reason": r.reason} for r in self.deny_rules
            ],
            "declassifications": {
                k: {"by": v.by} for k, v in self.declassifications.items()
            },
        }


# ---- @secure_tool decorator -------------------------------------------


def secure_tool(
    *,
    capabilities: Iterable[Capability | str] = (),
    required_integrity: Integrity = "UNTRUSTED",
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a Python function as an IFC-aware tool.

    The decorator records the tool's declared capability set and minimum
    integrity on the wrapped function's ``__aisafepy_tool__`` attribute.
    The interpreter consults this metadata when the tool is invoked.
    Tools that accept ``Tainted[T]`` arguments are recommended for
    callers that need fine-grained label propagation; tools that accept
    plain values will still be inspected at the call site.
    """
    declared_caps = frozenset(_as_str(c) for c in capabilities)

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        meta = ToolMetadata(
            name=name or fn.__name__,
            capabilities=declared_caps,
            required_integrity=required_integrity,
        )
        # Store metadata as an attribute. Wrapped function is still
        # directly callable; the interpreter inspects metadata at the
        # call site.
        setattr(fn, "__aisafepy_tool__", meta)
        return fn

    return deco


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    capabilities: frozenset[str]
    required_integrity: Integrity


def _as_str(value: Capability | str) -> str:
    if isinstance(value, Capability):
        return value.value
    return value
