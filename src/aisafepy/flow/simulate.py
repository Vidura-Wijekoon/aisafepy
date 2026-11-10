"""What-if analysis for flow policies.

A policy is a set of labels and requirements; what it *means* is the set
of data paths it lets through. :func:`simulate_policy` enumerates those
paths. For every labelled source, every tool the policy knows about, and
every control-flow integrity level, it builds a single ``Tainted`` probe
argument carrying the source's labels and runs it through
``evaluate_call`` in strict mode, recording whether the call was allowed
and, if not, why.

The result is a matrix a reviewer can read in one glance (sources as
rows, tools as columns) plus three derived lists:

* ``risky_paths()``: allowed paths where low-integrity or secret-bearing
  data reaches a tool that looks like a sink (by name or declared
  capabilities). These are the findings to look at first.
* ``uncovered_tools()``: tools with no requirement and no deny rule, so
  every path into them is allowed by default.
* ``denied_paths()``: the paths the policy actively stops, useful for
  confirming a rule does what its author thinks.

:func:`explain_path` narrates one (source, tool, cfi) combination step
by step in the order the interpreter checks it.

Limitations: the probe is one positional argument. ``deny_if`` rules
written for a specific keyword signature (``lambda to, body: ...``) will
raise on the probe, which the interpreter logs and treats as no match;
rules built from declarative specs, or written with ``*args, **kwargs``,
are evaluated faithfully. The report records that caveat per rule under
``notes``.
"""

from __future__ import annotations

import copy
import inspect
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from aisafepy.flow.interpreter import IFCContext, IFCViolationError, evaluate_call
from aisafepy.flow.policy import Capability, Policy
from aisafepy.flow.taint import Integrity, Tainted

_SINK_NAME_RE = re.compile(
    r"(send|transfer|wire|pay|write|post|publish|upload|exec|run|shell|delete|remove|"
    r"drop|email|mail|sms|notify|deploy|purchase|order|commit|push)",
    re.IGNORECASE,
)
_SINK_CAPS = frozenset({Capability.WRITE_EXTERNAL.value, Capability.EXECUTE_CODE.value})
_SENSITIVE_CAPS = frozenset({Capability.READ_SECRETS.value, Capability.READ_USER.value})

LITERAL_SOURCE = "<literal>"
"""Pseudo-source for an untainted argument (TRUSTED, no labels)."""


@dataclass(frozen=True)
class PathResult:
    """Outcome of one simulated call."""

    source: str
    tool: str
    cfi: Integrity
    allowed: bool
    reason: str
    integrity: Integrity
    capabilities: frozenset[str]

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.tool, self.cfi)


@dataclass
class SimulationReport:
    policy_mode: str
    sources: list[str]
    tools: list[str]
    cfis: tuple[Integrity, ...]
    results: list[PathResult]
    notes: list[str] = field(default_factory=list)
    _index: dict[tuple[str, str, str], PathResult] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {r.key: r for r in self.results}

    def get(self, source: str, tool: str, cfi: Integrity = "TRUSTED") -> PathResult | None:
        return self._index.get((source, tool, cfi))

    def allowed_paths(self) -> list[PathResult]:
        return [r for r in self.results if r.allowed]

    def denied_paths(self) -> list[PathResult]:
        return [r for r in self.results if not r.allowed]

    def risky_paths(self, policy: Policy | None = None) -> list[PathResult]:
        """Allowed paths where untrusted or sensitive data reaches a sink-like tool."""
        out: list[PathResult] = []
        for r in self.allowed_paths():
            if r.source == LITERAL_SOURCE:
                continue
            sensitive = r.integrity != "TRUSTED" or bool(r.capabilities & _SENSITIVE_CAPS)
            if sensitive and _is_sink(r.tool, policy):
                out.append(r)
        return out

    def uncovered_tools(self, policy: Policy | None = None) -> list[str]:
        """Tools with neither a requirement nor a deny rule."""
        if policy is None:
            # Without the policy, infer: a tool is uncovered if every path is allowed.
            return [
                t for t in self.tools if all(r.allowed for r in self.results if r.tool == t)
            ]
        covered = set(policy.requirements) | {r.tool for r in policy.deny_rules}
        return [t for t in self.tools if t not in covered]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_mode": self.policy_mode,
            "sources": self.sources,
            "tools": self.tools,
            "cfis": list(self.cfis),
            "paths": [
                {
                    "source": r.source,
                    "tool": r.tool,
                    "cfi": r.cfi,
                    "allowed": r.allowed,
                    "reason": r.reason,
                    "integrity": r.integrity,
                    "capabilities": sorted(r.capabilities),
                }
                for r in self.results
            ],
            "risky": [list(r.key) for r in self.risky_paths()],
            "uncovered_tools": self.uncovered_tools(),
            "notes": self.notes,
        }

    def to_markdown(self, cfi: Integrity = "TRUSTED") -> str:
        """Sources as rows, tools as columns, one matrix per requested CFI."""
        lines = [f"## Policy simulation (control flow {cfi})", ""]
        header = "| source | " + " | ".join(f"`{t}`" for t in self.tools) + " |"
        lines.append(header)
        lines.append("|---|" + "---|" * len(self.tools))
        for s in self.sources:
            cells = []
            for t in self.tools:
                r = self.get(s, t, cfi)
                cells.append("allow" if r is not None and r.allowed else "deny" if r is not None else "n/a")
            lines.append(f"| `{s}` | " + " | ".join(cells) + " |")
        risky = [r for r in self.risky_paths() if r.cfi == cfi]
        if risky:
            lines += ["", "### Risky allowed paths", ""]
            for r in risky:
                caps = ", ".join(sorted(r.capabilities)) or "none"
                lines.append(f"- `{r.source}` ({r.integrity}, caps: {caps}) -> `{r.tool}`")
        uncovered = self.uncovered_tools()
        if uncovered:
            lines += ["", "### Uncovered tools", ""]
            lines.extend(f"- `{t}`" for t in uncovered)
        if self.notes:
            lines += ["", "### Notes", ""]
            lines.extend(f"- {n}" for n in self.notes)
        return "\n".join(lines)


def _is_sink(tool: str, policy: Policy | None) -> bool:
    if _SINK_NAME_RE.search(tool):
        return True
    if policy is not None:
        req = policy.requirement(tool)
        if req is not None and req.capabilities & _SINK_CAPS:
            return True
    return False


def _probe(policy: Policy, source: str) -> Tainted[str]:
    if source == LITERAL_SOURCE:
        return Tainted(value="probe")
    label = policy.source(source)
    if label is None:
        return Tainted(value="probe", provenance=frozenset({source}), integrity="UNTRUSTED")
    return Tainted(
        value="probe",
        provenance=frozenset({source}),
        capabilities=label.capabilities,
        integrity=label.integrity,
    )


def _strict(policy: Policy) -> Policy:
    if policy.mode == "strict":
        return policy
    clone = copy.copy(policy)
    clone.mode = "strict"
    return clone


def _rule_notes(policy: Policy) -> list[str]:
    notes: list[str] = []
    for rule in policy.deny_rules:
        if getattr(rule, "spec", None) is not None:
            continue
        try:
            params = inspect.signature(rule.when).parameters.values()
        except (TypeError, ValueError):
            continue
        accepts_var = any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)
        positional_ok = sum(
            1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ) >= 1 and all(
            p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            for i, p in enumerate(params) if i > 0
        )
        if not (accepts_var or positional_ok):
            notes.append(
                f"deny rule {rule.reason!r} on {rule.tool!r} has a fixed signature; "
                "the single-argument probe cannot exercise it (treated as no match)"
            )
    return notes


def simulate_policy(
    policy: Policy,
    *,
    tools: Iterable[str] | None = None,
    sources: Iterable[str] | None = None,
    cfis: Iterable[Integrity] = ("TRUSTED", "UNTRUSTED"),
    include_literal: bool = True,
) -> SimulationReport:
    """Enumerate every (source, tool, cfi) path through ``policy``."""
    strict = _strict(policy)
    tool_list = sorted(
        set(tools) if tools is not None else set(policy.requirements) | {r.tool for r in policy.deny_rules}
    )
    source_list = sorted(set(sources) if sources is not None else set(policy.sources))
    if include_literal:
        source_list = [LITERAL_SOURCE, *source_list]
    cfi_list = tuple(cfis)

    results: list[PathResult] = []
    for cfi in cfi_list:
        for source in source_list:
            probe = _probe(strict, source)
            for tool in tool_list:
                ctx = IFCContext(control_flow_integrity=cfi)
                try:
                    evaluate_call(tool=tool, args=(probe,), kwargs={}, policy=strict, context=ctx)
                except IFCViolationError as exc:
                    results.append(
                        PathResult(
                            source=source,
                            tool=tool,
                            cfi=cfi,
                            allowed=False,
                            reason=exc.violation.reason,
                            integrity=probe.integrity,
                            capabilities=probe.capabilities,
                        )
                    )
                else:
                    results.append(
                        PathResult(
                            source=source,
                            tool=tool,
                            cfi=cfi,
                            allowed=True,
                            reason="allowed",
                            integrity=probe.integrity,
                            capabilities=probe.capabilities,
                        )
                    )
    report = SimulationReport(
        policy_mode=policy.mode,
        sources=source_list,
        tools=tool_list,
        cfis=cfi_list,
        results=results,
        notes=_rule_notes(policy),
    )
    return report


def explain_path(
    policy: Policy, source: str, tool: str, cfi: Integrity = "TRUSTED"
) -> list[str]:
    """Narrate the interpreter's checks for one path, in order."""
    probe = _probe(policy, source)
    steps: list[str] = []
    label = policy.source(source)
    if source == LITERAL_SOURCE:
        steps.append("argument is a plain literal: TRUSTED, no provenance, no capabilities")
    elif label is None:
        steps.append(f"source {source!r} is not labelled; probe is UNTRUSTED with no capabilities")
    else:
        caps = ", ".join(sorted(label.capabilities)) or "none"
        steps.append(f"source {source!r} is labelled {label.integrity} with capabilities: {caps}")
    steps.append(f"control-flow integrity of the call: {cfi}")

    req = policy.requirement(tool)
    if req is None:
        steps.append(f"tool {tool!r} has no requirement: integrity and capability checks are skipped")
    else:
        order = ("TRUSTED", "UNTRUSTED", "QUARANTINED")
        cfi_ok = order.index(cfi) <= order.index(req.control_flow_integrity)
        steps.append(
            f"requirement: control flow must be at least {req.control_flow_integrity}: "
            + ("ok" if cfi_ok else "DENY")
        )
        if cfi_ok:
            arg_ok = order.index(probe.integrity) <= order.index(req.arg_max_integrity)
            steps.append(
                f"requirement: arguments must be at least {req.arg_max_integrity}: "
                + ("ok" if arg_ok else f"DENY (argument is {probe.integrity})")
            )
            if arg_ok:
                forbidden = probe.capabilities - req.capabilities
                steps.append(
                    "requirement: argument capabilities within "
                    f"{sorted(req.capabilities) or 'none'}: "
                    + ("ok" if not forbidden else f"DENY (forbidden: {sorted(forbidden)})")
                )
    rules = policy.deny_rules_for(tool)
    if rules:
        for rule in rules:
            try:
                hit = bool(rule.when(probe))
                verdict = "DENY" if hit else "no match"
            except Exception as exc:  # fixed-signature lambda
                verdict = f"not evaluable with one probe argument ({type(exc).__name__})"
            steps.append(f"deny_if {rule.reason!r}: {verdict}")
    else:
        steps.append(f"no deny_if rules for {tool!r}")
    result = simulate_policy(policy, tools=[tool], sources=[source] if source != LITERAL_SOURCE else [], cfis=(cfi,)).get(source, tool, cfi)
    steps.append("result: " + ("ALLOW" if result and result.allowed else f"DENY ({result.reason if result else 'unknown'})"))
    return steps
