"""Static checks for :class:`aisafepy.flow.Policy` objects.

An IFC policy that *compiles* is not necessarily a policy that
*protects*. The classic failure modes are quiet ones: a sink that was
never given a ``require``, a deny rule whose lambda raises on real
arguments (the interpreter treats that as "no match"), a source that
is labelled but never consulted, a declassifier justified by an empty
string. ``lint_policy`` surfaces these before deployment.

Findings carry a stable ``code`` so teams can suppress specific checks
in CI, a ``severity`` (``info`` / ``warning`` / ``error``), and the
``subject`` (tool or source name) they are about.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from aisafepy.flow.policy import DenyRule, Policy
from aisafepy.flow.taint import Tainted

Severity = Literal["info", "warning", "error"]

_SEVERITY_RANK: dict[str, int] = {"error": 0, "warning": 1, "info": 2}

_SINK_HINT = re.compile(
    r"(^|[._-])(send|transfer|write|exec|execute|delete|remove|post|pay|publish|upload|deploy|run)"
    r"([._-]|$)",
    re.IGNORECASE,
)

MIN_JUSTIFICATION_CHARS = 12


@dataclass(frozen=True)
class LintFinding:
    """One linter result."""

    code: str
    severity: Severity
    message: str
    subject: str | None = None

    def __str__(self) -> str:
        where = f" [{self.subject}]" if self.subject else ""
        return f"{self.severity.upper():7} {self.code}{where}: {self.message}"


def lint_policy(
    policy: Policy,
    *,
    tools: Iterable[str] | None = None,
) -> list[LintFinding]:
    """Run every check against ``policy`` and return the findings.

    ``tools`` is the optional set of tool names the agent actually
    exposes. When given, tools without any policy coverage are reported
    (``uncovered-tool``) and policy entries that reference a tool the
    agent does not have are reported too (``rule-for-unknown-tool``).

    Findings are sorted by severity (errors first), then code, then subject.
    """
    findings: list[LintFinding] = []
    known_tools = set(tools) if tools is not None else None

    findings += _check_sources(policy)
    findings += _check_requirements(policy)
    findings += _check_deny_rules(policy)
    findings += _check_declassifiers(policy)
    findings += _check_mode(policy)
    if known_tools is not None:
        findings += _check_tool_coverage(policy, known_tools)

    findings.sort(key=lambda f: (_SEVERITY_RANK[f.severity], f.code, f.subject or ""))
    return findings


def has_errors(findings: Iterable[LintFinding]) -> bool:
    """``True`` when any finding has ``error`` severity."""
    return any(f.severity == "error" for f in findings)


def format_findings(findings: Iterable[LintFinding]) -> str:
    """Render findings as one line each, with a trailing summary line."""
    items = list(findings)
    if not items:
        return "no findings"
    counts = {"error": 0, "warning": 0, "info": 0}
    lines = []
    for f in items:
        counts[f.severity] += 1
        lines.append(str(f))
    lines.append(
        f"{len(items)} finding(s): {counts['error']} error, "
        f"{counts['warning']} warning, {counts['info']} info"
    )
    return "\n".join(lines)


# ---- individual checks -----------------------------------------------------


def _check_sources(policy: Policy) -> list[LintFinding]:
    out: list[LintFinding] = []
    referenced = _referenced_sources(policy)
    granted_caps: set[str] = set()
    for name, label in policy.sources.items():
        granted_caps |= set(label.capabilities)
        if name not in referenced:
            out.append(
                LintFinding(
                    code="unused-source",
                    severity="info",
                    subject=name,
                    message=(
                        "source is labelled but no deny rule or requirement refers to it; "
                        "labels still propagate, but nothing in this policy reacts to them"
                    ),
                )
            )
    for tool, req in policy.requirements.items():
        missing = set(req.capabilities) - granted_caps
        if missing and policy.sources:
            out.append(
                LintFinding(
                    code="capability-never-granted",
                    severity="info",
                    subject=tool,
                    message=(
                        f"requirement permits capabilities {sorted(missing)} that no "
                        "labelled source grants; the ceiling is unreachable"
                    ),
                )
            )
    return out


def _referenced_sources(policy: Policy) -> set[str]:
    """Sources something in the policy can react to.

    A source counts as referenced when a deny spec names it
    (``provenance_contains``), when a deny spec or a requirement ceiling
    mentions one of the capabilities it grants, when it is ``TRUSTED``
    and some requirement demands trusted arguments or control flow (the
    source is what satisfies that demand), or, as a heuristic for
    callable rules, when its name appears verbatim in a rule's reason.
    """
    refs: set[str] = set()
    caps_in_specs: set[str] = set()
    for rule in policy.deny_rules:
        refs |= _spec_names(rule.spec, "provenance_contains")
        caps_in_specs |= _spec_names(rule.spec, "capability_present")
        for src in policy.sources:
            if src in rule.reason:
                refs.add(src)
    caps_in_requirements: set[str] = set()
    demands_trust = False
    for req in policy.requirements.values():
        caps_in_requirements |= set(req.capabilities)
        if req.arg_max_integrity == "TRUSTED" or req.control_flow_integrity == "TRUSTED":
            demands_trust = True
    for name, label in policy.sources.items():
        if label.capabilities & (caps_in_specs | caps_in_requirements):
            refs.add(name)
        if label.integrity == "TRUSTED" and demands_trust:
            refs.add(name)
    return refs


def _spec_names(spec: dict[str, Any] | None, op: str) -> set[str]:
    if not isinstance(spec, dict):
        return set()
    out: set[str] = set()
    for key, value in spec.items():
        if key == op:
            out |= {value} if isinstance(value, str) else set(value)
        elif key in ("any_of", "all_of") and isinstance(value, list):
            for sub in value:
                out |= _spec_names(sub, op)
        elif key == "not":
            out |= _spec_names(value, op)
    return out


def _check_requirements(policy: Policy) -> list[LintFinding]:
    out: list[LintFinding] = []
    for tool, req in policy.requirements.items():
        permissive = (
            req.control_flow_integrity == "UNTRUSTED"
            and req.arg_max_integrity == "UNTRUSTED"
            and not req.capabilities
        )
        if permissive and _SINK_HINT.search(tool) and not policy.deny_rules_for(tool):
            out.append(
                LintFinding(
                    code="permissive-sink",
                    severity="warning",
                    subject=tool,
                    message=(
                        "tool name looks like an external sink but its requirement "
                        "accepts UNTRUSTED control flow and arguments with no capability "
                        "ceiling and no deny rules; this requirement enforces nothing"
                    ),
                )
            )
    return out


def _check_deny_rules(policy: Policy) -> list[LintFinding]:
    out: list[LintFinding] = []
    seen: dict[tuple[str, str], int] = {}
    for rule in policy.deny_rules:
        key = (rule.tool, rule.reason)
        seen[key] = seen.get(key, 0) + 1
        if rule.tool not in policy.requirements:
            out.append(
                LintFinding(
                    code="deny-without-require",
                    severity="warning",
                    subject=rule.tool,
                    message=(
                        f"deny rule {rule.reason!r} exists but the tool has no `require`; "
                        "control-flow and argument integrity are not checked for it"
                    ),
                )
            )
        error = _probe_rule(rule)
        if error is not None:
            out.append(
                LintFinding(
                    code="deny-rule-raises",
                    severity="error",
                    subject=rule.tool,
                    message=(
                        f"deny rule {rule.reason!r} raised {error} on a probe call; the "
                        "interpreter treats a raising rule as 'no match', so this rule "
                        "would silently allow"
                    ),
                )
            )
    for (tool, reason), count in seen.items():
        if count > 1:
            out.append(
                LintFinding(
                    code="duplicate-deny-reason",
                    severity="info",
                    subject=tool,
                    message=f"{count} deny rules share the reason {reason!r}",
                )
            )
    return out


def _probe_rule(rule: DenyRule) -> str | None:
    """Call ``rule.when`` with signature-shaped Tainted probe values.

    Returns a short description of the exception if the call raised,
    else ``None``. The probe mirrors what the interpreter does: every
    required parameter receives a ``Tainted`` value so rules that read
    ``.provenance`` / ``.value`` work, and ``**kwargs`` style rules get
    a single named probe so ``any(... for v in kw.values())`` has data.
    """
    probe = Tainted("probe", provenance=frozenset({"lint.probe"}), integrity="UNTRUSTED")
    args, kwargs = _probe_arguments(rule.when, probe)
    try:
        rule.when(*args, **kwargs)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _probe_arguments(fn: Callable[..., Any], probe: Any) -> tuple[list[Any], dict[str, Any]]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return [], {}
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            args.append(probe)
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            kwargs["probe"] = probe
        elif param.default is inspect.Parameter.empty:
            if param.kind is inspect.Parameter.KEYWORD_ONLY:
                kwargs[name] = probe
            else:
                args.append(probe)
    return args, kwargs


def _check_declassifiers(policy: Policy) -> list[LintFinding]:
    out: list[LintFinding] = []
    for tool, rule in policy.declassifications.items():
        if len(rule.by.strip()) < MIN_JUSTIFICATION_CHARS:
            out.append(
                LintFinding(
                    code="weak-declassification",
                    severity="warning",
                    subject=tool,
                    message=(
                        f"declassification justification {rule.by!r} is empty or shorter "
                        f"than {MIN_JUSTIFICATION_CHARS} characters; auditors need to know "
                        "why this tool is allowed to launder integrity"
                    ),
                )
            )
    return out


def _check_mode(policy: Policy) -> list[LintFinding]:
    if policy.mode == "mediated" and not policy.requirements:
        return [
            LintFinding(
                code="mediated-without-requirements",
                severity="warning",
                subject=None,
                message=(
                    "policy is in mediated mode but declares no requirements; "
                    "no call will ever reach the mediator"
                ),
            )
        ]
    return []


def _check_tool_coverage(policy: Policy, tools: set[str]) -> list[LintFinding]:
    out: list[LintFinding] = []
    # A labelled source that is also a tool is covered: its output is
    # labelled, which is the policy's whole statement about a reader.
    covered = policy.covered_tools() | set(policy.sources)
    for tool in sorted(tools - covered):
        out.append(
            LintFinding(
                code="uncovered-tool",
                severity="warning",
                subject=tool,
                message="tool is exposed to the agent but no requirement, deny rule, or "
                        "declassifier mentions it; calls are always allowed",
            )
        )
    for tool in sorted(policy.covered_tools() - tools):
        out.append(
            LintFinding(
                code="rule-for-unknown-tool",
                severity="warning",
                subject=tool,
                message="policy references a tool the agent does not expose "
                        "(typo, or stale policy?)",
            )
        )
    return out
