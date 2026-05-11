"""Policy document loader and a minimal policy-rule DSL.

This module owns the *runtime* representation of policy. The
emission-time representation (Cedar / OPA Rego) lives in
``aisafepy.adapt.compile.policy``. Per design principle 7, AIsafePy
exposes Python objects to developers and only emits DSLs as machine
artifacts produced by the compiler.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import yaml  # type: ignore[import-not-found]

    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False


@dataclass
class PolicyRule:
    """One rule inside a ``PolicyDocument``.

    A rule is matched by ``selector`` (a callable on a free-form context)
    and emits a ``decision`` when it matches. Rules carry a stable
    ``id`` and a textual ``rationale`` so the eval-to-guardrail compiler
    can track which rules came from which red-team finding.
    """

    id: str
    selector: Callable[[dict[str, Any]], bool]
    decision: Literal["allow", "block", "transform", "escalate"]
    rationale: str
    severity: Literal["info", "low", "medium", "high", "critical"] = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, context: dict[str, Any]) -> bool:
        try:
            return bool(self.selector(context))
        except Exception:
            # A misbehaving selector should never crash the whole evaluator.
            # The defensive default is "no match"; the compiler should
            # surface this in test results.
            return False


@dataclass
class PolicyDocument:
    """An ordered set of ``PolicyRule``s plus document-level metadata.

    The document is intentionally simple. Rule precedence is positional
    (first match wins), and the rules themselves carry all the
    interesting logic. This keeps the runtime trivial; expressiveness
    comes from the Python ``selector`` callables.
    """

    name: str
    version: str
    rules: list[PolicyRule] = field(default_factory=list)
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, rule: PolicyRule) -> PolicyDocument:
        self.rules.append(rule)
        return self

    def evaluate(self, context: dict[str, Any]) -> PolicyRule | None:
        """Return the first matching rule, or None."""
        for r in self.rules:
            if r.matches(context):
                return r
        return None

    # ---- serialization -------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicyDocument:
        """Load a policy document from YAML.

        The YAML schema is::

            name: company-safety
            version: 2.0
            description: ...
            rules:
              - id: pii-pin
                pattern: "\\b\\d{4}\\b"      # selector compiled from regex
                decision: block
                rationale: "PIN-like 4-digit numeric"
                severity: high
              - id: jailbreak-phrase
                contains: ["ignore previous instructions"]
                decision: block
                rationale: "common jailbreak preamble"
        """
        if not _HAS_YAML:
            raise RuntimeError(
                "PolicyDocument.from_yaml requires PyYAML. "
                "Install via `pip install pyyaml`."
            )
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        doc = cls(
            name=data["name"],
            version=str(data.get("version", "0.0.1")),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )
        for raw in data.get("rules", []):
            doc.add(_rule_from_yaml(raw))
        return doc


def _rule_from_yaml(raw: dict[str, Any]) -> PolicyRule:
    """Convert one YAML rule into a ``PolicyRule``.

    The supported selector forms are:

    - ``pattern``: a regex matched against ``context["content"]``.
    - ``contains``: a list of substrings ANY of which must appear.
    - ``equals``: an exact match on a chosen ``field`` (default ``content``).
    - ``not_empty``: ``True`` if a named ``field`` is present and non-empty.
    """
    import re

    rid = raw["id"]
    decision = raw["decision"]
    rationale = raw.get("rationale", rid)
    severity = raw.get("severity", "medium")
    field_name = raw.get("field", "content")
    metadata = raw.get("metadata", {})

    selector: Callable[[dict[str, Any]], bool]

    if "pattern" in raw:
        regex = re.compile(raw["pattern"], re.IGNORECASE | re.MULTILINE)

        def selector(ctx: dict[str, Any]) -> bool:
            v = ctx.get(field_name, "")
            return isinstance(v, str) and bool(regex.search(v))

    elif "contains" in raw:
        needles = [n.lower() for n in raw["contains"]]

        def selector(ctx: dict[str, Any]) -> bool:
            v = ctx.get(field_name, "")
            if not isinstance(v, str):
                return False
            v_low = v.lower()
            return any(n in v_low for n in needles)

    elif "equals" in raw:
        target = raw["equals"]

        def selector(ctx: dict[str, Any]) -> bool:
            return ctx.get(field_name) == target

    elif raw.get("not_empty") is True:

        def selector(ctx: dict[str, Any]) -> bool:
            v = ctx.get(field_name)
            return v not in (None, "", [], {}, ())

    else:
        # Fallback: never matches. Useful as a placeholder while authoring.
        def selector(_ctx: dict[str, Any]) -> bool:
            return False

    return PolicyRule(
        id=rid,
        selector=selector,
        decision=decision,
        rationale=rationale,
        severity=severity,
        metadata=metadata,
    )
