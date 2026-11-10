"""Policy document loader and a minimal policy-rule DSL.

This module owns the *runtime* representation of policy. The
emission-time representation (Cedar / OPA Rego) lives in
``aisafepy.adapt.compile.policy``. Per design principle 7, AIsafePy
exposes Python objects to developers and only emits DSLs as machine
artifacts produced by the compiler.

YAML rule vocabulary
--------------------
Every rule has ``id``, ``decision`` (allow / block / transform /
escalate), and optionally ``rationale``, ``severity``, ``field``
(default ``content``), ``when_role`` and ``metadata``. Exactly one
selector form::

    pattern: "regex"                  # re.search on the field
    regex_any: ["re1", "re2"]         # any regex matches
    contains: ["needle", ...]         # any substring (case-insensitive)
    equals: value                     # exact equality
    field_in: [a, b, c]               # field value is one of
    not_empty: true                   # field present and non-empty
    length_gt: 4000 / length_lt: 3    # len(field) comparison
    all_of: [<selector>, ...]         # nested selectors, all must match
    any_of: [<selector>, ...]         # nested selectors, any may match
    not: <selector>                   # negation

Nested selectors are mappings with the same keys (and may carry their
own ``field``). ``when_role`` restricts a rule to contexts whose
``role`` equals the given value; it composes with any selector.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import yaml  # type: ignore[import-not-found]

    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False

Decision = Literal["allow", "block", "transform", "escalate"]
SeverityLabel = Literal["info", "low", "medium", "high", "critical"]

_SELECTOR_KEYS = (
    "pattern",
    "regex_any",
    "contains",
    "equals",
    "field_in",
    "not_empty",
    "length_gt",
    "length_lt",
    "all_of",
    "any_of",
    "not",
)


@dataclass
class PolicyRule:
    """One rule inside a ``PolicyDocument``.

    A rule is matched by ``selector`` (a callable on a free-form context)
    and emits a ``decision`` when it matches. Rules carry a stable
    ``id`` and a textual ``rationale`` so the eval-to-guardrail compiler
    can track which rules came from which red-team finding.

    ``spec`` holds the declarative form the rule was built from (when
    it was built from YAML or a dict) so the document can be
    serialized back; rules built from a Python callable have no spec
    and serialize as opaque.
    """

    id: str
    selector: Callable[[dict[str, Any]], bool]
    decision: Decision
    rationale: str
    severity: SeverityLabel = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] | None = None
    when_role: str | None = None

    def matches(self, context: dict[str, Any]) -> bool:
        if self.when_role is not None and context.get("role") != self.when_role:
            return False
        try:
            return bool(self.selector(context))
        except Exception:
            # A misbehaving selector should never crash the whole evaluator.
            # The defensive default is "no match"; the compiler should
            # surface this in test results.
            return False

    def to_dict(self) -> dict[str, Any]:
        """Declarative form, or an opaque stub when built from a callable."""
        out: dict[str, Any] = {
            "id": self.id,
            "decision": self.decision,
            "rationale": self.rationale,
            "severity": self.severity,
        }
        if self.when_role is not None:
            out["when_role"] = self.when_role
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        if self.spec is not None:
            out.update(self.spec)
        else:
            out["opaque"] = True
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyRule:
        return _rule_from_yaml(raw)


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

    def evaluate_all(self, context: dict[str, Any]) -> list[PolicyRule]:
        """Return every matching rule, in document order."""
        return [r for r in self.rules if r.matches(context)]

    def explain(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Every rule with whether it matched. The first match is marked ``selected``."""
        out: list[dict[str, Any]] = []
        selected = False
        for r in self.rules:
            m = r.matches(context)
            out.append(
                {
                    "id": r.id,
                    "matched": m,
                    "selected": m and not selected,
                    "decision": r.decision,
                    "severity": r.severity,
                    "rationale": r.rationale,
                }
            )
            if m:
                selected = True
        return out

    def rule(self, rule_id: str) -> PolicyRule | None:
        for r in self.rules:
            if r.id == rule_id:
                return r
        return None

    def __len__(self) -> int:
        return len(self.rules)

    # ---- guard bridge --------------------------------------------------

    def to_guard_decision(self, rule: PolicyRule | None, *, tier: int = 1, latency_ms: float = 0.0):
        """Render a matched rule (or ``None``) as a ``GuardDecision``."""
        from aisafepy.core.decisions import GuardDecision

        name = f"policy:{self.name}"
        if rule is None:
            return GuardDecision.allow(
                tier=tier, guard_name=name, rationale="no policy rule matched", latency_ms=latency_ms
            )
        evidence = {"rule_id": rule.id, "policy": self.name, "version": self.version}
        if rule.decision == "allow":
            return GuardDecision.allow(
                tier=tier,
                guard_name=name,
                rationale=f"{rule.id}: {rule.rationale}",
                evidence=evidence,
                latency_ms=latency_ms,
            )
        if rule.decision == "block":
            return GuardDecision.block(
                tier=tier,
                confidence=1.0,
                rationale=f"{rule.id}: {rule.rationale}",
                severity=rule.severity,
                guard_name=name,
                evidence=evidence,
                latency_ms=latency_ms,
            )
        if rule.decision == "escalate":
            return GuardDecision.escalate(
                tier=tier,
                confidence=1.0,
                rationale=f"{rule.id}: {rule.rationale}",
                severity=rule.severity,
                guard_name=name,
                evidence=evidence,
                latency_ms=latency_ms,
            )
        # transform: the document does not know how to rewrite content; it
        # signals the intent and leaves the replacement to a downstream
        # redactor. ``transformed_content`` is the placeholder marker.
        return GuardDecision.transform(
            tier=tier,
            confidence=1.0,
            rationale=f"{rule.id}: {rule.rationale}",
            transformed_content=str(rule.metadata.get("replacement", "[REDACTED]")),
            severity=rule.severity,
            guard_name=name,
            evidence=evidence,
            latency_ms=latency_ms,
        )

    def as_guard(self, name: str | None = None, *, tier: int = 1) -> Any:
        """Return a Guard-protocol object evaluating this document.

        The guard builds its context as ``{"content": buffer or chunk,
        "prompt": prompt, "role": role, "metadata": ctx.metadata,
        **ctx.metadata}`` so YAML rules can target any of those fields,
        either flattened (``field: tenant``) or nested
        (``field: metadata.tenant``).
        """
        return _PolicyGuard(self, name=name or f"policy:{self.name}", tier=tier)

    # ---- serialization -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "version": self.version}
        if self.description:
            out["description"] = self.description
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        out["rules"] = [r.to_dict() for r in self.rules]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyDocument:
        doc = cls(
            name=str(data["name"]),
            version=str(data.get("version", "0.0.1")),
            description=data.get("description"),
            metadata=dict(data.get("metadata") or {}),
        )
        for raw in data.get("rules", []) or []:
            doc.add(_rule_from_yaml(raw))
        return doc

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
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialize to YAML; also write it to ``path`` when given.

        Rules built from Python callables cannot be serialized and are
        emitted as ``opaque: true`` stubs carrying their id, decision
        and rationale so the round trip is at least lossless on
        identity.
        """
        if not _HAS_YAML:
            raise RuntimeError("PolicyDocument.to_yaml requires PyYAML.")
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


class _PolicyGuard:
    """Guard-protocol adapter over a :class:`PolicyDocument`."""

    def __init__(self, document: PolicyDocument, *, name: str, tier: int):
        self.document = document
        self.name = name
        self.tier = tier

    async def __call__(self, ctx: Any):
        import time

        start = time.perf_counter()
        metadata = dict(getattr(ctx, "metadata", None) or {})
        context: dict[str, Any] = {
            "content": getattr(ctx, "buffer", None) or getattr(ctx, "chunk", None) or "",
            "prompt": getattr(ctx, "prompt", None),
            "role": getattr(ctx, "role", None),
            "metadata": metadata,
            **metadata,
        }
        rule = self.document.evaluate(context)
        return self.document.to_guard_decision(
            rule, tier=self.tier, latency_ms=(time.perf_counter() - start) * 1000.0
        )


# ---- selector compilation -------------------------------------------


def _field_value(ctx: dict[str, Any], field_name: str) -> Any:
    """Dotted-path lookup so ``metadata.user.tier`` style fields work."""
    if field_name in ctx:
        return ctx[field_name]
    cur: Any = ctx
    for seg in field_name.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur


def compile_selector(raw: dict[str, Any], default_field: str = "content") -> Callable[[dict[str, Any]], bool]:
    """Build a selector callable from a declarative mapping.

    Raises ``ValueError`` when more than one selector key is present or
    a nested selector is malformed; a mapping with no selector key
    yields a selector that never matches (handy as an authoring
    placeholder).
    """
    keys = [k for k in _SELECTOR_KEYS if k in raw]
    if len(keys) > 1:
        raise ValueError(f"rule {raw.get('id', '?')!r} has multiple selectors: {keys}")
    field_name = str(raw.get("field", default_field))
    if not keys:
        return lambda _ctx: False
    key = keys[0]
    value = raw[key]

    if key == "pattern":
        regex = re.compile(str(value), re.IGNORECASE | re.MULTILINE)

        def sel_pattern(ctx: dict[str, Any]) -> bool:
            v = _field_value(ctx, field_name)
            return isinstance(v, str) and bool(regex.search(v))

        return sel_pattern

    if key == "regex_any":
        if not isinstance(value, list):
            raise ValueError("regex_any expects a list of patterns")
        regexes = [re.compile(str(p), re.IGNORECASE | re.MULTILINE) for p in value]

        def sel_regex_any(ctx: dict[str, Any]) -> bool:
            v = _field_value(ctx, field_name)
            return isinstance(v, str) and any(r.search(v) for r in regexes)

        return sel_regex_any

    if key == "contains":
        needles = [str(n).lower() for n in (value if isinstance(value, list) else [value])]

        def sel_contains(ctx: dict[str, Any]) -> bool:
            v = _field_value(ctx, field_name)
            if not isinstance(v, str):
                return False
            v_low = v.lower()
            return any(n in v_low for n in needles)

        return sel_contains

    if key == "equals":
        target = value

        def sel_equals(ctx: dict[str, Any]) -> bool:
            return _field_value(ctx, field_name) == target

        return sel_equals

    if key == "field_in":
        if not isinstance(value, list):
            raise ValueError("field_in expects a list of values")
        allowed = list(value)

        def sel_field_in(ctx: dict[str, Any]) -> bool:
            return _field_value(ctx, field_name) in allowed

        return sel_field_in

    if key == "not_empty":
        if value is not True:
            return lambda _ctx: False

        def sel_not_empty(ctx: dict[str, Any]) -> bool:
            v = _field_value(ctx, field_name)
            return v not in (None, "", [], {}, ())

        return sel_not_empty

    if key in ("length_gt", "length_lt"):
        n = int(value)

        def sel_length(ctx: dict[str, Any]) -> bool:
            v = _field_value(ctx, field_name)
            try:
                length = len(v)
            except TypeError:
                return False
            return length > n if key == "length_gt" else length < n

        return sel_length

    if key in ("all_of", "any_of"):
        if not isinstance(value, list) or not value:
            raise ValueError(f"{key} expects a non-empty list of selectors")
        subs = [compile_selector(dict(s), field_name) for s in value]
        if key == "all_of":
            return lambda ctx: all(s(ctx) for s in subs)
        return lambda ctx: any(s(ctx) for s in subs)

    # key == "not"
    if not isinstance(value, dict):
        raise ValueError("not expects a selector mapping")
    inner = compile_selector(dict(value), field_name)
    return lambda ctx: not inner(ctx)


def _rule_from_yaml(raw: dict[str, Any]) -> PolicyRule:
    """Convert one YAML rule into a ``PolicyRule``.

    Supported selector forms are documented in the module docstring.
    A rule marked ``opaque: true`` (produced by :meth:`PolicyDocument.to_yaml`
    for callable-backed rules) never matches and is kept for identity only.
    """
    rid = str(raw["id"])
    decision = raw["decision"]
    if decision not in ("allow", "block", "transform", "escalate"):
        raise ValueError(f"rule {rid!r}: unknown decision {decision!r}")
    rationale = str(raw.get("rationale", rid))
    severity = raw.get("severity", "medium")
    metadata = dict(raw.get("metadata") or {})
    when_role = raw.get("when_role")

    if raw.get("opaque") is True:
        return PolicyRule(
            id=rid,
            selector=lambda _ctx: False,
            decision=decision,
            rationale=rationale,
            severity=severity,
            metadata=metadata,
            spec=None,
            when_role=when_role,
        )

    selector = compile_selector(raw)
    spec = {k: raw[k] for k in (*_SELECTOR_KEYS, "field") if k in raw}
    return PolicyRule(
        id=rid,
        selector=selector,
        decision=decision,
        rationale=rationale,
        severity=severity,
        metadata=metadata,
        spec=spec,
        when_role=when_role,
    )


def rules_from_dicts(raws: Iterable[dict[str, Any]]) -> list[PolicyRule]:
    """Convenience: compile several declarative rules."""
    return [_rule_from_yaml(r) for r in raws]
