"""Declarative policy serialization for :class:`aisafepy.flow.Policy`.

The fluent builder is the primary API, but production teams keep
policies in version-controlled configuration. This module provides a
lossless dict / YAML round trip for every part of a policy *except*
deny rules written as Python callables. For those, it offers a small
declarative predicate vocabulary that is expressive enough for the
CaMeL / FIDES patterns and compiles to the same ``when`` callables the
interpreter already understands.

Predicate vocabulary
--------------------

A predicate is a mapping with exactly one operator key::

    {"provenance_contains": "gmail.read"}        # any arg came from gmail.read
    {"provenance_contains": ["a", "b"]}          # ... from a OR b
    {"capability_present": "read.secrets"}       # any arg carries the capability
    {"arg_integrity_below": "TRUSTED"}           # any arg is worse than TRUSTED
    {"value_matches": "(?i)@evil\\.com$"}        # regex over any string arg
    {"any_of": [pred, ...]}
    {"all_of": [pred, ...]}
    {"not": pred}

A leaf predicate may be scoped to one argument with the optional
``arg`` key: a string selects a keyword argument, an integer selects a
positional index. An out-of-range or missing argument never matches.

Plain (non-``Tainted``) arguments are treated the way the interpreter
treats them: empty provenance, no capabilities, ``TRUSTED`` integrity.

YAML schema
-----------

::

    mode: strict                       # or mediated
    sources:
      user_prompt: {integrity: TRUSTED}
      gmail.read: {integrity: UNTRUSTED, capabilities: [read.user]}
    requirements:
      send_email:
        control_flow_integrity: TRUSTED
        capabilities: [write.external]
        arg_max_integrity: UNTRUSTED
    deny_rules:
      - tool: send_email
        reason: secret-to-external-sink
        when: {capability_present: read.secrets}
    declassifications:
      summarize: {by: "P-LLM only, structured output"}

Top-level ``name``, ``version``, and ``description`` keys are accepted
and ignored so a policy file can carry its own metadata.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, get_args

from aisafepy.flow.policy import DenyRule, Policy
from aisafepy.flow.taint import Integrity, labels_of, lower

try:
    import yaml  # type: ignore[import-not-found]

    _HAS_YAML = True
except Exception:  # pragma: no cover - depends on env
    _HAS_YAML = False

_INTEGRITY_ORDER: tuple[Integrity, ...] = ("TRUSTED", "UNTRUSTED", "QUARANTINED")
_VALID_INTEGRITY = frozenset(get_args(Integrity))
_VALID_MODES = frozenset({"strict", "mediated"})

_LEAF_OPS = frozenset(
    {"provenance_contains", "capability_present", "arg_integrity_below", "value_matches"}
)
_COMBINATOR_OPS = frozenset({"any_of", "all_of", "not"})
_ALL_OPS = _LEAF_OPS | _COMBINATOR_OPS

Predicate = Callable[..., bool]


class PredicateError(ValueError):
    """A declarative predicate spec is malformed."""


class SerializationError(ValueError):
    """A policy document cannot be (de)serialized faithfully."""


# ---- predicate compilation -----------------------------------------------


def compile_predicate(spec: dict[str, Any]) -> Predicate:
    """Compile a declarative predicate into a ``when`` callable.

    The returned callable has the ``(*args, **kwargs)`` signature the
    interpreter uses for ``deny_if`` rules. Raises :class:`PredicateError`
    on malformed input so mistakes surface at policy-load time rather
    than silently never matching at runtime.
    """
    return _compile(spec, depth=0)


def validate_predicate(spec: Any) -> list[str]:
    """Return a list of problems with ``spec`` (empty when valid)."""
    try:
        _compile(spec, depth=0)
    except PredicateError as exc:
        return [str(exc)]
    return []


def _compile(spec: Any, *, depth: int) -> Predicate:
    if depth > 16:
        raise PredicateError("predicate nesting exceeds 16 levels")
    if not isinstance(spec, dict):
        raise PredicateError(f"predicate must be a mapping (got {type(spec).__name__})")
    ops = [k for k in spec if k in _ALL_OPS]
    unknown = [k for k in spec if k not in _ALL_OPS and k != "arg"]
    if unknown:
        raise PredicateError(f"unknown predicate key(s): {sorted(unknown)}")
    if len(ops) != 1:
        raise PredicateError(
            f"predicate must have exactly one operator key, got {sorted(ops) or 'none'}"
        )
    op = ops[0]
    body = spec[op]
    scope = spec.get("arg")

    if op in _COMBINATOR_OPS:
        if scope is not None:
            raise PredicateError(f"'arg' scoping is only valid on leaf predicates (op={op})")
        if op == "not":
            inner = _compile(body, depth=depth + 1)
            return lambda *a, **kw: not inner(*a, **kw)
        if not isinstance(body, list) or not body:
            raise PredicateError(f"{op} requires a non-empty list of predicates")
        parts = [_compile(p, depth=depth + 1) for p in body]
        if op == "any_of":
            return lambda *a, **kw: any(p(*a, **kw) for p in parts)
        return lambda *a, **kw: all(p(*a, **kw) for p in parts)

    if scope is not None and (isinstance(scope, bool) or not isinstance(scope, (str, int))):
        raise PredicateError("'arg' must be a keyword name (str) or positional index (int)")

    test = _leaf_test(op, body)

    def predicate(*args: Any, **kwargs: Any) -> bool:
        return any(test(value) for value in _select(args, kwargs, scope))

    return predicate


def _select(args: tuple[Any, ...], kwargs: dict[str, Any], scope: str | int | None) -> list[Any]:
    if scope is None:
        return [*args, *kwargs.values()]
    if isinstance(scope, str):
        return [kwargs[scope]] if scope in kwargs else []
    return [args[scope]] if -len(args) <= scope < len(args) else []


def _leaf_test(op: str, body: Any) -> Callable[[Any], bool]:
    if op == "provenance_contains":
        names = _as_name_set(body, op)
        return lambda v: bool(labels_of(v)[0] & names)
    if op == "capability_present":
        names = _as_name_set(body, op)
        return lambda v: bool(labels_of(v)[1] & names)
    if op == "arg_integrity_below":
        if body not in _VALID_INTEGRITY:
            raise PredicateError(f"arg_integrity_below expects one of {sorted(_VALID_INTEGRITY)}")
        threshold = _INTEGRITY_ORDER.index(body)
        return lambda v: _INTEGRITY_ORDER.index(labels_of(v)[2]) > threshold
    if op == "value_matches":
        if not isinstance(body, str) or not body:
            raise PredicateError("value_matches expects a non-empty regex string")
        try:
            regex = re.compile(body)
        except re.error as exc:
            raise PredicateError(f"value_matches regex is invalid: {exc}") from exc

        def _match(v: Any) -> bool:
            raw = lower(v)
            return isinstance(raw, str) and regex.search(raw) is not None

        return _match
    raise PredicateError(f"unknown leaf operator {op!r}")  # pragma: no cover - guarded above


def _as_name_set(body: Any, op: str) -> frozenset[str]:
    if isinstance(body, str):
        items: Iterable[Any] = [body]
    elif isinstance(body, (list, tuple, set, frozenset)):
        items = body
    else:
        raise PredicateError(f"{op} expects a string or list of strings")
    names = frozenset(items)
    if not names or not all(isinstance(n, str) and n for n in names):
        raise PredicateError(f"{op} expects non-empty string names")
    return names


# ---- dict round trip -------------------------------------------------------


def policy_to_dict(policy: Policy) -> dict[str, Any]:
    """Render ``policy`` as a JSON / YAML compatible dict.

    Identical to :meth:`Policy.to_dict`; provided for symmetry with
    :func:`policy_from_dict`.
    """
    return policy.to_dict()


def policy_from_dict(data: dict[str, Any], *, strict: bool = True) -> Policy:
    """Rebuild a :class:`Policy` from the dict shape of :func:`policy_to_dict`.

    ``strict=True`` (default) raises :class:`SerializationError` for a
    deny rule that lacks a declarative ``when`` spec, since such a rule
    cannot be reconstructed. With ``strict=False`` those rules are
    dropped and the information is reported via the returned policy's
    ``deny_rules`` simply not containing them; callers that need to
    know can compare lengths before and after.
    """
    if not isinstance(data, dict):
        raise SerializationError(f"policy document must be a mapping (got {type(data).__name__})")
    allowed = {
        "mode", "sources", "requirements", "deny_rules", "declassifications",
        "name", "version", "description",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SerializationError(f"unknown top-level key(s): {unknown}")

    policy = Policy()
    mode = data.get("mode", "strict")
    if mode not in _VALID_MODES:
        raise SerializationError(f"mode must be one of {sorted(_VALID_MODES)} (got {mode!r})")
    policy.with_mode(mode)

    for name, raw in (data.get("sources") or {}).items():
        raw = raw or {}
        _expect_mapping(raw, f"sources.{name}")
        policy.label_source(
            str(name),
            integrity=_integrity(raw.get("integrity", "UNTRUSTED"), f"sources.{name}.integrity"),
            caps=_caps(raw.get("capabilities", ()), f"sources.{name}.capabilities"),
        )

    for tool, raw in (data.get("requirements") or {}).items():
        raw = raw or {}
        _expect_mapping(raw, f"requirements.{tool}")
        policy.require(
            str(tool),
            control_flow_integrity=_integrity(
                raw.get("control_flow_integrity", "UNTRUSTED"),
                f"requirements.{tool}.control_flow_integrity",
            ),
            caps=_caps(raw.get("capabilities", ()), f"requirements.{tool}.capabilities"),
            arg_max_integrity=_integrity(
                raw.get("arg_max_integrity", "UNTRUSTED"),
                f"requirements.{tool}.arg_max_integrity",
            ),
        )

    for i, raw in enumerate(data.get("deny_rules") or []):
        _expect_mapping(raw, f"deny_rules[{i}]")
        tool = raw.get("tool")
        if not isinstance(tool, str) or not tool:
            raise SerializationError(f"deny_rules[{i}].tool must be a non-empty string")
        reason = str(raw.get("reason") or f"deny-rule-{i}")
        spec = raw.get("when")
        if spec is None:
            if strict:
                raise SerializationError(
                    f"deny_rules[{i}] ({tool!r}: {reason!r}) has no declarative 'when' "
                    "predicate and cannot be reconstructed; pass strict=False to drop it"
                )
            continue
        try:
            predicate = compile_predicate(spec)
        except PredicateError as exc:
            raise SerializationError(f"deny_rules[{i}].when: {exc}") from exc
        policy.deny_rules.append(DenyRule(tool=tool, when=predicate, reason=reason, spec=spec))

    for tool, raw in (data.get("declassifications") or {}).items():
        raw = raw or {}
        _expect_mapping(raw, f"declassifications.{tool}")
        policy.declassify(str(tool), by=str(raw.get("by", "")))

    return policy


def _expect_mapping(value: Any, where: str) -> None:
    if not isinstance(value, dict):
        raise SerializationError(f"{where} must be a mapping (got {type(value).__name__})")


def _integrity(value: Any, where: str) -> Integrity:
    if value not in _VALID_INTEGRITY:
        raise SerializationError(
            f"{where} must be one of {sorted(_VALID_INTEGRITY)} (got {value!r})"
        )
    return value  # type: ignore[return-value]


def _caps(value: Any, where: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise SerializationError(f"{where} must be a list of capability strings")
    out: list[str] = []
    for c in value:
        if not isinstance(c, str) or not c:
            raise SerializationError(f"{where} entries must be non-empty strings")
        out.append(c)
    return out


# ---- YAML -----------------------------------------------------------------


def _require_yaml() -> None:
    if not _HAS_YAML:
        raise RuntimeError(
            "YAML policy support requires PyYAML. Install via `pip install pyyaml`."
        )


def policy_to_yaml(policy: Policy, path: str | Path | None = None) -> str:
    """Serialize ``policy`` to YAML text; write it to ``path`` when given.

    Deny rules without a declarative spec are emitted as reason-only
    entries prefixed with a ``# opaque`` marker in a trailing comment
    block, so a reviewer can see what the file does *not* capture.
    """
    _require_yaml()
    data = policy.to_dict()
    opaque = [r for r in data["deny_rules"] if "when" not in r]
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
    if opaque:
        notes = "\n".join(
            f"#   - {r['tool']}: {r['reason']}" for r in opaque
        )
        text += (
            "# NOTE: the following deny rules were written as Python callables and\n"
            "# are NOT reproducible from this file:\n" + notes + "\n"
        )
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def policy_from_yaml(source: str | Path, *, strict: bool = True) -> Policy:
    """Load a policy from a YAML file path or a YAML string.

    A ``str`` that names an existing file is read from disk; any other
    string is parsed as YAML text. ``Path`` objects are always read.
    """
    _require_yaml()
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    elif isinstance(source, str) and "\n" not in source and Path(source).is_file():
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = str(source)
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    return policy_from_dict(data, strict=strict)
