"""OpenTelemetry conventions for AIsafePy.

We piggyback on the emerging ``gen_ai.*`` semantic conventions for LLM
spans, and add ``aisafepy.*`` attributes for guard-specific data. The
goal is that any compatible backend (Langfuse, Arize Phoenix, Helicone,
Opik, Datadog) can show "why was this blocked?" without writing a
custom integration.

If the user never configures an OpenTelemetry SDK, ``get_tracer`` falls
back to a no-op tracer and the attribute setters are effectively free.

Redaction
---------

Everything that leaves the process through a span or a structured log
line passes through :func:`redact`. Two independent defences apply:

* **Key-based.** Dict keys that name credentials (``api_key``,
  ``authorization``, ``password``, ...) have their values replaced
  wholesale. This is the liteLLM-incident defence: guardrail evidence
  that happened to include request headers must not leak them.
* **Value-based.** String values are scanned for well-known credential
  shapes (OpenAI / Anthropic keys, GitHub and Slack tokens, AWS access
  keys, JWTs, bearer headers, PEM blocks). Matches are replaced with a
  typed placeholder such as ``[REDACTED:openai_key]``. This catches the
  far more common case where a secret is embedded in free text, e.g. a
  model echoing a key that appeared in a retrieved document, under a
  perfectly innocent key like ``"snippet"``.

Both lists can be extended at runtime with :func:`configure_redaction`.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

if TYPE_CHECKING:  # pragma: no cover
    from opentelemetry.trace import Span

    from aisafepy.core.decisions import GuardDecision, IFCViolation

GEN_AI_NS = "gen_ai"
AISAFEPY_NS = "aisafepy"

_logger = logging.getLogger("aisafepy")

# Provide a basic stderr handler if the application hasn't configured one.
if not _logger.handlers:  # pragma: no cover - depends on env
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)


def get_tracer(name: str = "aisafepy", version: str | None = None) -> trace.Tracer:
    """Return an OpenTelemetry tracer.

    Always succeeds; if the SDK is not configured this returns the
    no-op tracer.
    """
    return trace.get_tracer(name, version)


def _set_attrs(span: Span, attrs: dict[str, Any]) -> None:
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (str, bool, int, float)):
            span.set_attribute(k, v)
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v):
            span.set_attribute(k, list(v))
        else:
            # Fall back to JSON for complex evidence dicts.
            try:
                span.set_attribute(k, json.dumps(v, default=str))
            except Exception:  # pragma: no cover - defensive
                span.set_attribute(k, str(v))


@contextmanager
def span_for_decision(
    name: str,
    *,
    tracer: trace.Tracer | None = None,
    extra: dict[str, Any] | None = None,
) -> Iterator[Span]:
    """Open a span that will be enriched with a ``GuardDecision`` on exit.

    Usage::

        with span_for_decision("aisafepy.stream.tier2") as span:
            decision = await guard(context)
            from aisafepy.core.telemetry import attach_decision
            attach_decision(span, decision)
    """
    tracer = tracer or get_tracer()
    with tracer.start_as_current_span(name) as span:
        if extra:
            _set_attrs(span, extra)
        yield span


def attach_decision(span: Span, decision: GuardDecision) -> None:
    """Write the canonical ``aisafepy.guard.*`` attributes onto a span.

    ``evidence`` and ``rationale`` are run through :func:`redact` first
    so secrets and PII do not leak into trace backends. This addresses
    the liteLLM-class incident where guardrail logs captured
    Authorization headers and other sensitive fields.
    """
    _set_attrs(
        span,
        {
            f"{AISAFEPY_NS}.guard.action": decision.action.value,
            f"{AISAFEPY_NS}.guard.confidence": decision.confidence,
            f"{AISAFEPY_NS}.guard.tier": decision.tier,
            f"{AISAFEPY_NS}.guard.rationale": redact(decision.rationale),
            f"{AISAFEPY_NS}.guard.severity": decision.severity,
            f"{AISAFEPY_NS}.guard.latency_ms": decision.latency_ms,
            f"{AISAFEPY_NS}.guard.name": decision.guard_name,
            f"{AISAFEPY_NS}.guard.decision_id": decision.decision_id,
            f"{AISAFEPY_NS}.guard.evidence": redact(decision.evidence),
            f"{AISAFEPY_NS}.guard.fallback": decision.fallback,
        },
    )


def attach_violation(span: Span, violation: IFCViolation) -> None:
    """Write the canonical ``aisafepy.flow.*`` attributes onto a span."""
    _set_attrs(
        span,
        {
            f"{AISAFEPY_NS}.flow.reason": redact(violation.reason),
            f"{AISAFEPY_NS}.flow.tool": violation.tool,
            f"{AISAFEPY_NS}.flow.provenance": sorted(violation.provenance),
            f"{AISAFEPY_NS}.flow.integrity": violation.integrity,
            f"{AISAFEPY_NS}.flow.required_integrity": violation.required_integrity,
            f"{AISAFEPY_NS}.flow.capabilities": sorted(violation.capabilities),
            f"{AISAFEPY_NS}.flow.required_capabilities": sorted(
                violation.required_capabilities
            ),
            f"{AISAFEPY_NS}.flow.taint_chain": [list(t) for t in violation.taint_chain],
            f"{AISAFEPY_NS}.flow.violation_id": violation.violation_id,
        },
    )


def structured_log(
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured log line.

    Used by guards and adapters when an OTel tracer is not configured,
    or for events that don't naturally live on a span (e.g. budget
    snapshots emitted periodically). Fields are redacted with the same
    rules as span attributes.
    """
    try:
        payload = json.dumps({"event": event, **redact(fields)}, default=str)
    except Exception:  # pragma: no cover
        payload = f"event={event} fields={fields!r}"
    _logger.log(level, payload)


# ---- redaction ----------------------------------------------------------

# Keys that commonly carry sensitive data. Values for these keys are
# replaced with "[REDACTED]" before they reach an OTel span. This is a
# defense against the liteLLM-class incident where guardrail logs
# captured Authorization headers and other secrets.
_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "api-key",
    "authorization", "auth", "token", "access_token", "refresh_token",
    "password", "passwd", "secret", "private_key", "ssh_key",
    "cookie", "session", "sessionid", "session_id",
    "credit_card", "card_number", "ssn", "social_security",
    "x-api-key", "x_api_key", "openai_api_key", "anthropic_api_key",
})

# Substring patterns indicating sensitive context (case-insensitive).
_SENSITIVE_SUBSTRINGS = (
    "api_key", "apikey", "secret", "password", "token", "authorization",
)

REDACTED_PLACEHOLDER = "[REDACTED]"

# Value shapes that are credentials regardless of which key holds them.
# Ordered from most to least specific so the typed placeholder names the
# narrowest match (``sk-ant-`` before ``sk-``, for instance).
_SECRET_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "pem_private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
        ),
    ),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{16,}")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("stripe_key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("basic_auth_url", re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:([^@\s/]+)@")),
)


@dataclass
class RedactionConfig:
    """Mutable, process-wide redaction settings. See :func:`configure_redaction`."""

    enabled: bool = True
    extra_keys: set[str] = field(default_factory=set)
    extra_substrings: tuple[str, ...] = ()
    extra_patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=list)
    max_string_chars: int = 4096
    max_depth: int = 8
    placeholder: str = REDACTED_PLACEHOLDER


_config = RedactionConfig()
_config_lock = threading.Lock()


def configure_redaction(
    *,
    enabled: bool | None = None,
    extra_keys: Iterable[str] = (),
    extra_substrings: Iterable[str] = (),
    extra_patterns: dict[str, str | re.Pattern[str]] | None = None,
    max_string_chars: int | None = None,
    max_depth: int | None = None,
    reset: bool = False,
) -> RedactionConfig:
    """Adjust the global redaction rules.

    ``extra_keys`` are exact (case-insensitive) dict keys to scrub;
    ``extra_substrings`` are key fragments; ``extra_patterns`` map a
    label to a regex applied to string *values*. ``reset=True`` restores
    the defaults before applying the other arguments. Returns the live
    config object.
    """
    global _config
    with _config_lock:
        if reset:
            _config = RedactionConfig()
        cfg = _config
        if enabled is not None:
            cfg.enabled = enabled
        cfg.extra_keys.update(k.lower() for k in extra_keys)
        if extra_substrings:
            cfg.extra_substrings = (*cfg.extra_substrings, *(s.lower() for s in extra_substrings))
        for label, pat in (extra_patterns or {}).items():
            compiled = pat if isinstance(pat, re.Pattern) else re.compile(pat)
            cfg.extra_patterns.append((label, compiled))
        if max_string_chars is not None:
            cfg.max_string_chars = max(16, int(max_string_chars))
        if max_depth is not None:
            cfg.max_depth = max(1, int(max_depth))
        return cfg


def redaction_config() -> RedactionConfig:
    """Return the live redaction configuration (read-only use intended)."""
    return _config


def _is_sensitive_key(key: str) -> bool:
    """Return True if a dict key looks sensitive."""
    if not isinstance(key, str):
        return False
    low = key.lower()
    if low in _SENSITIVE_KEYS or low in _config.extra_keys:
        return True
    if any(s in low for s in _SENSITIVE_SUBSTRINGS):
        return True
    return any(s in low for s in _config.extra_substrings)


def find_secret_values(text: str) -> list[tuple[str, int, int]]:
    """Return ``(label, start, end)`` for every credential-shaped span in ``text``."""
    if not text:
        return []
    found: list[tuple[str, int, int]] = []
    for label, pat in (*_SECRET_VALUE_PATTERNS, *_config.extra_patterns):
        for m in pat.finditer(text):
            # ``basic_auth_url`` only wants the password group masked.
            if label == "basic_auth_url" and m.lastindex:
                found.append((label, m.start(1), m.end(1)))
            else:
                found.append((label, m.start(), m.end()))
    found.sort(key=lambda t: (t[1], -t[2]))
    # Drop spans fully contained in an earlier, wider span.
    kept: list[tuple[str, int, int]] = []
    for item in found:
        if kept and item[1] >= kept[-1][1] and item[2] <= kept[-1][2]:
            continue
        kept.append(item)
    return kept


def redact_text(text: str) -> str:
    """Replace credential-shaped substrings with ``[REDACTED:<label>]``."""
    spans = find_secret_values(text)
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for label, start, end in spans:
        if start < cursor:
            continue
        out.append(text[cursor:start])
        out.append(f"[REDACTED:{label}]")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def redact(value, _depth: int = 0):
    """Recursively redact sensitive values in a JSON-shaped structure.

    Returns a new structure; the input is not mutated. Strings are
    scanned for credential shapes, then truncated to
    ``max_string_chars`` to avoid log explosion. Recursion depth is
    capped to defend against pathological nesting. Disabled entirely
    when ``configure_redaction(enabled=False)`` was called.
    """
    cfg = _config
    if not cfg.enabled:
        return value
    if _depth > cfg.max_depth:
        return cfg.placeholder
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if _is_sensitive_key(k):
                out[k] = cfg.placeholder
            else:
                out[k] = redact(v, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v, _depth + 1) for v in value]
    if isinstance(value, str):
        scrubbed = redact_text(value)
        if len(scrubbed) > cfg.max_string_chars:
            return scrubbed[: cfg.max_string_chars - 3] + "..."
        return scrubbed
    return value
