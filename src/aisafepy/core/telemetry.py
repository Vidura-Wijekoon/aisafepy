"""OpenTelemetry conventions for AIsafePy.

We piggyback on the emerging ``gen_ai.*`` semantic conventions for LLM
spans, and add ``aisafepy.*`` attributes for guard-specific data. The
goal is that any compatible backend (Langfuse, Arize Phoenix, Helicone,
Opik, Datadog) can show "why was this blocked?" without writing a
custom integration.

If the user never configures an OpenTelemetry SDK, ``get_tracer`` falls
back to a no-op tracer and the attribute setters are effectively free.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
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
            f"{AISAFEPY_NS}.flow.reason": violation.reason,
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
    snapshots emitted periodically).
    """
    try:
        payload = json.dumps({"event": event, **fields}, default=str)
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


def _is_sensitive_key(key: str) -> bool:
    """Return True if a dict key looks sensitive."""
    if not isinstance(key, str):
        return False
    low = key.lower()
    if low in _SENSITIVE_KEYS:
        return True
    return any(s in low for s in _SENSITIVE_SUBSTRINGS)


def redact(value, _depth: int = 0):
    """Recursively redact sensitive values in a JSON-shaped structure.

    Returns a new structure; the input is not mutated. Strings that
    exceed 4 KB are truncated to avoid log explosion. Recursion depth
    is capped at 8 to defend against pathological nesting.
    """
    if _depth > 8:
        return REDACTED_PLACEHOLDER
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if _is_sensitive_key(k):
                out[k] = REDACTED_PLACEHOLDER
            else:
                out[k] = redact(v, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v, _depth + 1) for v in value]
    if isinstance(value, str) and len(value) > 4096:
        return value[:4093] + "..."
    return value
