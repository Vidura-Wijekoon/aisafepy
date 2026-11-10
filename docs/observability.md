# Observability

AIsafePy is built so that "why was this blocked?" is answerable from
your existing tracing stack without a custom integration. Every decision
is a structured `GuardDecision`, every enforcement point opens an
OpenTelemetry span, and sensitive data is redacted before it leaves the
process.

## Spans

| Span name | Opened by | When |
|-----------|-----------|------|
| `aisafepy.flow.<tool>` | `flow.interpreter.evaluate_call` | once per tool-call evaluation |
| `aisafepy.stream.tier1` | `stream.pipeline` | when Tier 1 runs |
| `aisafepy.stream.tier2` | `stream.pipeline` | when Tier 2 runs |
| `aisafepy.stream.tier3` | `stream.pipeline` | when Tier 3 runs |

Spans are created through `span_for_decision`. If no OpenTelemetry SDK
is configured, `get_tracer` returns the no-op tracer and the span and
attribute calls cost effectively nothing.

## Guard attributes (`aisafepy.guard.*`)

Written by `attach_decision` for every `GuardDecision`.

| Attribute | Meaning |
|-----------|---------|
| `aisafepy.guard.action` | allow / block / transform / escalate |
| `aisafepy.guard.confidence` | 0.0 to 1.0 |
| `aisafepy.guard.tier` | cascade tier that produced the decision |
| `aisafepy.guard.rationale` | human-readable reason (redacted) |
| `aisafepy.guard.severity` | info / low / medium / high / critical |
| `aisafepy.guard.latency_ms` | guard latency |
| `aisafepy.guard.name` | guard identifier |
| `aisafepy.guard.decision_id` | unique id for correlating logs |
| `aisafepy.guard.evidence` | JSON-encoded evidence (redacted) |
| `aisafepy.guard.fallback` | user-facing fallback message, if any |

## Flow attributes (`aisafepy.flow.*`)

Written by `attach_violation` for every `IFCViolation`.

| Attribute | Meaning |
|-----------|---------|
| `aisafepy.flow.reason` | why the call was denied |
| `aisafepy.flow.tool` | the tool that was blocked |
| `aisafepy.flow.provenance` | sorted source identifiers of the arguments |
| `aisafepy.flow.integrity` | joined integrity label of the arguments |
| `aisafepy.flow.required_integrity` | the integrity the tool required |
| `aisafepy.flow.capabilities` | capability tags present in the arguments |
| `aisafepy.flow.required_capabilities` | capabilities the tool permits |
| `aisafepy.flow.taint_chain` | the (source, op, dest) trail reconstructed by the interpreter |
| `aisafepy.flow.violation_id` | unique id for correlating logs |

## Structured log events

Emitted by `structured_log` as JSON lines through the `aisafepy` logger,
for events that do not naturally live on a span or when no tracer is
configured.

| Event | Fields | Emitted when |
|-------|--------|--------------|
| `budget.checkpoint` | budget, kind, used, limit, pct | a budget crosses 25 / 50 / 75 / 100 percent |
| `guard.error` | guard, error, latency_ms | a guard raised and the pipeline failed open |
| `on_violation.error` | error | an `on_violation` callback raised |

## Redaction

`attach_decision` runs `rationale` and `evidence` through `redact`
before they reach a span, and you can call `redact` directly on anything
you log yourself. The rules:

- Dict keys that look sensitive (`api_key`, `authorization`, `token`,
  `password`, `secret`, `cookie`, `ssn`, `credit_card`, and the common
  variants and provider-specific key names) have their values replaced
  with `[REDACTED]`.
- Keys containing the substrings `api_key`, `apikey`, `secret`,
  `password`, `token`, or `authorization` are treated as sensitive.
- Strings longer than 4 KB are truncated to avoid log explosion.
- Recursion is capped at depth 8 against pathological nesting.
- The input is never mutated; a redacted copy is returned.

This defends against the class of incident where guardrail logs captured
authorization headers and other secrets.

## Pointing a backend at AIsafePy

AIsafePy depends only on `opentelemetry-api`, so it emits to whatever
SDK and exporter your application configures. There is nothing
AIsafePy-specific to wire up: configure the OpenTelemetry SDK once and
the `aisafepy.*` spans flow to your backend alongside everything else.

Generic setup (OTLP, which Langfuse, Arize Phoenix, Grafana, and Datadog
all accept through a collector or native endpoint):

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="...")))
trace.set_tracer_provider(provider)

# From here on, every aisafepy guard decision is a span on your backend.
```

For Langfuse and Arize Phoenix, point the OTLP exporter at their
ingestion endpoint. For Datadog, export through the OpenTelemetry
Collector or the Datadog Agent's OTLP intake. The `gen_ai.*` and
`aisafepy.*` attributes appear as span attributes you can filter on.

## Working with decisions outside a tracer

Guards return `GuardDecision` objects whether or not a tracer is
configured. A simple pattern is to write `decision.model_dump()` as one
JSON line per decision and analyse the file with the CLI:

```bash
aisafepy decisions summarize decisions.jsonl
aisafepy decisions filter decisions.jsonl --action block --min-confidence 0.9
```

### Sample decision log

```json
{"action": "allow", "confidence": 1.0, "tier": 1, "rationale": "no regex matches", "severity": "info", "latency_ms": 0.42, "guard_name": "pii", "decision_id": "a1b2c3", "evidence": {}}
{"action": "block", "confidence": 1.0, "tier": 1, "rationale": "regex match: ['ssn']", "severity": "high", "latency_ms": 0.91, "guard_name": "pii", "decision_id": "d4e5f6", "evidence": {"matches": [["ssn", "123-45-6789"]]}}
{"action": "escalate", "confidence": 0.55, "tier": 3, "rationale": "judge unsure", "severity": "medium", "latency_ms": 312.0, "guard_name": "judge", "decision_id": "7890ab", "evidence": {}}
```
