# `aisafepy.core`: shared primitives

Everything in `core` is dependency-light (pydantic, OpenTelemetry API,
anyio, tenacity) and used by all three main modules.

## Decisions

`GuardDecision` is the type every guard, IFC check and compiled
artifact returns. Constructors: `allow`, `block`, `transform`,
`escalate`. Predicates: `is_terminal` (block or escalate), `is_blocked`.

Decisions are frozen pydantic models, so they can be hashed, logged and
compared safely. `IFCViolation.to_guard_decision()` renders an
information-flow violation in the same shape.

### Decision records and sinks (`core.sinks`)

`decision_to_record(decision, extra=...)` renders a decision as a
JSON-safe dict with a stable schema:

```json
{"schema_version": 1, "timestamp": 1731240000.1, "decision_id": "…",
 "action": "block", "confidence": 0.97, "tier": 2, "severity": "high",
 "guard_name": "prompt-guard-2", "rationale": "…", "evidence": {"…": "…"},
 "latency_ms": 12.4, "fallback": "I can't help with that.",
 "transformed": false, "extra": {"request_id": "r-1"}}
```

`rationale`, `evidence` and `extra` pass through
`aisafepy.core.telemetry.redact` unless `redact_fields=False`, so API
keys and tokens captured as evidence never reach disk.
`record_to_decision` is the (lossy) inverse.

| sink | behaviour |
|---|---|
| `InMemorySink(maxlen)` | bounded ring buffer; `actions()` counts per action |
| `JSONLSink(path)` | one JSON object per line, lock-protected appends, `read()` |
| `CallbackSink(fn)` | forwards records; exceptions are counted on `errors` |
| `MultiSink([...])` | fan-out; a failing sink never blocks the others |
| `FilterSink(inner, min_severity=, actions=, guard_names=)` | forwards only matching decisions |

`severity_rank()` orders `info < low < medium < high < critical`.

## Budgets (`core.budgets`)

| class | caps | consume via |
|---|---|---|
| `IterationBudget` | agent steps / tool calls | `consume()` |
| `TokenBudget` | cumulative tokens | `consume(n)` |
| `DollarBudget` | cumulative USD | `consume(usd)` or `charge(model, prompt_tokens, completion_tokens)` |
| `WallClockBudget` | elapsed seconds | `start()` then `check()` |
| `RateBudget` | events per sliding window | `try_acquire()` / `acquire()`; `retry_after()` |

All budgets raise `BudgetExceeded` (with `.budget` attached) when the
cap is crossed, emit a `budget.checkpoint` structured log the first
time usage passes 25/50/75/100 percent, and call every callback in
`budget.on_checkpoint` with `(snapshot, pct)`.

### Pricing

`TokenPricing` maps model ids to USD per million prompt/completion
tokens. Lookups are exact first, then longest registered prefix, so a
dated snapshot such as `gpt-4o-2024-08-06` inherits the `gpt-4o`
price. Unknown models raise `KeyError` rather than costing zero.
Register your negotiated rates:

```python
pricing = TokenPricing()
pricing.register("my-finetune", prompt_per_million=1.2, completion_per_million=4.8)
budget = DollarBudget("session", limit=0.50)
budget.charge("my-finetune", usage.prompt_tokens, usage.completion_tokens, pricing=pricing)
```

### Grouping and enforcement

```python
from aisafepy.core.budgets import BudgetSet, budgeted, IterationBudget, WallClockBudget, RateBudget

run = BudgetSet("run", [IterationBudget("steps", 50), WallClockBudget("wall", 120),
                        RateBudget("tool-rps", max_events=5, per_seconds=1)])

@budgeted(run)                  # per call: check wall clock, acquire a rate slot, consume a step
async def call_tool(name, **kwargs): ...

with run:                       # starts clocks on entry, check_all() on clean exit
    await agent.run()
run.snapshots()                 # {name: BudgetSnapshot}
run.exceeded()                  # budgets currently over cap, never raises
```

`BudgetSet.get()` accepts a budget name or, when unambiguous, a kind
(`"iteration"`, `"tokens"`, `"dollars"`, `"wall_clock"`, `"rate"`).

## Progress / loop detection (`core.progress`)

`ProgressTracker.observe(Step(tool, args, observation))` returns a
`LoopDetected` (an `escalate` decision) when a detector fires:

| reason | trigger |
|---|---|
| `max_steps` | total steps reached `max_steps` |
| `duplicate_tool_call` | the same fingerprint `duplicate_threshold` times inside `window` |
| `cyclic_tool_calls` | an A-B-A-B or A-B-C-A-B-C pattern (`cycle_lengths`) |
| `stagnation` | identical observations for `stagnation_window` steps |
| `semantic_no_progress` | `stagnation_window` consecutive similarity scores at or above `semantic_threshold` |

Fingerprints can ignore volatile keys (`ignore_args={"request_id"}`)
and fold string case/whitespace (`normalize_strings=True`). Similarity
scores come from the caller (`observe_similarity(score)` from
embeddings) or from `observe_text(observation)`, which computes
character 3-gram Jaccard against the previous observation and needs no
model. `explain()` returns every detector's current state; `reset()`
clears history at a task boundary.

## Policy documents (`core.policies`)

`PolicyDocument` is an ordered, first-match-wins list of `PolicyRule`s
evaluated over a plain dict context. Load from YAML, build in Python,
or mix.

### YAML reference

```yaml
name: company-safety
version: "2.1"
description: optional
metadata: {owner: trust-and-safety}
rules:
  - id: destructive-sql
    regex_any: ['\bdrop\s+table\b', '\btruncate\b']
    decision: block
    severity: critical
    rationale: destructive statement
  - id: refusal-leak
    contains: ["as an ai language model"]
    when_role: assistant            # only when context["role"] == "assistant"
    decision: escalate
  - id: free-tier-redaction
    field: metadata.tenant          # dotted paths walk nested dicts
    field_in: [free, trial]
    decision: transform
    metadata: {replacement: "[hidden]"}
  - id: oversized
    length_gt: 20000
    decision: block
  - id: wire-without-invoice
    all_of:
      - contains: [wire]
      - not: {contains: [invoice]}
    decision: block
  - id: handshake
    any_of:
      - equals: ping
      - {field: prompt, equals: pong}
    decision: allow
```

Selector keys (exactly one per rule or nested selector): `pattern`,
`regex_any`, `contains`, `equals`, `field_in`, `not_empty`,
`length_gt`, `length_lt`, `all_of`, `any_of`, `not`. Every rule may
carry `field` (default `content`), `when_role`, `severity`,
`rationale` and `metadata`.

### API

| method | purpose |
|---|---|
| `evaluate(ctx)` | first matching rule or `None` |
| `evaluate_all(ctx)` | every matching rule in order |
| `explain(ctx)` | every rule with `matched` / `selected` flags |
| `to_guard_decision(rule)` | render a match as a `GuardDecision` |
| `as_guard(name, tier=1)` | a Guard-protocol object for `GuardPipeline`; context is `{"content", "prompt", "role", "metadata", **metadata}` |
| `to_yaml(path)` / `from_yaml(path)` / `to_dict()` / `from_dict()` | serialization; Python-callable rules serialize as `opaque: true` stubs |

## Telemetry

`get_tracer()`, `span_for_decision()`, `attach_decision()` and
`attach_violation()` write `aisafepy.guard.*` / `aisafepy.flow.*`
attributes onto OpenTelemetry spans; `structured_log()` emits JSON log
lines; `redact()` strips sensitive keys and truncates long strings
before anything leaves the process.
