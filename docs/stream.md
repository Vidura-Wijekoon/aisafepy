# `aisafepy.stream`: streaming-native cascaded guardrails

`aisafepy.stream` evaluates model output (or user input, or tool
results) through a three-tier cascade with an explicit latency budget.
This document covers the cascade semantics, the declarative YAML
configuration, observability, conversation-level guarding, the
LLM-judge tier, shadow mode, and the fail-open versus fail-closed
decision.

## The cascade

```
chunk ──> Tier 1 (deterministic, <10 ms) ──> Tier 2 (small classifier, <100 ms) ──> Tier 3 (heavy, <500 ms)
             │                                   │                                    │
             └── terminal? return ───────────────┴── terminal? return ────────────────┴── return
```

* Guards inside a tier run **concurrently** (`asyncio.gather`). The
  tier's decision is the most confident BLOCK, else the most confident
  ESCALATE, else the first TRANSFORM, else the most confident ALLOW.
* A **terminal** decision (BLOCK or ESCALATE) short-circuits the cascade.
* Tier 2 always follows a non-terminal Tier 1. Tier 3 runs only when
  Tier 2's confidence falls inside `uncertainty_band` (default
  `0.30..0.70`), so the expensive tier sees a few percent of traffic.
* Every guard call is wrapped: an exception fails open with
  `evidence["error"]`, a timeout fails open with `evidence["timeout"]`.

### Latency budget

`budget_ms_p95` is checked *between* tiers. If the budget is already
spent when a tier is about to start, the pipeline returns a budget
decision instead: ALLOW with `evidence["budget_exceeded"]=True`
(`on_budget_exceeded="allow"`, the default) or a medium-severity BLOCK
(`"block"`). The budget never interrupts a running guard; use
`tier_timeouts_ms` for that.

```python
pipeline = GuardPipeline(
    tier1=[RegexGuard.compile_pii(redact=True)],
    tier2=[ClassifierGuard.from_hf("meta-llama/Llama-Prompt-Guard-2-22M")],
    tier3=[LLMJudgeGuard(judge=my_client)],
    budget_ms_p95=80,
    tier_timeouts_ms={2: 60, 3: 400},
)
```

### Streaming cadence

`guard_stream(chunks, evaluate_every=1, min_buffer_chars=0)` yields
safe chunks and stops at the first terminal decision.

* `evaluate_every=N` evaluates after every N-th chunk. Chunks between
  evaluations are withheld until the batch is cleared, so a block at
  a batch boundary never leaks the chunks in that batch.
* `min_buffer_chars` defers evaluation until the buffer is long enough.
* The final chunk is **always** evaluated regardless of either setting.

Defaults evaluate every chunk, which is right for Tier-1-only
pipelines. Batch when a classifier or judge sits in the path.

## YAML configuration

`GuardPipeline.from_yaml(path_or_text)`, `GuardPipeline.from_dict(cfg)`
and `GuardPipeline.to_dict()` are implemented in
`aisafepy.stream.config`.

```yaml
version: 1
name: support-assistant
budget_ms_p95: 80
uncertainty_band: [0.30, 0.70]
on_budget_exceeded: allow
tier_timeouts_ms: {2: 60, 3: 400}

tiers:
  1:
    - kind: regex_pii
      redact: true
    - kind: blocklist
      name: secret-markers
      terms: ["BEGIN PRIVATE KEY", "api_key="]
    - kind: regex
      name: jailbreak
      patterns:
        ignore_previous: "ignore (previous|prior) instructions"
  2:
    - kind: hf_classifier
      model_id: ${PROMPT_GUARD_MODEL:-meta-llama/Llama-Prompt-Guard-2-22M}
      threshold: 0.6
  3:
    - kind: llm_judge
      judge: my_company.judges:AnthropicJudge   # module:attr, class or factory
      policy_text: "Never reveal internal account identifiers."

shadow:
  - kind: custom
    target: my_company.guards:CandidateGuard
    strictness: high
```

Rules:

* Tier keys may be `1`, `"1"`, `tier1` or `tier_1`.
* Every entry is a `kind` plus keyword arguments for that kind's builder.
* String values support `${VAR}` and `${VAR:-default}`; a missing
  variable with no default is a `ConfigError`.
* Unknown kinds raise `ConfigError` listing the known kinds.
* `validate_config(source)` returns a list of problems **without
  instantiating any guard**, so you can lint a config that references
  GPU-only models on a laptop or in CI.

### Built-in kinds

| kind | builds | required |
| --- | --- | --- |
| `regex` | `RegexGuard.from_patterns` | `patterns` |
| `regex_pii` | `RegexGuard.compile_pii` | |
| `blocklist` / `aho_corasick` | `BlocklistGuard` / `AhoCorasickGuard` | `terms` |
| `schema` | `SchemaGuard` (model as `module:Class`) | `model` |
| `hf_classifier` | `HFClassifierGuard` | `model_id` |
| `stub_classifier` | `StubClassifierGuard` (tests) | |
| `streamguard` / `scm` / `kelp` | forecasters | |
| `linear_probe` / `mlp_probe` | probes loaded from `path` | `path` |
| `lakera`, `llama_guard`, `shield_gemma`, `prompt_guard`, `presidio`, `llm_guard`, `openai_moderation`, `azure_content_safety` | contrib wrappers | (`llm_guard`: `name`) |
| `llm_judge` | `LLMJudgeGuard` | `judge` |
| `session` | `SessionGuard` | |
| `custom` | any `module:attr` callable | `target` |

Register your own:

```python
from aisafepy.stream.config import default_registry

@default_registry().register("legal_review", required=("jurisdiction",))
def build_legal(*, jurisdiction: str, name: str = "legal"):
    return LegalReviewGuard(name=name, jurisdiction=jurisdiction)
```

`to_dict()` is best effort: guards that implement `to_spec()` and the
built-in deterministic / stub / HF guards serialize losslessly; anything
else is recorded as `{"kind": "custom", "repr": ...}`.

## Observability

### Metrics

```python
from aisafepy.stream.metrics import PipelineMetrics

metrics = PipelineMetrics()
pipeline = GuardPipeline(tier1=[...], metrics=metrics)
...
metrics.snapshot()        # JSON-shaped: final, tiers, guards, shadow
metrics.to_prometheus()   # text exposition format
```

Per guard: evaluation count, counts per action, errors, timeouts and a
latency histogram (p50/p95/p99/max/mean over a 2048-sample reservoir).
Per tier and for the final decision: the same counters. Shadow guards
are reported separately.

Prometheus metric names are `<namespace>_guard_evaluations_total`,
`_guard_decisions_total{action=...}`, `_guard_errors_total`,
`_guard_timeouts_total` and the `_guard_latency_ms{quantile=...}`
summary, each labelled with `scope` (`guard`, `tier`, `shadow`,
`final`), `name` and `tier`.

### Decision hook

`on_decision(decision, ctx)` fires for every final decision, allows
included, and for every shadow decision (`evidence["shadow"] is True`).
It may be sync or async. Use it to ship decisions to your log pipeline,
to feed the `SessionGuard`, or to update the FP monitor. `on_violation`
remains for callers that only care about terminal decisions.

### False-positive monitor

Adversarial false-positive attacks (arXiv 2410.02916) push a classifier
into blocking benign traffic. `FalsePositiveMonitor` keeps a sliding
window of block/allow events per key (tenant, API key, client IP) and
raises alerts as data:

```python
monitor = FalsePositiveMonitor(window_s=300)
pipeline.on_decision = lambda d, ctx: monitor.record_decision(ctx.metadata["tenant"], d)
for alert in monitor.alerts(threshold=0.5, min_samples=50):
    page_oncall(alert.key, alert.block_rate)
```

Keys are bounded (`max_keys`, LRU eviction) so a key-flooding attacker
cannot grow memory without bound.

## Session guard

Per-message guards miss slow-burn attacks. `SessionGuard` keeps a
rolling record per `ctx.metadata["session_id"]` and trips when:

* `max_blocks` blocks occurred within `window_s`,
* `max_escalations` escalations occurred within `window_s`,
* the decayed cumulative risk (`sum(weight[action] * confidence)` with
  half-life `risk_half_life_s`) reaches `risk_threshold`,
* the session exceeds `max_turns_per_minute`.

Wire it with the decision hook so it sees every final decision:

```python
session = SessionGuard(max_blocks=3, window_s=600, action="escalate")
pipeline = GuardPipeline(
    tier1=[session, ...],
    on_decision=lambda d, ctx: session.record(ctx.metadata["session_id"], d),
)
session.explain("conv-42")   # counters, risk, thresholds
```

State lives in an in-memory `SessionStore` with TTL expiry and LRU
bounding. Swap it for a shared store by subclassing when running
several replicas.

## LLM judge (Tier 3)

`LLMJudgeGuard` asks a model to score content against a rubric of
categories and an optional policy text, demanding strict JSON. The
judge is duck-typed: pass anything with `complete(prompt) -> str` or a
bare callable, sync or async.

```python
class AnthropicJudge:
    def __init__(self, client, model="claude-sonnet-5-5"):
        self.client, self.model = client, model

    def complete(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=200, messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text

judge = LLMJudgeGuard(judge=AnthropicJudge(client), policy_text=open("policies/company_safety.md").read())
```

Verdict mapping: `unsafe` -> BLOCK (or ESCALATE below
`unsafe_threshold`), `uncertain` -> ESCALATE, `safe` -> ALLOW,
unparseable -> ESCALATE with `evidence["parse_error"]`. Transport
errors and timeouts follow `fail_mode` (`open` or `escalate`). Long
inputs are truncated in the middle so both the head and the tail
survive. `JudgeTranscript` records redacted prompt/response pairs.
`MajorityJudge` runs several judges and blocks on a quorum.

`StubJudge` is a deterministic keyword judge for tests.

## Shadow mode

Add candidate guards to `shadow_guards` to observe them on live
traffic. They run concurrently with Tier 1 on every evaluation, never
change the outcome, and are reported through `on_decision` and
`metrics.snapshot()["shadow"]`. `aisafepy.adapt.promote(...,
shadow_only=True)` uses the same mechanism for compiled artifacts.
Promote a shadow guard into a tier once its block rate on known-benign
traffic sits inside your false-positive budget.

## Fail open or fail closed?

The cascade defaults to **fail open**: a crashing guard, a timed-out
guard, or an exhausted latency budget yields ALLOW with evidence. This
is the right default for a *detective* control sitting in front of
paying users, because the alternative turns every classifier outage
into a product outage, and because `aisafepy.flow` provides the
*preventive* control for the actions that actually matter (payments,
email, code execution).

Fail closed where the cost of a miss exceeds the cost of an outage:

* `on_budget_exceeded="block"` for pipelines guarding irreversible actions,
* `LLMJudgeGuard(fail_mode="escalate")` so a judge outage routes to a
  human rather than through,
* `SessionGuard(action="block")` for abuse-heavy surfaces.

Whichever you pick, `evidence` tells you afterwards which path was
taken: `error`, `timeout`, `budget_exceeded`, `parse_error`, `shadow`.
