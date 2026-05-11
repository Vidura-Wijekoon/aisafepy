# AIsafePy

**Capability-based information-flow control, streaming-native cascaded guardrails, and a continuous eval-to-guardrail compiler for LLM agents.**

AIsafePy fills three gaps that the existing OSS guardrails ecosystem (NeMo, Guardrails AI, llm-guard, LlamaFirewall, OpenAI Guardrails) has not closed:

1. **`aisafepy.flow`** — capability-based, taint-propagating runtime around tool-calling agents (CaMeL / FIDES / RTBAS-style information-flow control), packaged as drop-in adapters for OpenAI Agents SDK, LangGraph, LlamaIndex, Anthropic tools, and MCP servers.
2. **`aisafepy.stream`** — streaming-native cascaded guardrails with deterministic Tier-1, small-classifier Tier-2, and optional white-box activation probes / LLM-judge Tier-3, plus an explicit p95 latency budget and structured `GuardDecision`s.
3. **`aisafepy.adapt`** — a continuous eval-to-guardrail compiler that promotes PyRIT / Garak / Inspect failures into runtime guards: distilled classifiers, synthesized regexes, Cedar/OPA policy rules, steering vectors (for self-hosted models), and deliberative cases.

## Status

**Alpha (v0.1).** API surface is stable enough to build against, but expect rough edges and missing optional dependencies in the heavier extras.

## Install

```bash
pip install aisafepy                       # core only
pip install "aisafepy[stream]"             # + HF classifiers, regex, deterministic Tier 1/2
pip install "aisafepy[probes]"             # + linear/MLP activation probes for HF models
pip install "aisafepy[adapt]"              # + clustering and compiler targets
pip install "aisafepy[flow-openai]"        # + OpenAI Agents SDK adapter
pip install "aisafepy[all]"                # everything except contrib-* extras
```

For development:

```bash
uv venv
uv pip install -e ".[dev,all]"
uv run pytest
```

## Quickstart

### `flow`: defeating indirect prompt injection by construction

```python
from aisafepy.flow import Policy, Capability, secure_agent, Tainted
from agents import Agent, Runner  # openai-agents

policy = (
    Policy()
    .label_source("web.fetch", integrity="UNTRUSTED")
    .label_source("gmail.read", integrity="UNTRUSTED", caps={Capability.READ_USER})
    .label_source("user_prompt", integrity="TRUSTED")
    .require("send_email", control_flow_integrity="TRUSTED")
    .require("payments.transfer", control_flow_integrity="TRUSTED",
             caps={Capability.WRITE_EXTERNAL})
    .deny_if("send_email",
             when=lambda to, body: "read.secrets" in body.provenance,
             reason="secret-to-external-sink")
)

agent = Agent(name="ops-bot", tools=[gmail_read, web_fetch, send_email, transfer])
safe_agent = secure_agent(agent, policy=policy)
result = Runner.run_sync(safe_agent, "Read my last email and act on it.")
```

### `stream`: cascaded guardrails with a latency budget

```python
from aisafepy.stream import (
    GuardPipeline, RegexGuard, ClassifierGuard, probes,
)

pipeline = GuardPipeline(
    tier1=[
        RegexGuard.compile_pii(),
        RegexGuard.blocklist(["api_key=", "BEGIN PRIVATE"]),
    ],
    tier2=[ClassifierGuard.from_hf("meta-llama/Llama-Prompt-Guard-2-22M")],
    tier3=[ClassifierGuard.from_hf("meta-llama/Llama-Guard-4")],
    budget_ms_p95=80,
)

async for chunk_or_decision in pipeline.guard_stream(model.generate_stream(prompt)):
    if hasattr(chunk_or_decision, "action"):
        log_otel(chunk_or_decision)
        break
    yield chunk_or_decision
```

### `adapt`: PyRIT failures → deployed guard pipeline

```python
from aisafepy.adapt import PyRITSource, GuardCompiler, Target, promote
from aisafepy.stream import GuardPipeline

source = PyRITSource(memory_db="pyrit_memory.duckdb")
compiler = GuardCompiler(
    source=source,
    targets=[
        Target.distill_classifier(base="meta-llama/Llama-Prompt-Guard-2-22M"),
        Target.synthesize_regex(min_precision=0.99),
        Target.steering_vector(model="Qwen/Qwen3-8B-Instruct"),
        Target.deliberative_case(policy="policies/company_safety.md"),
    ],
    min_attack_success_rate=0.05,
)
report = compiler.compile()
promote(report, to=GuardPipeline.from_yaml("guards.yaml"),
        canary_traffic_pct=1.0, fp_budget=0.005)
```

## Layout

```
src/aisafepy/
├── core/           # shared primitives: GuardDecision, telemetry, budgets, progress, policies
├── flow/           # Gap 1 — capability-based IFC
│   └── adapters/   # openai_agents, langgraph, llamaindex, anthropic_tools, mcp
├── stream/         # Gap 2 — streaming cascade
│   └── adapters/   # openai_agents, langchain, llamaindex
├── adapt/          # Gap 3 — eval-to-guardrail compiler
│   └── compile/    # classifier, regex, policy, steering, deliberative
└── contrib/        # thin wrappers: presidio, llama_guard, shield_gemma, prompt_guard, llm_guard, lakera
```

## Design principles

1. **Pythonic, not DSL-first.** Decorators and types, not Colang. Cedar / OPA appears only as an emission target inside `adapt.compile.policy`.
2. **Composable primitives.** Every guard is a `Callable[[Context], Awaitable[GuardDecision]]`. Pipelines, IFC, and `adapt` all consume and produce this type.
3. **Bring your own model.** No proprietary models are shipped. `contrib/` wraps Llama Guard 4, ShieldGemma, Prompt Guard 2, llm-guard, Lakera, Presidio.
4. **Defense in depth.** `flow` (architectural) + `stream` (detective) + `adapt` (continuous) compose.
5. **Observability is a first-class output.** Structured `GuardDecision` / `IFCViolation`, OpenTelemetry-native, with explicit `why_blocked` + `evidence`.
6. **Self-hosted parity.** Probe-based and steering-based features work on HF Transformers; hosted APIs fall back to classifier guards.

## Caveats

Capability-based defenses reduce risk dramatically but are not free — CaMeL reports ~2.7× tokens, RTBAS ~2% utility loss. Streaming forecasters require MC rollouts or token-level supervision to train. Activation probes are model-specific. AIsafePy does not solve sleeper-agent / deceptive-alignment problems. See `docs/CAVEATS.md`.

## License

Apache-2.0. See `LICENSE`.
