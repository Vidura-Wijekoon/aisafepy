# FAQ

### Does a guard fail open or fail closed?

Guards fail open by default. If a guard raises or a dependency is
missing, the pipeline logs a `guard.error` event and returns an allow
with an `evidence` flag, rather than blocking legitimate traffic on an
internal fault. You can invert this per pipeline: set
`GuardPipeline(on_budget_exceeded="block")` to fail closed when the
latency budget is blown, and a guard you write can return a block on its
own errors if your risk posture calls for it. `flow` is the opposite by
design: an IFC violation denies the action.

### What exactly does the latency budget do?

`GuardPipeline(budget_ms_p95=80)` sets a per-evaluation deadline. The
pipeline checks the clock between tiers; if the deadline has passed
before an expensive tier runs, it emits a `budget_exceeded` evidence
flag and either allows (default) or blocks, depending on
`on_budget_exceeded`. It bounds added latency; it does not interrupt a
guard already running.

### Why no Colang or a bespoke DSL?

Design principle one: Pythonic, not DSL-first. Policies are Python
objects and decorators, which compose with your code, type-check, and
need no new language. Cedar and OPA Rego appear only as emission targets
inside `adapt.compile.policy`, as machine artifacts for governance
systems, never as something you author by hand.

### How does taint survive string operations?

`Tainted[T]` propagates labels through the operations that matter.
Concatenation (`+`), slicing (`[]`), iteration, and `map` all return a
new `Tainted` whose provenance and capabilities are the union of the
inputs and whose integrity is the worst (lowest-trust) of them. So
`greeting + untrusted_name` is UNTRUSTED even though `greeting` was
TRUSTED. The lattice is three integrity levels and a flat capability
set, which is enough to express the CaMeL / FIDES / RTBAS patterns.

### What does `lower()` do, and when should I use it?

`lower(t)` strips all taint and returns the bare value. It is the unsafe
escape hatch: it bypasses IFC entirely. Use it only at a
declassification boundary your policy explicitly approved, for example
after a structured, P-LLM-only summarisation step you recorded with
`Policy.declassify`. If you find yourself reaching for `lower` to make a
tool call pass, that is usually the sign the data flow is the thing to
fix.

### What is the overhead of the `flow` IFC checks?

Per allowed `evaluate_call`, the interpreter adds tens of microseconds
over a bare call, rising modestly with the number of deny rules; run
`benchmarks/bench_flow_overhead.py` on your hardware for exact numbers.
At the model level, CaMeL-style designs report roughly 2.7x token
overhead and RTBAS reports about 2% utility loss with human-approval
mediation. Plan for both; see `docs/CAVEATS.md`.

### Does `stream` really only run Tier 3 on a few percent of traffic?

That is the intent of the cascade: Tier 1 is deterministic and cheap,
Tier 2 is a small classifier, and Tier 3 (a judge or white-box probe)
runs only when Tier 2's confidence falls inside the uncertainty band.
The band is configurable via `uncertainty_band`. How small the Tier-3
fraction actually is depends on your traffic and thresholds.

### How good is Tier 1 on its own?

Deliberately shallow. `benchmarks/bench_detection.py` measures it: high
precision, low recall, and near-zero recall on encoded and obfuscated
families. That is the argument for the Tier-2 classifier, and the
benchmark lets you quantify the lift a specific classifier adds with
`--guard`.

### Do probes and steering vectors work with hosted APIs?

No. White-box activation probes and steering vectors need the model's
hidden states and residual stream, which hosted APIs do not expose. They
work on self-hosted Hugging Face models. For hosted APIs, the compiler
and cascade fall back to classifier, regex, and deliberative targets.
This is the self-hosted-parity principle: the architectural and
black-box detective layers work everywhere; the white-box layers are a
self-hosted bonus.

### Is a probe trained on one model reusable on another?

No. Activation probes are model-specific; a probe trained on Llama-3-8B
does not transfer to Qwen3-8B. Calibrate per backbone.

### What does AIsafePy not protect against?

Deceptive alignment and sleeper agents in frontier models, compliance
certification (it produces artifacts that help, not a certificate), and
a compromised host process. See `docs/threat-model.md` and
`docs/CAVEATS.md` for the full list.

### How are versions and compatibility handled?

The project aims to follow semantic versioning. The current line is
0.1.x (alpha): the API surface is stable enough to build against, but
minor pre-1.0 releases may make breaking changes, which the
`CHANGELOG.md` will call out. Only the latest minor line receives
security patches; see `SECURITY.md`.

### Which Python versions are supported?

Python 3.10, 3.11, and 3.12. CI runs the suite on all three.

### Does it run on Windows?

Yes. The library, the CLI, and the offline benchmarks run on Windows,
macOS, and Linux. The audit log uses `msvcrt` locking on Windows and
`fcntl` elsewhere, and CLI output is kept ASCII-safe for the Windows
console code page. Some optional ML extras have their own platform
constraints from upstream.

### Can I use my existing guardrail model?

Yes, that is the bring-your-own-model principle. AIsafePy ships no
proprietary models. `contrib/` wraps Llama Guard 4, ShieldGemma, Prompt
Guard 2, llm-guard, Presidio, and Lakera, and `HFClassifierGuard.from_hf`
wraps any Hugging Face sequence classifier behind the Guard protocol.

### How do I turn red-team findings into guards?

Point `aisafepy.adapt` at a PyRIT, Garak, Inspect, or Langfuse source,
compile, and promote behind a false-positive budget. For a quick offline
pass without writing code, use `aisafepy compile --source ... --target
regex --target policy`. See `docs/architecture.md` for the compile loop.

### Where do I report a security issue?

Not in a public issue. Follow the process in `SECURITY.md`.
