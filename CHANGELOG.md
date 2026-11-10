# Changelog

All notable changes to AIsafePy are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This cycle broadens AIsafePy from a library into a toolkit without
changing the three-module design.

### Added

**`flow`**

- Declarative policies: `Policy.from_yaml` / `to_yaml` / `from_dict`, a
  predicate vocabulary for `deny_if` (`provenance_contains`,
  `capability_present`, `arg_integrity_below`, `value_matches`, `arg`
  scoping, `any_of` / `all_of` / `not`) that round-trips through YAML.
- `run_plan`, the restricted CaMeL-style evaluator for planner-emitted
  action lists (`$ref`, `$get`, `$concat`, `$format`, `$lit`, optional
  per-step `if`), with control-flow integrity computed from the labels
  each step depends on and a `PlanTrace` for the audit log.
- `lint_policy` with findings for unused sources, uncovered sinks,
  permissive sink requirements, weak declassifier justifications and
  deny rules that raise.
- `secure_callables` and `ToolRegistry`, a framework-free adapter for
  plain Python tools; `secure_agent(..., framework="generic")`.
- `DecisionRecorder` and `IFCContext.on_decision` so every allow,
  violation and mediated approval can be recorded (ring buffer, JSONL,
  callback) and flushed by `ifc_run`.
- MCP: `build_manifest`, `diff_manifest`, JSON manifest helpers,
  screening of parameter descriptions inside input schemas, and async
  handler support.
- `Tainted` helpers: `collect`, `map_values`, `tainted_join`,
  `labels_of`, `is_tainted`, repetition and truthiness operators.

**`stream`**

- Tier-1 guards: `SecretsGuard`, `EncodingGuard`, `ToolCallGuard` (with
  `URLPolicy`, `PathPolicy`, `ShellPolicy`), `InjectionHeuristicGuard`,
  `LengthGuard`, `RepetitionGuard`, `ScriptMixGuard`, `CharsetGuard`,
  `URLGuard`, `MarkdownImageExfilGuard`, `JSONDepthGuard`, and the
  offset-preserving `Normalizer` they share.
- YAML pipeline configuration: `GuardPipeline.from_yaml` / `from_dict`
  / `to_dict`, `load_pipeline`, `validate_config`, `GuardRegistry` with
  builders for every shipped guard and custom dotted-path targets,
  environment-variable interpolation.
- Pipeline features: shadow guards, `on_decision` hook, per-tier
  timeouts, `describe()`, `evaluate_text()`, batched streaming via
  `evaluate_every` / `min_buffer_chars`.
- Observability: `PipelineMetrics` with exact percentiles and
  Prometheus exposition, `FalsePositiveMonitor` keyed by tenant or
  client.
- `SessionGuard`, `SessionStore` and `SessionState` for multi-turn risk
  with exponential decay and per-session rate limits.
- `LLMJudgeGuard`, `StubJudge`, `MajorityJudge` and `JudgeTranscript`
  for a Tier-3 LLM-as-judge with robust verdict parsing.
- Adapters: `openai_chat` (streaming and non-streaming Chat
  Completions), `anthropic` (Messages event streams and objects), and
  `asgi.GuardrailMiddleware` for any ASGI 3 framework.
- `RegexGuard` validators (Luhn, IPv4 octets, key-material entropy),
  all-occurrence redaction with overlap merging, `scope`, escalate
  action, per-pattern match caps; `AhoCorasickGuard` whole-word mode
  and offsets; `SchemaGuard` partial mode implemented.

**`adapt`**

- Sources: `JSONLSource`, `CSVSource`, `PromptfooSource`,
  `DeepEvalSource`, `InMemorySource`, `merge_sources`, `dedupe_records`,
  `RecordFilter`.
- Regex target precision against a negative pool, with recall and
  support, generalisation of digit runs, dominance collapse, and
  `safe_regex` / `timed_search` to reject catastrophic patterns.
- `GuardCompiler(negatives=, max_clusters=, hooks=, dry_run=)`;
  `CompilationReport.save` / `load` / `to_json` / `to_markdown` /
  `filter`.
- `CanaryController` with deterministic traffic routing, false-positive
  accounting against labelled benign traffic, automatic rollback over
  the budget, shadow-to-live promotion and persisted state;
  `ShadowGuard`.
- `evaluate_guard`, `threshold_sweep`, `best_threshold` and
  `EvaluationReport` for offline measurement.
- `AuditLog`: `verify_detailed`, optional HMAC signing, `tail`,
  `iter_entries`, `export_csv`, `append_decision`, and the in-lock
  chain re-read the original comment promised.

**`core`**

- Budgets: `BudgetSet`, `budgeted` decorator, `RateBudget`,
  `TokenPricing` / `ModelPrice` and `DollarBudget.charge`,
  `on_checkpoint` callbacks.
- Progress: `max_steps`, canonicalisation options, `observe_text` with
  Jaccard shingles, `explain`, `reset`.
- Policy documents: `regex_any`, nested boolean combinators,
  `field_in`, length bounds, role scoping, `to_yaml`, `evaluate_all`,
  `explain`, `as_guard`.
- `sinks`: `DecisionSink` protocol with JSONL, in-memory, callback,
  multi and filter sinks and a versioned record schema.
- Telemetry: value-based credential redaction with typed placeholders,
  `configure_redaction`, redaction of `structured_log` fields.

**`contrib`**: `OpenAIModerationGuard`, `AzureContentSafetyGuard`.

**Tooling and project**

- The `aisafepy` command-line interface (`scan`, `policy`, `audit`,
  `manifest`, `compile`, `decisions`) and `python -m aisafepy`.
- An offline benchmark suite (`benchmarks/`) with a 178-record
  labelled prompt-injection dataset.
- Guides for every module plus architecture, threat model,
  observability, integrations, CLI and FAQ; contributor and conduct
  guidelines; issue and pull-request templates.
- Tests for the framework adapters, contrib wrappers, model-backed
  guards and compiler targets using fakes, and a test that executes
  every offline example; CI runs the examples on each Python version
  and isolates the model-download test.
- `.gitattributes` for LF line endings, the `py.typed` marker, and a
  `yaml` extra for PyYAML.

### Fixed

- `RegexGuard` redaction masked only the first occurrence of each
  pattern.
- `contrib.lakera` treated any non-zero category score as a hit.
- `adapt.compile.steering` recorded an empty negatives list in its
  deferred training spec when it had fallen back to self-contrast.
- The MCP description screen missed "ignore all previous instructions".
- `AuditLog` on Windows locked byte 0, which blocked concurrent reads.

## [0.1.0] - 2026-05-11

Initial release.

### Added

- **`aisafepy.core`**: the `GuardDecision` and `IFCViolation` decision
  types, the `PolicyDocument` runtime policy loader, iteration / token /
  dollar / wall-clock budgets, agent loop and no-progress detection, and
  OpenTelemetry-native telemetry with secret redaction.
- **`aisafepy.flow`**: capability-based information-flow control with
  `Tainted[T]` label propagation, a fluent `Policy` builder, the
  `evaluate_call` interpreter, and adapters for the OpenAI Agents SDK,
  LangGraph, LlamaIndex, Anthropic tools, and MCP servers (with tool
  description screening and manifest pinning).
- **`aisafepy.stream`**: a streaming-native three-tier cascade with
  deterministic Tier-1 guards (regex, Aho-Corasick, schema), Hugging
  Face classifier wrappers, streaming forecasters, and white-box
  activation probes, under an explicit p95 latency budget.
- **`aisafepy.adapt`**: an eval-to-guardrail compiler that normalises
  PyRIT, Garak, Inspect, and Langfuse sources into failure records,
  clusters them, and compiles classifier, regex, Cedar/OPA policy,
  steering-vector, and deliberative-case targets, with a hash-chained
  audit log and canary promotion.
- **`aisafepy.contrib`**: thin Guard wrappers for Llama Guard 4,
  ShieldGemma, Prompt Guard 2, llm-guard, Presidio, and Lakera.
- Apache-2.0 licensing, a hardened CI and publish pipeline (pinned
  action SHAs, least-privilege tokens, egress auditing), and a security
  policy.

[Unreleased]: https://github.com/Vidura-Wijekoon/aisafepy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Vidura-Wijekoon/aisafepy/releases/tag/v0.1.0
