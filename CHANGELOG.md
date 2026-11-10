# Changelog

All notable changes to AIsafePy are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This cycle broadens AIsafePy from a library into a toolkit without
changing the three-module design. Areas added:

### Added

- **Command-line interface** (`aisafepy`). Subcommands for scanning
  files and stdin with Tier-1 guards (`scan`), evaluating policy
  documents (`policy`), verifying and summarising audit logs (`audit`),
  pinning and verifying MCP tool manifests (`manifest`), compiling
  red-team failures into guard artifacts offline (`compile`), and
  analysing `GuardDecision` logs (`decisions`). Every command returns
  explicit exit codes (0 ok, 1 error, 2 flagged).
- **Offline benchmark suite** under `benchmarks/`. A labelled
  prompt-injection dataset of 160+ records across nine attack families
  and hard negatives, plus latency, detection-quality, flow-overhead,
  and budget-throughput benchmarks that run without a GPU.
- **Project documentation**: architecture, threat model, observability
  reference, CLI reference, and an FAQ, along with contributor and
  conduct guidelines and issue and pull-request templates.
- **New `stream` guards** extending the Tier-1 and Tier-2 surface.
- **`flow` additions**: a restricted planner action DSL and loading of
  IFC policies from YAML.
- **`adapt` additions**: a canary rollout controller and offline
  evaluation of compiled artifacts.
- **`core` additions**: output sink primitives and budget helpers shared
  across modules.
- **Typing**: ship the `py.typed` marker so downstream type checkers see
  the inline annotations the package already carries.

### Notes

- Items beyond the CLI, benchmarks, and documentation are delivered
  across several workstreams in this cycle; see the module docstrings
  and `docs/` for the authoritative description of each.

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
