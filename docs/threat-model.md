# Threat model

This document states what AIsafePy defends, against whom, and where each
module helps or does not. It is deliberately specific about limits:
capability-based controls reduce risk sharply but never deliver provable
security, and the detective layers are probabilistic. Read
`docs/CAVEATS.md` alongside this.

## Assets

- **User and organisation secrets**: API keys, credentials, PII, private
  documents the agent can read.
- **Privileged actions**: sending email, moving money, writing to
  external systems, executing code, deleting records.
- **The model's instructions**: the system prompt and tool definitions.
- **Service availability**: the agent's ability to keep serving
  legitimate traffic at acceptable cost and latency.
- **The audit trail**: the record of what guards decided and why.

## Adversaries

- **Indirect attacker**: cannot talk to the agent directly but controls
  content the agent ingests (a web page, an email, a document, a
  retrieval hit, an MCP tool's output).
- **Direct attacker**: a malicious or careless user crafting prompts.
- **Malicious tool or supply chain**: a poisoned MCP server or a tool
  whose description smuggles instructions.
- **Adversarial evader**: an attacker tuning inputs to defeat a
  classifier, in either direction (slip an attack through, or drive
  benign traffic to be blocked).

## Attack surfaces and mitigations

The mapping below uses the OWASP Top 10 for LLM Applications (2025).

### Indirect prompt injection (LLM01: Prompt Injection)

Untrusted content tells the agent to do something the user did not ask
for, such as exfiltrate data or take a privileged action.

- **`flow` mitigates** by construction. Untrusted data carries an
  UNTRUSTED integrity label and a provenance set; a tool that requires
  TRUSTED control-flow integrity or forbids certain capability-tagged
  arguments is denied when the labels do not meet the requirement. The
  attack does not need to be recognised as malicious; the data flow is
  simply not permitted.
- **`stream` mitigates** detectively, catching known injection phrasing
  at Tier 1 and Tier 2 before it reaches the model or the user.
- **Does not fully solve**: if the privileged planner is itself
  jailbroken into emitting raw actions outside the constrained path,
  guarantees degrade. Obfuscated or encoded payloads can evade Tier-1
  literal guards; that gap is why Tier 2 exists and is measured in
  `benchmarks/bench_detection.py`.

### Tool poisoning and schema drift (LLM01; LLM03: Supply Chain)

A malicious MCP tool hides instructions in its description, or a tool's
schema changes silently after it was reviewed.

- **`flow.adapters.mcp` mitigates** by screening every tool description
  for hidden-instruction patterns at registration time and by pinning a
  hash of each tool's schema. Drift against the pinned manifest is
  denied. The `aisafepy manifest` CLI makes pinning and verification a
  one-liner.
- **Does not solve**: if you never pin a manifest, drift checks are
  inert. Screening is pattern-based and can miss novel phrasings.

### Data exfiltration (LLM02: Sensitive Information Disclosure)

Secrets or PII leave through a tool argument, a markdown image URL, or
the model's output.

- **`flow` mitigates** by denying tool calls whose arguments carry a
  `read.secrets` provenance into an external-write sink.
- **`stream` mitigates** with PII and secret regex guards that block or
  redact, and with the markdown-image exfiltration patterns in the
  detection benchmark.
- **`core` telemetry mitigates** log-side leakage by redacting sensitive
  keys and long strings before `evidence` reaches a trace backend.

### System prompt disclosure (LLM07: System Prompt Leakage)

An attacker asks the model to repeat its instructions or tool
definitions.

- **`stream` mitigates** at Tier 1 and Tier 2 with exfiltration-phrasing
  guards.
- **Does not solve**: a sufficiently novel phrasing can evade literal
  guards; treat the system prompt as discoverable and keep secrets out
  of it.

### Resource exhaustion and loops (LLM10: Unbounded Consumption)

An agent loops, retries forever, or is driven to burn tokens and money.

- **`core.budgets` mitigates** with iteration, token, dollar, and
  wall-clock caps that raise `BudgetExceeded` and emit checkpoint logs.
- **`core.progress` mitigates** by detecting duplicate tool calls,
  cycles, stagnation, and semantic no-progress, and halting the agent.
- **`stream` mitigates** input-length denial of service with per-guard
  input caps.

### Adversarial false positives (LLM01; availability)

An attacker crafts benign-looking traffic that a classifier blocks,
denying service to real users.

- **`stream` mitigates** with input caps and a cascade that gates the
  expensive, more error-prone tiers behind cheap deterministic ones.
- **`adapt` mitigates** by promoting new guards behind a false-positive
  budget and rolling back automatically when a labelled benign baseline
  shows the budget breached.
- **Does not solve**: false-positive attacks (arXiv 2410.02916) can be
  severe and need operational tuning per deployment.

### Guardrail log leakage (LLM02; operational)

Guardrail logs capture the very secrets the guard saw.

- **`core.telemetry` mitigates** by redacting sensitive keys and
  truncating long strings in `evidence` and `rationale` before they
  reach a span. This addresses the class of incident where guardrail
  logs captured authorization headers.

## Out of scope

- **Deceptive alignment and sleeper agents.** Activation probes support
  detection on toy backdoors but no current technique reliably detects
  deception in frontier models.
- **Compliance certification.** AIsafePy emits audit artifacts that help
  a compliance process; certification is a separate activity.
- **Protecting against a compromised host.** If the process running the
  agent is itself compromised, in-process guards cannot help.

## Trust boundaries in one line

User input is trusted only as far as the policy says; model output is
untrusted; retrieved and tool-returned content is untrusted; the audit
log is append-only and verifiable; secrets never enter the system prompt
or the logs.
