# Caveats and limits

AIsafePy reduces risk in measurable, well-scoped ways. It does not provide
"provable security." This document records the caveats that should temper
how the library is used, marketed, and integrated.

## flow (IFC)

- **CaMeL-style IFC depends on the assumption that the privileged LLM (P-LLM)
  emits only the constrained action DSL.** If the planner escapes that
  sandbox (e.g., is jailbroken into emitting raw natural-language reasoning
  that the host code then evaluates), guarantees degrade.
- CaMeL reports ~2.7× token overhead. RTBAS reports ~2% utility loss with
  human-approval mediation. Plan for both.
- *Are Firewalls All You Need?* (arXiv 2510.05244) shows even system-level
  defenses can be bypassed by obfuscation or Braille payloads.
  `flow` reduces attack surface significantly but never eliminates it.
- The MCP tool-poisoning defense relies on a hashed tool manifest. If you
  do not pin manifests, drift checks are inert.

## stream (cascade + probes)

- Forecasting-based streaming detectors (StreamGuard / SCM / Kelp) require
  either MC rollouts at training time or token-level supervision. The
  v0.1 release ships pre-trained-model wrappers and a recipe, not a fully
  automated training pipeline.
- Activation probes are model-specific. A probe trained on Llama-3-8B
  does not transfer to Qwen3-8B. Calibrate per backbone.
- Adversarial false-positive attacks (arXiv 2410.02916) can drive a
  classifier to block >97% of benign traffic. `stream` ships DoS defenses
  (input length caps, regex catastrophe protection, per-IP FP monitoring),
  but they require operational tuning.

## adapt (eval-to-guardrail)

- Steering-vector deployment works for self-hosted models only. For
  hosted APIs (OpenAI, Anthropic, Gemini) the compiler falls back to
  classifier + regex + deliberative targets.
- The cluster summarizer step uses an LLM-as-judge for labeling. Cluster
  labels are descriptive, not authoritative.
- Canary promotion checks the FP rate against a *labeled* benign baseline.
  If your baseline is unrepresentative, the rollback signal is unreliable.

## Out of scope

- Sleeper-agent / deceptive-alignment detection. Anthropic's *Simple
  probes catch sleeper agents* is reproducible on toy backdoors and
  supports `stream.probes`, but no current technique reliably detects
  deception in frontier models.
- Compliance certification (OWASP, NIST AI RMF, EU AI Act). AIsafePy
  emits audit-log artifacts that *help* a compliance team. Certification
  belongs to a separate process.
- Hosted SaaS. The library is fully self-hostable. Managed canary /
  governance dashboards are not part of the OSS roadmap.
