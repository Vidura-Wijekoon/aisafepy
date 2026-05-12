# Security Policy

AIsafePy is a security-adjacent library. If you find a vulnerability,
please disclose it responsibly using the process below.

## Reporting a vulnerability

**Do not file a public GitHub issue.** Instead, email
`businessaividura@viduraaitech.space` with:

- A clear description of the vulnerability.
- Steps to reproduce, including a minimal proof-of-concept if possible.
- The version of AIsafePy you tested against (`pip show aisafepy`).
- The Python version and OS.
- Your name and a way to credit you, if you wish.

You should receive an acknowledgement within 5 business days. We aim
to publish a coordinated patch within 30 days for critical and high
severity issues, 90 days for medium and low.

## Supported versions

Only the latest minor release line of AIsafePy receives security
patches. Older releases are end-of-life on publication of a newer
minor version.

| Version | Supported |
|---------|-----------|
| 0.1.x   | yes       |
| < 0.1   | no        |

## Threat model summary

AIsafePy is defense-in-depth tooling. It assumes:

- The host process is trusted. An attacker who can modify
  AIsafePy bytecode at runtime bypasses every guarantee.
- The policy author is trusted. A misconfigured `Policy` (e.g. a
  `deny_if` predicate that always returns False, or a source labelled
  TRUSTED when it should be UNTRUSTED) will let attacks through.
- Probe and steering checkpoints are loaded from a trusted source.
  AIsafePy uses `torch.load(weights_only=True)` to mitigate pickle
  code execution, but a malicious model file may still produce
  incorrect predictions.

Out of scope for the threat model:

- Sleeper-agent / deceptive-alignment detection in frontier models.
- Side-channel attacks against the model weights themselves.
- Compliance certification (EU AI Act, NIST AI RMF). The library
  provides audit-log artifacts but certification is a separate
  process.

See `docs/CAVEATS.md` for the operational caveats that complement
this threat model.

## Hardening checklist for production deployments

If you are running AIsafePy in production:

1. Pin dependency versions in your lockfile. Re-pin only after a
   security review of the diff.
2. Run `pip-audit` weekly against your environment.
3. Configure an OpenTelemetry exporter and forward
   `aisafepy.flow.*` / `aisafepy.guard.*` spans to a SIEM. The
   library redacts known-sensitive keys (`api_key`, `authorization`,
   etc.) but you should review your own evidence dicts.
4. Use `Policy.with_mode("mediated")` for high-risk tools so a
   human approves every borderline call.
5. Run benchmarks against AgentDojo before and after a policy
   change to catch regressions.
6. Keep `aisafepy.adapt.canary` rollouts at `<= 1%` until the false
   positive rate has stabilized over at least 1,000 production
   requests.

## Acknowledgements

Thanks to the researchers behind CaMeL, FIDES, RTBAS, SAFEFLOW,
Constitutional Classifiers, and the Anthropic Alignment Science Team
for the techniques this library packages.
