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


## Supply chain integrity

AIsafePy treats the path from source to installed package as part of
its security perimeter. The May 2026 "Mini Shai-Hulud" worm
(CVE-2026-45321, CVSS 9.6) compromised 172 npm and PyPI packages by
hijacking GitHub Actions OIDC tokens at publication time. The hardening
below makes AIsafePy resistant to the same attack class.

### How we publish

- The publish workflow (`.github/workflows/publish.yml`) is triggered
  only on a **published GitHub Release**, never on a push or PR.
- Publication runs inside the `pypi` environment, which requires a
  human reviewer in the GitHub UI before the deploy step runs.
- PyPI Trusted Publishing (OIDC) is used. There is no long-lived
  PyPI API token stored in repository secrets.
- Every action in every workflow is pinned to a commit SHA, not a
  mutable tag. A version comment is included alongside the SHA so
  humans can review what they pin to.
- `GITHUB_TOKEN` defaults to no permissions; each job opts in to the
  minimum scopes it needs.
- The `step-security/harden-runner` action restricts outbound network
  egress on every runner.
- Every release artifact is signed with sigstore and accompanied by
  a SLSA-style build provenance attestation. Both are uploaded to the
  GitHub Release.
- A CycloneDX SBOM is generated for each release and attached as a
  release asset.

### How to verify a release

You can verify any v0.1.1 or later AIsafePy release before installing.

1. **Verify the sigstore signature**:
   ```bash
   pip install sigstore
   sigstore verify identity \
     --cert-identity "https://github.com/Vidura-Wijekoon/aisafepy/.github/workflows/publish.yml@refs/tags/v0.1.1" \
     --cert-oidc-issuer "https://token.actions.githubusercontent.com" \
     aisafepy-0.1.1-py3-none-any.whl
   ```
2. **Verify the build provenance**:
   ```bash
   gh attestation verify aisafepy-0.1.1-py3-none-any.whl \
     --repo Vidura-Wijekoon/aisafepy
   ```
3. **Pin by hash in your own lockfile**:
   ```toml
   # pyproject.toml or requirements.txt
   aisafepy==0.1.1 \
       --hash=sha256:<paste hash from PyPI release page>
   ```
   `pip install --require-hashes` will refuse to install anything else.

### What an attacker would need to defeat this stack

To publish a malicious AIsafePy version to PyPI, an attacker would
need to compromise *all* of:

- The maintainer's GitHub account (2FA-protected).
- The `pypi` environment approver (a different human, in practice).
- The pinned SHA of every action in the publish workflow.
- The sigstore signing certificate chain (issued ephemerally per run).

Each layer alone is bypassable. Combined, this is much harder than
the single-OIDC-token theft pattern that Shai-Hulud exploited.

### What we ask of users

If you depend on AIsafePy in production:

1. Pin to a specific version with `--require-hashes`.
2. Subscribe to GitHub Releases for `Vidura-Wijekoon/aisafepy`.
3. Run `pip-audit` weekly against your environment.
4. Treat any prerelease (`X.Y.Z-alpha`) as alpha. Do not auto-update
   to prereleases.

If anything in this section is unclear or you find a gap, please
report it via the disclosure address at the top of this file.
