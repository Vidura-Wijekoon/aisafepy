"""Tier-1 credential and payment-card detection.

Three detectors, in decreasing order of precision:

1. **Known-prefix patterns.** Vendors increasingly prefix their tokens
   (``sk-``, ``ghp_``, ``xoxb-``, ``AKIA``). These matches are
   near-certain and carry a per-kind severity.
2. **Luhn-validated card numbers.** The crude ``13-19 digits`` regex in
   :mod:`aisafepy.stream.deterministic` flags order numbers, tracking
   numbers and phone numbers. Here a candidate must pass the Luhn
   checksum *and* carry a recognised brand prefix.
3. **Shannon-entropy tokens.** A fallback for unprefixed secrets. It is
   the noisiest of the three, so thresholds follow the widely used
   ``detect-secrets`` defaults (4.5 bits/char for base64 charsets, 3.0
   for hex) and common false positives (UUIDs, git SHAs) are excluded.

Privacy rule: the raw secret never appears in a ``GuardDecision``.
Rationale and evidence carry only the kind, a masked preview, the span
in the original text and, for entropy hits, the measured entropy.
Redaction mode rewrites the content with ``[REDACTED:<kind>]`` markers.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from aisafepy.core.decisions import GuardDecision, Severity
from aisafepy.stream.pipeline import Context

_SEVERITY_RANK: dict[str, int] = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass(frozen=True)
class SecretPattern:
    """A named regex for a known credential format.

    If the pattern defines a ``secret`` named group, only that group is
    treated as the secret (and redacted); otherwise the whole match is.
    """

    name: str
    regex: re.Pattern[str]
    severity: Severity
    description: str


def _p(name: str, pattern: str, severity: Severity, description: str, flags: int = 0) -> SecretPattern:
    return SecretPattern(name, re.compile(pattern, flags), severity, description)


KNOWN_SECRET_PATTERNS: tuple[SecretPattern, ...] = (
    _p(
        "anthropic_api_key",
        r"\bsk-ant-(?:api|admin)\d{2}-[A-Za-z0-9_-]{20,}",
        "critical",
        "Anthropic API key",
    ),
    _p(
        "openai_api_key",
        r"\bsk-(?!ant-)(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}\b",
        "critical",
        "OpenAI API key",
    ),
    _p(
        "github_token",
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{80,}\b",
        "critical",
        "GitHub personal access / OAuth / app token",
    ),
    _p("slack_token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", "critical", "Slack token"),
    _p(
        "slack_webhook",
        r"https://hooks\.slack\.com/services/T[A-Za-z0-9_]+/B[A-Za-z0-9_]+/[A-Za-z0-9_]+",
        "high",
        "Slack incoming webhook URL",
    ),
    _p(
        "aws_access_key_id",
        r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{16}\b",
        "critical",
        "AWS access key id",
    ),
    _p(
        "aws_secret_access_key",
        r"\baws_?(?:secret_?)?(?:access_?)?key\b\s*[:=]\s*[\"']?(?P<secret>[A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])",
        "critical",
        "AWS secret access key assignment",
        re.IGNORECASE,
    ),
    _p("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}\b", "high", "Google API key"),
    _p("google_oauth_token", r"\bya29\.[0-9A-Za-z_-]{20,}", "critical", "Google OAuth access token"),
    _p(
        "stripe_key",
        r"\b(?:sk|rk)_(?:live|test)_[0-9a-zA-Z]{24,}\b",
        "critical",
        "Stripe secret / restricted key",
    ),
    _p("twilio_api_key", r"\bSK[0-9a-fA-F]{32}\b", "high", "Twilio API key"),
    _p(
        "sendgrid_api_key",
        r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b",
        "critical",
        "SendGrid API key",
    ),
    _p("npm_token", r"\bnpm_[A-Za-z0-9]{36}\b", "critical", "npm access token"),
    _p("pypi_token", r"\bpypi-AgEI[A-Za-z0-9_-]{50,}", "critical", "PyPI API token"),
    _p("huggingface_token", r"\bhf_[A-Za-z0-9]{30,}\b", "high", "Hugging Face token"),
    _p(
        "jwt",
        r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
        "high",
        "JSON Web Token",
    ),
    _p(
        "private_key_block",
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----",
        "critical",
        "PEM private key header",
    ),
    _p(
        "azure_storage_key",
        r"AccountKey=(?P<secret>[A-Za-z0-9+/=]{60,})",
        "critical",
        "Azure storage account key",
        re.IGNORECASE,
    ),
    _p(
        "basic_auth_url",
        r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:(?P<secret>[^\s/@]+)@[^\s/]+",
        "high",
        "URL with embedded credentials",
        re.IGNORECASE,
    ),
    _p(
        "credential_assignment",
        r"\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key)\b\s*[:=]\s*[\"']?(?P<secret>[^\s\"',;]{8,})",
        "high",
        "credential-looking assignment",
        re.IGNORECASE,
    ),
)

# Values that look like documentation placeholders rather than secrets.
_PLACEHOLDER_RE = re.compile(
    r"^(?:<[^>]+>|\$\{?[A-Z_]+\}?|\{\{.*\}\}|x{4,}|\*{4,}|#{4,}|\.{3,}|"
    r"(?:your|my|the|a|an)[-_]?(?:password|secret|key|token).*|"
    r"(?:changeme|password|passw0rd|example|placeholder|redacted|hunter2|secret|123456(?:78)?)\W*)$",
    re.IGNORECASE,
)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_B64_RE = re.compile(r"^[A-Za-z0-9+/=_-]+$")
_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s<>\"']+", re.IGNORECASE)
_CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")


# --- helpers ---------------------------------------------------------------


def shannon_entropy(s: str) -> float:
    """Shannon entropy of ``s`` in bits per character (0.0 for empty)."""
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def luhn_valid(digits: str) -> bool:
    """Luhn checksum (ISO/IEC 7812-1) over a string of ASCII digits."""
    if not digits.isdigit() or len(digits) < 2:
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def card_brand(digits: str) -> str | None:
    """Best-effort issuer from the IIN prefix, or ``None`` if unrecognised."""
    n = len(digits)
    two = int(digits[:2])
    four = int(digits[:4])
    if digits[0] == "4" and n in (13, 16, 19):
        return "visa"
    if n == 16 and (51 <= two <= 55 or 2221 <= four <= 2720):
        return "mastercard"
    if n == 15 and two in (34, 37):
        return "amex"
    if n in (16, 19) and (four == 6011 or two == 65 or 644 <= int(digits[:3]) <= 649):
        return "discover"
    if n in (16, 17, 18, 19) and 3528 <= four <= 3589:
        return "jcb"
    if n == 14 and (300 <= int(digits[:3]) <= 305 or two in (36, 38)):
        return "diners"
    if n in (16, 18, 19) and (two == 62 or two == 81):
        return "unionpay"
    return None


def mask_secret(secret: str, keep_prefix: int = 2, keep_suffix: int = 2) -> str:
    """``abcdefghij`` -> ``ab***ij``. Short secrets collapse to ``***``."""
    if len(secret) <= keep_prefix + keep_suffix + 2:
        return "***"
    return f"{secret[:keep_prefix]}***{secret[-keep_suffix:] if keep_suffix else ''}"


@dataclass(frozen=True)
class SecretMatch:
    """One detected secret. ``start``/``end`` index the *original* text."""

    kind: str
    severity: Severity
    start: int
    end: int
    preview: str
    detector: Literal["pattern", "luhn", "entropy"]
    entropy: float | None = None
    brand: str | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "kind": self.kind,
            "severity": self.severity,
            "span": [self.start, self.end],
            "preview": self.preview,
            "detector": self.detector,
        }
        if self.entropy is not None:
            out["entropy"] = round(self.entropy, 3)
        if self.brand is not None:
            out["brand"] = self.brand
        return out


# --- guard -----------------------------------------------------------------


@dataclass
class SecretsGuard:
    """Detect (and optionally redact) credentials and card numbers.

    Parameters
    ----------
    mode:
        ``"block"`` returns ``GuardDecision.block``; ``"redact"`` returns
        ``GuardDecision.transform`` with every match replaced by
        ``redact_with`` (``{kind}`` is substituted).
    patterns:
        Known-format patterns; defaults to :data:`KNOWN_SECRET_PATTERNS`.
        Pass your own tuple to add vendor-internal formats.
    detect_entropy / detect_cards:
        Toggle the two heuristic detectors.
    entropy_base64_threshold / entropy_hex_threshold:
        Bits-per-character thresholds for base64-charset and hex tokens.
    min_base64_len / min_hex_len:
        Minimum token length for the entropy detector.
    allow_git_sha:
        Skip 40- and 64-char hex tokens (git / sha256 digests).
    skip_urls:
        Do not run the entropy detector over URL spans.
    scan:
        ``"chunk"`` (default, like :class:`RegexGuard`) scans the current
        chunk when streaming; ``"buffer"`` scans the whole buffer.
    """

    name: str = "secrets"
    tier: int = 1
    mode: Literal["block", "redact"] = "block"
    patterns: tuple[SecretPattern, ...] = KNOWN_SECRET_PATTERNS
    detect_entropy: bool = True
    detect_cards: bool = True
    require_known_brand: bool = True
    entropy_base64_threshold: float = 4.5
    entropy_hex_threshold: float = 3.0
    min_base64_len: int = 20
    min_hex_len: int = 24
    allow_git_sha: bool = True
    skip_urls: bool = False
    keep_prefix: int = 2
    keep_suffix: int = 2
    redact_with: str = "[REDACTED:{kind}]"
    max_input_chars: int = 50_000
    scan_target: Literal["chunk", "buffer"] = "chunk"
    _token_re: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        min_len = max(8, min(self.min_base64_len, self.min_hex_len))
        self._token_re = re.compile(rf"[A-Za-z0-9+/=_\-]{{{min_len},}}")

    # ---- public API ----------------------------------------------------

    def scan(self, text: str) -> list[SecretMatch]:
        """Return all matches in ``text``, sorted by position, de-overlapped."""
        matches: list[SecretMatch] = []
        matches.extend(self._scan_patterns(text))
        if self.detect_cards:
            matches.extend(self._scan_cards(text))
        if self.detect_entropy:
            matches.extend(self._scan_entropy(text))
        return _dedupe(matches)

    def redact(self, text: str, matches: Iterable[SecretMatch] | None = None) -> str:
        """Replace every match in ``text`` with the redaction marker."""
        found = list(matches) if matches is not None else self.scan(text)
        out = text
        for m in sorted(found, key=lambda m: m.start, reverse=True):
            out = out[: m.start] + self.redact_with.format(kind=m.kind) + out[m.end :]
        return out

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        if self.scan_target == "buffer":
            text = ctx.buffer or ctx.chunk or ""
        else:
            text = ctx.chunk or ctx.buffer or ""
        if len(text) > self.max_input_chars:
            text = text[-self.max_input_chars :]
        try:
            matches = self.scan(text)
        except Exception as exc:  # pragma: no cover - defensive
            return GuardDecision.allow(
                tier=self.tier,
                rationale=f"{self.name}: scanner error; failing open",
                guard_name=self.name,
                evidence={"error": repr(exc)},
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not matches:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="no secrets detected",
                guard_name=self.name,
                latency_ms=latency_ms,
            )

        kinds = sorted({m.kind for m in matches})
        severity = max((m.severity for m in matches), key=lambda s: _SEVERITY_RANK[s])
        confidence = 1.0 if any(m.detector != "entropy" for m in matches) else 0.8
        evidence = {"matches": [m.to_dict() for m in matches], "kinds": kinds}

        if self.mode == "redact":
            return GuardDecision.transform(
                tier=self.tier,
                confidence=confidence,
                rationale=f"secrets redacted: {kinds}",
                transformed_content=self.redact(text, matches),
                severity=severity,
                guard_name=self.name,
                evidence=evidence,
                latency_ms=latency_ms,
            )
        return GuardDecision.block(
            tier=self.tier,
            confidence=confidence,
            rationale=f"secrets detected: {kinds}",
            fallback="The response contained credential material and was withheld.",
            severity=severity,
            guard_name=self.name,
            evidence=evidence,
            latency_ms=latency_ms,
        )

    # ---- detectors -----------------------------------------------------

    def _scan_patterns(self, text: str) -> list[SecretMatch]:
        out: list[SecretMatch] = []
        for pat in self.patterns:
            for m in pat.regex.finditer(text):
                if "secret" in pat.regex.groupindex and m.group("secret") is not None:
                    start, end = m.span("secret")
                else:
                    start, end = m.span()
                value = text[start:end]
                if pat.name == "credential_assignment" and _PLACEHOLDER_RE.match(value):
                    continue
                out.append(
                    SecretMatch(
                        kind=pat.name,
                        severity=pat.severity,
                        start=start,
                        end=end,
                        preview=mask_secret(value, self.keep_prefix, self.keep_suffix),
                        detector="pattern",
                    )
                )
        return out

    def _scan_cards(self, text: str) -> list[SecretMatch]:
        out: list[SecretMatch] = []
        for m in _CARD_RE.finditer(text):
            digits = re.sub(r"[ -]", "", m.group(0))
            if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
                continue
            if not luhn_valid(digits):
                continue
            brand = card_brand(digits)
            if self.require_known_brand and brand is None:
                continue
            out.append(
                SecretMatch(
                    kind="payment_card",
                    severity="critical",
                    start=m.start(),
                    end=m.end(),
                    preview=f"{digits[:4]}***{digits[-4:]}",
                    detector="luhn",
                    brand=brand,
                )
            )
        return out

    def _scan_entropy(self, text: str) -> list[SecretMatch]:
        url_spans = [m.span() for m in _URL_RE.finditer(text)] if self.skip_urls else []
        out: list[SecretMatch] = []
        for m in self._token_re.finditer(text):
            token = m.group(0).strip("=-_")
            if not token or any(s <= m.start() and m.end() <= e for s, e in url_spans):
                continue
            classification = self._classify_token(token)
            if classification is None:
                continue
            charset, entropy = classification
            out.append(
                SecretMatch(
                    kind=f"high_entropy_{charset}",
                    severity="medium",
                    start=m.start(),
                    end=m.end(),
                    preview=mask_secret(token, self.keep_prefix, self.keep_suffix),
                    detector="entropy",
                    entropy=entropy,
                )
            )
        return out

    def _classify_token(self, token: str) -> tuple[str, float] | None:
        if _UUID_RE.match(token):
            return None
        if _HEX_RE.match(token):
            if len(token) < self.min_hex_len:
                return None
            if self.allow_git_sha and len(token) in (40, 64):
                return None
            if token.isdigit():
                return None
            entropy = shannon_entropy(token)
            return ("hex", entropy) if entropy >= self.entropy_hex_threshold else None
        if _B64_RE.match(token):
            if len(token) < self.min_base64_len:
                return None
            if not any(c.isdigit() for c in token) or not any(c.isalpha() for c in token):
                return None
            entropy = shannon_entropy(token)
            return ("base64", entropy) if entropy >= self.entropy_base64_threshold else None
        return None


def _dedupe(matches: list[SecretMatch]) -> list[SecretMatch]:
    """Drop matches overlapping an earlier, higher-priority match.

    Priority: pattern > luhn > entropy, then longer span, then position.
    """
    rank = {"pattern": 0, "luhn": 1, "entropy": 2}
    ordered = sorted(matches, key=lambda m: (rank[m.detector], -(m.end - m.start), m.start))
    kept: list[SecretMatch] = []
    for m in ordered:
        if any(m.start < k.end and k.start < m.end for k in kept):
            continue
        kept.append(m)
    kept.sort(key=lambda m: m.start)
    return kept
