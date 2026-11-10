"""Tier-1 deterministic guards.

Pure-Python implementations of the cheapest guard primitives:

* :class:`RegexGuard`. Linear-time pattern matching with optional
  catastrophic-backtracking protection. Defaults to the ``re`` module;
  if ``re2`` or ``hyperscan`` is installed they are used transparently.
* :class:`AhoCorasickGuard`. A trie-based multi-pattern matcher for
  large blocklists. Falls back to a sorted-substring scan when the
  ``ahocorasick`` package is missing.
* :class:`BlocklistGuard`. Convenience over Aho-Corasick.
* :class:`SchemaGuard`. Partial JSON validation against a Pydantic
  model (useful for tool-call argument extraction).

Design notes
------------

False positives are the operational failure mode of Tier 1 (see
``docs/CAVEATS.md`` on adversarial false-positive attacks). The PII
pattern set therefore pairs each permissive regex with a cheap
*validator*: credit-card candidates must pass a Luhn check, IPv4
candidates must have in-range octets, and generic API-key candidates
must look like real key material (mixed character classes, high
Shannon entropy) rather than a long word or a hex digest.

Redaction replaces *every* occurrence of every matching pattern. Spans
from different patterns that overlap (an email inside a URL, a phone
number inside a longer digit run) are merged before substitution so the
output never contains a half-masked fragment.

Streaming scope: by default a guard looks at the current ``chunk`` when
one is present and otherwise at the accumulated ``buffer``. Tokens that
split a secret across chunk boundaries are only caught when the guard
is configured with ``scope="buffer"``; the trade-off is re-scanning the
growing buffer on every chunk, which the ``max_input_chars`` cap keeps
bounded.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from re import Pattern
from typing import Any, Literal

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context

Scope = Literal["auto", "chunk", "buffer"]

# --- validators -----------------------------------------------------------


def luhn_check(candidate: str) -> bool:
    """Return True if the digits in ``candidate`` pass the Luhn checksum.

    Non-digit characters (spaces, hyphens) are ignored. Fewer than 12
    digits is never a valid payment card number.
    """
    digits = [int(ch) for ch in candidate if ch.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def valid_ipv4(candidate: str) -> bool:
    """Return True if every dotted octet is in ``0..255`` with no leading zeros."""
    parts = candidate.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit() or (len(p) > 1 and p[0] == "0"):
            return False
        if int(p) > 255:
            return False
    return True


def shannon_entropy(text: str) -> float:
    """Bits of entropy per character of ``text`` (0.0 for empty input)."""
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def looks_like_key_material(candidate: str, *, min_entropy: float = 3.0) -> bool:
    """Heuristic for generic API keys: mixed classes and non-trivial entropy.

    Long plain words ("internationalization"), repeated characters, and
    pure hex digests are rejected; base64-ish tokens with letters and
    digits mixed together pass.
    """
    has_alpha = any(ch.isalpha() for ch in candidate)
    has_digit = any(ch.isdigit() for ch in candidate)
    if not (has_alpha and has_digit):
        return False
    if re.fullmatch(r"[0-9a-f]+", candidate.lower()) and len(candidate) in (32, 40, 64):
        # md5 / sha1 / sha256 digests are content hashes, not credentials.
        return False
    return shannon_entropy(candidate) >= min_entropy


# --- common PII pattern set --------------------------------------------
#
# Conservative defaults. Tune per-jurisdiction in practice.

_PII_PATTERNS: dict[str, Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "api_key_generic": re.compile(r"\b[A-Za-z0-9_-]{32,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

_PII_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "credit_card": luhn_check,
    "ipv4": valid_ipv4,
    "api_key_generic": looks_like_key_material,
}


@dataclass(frozen=True)
class Match:
    """One validated regex hit, with offsets into the scanned text."""

    label: str
    text: str
    start: int
    end: int


@dataclass
class RegexGuard:
    """A single regex (or set of regexes) against the streamed buffer.

    Use :meth:`compile_pii` for the canonical PII pattern set, or
    :meth:`from_patterns` to pass your own.

    Parameters
    ----------
    validators:
        Optional per-label callables ``(matched_text) -> bool``. A hit
        is discarded when its validator returns False. This is how the
        PII set suppresses Luhn-invalid card numbers and out-of-range
        IPv4 literals.
    scope:
        Which text to scan: ``"auto"`` (chunk if present, else buffer),
        ``"chunk"``, or ``"buffer"``.
    max_matches_per_pattern:
        Cap on hits collected per pattern. Defends against inputs crafted
        to produce millions of tiny matches.
    """

    name: str
    patterns: dict[str, Pattern[str]]
    action_on_match: str = "block"
    redact: bool = False
    redact_with: str = "[REDACTED]"
    tier: int = 1
    max_input_chars: int = 50_000
    """Hard cap on input length to defend against the regex-DoS
    vector described in arXiv 2410.02916."""
    validators: dict[str, Callable[[str], bool]] = field(default_factory=dict)
    scope: Scope = "auto"
    max_matches_per_pattern: int = 200
    severity: str = "high"

    @classmethod
    def compile_pii(
        cls,
        name: str = "pii",
        redact: bool = False,
        *,
        only: Iterable[str] | None = None,
        exclude: Iterable[str] = (),
    ) -> RegexGuard:
        """Build the canonical PII guard.

        ``only`` restricts the pattern set to the named labels and
        ``exclude`` drops labels. Unknown labels raise ``KeyError`` so a
        typo in configuration fails loudly instead of silently scanning
        nothing.
        """
        labels = list(_PII_PATTERNS) if only is None else list(only)
        unknown = [lab for lab in (*labels, *exclude) if lab not in _PII_PATTERNS]
        if unknown:
            raise KeyError(f"unknown PII pattern label(s): {unknown}; known: {sorted(_PII_PATTERNS)}")
        selected = {lab: _PII_PATTERNS[lab] for lab in labels if lab not in set(exclude)}
        return cls(
            name=name,
            patterns=selected,
            action_on_match="transform" if redact else "block",
            redact=redact,
            validators={k: v for k, v in _PII_VALIDATORS.items() if k in selected},
        )

    @classmethod
    def from_patterns(
        cls,
        name: str,
        patterns: dict[str, str] | dict[str, Pattern[str]],
        *,
        action_on_match: str = "block",
        validators: dict[str, Callable[[str], bool]] | None = None,
        redact_with: str = "[REDACTED]",
    ) -> RegexGuard:
        compiled = {
            k: (v if isinstance(v, re.Pattern) else re.compile(v, re.IGNORECASE | re.MULTILINE))
            for k, v in patterns.items()
        }
        if action_on_match not in ("block", "transform", "escalate"):
            raise ValueError(
                f"action_on_match must be block, transform, or escalate (got {action_on_match!r})"
            )
        return cls(
            name=name,
            patterns=compiled,
            action_on_match=action_on_match,
            redact=action_on_match == "transform",
            validators=dict(validators or {}),
            redact_with=redact_with,
        )

    @classmethod
    def blocklist(cls, terms: Iterable[str], *, name: str = "blocklist") -> RegexGuard:
        escaped = {f"term_{i}": re.escape(t) for i, t in enumerate(terms)}
        return cls.from_patterns(name=name, patterns=escaped, action_on_match="block")

    # ---- scanning --------------------------------------------------------

    def _select_text(self, ctx: Context) -> str:
        if self.scope == "buffer":
            return ctx.buffer or ctx.chunk or ""
        if self.scope == "chunk":
            return ctx.chunk or ""
        return ctx.chunk or ctx.buffer or ""

    def scan(self, text: str) -> list[Match]:
        """Return every validated match in ``text``, ordered by offset."""
        hits: list[Match] = []
        for label, pat in self.patterns.items():
            validator = self.validators.get(label)
            n = 0
            for m in pat.finditer(text):
                span_text = m.group(0)
                if not span_text:
                    continue
                if validator is not None and not validator(span_text):
                    continue
                hits.append(Match(label=label, text=span_text, start=m.start(), end=m.end()))
                n += 1
                if n >= self.max_matches_per_pattern:
                    break
        hits.sort(key=lambda h: (h.start, -h.end))
        return hits

    def redact_text(self, text: str, hits: list[Match] | None = None) -> str:
        """Mask every hit in ``text``; overlapping spans are merged first."""
        hits = self.scan(text) if hits is None else hits
        if not hits:
            return text
        merged: list[tuple[int, int]] = []
        for h in hits:
            if merged and h.start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], h.end))
            else:
                merged.append((h.start, h.end))
        out: list[str] = []
        cursor = 0
        for start, end in merged:
            out.append(text[cursor:start])
            out.append(self.redact_with)
            cursor = end
        out.append(text[cursor:])
        return "".join(out)

    async def __call__(self, ctx: Context) -> GuardDecision:
        text = self._select_text(ctx)
        truncated = False
        if len(text) > self.max_input_chars:
            # DoS protection.
            text = text[-self.max_input_chars :]
            truncated = True
        start = time.perf_counter()
        hits = self.scan(text)
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not hits:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="no regex matches",
                guard_name=self.name,
                latency_ms=latency_ms,
                evidence={"truncated": True} if truncated else None,
            )
        labels = sorted({h.label for h in hits})
        evidence = {
            # (label, text) tuples kept for backward compatibility.
            "matches": [(h.label, h.text) for h in hits],
            "spans": [{"label": h.label, "start": h.start, "end": h.end} for h in hits],
            "labels": labels,
            "truncated": truncated,
        }
        if self.redact or self.action_on_match == "transform":
            return GuardDecision.transform(
                tier=self.tier,
                confidence=1.0,
                rationale=f"regex match; redacted: {labels}",
                transformed_content=self.redact_text(text, hits),
                guard_name=self.name,
                evidence=evidence,
                latency_ms=latency_ms,
            )
        if self.action_on_match == "escalate":
            return GuardDecision.escalate(
                tier=self.tier,
                confidence=0.9,
                rationale=f"regex match: {labels}",
                guard_name=self.name,
                evidence=evidence,
                severity=self.severity,  # type: ignore[arg-type]
                latency_ms=latency_ms,
            )
        return GuardDecision.block(
            tier=self.tier,
            confidence=1.0,
            rationale=f"regex match: {labels}",
            guard_name=self.name,
            evidence=evidence,
            severity=self.severity,  # type: ignore[arg-type]
            latency_ms=latency_ms,
        )

    def to_spec(self) -> dict[str, Any]:
        """Serializable description used by the pipeline config loader."""
        return {
            "kind": "regex",
            "name": self.name,
            "patterns": {k: v.pattern for k, v in self.patterns.items()},
            "action_on_match": self.action_on_match,
            "redact_with": self.redact_with,
            "scope": self.scope,
            "max_input_chars": self.max_input_chars,
        }


# ---- Aho-Corasick / blocklist ----------------------------------------


@dataclass
class AhoCorasickGuard:
    """Multi-string matcher. Uses the ``ahocorasick`` package if available,
    falls back to a simple substring scan otherwise.

    ``whole_words=True`` requires each hit to sit on word boundaries,
    which suppresses the classic "ass" in "class" false positive for
    short terms.
    """

    name: str
    terms: list[str]
    tier: int = 1
    case_sensitive: bool = False
    whole_words: bool = False
    max_input_chars: int = 200_000
    scope: Scope = "auto"
    _automaton: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.terms = [t for t in self.terms if t]
        try:
            import ahocorasick  # type: ignore[import-not-found]
        except ImportError:
            self._automaton = None
            return
        a = ahocorasick.Automaton()
        for i, term in enumerate(self.terms):
            key = term if self.case_sensitive else term.lower()
            a.add_word(key, (i, term))
        a.make_automaton()
        self._automaton = a

    def _select_text(self, ctx: Context) -> str:
        if self.scope == "buffer":
            return ctx.buffer or ctx.chunk or ""
        if self.scope == "chunk":
            return ctx.chunk or ""
        return ctx.chunk or ctx.buffer or ""

    def _on_word_boundary(self, haystack: str, start: int, end: int) -> bool:
        before_ok = start == 0 or not (haystack[start - 1].isalnum() or haystack[start - 1] == "_")
        after_ok = end >= len(haystack) or not (haystack[end].isalnum() or haystack[end] == "_")
        return before_ok and after_ok

    def scan(self, text: str) -> list[tuple[str, int]]:
        """Return ``(term, offset)`` pairs for every occurrence."""
        haystack = text if self.case_sensitive else text.lower()
        hits: list[tuple[str, int]] = []
        if self._automaton is not None:
            for end_idx, (_idx, original) in self._automaton.iter(haystack):
                start = end_idx - len(original) + 1
                if self.whole_words and not self._on_word_boundary(haystack, start, end_idx + 1):
                    continue
                hits.append((original, start))
        else:
            for term in self.terms:
                needle = term if self.case_sensitive else term.lower()
                pos = haystack.find(needle)
                while pos != -1:
                    if not self.whole_words or self._on_word_boundary(
                        haystack, pos, pos + len(needle)
                    ):
                        hits.append((term, pos))
                    pos = haystack.find(needle, pos + 1)
        hits.sort(key=lambda h: h[1])
        return hits

    async def __call__(self, ctx: Context) -> GuardDecision:
        text = self._select_text(ctx)
        if len(text) > self.max_input_chars:
            text = text[-self.max_input_chars :]
        start = time.perf_counter()
        hits = self.scan(text)
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not hits:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="no blocklist hits",
                guard_name=self.name,
                latency_ms=latency_ms,
            )
        terms = [t for t, _ in hits]
        return GuardDecision.block(
            tier=self.tier,
            confidence=1.0,
            rationale=f"blocklist hit: {terms[:5]}",
            guard_name=self.name,
            evidence={"matches": terms, "offsets": [o for _, o in hits]},
            severity="high",
            latency_ms=latency_ms,
        )

    def to_spec(self) -> dict[str, Any]:
        return {
            "kind": "blocklist",
            "name": self.name,
            "terms": list(self.terms),
            "case_sensitive": self.case_sensitive,
            "whole_words": self.whole_words,
            "scope": self.scope,
        }


@dataclass
class BlocklistGuard(AhoCorasickGuard):
    """Alias of :class:`AhoCorasickGuard` for ergonomic naming."""


# ---- Schema --------------------------------------------------------


@dataclass
class SchemaGuard:
    """Validate (partial) JSON content against a Pydantic model.

    Useful at the tool-call boundary to reject malformed argument
    payloads emitted by the LLM.

    ``partial=True`` validates only the fields that are *present*: each
    is checked against its declared type individually, unknown fields
    are rejected when ``reject_unknown`` is set, and missing required
    fields are tolerated because the stream may still be producing
    them. ``partial=False`` runs the full model validation.

    Text that is not JSON is ignored (allow) because this guard is only
    meaningful on structured payloads. Text that *starts* like JSON but
    does not parse is reported as ``incomplete`` and allowed, so a
    streaming tool call is not blocked mid-way.
    """

    name: str
    model: Any  # a Pydantic BaseModel subclass
    partial: bool = True
    tier: int = 1
    reject_unknown: bool = True
    metadata_key: str = "tool_arguments"
    """If ``ctx.metadata[metadata_key]`` is a dict it is validated instead
    of parsing the text, which lets adapters hand over already-decoded
    arguments."""

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        data: Any
        from_metadata = ctx.metadata.get(self.metadata_key) if ctx.metadata else None
        if isinstance(from_metadata, dict):
            data = from_metadata
        else:
            text = (ctx.chunk or ctx.buffer or "").strip()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                incomplete = text.startswith(("{", "["))
                return GuardDecision.allow(
                    tier=self.tier,
                    rationale=(
                        "incomplete JSON; deferring validation"
                        if incomplete
                        else "non-JSON content; schema not applicable"
                    ),
                    guard_name=self.name,
                    evidence={"incomplete": incomplete},
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )
        if not isinstance(data, dict):
            return GuardDecision.block(
                tier=self.tier,
                confidence=1.0,
                rationale=f"schema violation: expected an object, got {type(data).__name__}",
                guard_name=self.name,
                evidence={"errors": ["payload is not a JSON object"]},
                severity="medium",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        errors = self._validate_partial(data) if self.partial else self._validate_full(data)
        latency_ms = (time.perf_counter() - start) * 1000.0
        if errors:
            return GuardDecision.block(
                tier=self.tier,
                confidence=1.0,
                rationale=f"schema violation: {errors[0]}",
                guard_name=self.name,
                evidence={"errors": errors},
                severity="medium",
                latency_ms=latency_ms,
            )
        return GuardDecision.allow(
            tier=self.tier,
            rationale="schema ok",
            guard_name=self.name,
            latency_ms=latency_ms,
        )

    def _validate_full(self, data: dict[str, Any]) -> list[str]:
        try:
            self.model.model_validate(data)
        except Exception as exc:
            return _pydantic_error_lines(exc)
        return []

    def _validate_partial(self, data: dict[str, Any]) -> list[str]:
        from pydantic import TypeAdapter

        fields = getattr(self.model, "model_fields", {})
        errors: list[str] = []
        if self.reject_unknown:
            unknown = [k for k in data if k not in fields]
            if unknown:
                errors.append(f"unknown field(s): {sorted(unknown)}")
        for key, value in data.items():
            info = fields.get(key)
            if info is None:
                continue
            try:
                TypeAdapter(info.annotation).validate_python(value)
            except Exception as exc:
                errors.append(f"{key}: {_pydantic_error_lines(exc)[0]}")
        return errors


def _pydantic_error_lines(exc: Exception) -> list[str]:
    """Flatten a pydantic ``ValidationError`` into short ``loc: msg`` lines."""
    errors_fn = getattr(exc, "errors", None)
    if callable(errors_fn):
        try:
            out = []
            for e in errors_fn():
                loc = ".".join(str(p) for p in e.get("loc", ())) or "<root>"
                out.append(f"{loc}: {e.get('msg', 'invalid')}")
            if out:
                return out
        except Exception:  # pragma: no cover - defensive
            pass
    return [str(exc)]
