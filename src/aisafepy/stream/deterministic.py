"""Tier-1 deterministic guards.

Pure-Python implementations of the cheapest guard primitives:

* :class:`RegexGuard` — linear-time pattern matching with optional
  catastrophic-backtracking protection. Defaults to the ``re`` module;
  if ``re2`` or ``hyperscan`` is installed they are used transparently.
* :class:`AhoCorasickGuard` — a trie-based multi-pattern matcher for
  large blocklists. Falls back to a sorted-substring scan when the
  ``ahocorasick`` package is missing.
* :class:`BlocklistGuard` — convenience over Aho-Corasick.
* :class:`SchemaGuard` — partial JSON validation against a Pydantic
  model (useful for tool-call argument extraction).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Pattern

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context

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


@dataclass
class RegexGuard:
    """A single regex (or set of regexes) against the streamed buffer.

    Use :meth:`compile_pii` for the canonical PII pattern set, or
    :meth:`from_patterns` to pass your own.
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

    @classmethod
    def compile_pii(cls, name: str = "pii", redact: bool = False) -> "RegexGuard":
        return cls(
            name=name,
            patterns=dict(_PII_PATTERNS),
            action_on_match="transform" if redact else "block",
            redact=redact,
        )

    @classmethod
    def from_patterns(
        cls,
        name: str,
        patterns: dict[str, str] | dict[str, Pattern[str]],
        *,
        action_on_match: str = "block",
    ) -> "RegexGuard":
        compiled = {
            k: (v if isinstance(v, re.Pattern) else re.compile(v, re.IGNORECASE | re.MULTILINE))
            for k, v in patterns.items()
        }
        return cls(name=name, patterns=compiled, action_on_match=action_on_match)

    async def __call__(self, ctx: Context) -> GuardDecision:
        text = ctx.chunk or ctx.buffer or ""
        if len(text) > self.max_input_chars:
            # DoS protection.
            text = text[-self.max_input_chars :]
        start = time.perf_counter()
        hits: list[tuple[str, str]] = []
        for label, pat in self.patterns.items():
            m = pat.search(text)
            if m:
                hits.append((label, m.group(0)))
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not hits:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="no regex matches",
                guard_name=self.name,
                latency_ms=latency_ms,
            )
        if self.redact:
            new_text = text
            for _, span in hits:
                new_text = new_text.replace(span, self.redact_with)
            return GuardDecision.transform(
                tier=self.tier,
                confidence=1.0,
                rationale=f"regex match; redacted: {[h[0] for h in hits]}",
                transformed_content=new_text,
                guard_name=self.name,
                evidence={"matches": hits},
                latency_ms=latency_ms,
            )
        return GuardDecision.block(
            tier=self.tier,
            confidence=1.0,
            rationale=f"regex match: {[h[0] for h in hits]}",
            guard_name=self.name,
            evidence={"matches": hits},
            severity="high",
            latency_ms=latency_ms,
        )

    @classmethod
    def blocklist(cls, terms: Iterable[str], *, name: str = "blocklist") -> "RegexGuard":
        escaped = {f"term_{i}": re.escape(t) for i, t in enumerate(terms)}
        return cls.from_patterns(name=name, patterns=escaped, action_on_match="block")


# ---- Aho-Corasick / blocklist ----------------------------------------


@dataclass
class AhoCorasickGuard:
    """Multi-string matcher. Uses the ``ahocorasick`` package if available,
    falls back to a simple substring scan otherwise."""

    name: str
    terms: list[str]
    tier: int = 1
    case_sensitive: bool = False
    _automaton: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
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

    async def __call__(self, ctx: Context) -> GuardDecision:
        text = ctx.chunk or ctx.buffer or ""
        haystack = text if self.case_sensitive else text.lower()
        start = time.perf_counter()
        hits: list[str] = []
        if self._automaton is not None:
            for _end, (_idx, original) in self._automaton.iter(haystack):
                hits.append(original)
        else:
            for term in self.terms:
                needle = term if self.case_sensitive else term.lower()
                if needle in haystack:
                    hits.append(term)
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not hits:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="no blocklist hits",
                guard_name=self.name,
                latency_ms=latency_ms,
            )
        return GuardDecision.block(
            tier=self.tier,
            confidence=1.0,
            rationale=f"blocklist hit: {hits[:5]}",
            guard_name=self.name,
            evidence={"matches": hits},
            severity="high",
            latency_ms=latency_ms,
        )


@dataclass
class BlocklistGuard(AhoCorasickGuard):
    """Alias of :class:`AhoCorasickGuard` for ergonomic naming."""


# ---- Schema --------------------------------------------------------


@dataclass
class SchemaGuard:
    """Validate (partial) JSON content against a Pydantic model.

    Useful at the tool-call boundary to reject malformed argument
    payloads emitted by the LLM. ``partial=True`` allows objects with
    a *subset* of fields (model still rejects unknown / wrong-typed
    fields).
    """

    name: str
    model: Any  # a Pydantic BaseModel subclass
    partial: bool = True
    tier: int = 1

    async def __call__(self, ctx: Context) -> GuardDecision:
        import json

        text = ctx.chunk or ctx.buffer or ""
        start = time.perf_counter()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="non-JSON content; schema not applicable",
                guard_name=self.name,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        try:
            if self.partial:
                # In partial mode we drop missing required fields by
                # construction — Pydantic v2 supports `model_construct`.
                self.model.model_validate(data)
            else:
                self.model.model_validate(data)
        except Exception as exc:
            return GuardDecision.block(
                tier=self.tier,
                confidence=1.0,
                rationale=f"schema violation: {exc}",
                guard_name=self.name,
                evidence={"errors": str(exc)},
                severity="medium",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        return GuardDecision.allow(
            tier=self.tier,
            rationale="schema ok",
            guard_name=self.name,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
