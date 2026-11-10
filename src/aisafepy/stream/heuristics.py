"""Small, cheap Tier-1 guards for structural and surface properties.

None of these guards understands meaning. Each checks one cheap,
well-defined property that is either a known attack shape (markdown
image exfiltration, homoglyph words, URLs to private networks) or an
operational safety limit (length, nesting depth, degenerate
repetition). They run in microseconds and are intended to sit in
``GuardPipeline.tier1`` next to :class:`RegexGuard`.

Guards defined here:

* :class:`LengthGuard`: character / estimated-token caps.
* :class:`RepetitionGuard`: degenerate generation (n-gram repetition and
  tail loops).
* :class:`ScriptMixGuard`: homoglyph attacks via word-level script mixing.
* :class:`CharsetGuard`: restrict output to a set of Unicode scripts.
* :class:`URLGuard`: apply :class:`aisafepy.stream.toolcalls.URLPolicy`
  to URLs found in free text.
* :class:`MarkdownImageExfilGuard`: zero-click data exfiltration through
  rendered markdown images and links.
* :class:`JSONDepthGuard`: nesting depth and key-count limits for JSON.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.normalize import (
    CONFUSABLE_SCRIPTS,
    mixed_script_ratio,
    mixed_script_words,
    script_of,
)
from aisafepy.stream.pipeline import Context
from aisafepy.stream.secrets import shannon_entropy
from aisafepy.stream.toolcalls import URLPolicy

Action = Literal["block", "escalate"]


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _decide(
    guard: Any,
    action: Action,
    *,
    confidence: float,
    rationale: str,
    evidence: dict[str, Any],
    latency_ms: float,
    severity: str = "medium",
) -> GuardDecision:
    if action == "block":
        return GuardDecision.block(
            tier=guard.tier,
            confidence=confidence,
            rationale=rationale,
            severity=severity,  # type: ignore[arg-type]
            guard_name=guard.name,
            evidence=evidence,
            latency_ms=latency_ms,
        )
    return GuardDecision.escalate(
        tier=guard.tier,
        confidence=confidence,
        rationale=rationale,
        severity=severity,  # type: ignore[arg-type]
        guard_name=guard.name,
        evidence=evidence,
        latency_ms=latency_ms,
    )


# --- length -----------------------------------------------------------------


@dataclass
class LengthGuard:
    """Cap the cumulative buffer by characters and/or estimated tokens.

    Token count is estimated as ``chars / chars_per_token``; the default
    of 4.0 is a reasonable average for English with BPE tokenizers.
    """

    name: str = "length"
    tier: int = 1
    max_chars: int | None = None
    max_tokens: int | None = None
    chars_per_token: float = 4.0
    action: Action = "block"

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        text = ctx.buffer or ctx.chunk or ""
        chars = len(text)
        tokens = chars / self.chars_per_token if self.chars_per_token > 0 else 0.0
        over: list[str] = []
        if self.max_chars is not None and chars > self.max_chars:
            over.append(f"chars {chars} > {self.max_chars}")
        if self.max_tokens is not None and tokens > self.max_tokens:
            over.append(f"tokens ~{tokens:.0f} > {self.max_tokens}")
        if not over:
            return GuardDecision.allow(tier=self.tier, rationale="within length limits", guard_name=self.name, latency_ms=_ms(start))
        return _decide(
            self,
            self.action,
            confidence=1.0,
            rationale="length limit exceeded: " + "; ".join(over),
            evidence={"chars": chars, "estimated_tokens": round(tokens), "limits": over},
            latency_ms=_ms(start),
        )


# --- repetition -------------------------------------------------------------


def tail_period_repeat(text: str, *, min_unit: int = 3, max_unit: int | None = None) -> tuple[str, int]:
    """Find the shortest unit that repeats consecutively at the end of ``text``.

    Returns ``(unit, count)`` with ``count >= 1``; ``count == 1`` means no
    repetition. Used to catch generation loops such as ``"and then and
    then and then..."`` which n-gram ratios miss when the buffer is
    otherwise long and varied.
    """
    n = len(text)
    limit = max_unit or n // 2
    best: tuple[str, int] = (text[-min_unit:] if n >= min_unit else text, 1)
    for p in range(min_unit, min(limit, n // 2) + 1):
        unit = text[n - p :]
        count = 1
        while n - (count + 1) * p >= 0 and text[n - (count + 1) * p : n - count * p] == unit:
            count += 1
        if count >= 3 and count * p > best[1] * len(best[0]):
            best = (unit, count)
            break
    return best


@dataclass
class RepetitionGuard:
    """Detect degenerate, looping generation.

    Two signals: the share of word n-grams that are repeats
    (``max_repeat_ratio``) and a repeating tail unit covering at least
    ``min_tail_share`` of the buffer. Short buffers (under ``min_words``)
    are never flagged, because a short answer legitimately repeats.
    """

    name: str = "repetition"
    tier: int = 1
    ngram: int = 4
    min_words: int = 40
    max_repeat_ratio: float = 0.5
    min_tail_repeats: int = 4
    min_tail_share: float = 0.4
    action: Action = "escalate"

    def analyze(self, text: str) -> dict[str, Any]:
        words = text.split()
        out: dict[str, Any] = {"words": len(words), "repeat_ratio": 0.0, "tail_unit": "", "tail_repeats": 1}
        if len(words) >= self.ngram:
            grams = [" ".join(words[i : i + self.ngram]) for i in range(len(words) - self.ngram + 1)]
            out["repeat_ratio"] = 1.0 - len(set(grams)) / len(grams)
        unit, count = tail_period_repeat(text.rstrip())
        out["tail_unit"] = unit[:40]
        out["tail_repeats"] = count
        out["tail_share"] = (len(unit) * count) / max(len(text), 1)
        return out

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        text = ctx.buffer or ctx.chunk or ""
        info = self.analyze(text)
        if info["words"] < self.min_words:
            return GuardDecision.allow(tier=self.tier, rationale="buffer too short to judge repetition", guard_name=self.name, latency_ms=_ms(start))
        looping_tail = info["tail_repeats"] >= self.min_tail_repeats and info["tail_share"] >= self.min_tail_share
        if info["repeat_ratio"] <= self.max_repeat_ratio and not looping_tail:
            return GuardDecision.allow(tier=self.tier, rationale="no degenerate repetition", guard_name=self.name, evidence=info, latency_ms=_ms(start))
        return _decide(
            self,
            self.action,
            confidence=max(info["repeat_ratio"], info["tail_share"]),
            rationale=f"degenerate repetition: ratio={info['repeat_ratio']:.2f}, tail x{info['tail_repeats']}",
            evidence=info,
            latency_ms=_ms(start),
        )


# --- scripts ----------------------------------------------------------------


@dataclass
class ScriptMixGuard:
    """Flag homoglyph attacks via mixed Unicode scripts.

    Word-level mixing of the confusable scripts (Latin, Cyrillic, Greek,
    Cherokee, Armenian) is treated as an attack and takes ``action``.
    Document-level mixing above ``max_ratio`` without any mixed word is
    merely escalated, because bilingual text is normal.
    """

    name: str = "script-mix"
    tier: int = 1
    max_ratio: float = 0.15
    min_letters: int = 8
    max_mixed_words: int = 0
    scripts: frozenset[str] = CONFUSABLE_SCRIPTS
    action: Action = "block"

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        text = ctx.buffer or ctx.chunk or ""
        letters = sum(1 for c in text if c.isalpha())
        if letters < self.min_letters:
            return GuardDecision.allow(tier=self.tier, rationale="too few letters to judge", guard_name=self.name, latency_ms=_ms(start))
        mixed_words = mixed_script_words(text, scripts=self.scripts)
        ratio = mixed_script_ratio(text, scripts=self.scripts)
        evidence = {"mixed_script_words": mixed_words[:10], "mixed_script_ratio": round(ratio, 4)}
        if len(mixed_words) > self.max_mixed_words:
            return _decide(
                self,
                self.action,
                confidence=0.95,
                rationale=f"{len(mixed_words)} word(s) mix confusable scripts",
                evidence=evidence,
                latency_ms=_ms(start),
                severity="high",
            )
        if ratio > self.max_ratio:
            return GuardDecision.escalate(
                tier=self.tier,
                confidence=min(1.0, ratio * 2),
                rationale=f"mixed-script ratio {ratio:.2f} > {self.max_ratio}",
                guard_name=self.name,
                evidence=evidence,
                latency_ms=_ms(start),
            )
        return GuardDecision.allow(tier=self.tier, rationale="scripts consistent", guard_name=self.name, latency_ms=_ms(start))


@dataclass
class CharsetGuard:
    """Only allow letters from ``allowed_scripts`` (plus Common)."""

    name: str = "charset"
    tier: int = 1
    allowed_scripts: frozenset[str] = frozenset({"Latin"})
    max_foreign_ratio: float = 0.0
    min_letters: int = 1
    action: Action = "block"

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        text = ctx.buffer or ctx.chunk or ""
        counts: Counter[str] = Counter(script_of(c) for c in text if c.isalpha())
        total = sum(counts.values())
        if total < self.min_letters:
            return GuardDecision.allow(tier=self.tier, rationale="no letters", guard_name=self.name, latency_ms=_ms(start))
        foreign = {s: n for s, n in counts.items() if s not in self.allowed_scripts and s != "Common"}
        ratio = sum(foreign.values()) / total
        if ratio <= self.max_foreign_ratio:
            return GuardDecision.allow(tier=self.tier, rationale="charset ok", guard_name=self.name, latency_ms=_ms(start))
        return _decide(
            self,
            self.action,
            confidence=1.0,
            rationale=f"disallowed scripts {sorted(foreign)} ({ratio:.0%} of letters)",
            evidence={"foreign_scripts": foreign, "foreign_ratio": round(ratio, 4)},
            latency_ms=_ms(start),
        )


# --- URLs -------------------------------------------------------------------

_URL_IN_TEXT_RE = re.compile(r"\b[a-z][a-z0-9+.-]{1,15}://[^\s<>\"'`)\]}]+", re.IGNORECASE)


def extract_urls(text: str) -> list[str]:
    """URLs with an explicit scheme, trailing punctuation trimmed."""
    out: list[str] = []
    for m in _URL_IN_TEXT_RE.finditer(text):
        url = m.group(0).rstrip(".,;:!?")
        out.append(url)
    return out


@dataclass
class URLGuard:
    """Apply a :class:`URLPolicy` to every URL in free text."""

    name: str = "urls"
    tier: int = 1
    policy: URLPolicy = field(default_factory=URLPolicy)
    max_urls: int = 200
    action: Action = "block"

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        text = ctx.buffer or ctx.chunk or ""
        urls = extract_urls(text)[: self.max_urls]
        violations: list[dict[str, Any]] = []
        for url in urls:
            rules = self.policy.check(url)
            if rules:
                host = urlsplit(url).hostname or ""
                violations.append({"host": host, "rules": rules})
        if not violations:
            return GuardDecision.allow(tier=self.tier, rationale=f"{len(urls)} url(s) within policy", guard_name=self.name, latency_ms=_ms(start))
        critical = any("url_private_network" in v["rules"] for v in violations)
        return _decide(
            self,
            self.action,
            confidence=1.0,
            rationale=f"{len(violations)} url(s) violate policy: {violations[0]['rules']}",
            evidence={"violations": violations, "urls_seen": len(urls)},
            latency_ms=_ms(start),
            severity="critical" if critical else "high",
        )


# --- markdown exfiltration --------------------------------------------------

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_MD_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_HTML_IMG_RE = re.compile(r"<img\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
_URL_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


@dataclass
class MarkdownImageExfilGuard:
    """Catch data exfiltration through rendered markdown.

    A markdown image to an attacker's host is fetched by the client the
    moment it renders, with no click; the query string or path carries
    whatever the model was tricked into putting there. This guard blocks
    external images whose URL carries a payload (long or high-entropy
    query / path, percent-encoding, base64) and escalates ordinary
    external images and payload-bearing links. Images on
    ``allowed_domains`` are ignored.
    """

    name: str = "markdown-exfil"
    tier: int = 1
    allowed_domains: tuple[str, ...] = ()
    max_payload_chars: int = 48
    payload_entropy: float = 3.8
    block_all_external_images: bool = False
    check_links: bool = True
    link_action: Action = "escalate"

    def _is_allowed(self, host: str) -> bool:
        host = host.lower()
        return any(host == d.lower() or host.endswith("." + d.lower()) for d in self.allowed_domains)

    def _payload(self, url: str) -> tuple[bool, dict[str, Any]]:
        parts = urlsplit(url)
        query = unquote(parts.query)
        path = unquote(parts.path)
        longest_segment = max((len(s) for s in path.split("/")), default=0)
        info = {
            "query_chars": len(query),
            "longest_path_segment": longest_segment,
            "query_entropy": round(shannon_entropy(query), 3) if query else 0.0,
            "percent_encoded": "%" in parts.query or "%" in parts.path,
        }
        suspicious = (
            len(query) > self.max_payload_chars
            or longest_segment > self.max_payload_chars
            or (len(query) >= 16 and shannon_entropy(query) >= self.payload_entropy)
            or bool(re.search(r"[A-Za-z0-9+/_-]{32,}={0,2}", parts.query))
        )
        return suspicious, info

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        text = ctx.buffer or ctx.chunk or ""
        images = [m.group(1) for m in _MD_IMAGE_RE.finditer(text)] + [m.group(1) for m in _HTML_IMG_RE.finditer(text)]
        links = [m.group(1) for m in _MD_LINK_RE.finditer(text)] if self.check_links else []
        findings: list[dict[str, Any]] = []
        worst: str | None = None
        for kind, urls in (("image", images), ("link", links)):
            for url in urls:
                if not _URL_SCHEME_RE.match(url):
                    continue  # relative paths and data: URIs are not exfil channels
                host = urlsplit(url).hostname or ""
                if not host or self._is_allowed(host):
                    continue
                suspicious, info = self._payload(url)
                if kind == "image" and (suspicious or self.block_all_external_images):
                    verdict = "block"
                elif kind == "image" or suspicious:
                    verdict = self.link_action if kind == "link" else "escalate"
                else:
                    continue
                findings.append({"kind": kind, "host": host, "verdict": verdict, **info})
                if verdict == "block" or worst is None:
                    worst = verdict
        if not findings:
            return GuardDecision.allow(tier=self.tier, rationale="no markdown exfiltration vectors", guard_name=self.name, latency_ms=_ms(start))
        return _decide(
            self,
            "block" if worst == "block" else "escalate",
            confidence=0.9 if worst == "block" else 0.6,
            rationale=f"{len(findings)} external markdown {findings[0]['kind']}(s) to {findings[0]['host']}",
            evidence={"findings": findings},
            latency_ms=_ms(start),
            severity="high" if worst == "block" else "medium",
        )


# --- JSON structure ---------------------------------------------------------


def json_shape(value: Any) -> tuple[int, int]:
    """Return ``(max_depth, total_keys)`` for a parsed JSON value, iteratively."""
    max_depth = 0
    keys = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, dict):
            max_depth = max(max_depth, depth)
            keys += len(node)
            stack.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, list):
            max_depth = max(max_depth, depth)
            stack.extend((v, depth + 1) for v in node)
    return max_depth, keys


@dataclass
class JSONDepthGuard:
    """Reject JSON that is too deeply nested or too wide.

    Protects downstream parsers and tool handlers from resource
    exhaustion. Non-JSON content is allowed; content longer than
    ``max_chars`` is rejected without parsing.
    """

    name: str = "json-depth"
    tier: int = 1
    max_depth: int = 32
    max_keys: int = 10_000
    max_chars: int = 1_000_000
    action: Action = "block"

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        text = (ctx.buffer or ctx.chunk or "").strip()
        if not text or text[0] not in "[{":
            return GuardDecision.allow(tier=self.tier, rationale="not JSON", guard_name=self.name, latency_ms=_ms(start))
        if len(text) > self.max_chars:
            return _decide(self, self.action, confidence=1.0, rationale=f"JSON exceeds {self.max_chars} chars", evidence={"chars": len(text)}, latency_ms=_ms(start))
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, RecursionError) as exc:
            if isinstance(exc, RecursionError):
                return _decide(self, self.action, confidence=1.0, rationale="JSON nesting overflowed the parser", evidence={"recursion_error": True}, latency_ms=_ms(start))
            return GuardDecision.allow(tier=self.tier, rationale="not valid JSON (yet)", guard_name=self.name, latency_ms=_ms(start))
        depth, keys = json_shape(value)
        evidence = {"depth": depth, "keys": keys}
        if depth > self.max_depth or keys > self.max_keys:
            return _decide(
                self,
                self.action,
                confidence=1.0,
                rationale=f"JSON shape exceeds limits (depth={depth}, keys={keys})",
                evidence=evidence,
                latency_ms=_ms(start),
            )
        return GuardDecision.allow(tier=self.tier, rationale="JSON shape ok", guard_name=self.name, evidence=evidence, latency_ms=_ms(start))


__all__ = [
    "CharsetGuard",
    "JSONDepthGuard",
    "LengthGuard",
    "MarkdownImageExfilGuard",
    "RepetitionGuard",
    "ScriptMixGuard",
    "URLGuard",
    "extract_urls",
    "json_shape",
    "tail_period_repeat",
]
