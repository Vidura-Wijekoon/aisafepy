"""Tier-1 detection of encoded and obfuscated payloads.

Keyword guards are only as good as the text they see. The bypasses
catalogued in *Are Firewalls All You Need?* (arXiv 2510.05244, see
``docs/CAVEATS.md``) wrap the payload in a transport the filter does not
decode: base64, hex, percent-encoding, ROT13, Unicode Braille cells,
zero-width binary, TAG characters, homoglyphs and leetspeak. This guard
decodes each of those transports (recursively, up to
``max_decode_depth``) and scans the result for a small, high-precision
phrase list (:data:`INJECTION_PHRASES`).

Decision policy:

* a decoded payload that contains an injection phrase is a **block**;
* a transport that decodes to *unreadable* bytes, or an invisible
  carrier with no decodable content, is *opaque*. Opaque payloads are
  escalated by default (``on_opaque="escalate"``) because a model has
  no legitimate reason to emit binary blobs mid-conversation, but
  ``on_opaque="allow"`` keeps the guard purely detective;
* a transport that decodes to ordinary readable text (a base64 JSON
  config, a percent-encoded URL) is allowed, with evidence.

False-positive discipline: hex digests that are not text are ignored
entirely (hashes are everywhere), base64 candidates must look like
base64 (length multiple of four, digits present, entropy above a word-
like threshold), and image ``data:`` URIs are skipped by default.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import time
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import unquote

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.normalize import (
    ZERO_WIDTH_CHARS,
    Normalizer,
    is_tag_char,
)
from aisafepy.stream.pipeline import Context
from aisafepy.stream.secrets import shannon_entropy

INJECTION_PHRASES: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore all prior instructions",
    "ignore prior instructions",
    "ignore the above",
    "ignore your instructions",
    "disregard the system prompt",
    "disregard your instructions",
    "disregard previous instructions",
    "forget your instructions",
    "forget all previous instructions",
    "new instructions",
    "you are now",
    "system prompt",
    "reveal your instructions",
    "print your system prompt",
    "repeat your system prompt",
    "do not tell the user",
    "don't tell the user",
    "developer mode",
    "jailbreak",
    "act as dan",
    "do anything now",
    "override your",
    "bypass your",
    "without restrictions",
)
"""Phrases searched for inside *decoded* payloads.

Deliberately short and specific: a phrase only has to be convincing once
it has been found behind an encoding layer, where benign text is rare.
"""


def _phrase_regex(phrase: str) -> re.Pattern[str]:
    words = [re.escape(w) for w in phrase.split()]
    return re.compile(r"\b" + r"\W+".join(words) + r"\b", re.IGNORECASE)


_PHRASE_RES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (p, _phrase_regex(p)) for p in INJECTION_PHRASES
)


def find_injection_phrases(text: str, phrases: tuple[tuple[str, re.Pattern[str]], ...] = _PHRASE_RES) -> list[str]:
    """Return the phrases from :data:`INJECTION_PHRASES` present in ``text``."""
    return [p for p, rx in phrases if rx.search(text)]


# --- candidate patterns ----------------------------------------------------

_HEX_RE = re.compile(r"(?<![0-9A-Za-z])(?:0x)?((?:[0-9a-fA-F]{2}){12,})(?![0-9A-Za-z])")
_ESCAPED_HEX_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2}){8,}")
_PERCENT_RE = re.compile(r"[A-Za-z0-9._~+%-]{8,}")
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_DATA_URI_RE = re.compile(r"data:(?:image|audio|video|font|application/octet-stream)[^,]*;base64,", re.I)

# Braille ASCII (North American) table: Unicode cell offset -> ASCII char.
_BRAILLE_ASCII: dict[int, str] = {
    0x00: " ", 0x2E: "!", 0x10: '"', 0x3C: "#", 0x2B: "$", 0x29: "%", 0x2F: "&",
    0x04: "'", 0x37: "(", 0x3E: ")", 0x21: "*", 0x2C: "+", 0x20: ",", 0x24: "-",
    0x28: ".", 0x0C: "/", 0x34: "0", 0x02: "1", 0x06: "2", 0x12: "3", 0x32: "4",
    0x22: "5", 0x16: "6", 0x36: "7", 0x26: "8", 0x14: "9", 0x31: ":", 0x30: ";",
    0x23: "<", 0x3F: "=", 0x1C: ">", 0x39: "?", 0x08: "@",
    0x01: "a", 0x03: "b", 0x09: "c", 0x19: "d", 0x11: "e", 0x0B: "f", 0x1B: "g",
    0x13: "h", 0x0A: "i", 0x1A: "j", 0x05: "k", 0x07: "l", 0x0D: "m", 0x1D: "n",
    0x15: "o", 0x0F: "p", 0x1F: "q", 0x17: "r", 0x0E: "s", 0x1E: "t", 0x25: "u",
    0x27: "v", 0x3A: "w", 0x2D: "x", 0x3D: "y", 0x35: "z",
    0x2A: "[", 0x33: "\\", 0x3B: "]", 0x18: "^", 0x38: "_",
}


def decode_braille(text: str) -> str:
    """Decode six-dot Unicode Braille cells via the Braille ASCII table.

    Eight-dot cells (dots 7/8 set) and non-Braille characters are passed
    through unchanged so surrounding prose survives the decode.
    """
    out: list[str] = []
    for ch in text:
        o = ord(ch)
        if 0x2800 <= o <= 0x28FF:
            cell = o - 0x2800
            if cell < 0x40:
                out.append(_BRAILLE_ASCII.get(cell, "?"))
                continue
        out.append(ch)
    return "".join(out)


def decode_tag_chars(text: str) -> str:
    """Extract the ASCII hidden in Unicode TAG characters (U+E0020..U+E007E)."""
    return "".join(chr(ord(ch) - 0xE0000) for ch in text if 0xE0020 <= ord(ch) <= 0xE007E)


def decode_zero_width_binary(text: str) -> str | None:
    """Decode the common ZWSP=0 / ZWNJ=1 (or ZWJ=1) binary steganography.

    Returns ``None`` when the carrier does not form whole, printable
    bytes.
    """
    bits: list[str] = []
    for ch in text:
        if ch == "\u200b":
            bits.append("0")
        elif ch in ("\u200c", "\u200d"):
            bits.append("1")
    if len(bits) < 16:
        return None
    bits = bits[: len(bits) - len(bits) % 8]
    try:
        raw = int("".join(bits), 2).to_bytes(len(bits) // 8, "big")
        decoded = raw.decode("ascii")
    except (ValueError, UnicodeDecodeError):
        return None
    return decoded if _printable_ratio(decoded) >= 0.9 else None


def _printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    ok = sum(1 for c in s if c.isprintable() or c in "\n\r\t")
    return ok / len(s)


@dataclass(frozen=True)
class EncodedPayload:
    """One transport detected in the text."""

    kind: str
    start: int
    end: int
    depth: int
    decoded_preview: str
    injection_phrases: tuple[str, ...] = ()
    opaque: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "span": [self.start, self.end],
            "depth": self.depth,
            "decoded_preview": self.decoded_preview,
            "injection_phrases": list(self.injection_phrases),
            "opaque": self.opaque,
        }


@dataclass
class EncodingGuard:
    """Decode smuggled payloads and scan them for injection phrases.

    Parameters
    ----------
    min_base64_len:
        Minimum length of a base64 run to consider (default 24, i.e. at
        least 18 decoded bytes).
    max_decode_depth:
        Recursive decoding depth for nested transports (base64 inside
        base64, hex inside base64, ...).
    on_opaque:
        What to do with transports that decode to unreadable bytes or
        invisible carriers with no recoverable text.
    braille_min_chars / braille_ratio:
        Minimum count and share of Braille cells among non-space
        characters before the Braille decoder runs.
    zero_width_min:
        Count of zero-width characters that constitutes a carrier.
    homoglyph_min:
        Minimum folded confusables before the homoglyph path runs.
    check_rot13 / check_leetspeak:
        Toggle the two transforms that have no detectable "shape" and
        are therefore only ever reported when they expose a phrase.
    """

    name: str = "encoding"
    tier: int = 1
    min_base64_len: int = 24
    max_decode_depth: int = 2
    on_opaque: Literal["escalate", "allow"] = "escalate"
    ignore_data_uris: bool = True
    printable_ratio: float = 0.9
    braille_min_chars: int = 8
    braille_ratio: float = 0.3
    zero_width_min: int = 8
    homoglyph_min: int = 3
    check_rot13: bool = True
    check_leetspeak: bool = True
    phrases: tuple[str, ...] = INJECTION_PHRASES
    max_input_chars: int = 50_000
    scan_target: Literal["chunk", "buffer"] = "chunk"
    _phrase_res: tuple[tuple[str, re.Pattern[str]], ...] = field(init=False, repr=False)
    _b64_re: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._phrase_res = (
            _PHRASE_RES
            if self.phrases is INJECTION_PHRASES
            else tuple((p, _phrase_regex(p)) for p in self.phrases)
        )
        n = max(8, self.min_base64_len)
        self._b64_re = re.compile(rf"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{{{n},}}={{0,2}}")

    # ---- public API ----------------------------------------------------

    def analyze(self, text: str, *, _depth: int = 0) -> list[EncodedPayload]:
        """Return every detected transport in ``text`` (recursively)."""
        payloads: list[EncodedPayload] = []
        consumed: list[tuple[int, int]] = []

        def overlaps(s: int, e: int) -> bool:
            return any(s < ce and cs < e for cs, ce in consumed)

        # Order matters: hex before base64 (hex charset is a subset).
        for m in _HEX_RE.finditer(text):
            try:
                raw = bytes.fromhex(m.group(1))
            except ValueError:
                continue
            decoded = self._as_text(raw)
            if decoded is None:
                continue  # a digest, not a transport
            payloads.extend(self._emit("hex", m.start(), m.end(), decoded, _depth))
            consumed.append(m.span())

        for m in _ESCAPED_HEX_RE.finditer(text):
            if overlaps(*m.span()):
                continue
            raw = bytes(int(h, 16) for h in re.findall(r"\\x([0-9a-fA-F]{2})", m.group(0)))
            decoded = self._as_text(raw)
            if decoded is not None:
                payloads.extend(self._emit("escaped_hex", m.start(), m.end(), decoded, _depth))
                consumed.append(m.span())

        for m in self._b64_re.finditer(text):
            if overlaps(*m.span()):
                continue
            token = m.group(0)
            if self.ignore_data_uris and _DATA_URI_RE.search(text[max(0, m.start() - 64) : m.start()]):
                continue
            if not self._looks_like_base64(token):
                continue
            raw = self._b64decode(token)
            if raw is None:
                continue
            decoded = self._as_text(raw)
            if decoded is None:
                payloads.append(
                    EncodedPayload("base64", m.start(), m.end(), _depth, "<binary>", opaque=True)
                )
            else:
                payloads.extend(self._emit("base64", m.start(), m.end(), decoded, _depth))
            consumed.append(m.span())

        for m in _PERCENT_RE.finditer(text):
            if overlaps(*m.span()) or len(_PERCENT_ESCAPE_RE.findall(m.group(0))) < 3:
                continue
            decoded = unquote(m.group(0))
            phrases = tuple(find_injection_phrases(decoded, self._phrase_res))
            if phrases:
                payloads.append(
                    EncodedPayload("percent", m.start(), m.end(), _depth, decoded[:80], phrases)
                )
            consumed.append(m.span())

        if self.check_rot13:
            rot = codecs.decode(text, "rot13")
            phrases = tuple(find_injection_phrases(rot, self._phrase_res))
            if phrases:
                payloads.append(EncodedPayload("rot13", 0, len(text), _depth, rot[:80], phrases))

        payloads.extend(self._invisible_carriers(text, _depth))
        payloads.extend(self._visual_obfuscation(text, _depth))
        return payloads

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        if self.scan_target == "buffer":
            text = ctx.buffer or ctx.chunk or ""
        else:
            text = ctx.chunk or ctx.buffer or ""
        if len(text) > self.max_input_chars:
            text = text[-self.max_input_chars :]
        try:
            payloads = self.analyze(text)
        except Exception as exc:  # pragma: no cover - defensive
            return GuardDecision.allow(
                tier=self.tier,
                rationale=f"{self.name}: analyzer error; failing open",
                guard_name=self.name,
                evidence={"error": repr(exc)},
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        evidence = {"payloads": [p.to_dict() for p in payloads]}
        hits = [p for p in payloads if p.injection_phrases]
        if hits:
            kinds = sorted({p.kind for p in hits})
            return GuardDecision.block(
                tier=self.tier,
                confidence=0.95,
                rationale=f"encoded injection payload ({kinds}): {sorted({ph for p in hits for ph in p.injection_phrases})[:3]}",
                fallback="The content contained an obfuscated instruction payload and was withheld.",
                severity="high",
                guard_name=self.name,
                evidence=evidence,
                latency_ms=latency_ms,
            )
        opaque = [p for p in payloads if p.opaque]
        if opaque and self.on_opaque == "escalate":
            return GuardDecision.escalate(
                tier=self.tier,
                confidence=0.6,
                rationale=f"opaque encoded payload ({sorted({p.kind for p in opaque})})",
                severity="medium",
                guard_name=self.name,
                evidence=evidence,
                latency_ms=latency_ms,
            )
        return GuardDecision.allow(
            tier=self.tier,
            rationale="no encoded injection payload" if not payloads else "encoded content is benign",
            guard_name=self.name,
            evidence=evidence if payloads else {},
            latency_ms=latency_ms,
        )

    # ---- internals -----------------------------------------------------

    def _emit(self, kind: str, start: int, end: int, decoded: str, depth: int) -> list[EncodedPayload]:
        phrases = tuple(find_injection_phrases(decoded, self._phrase_res))
        out = [EncodedPayload(kind, start, end, depth, decoded[:80], phrases)]
        if depth + 1 < self.max_decode_depth:
            for inner in self.analyze(decoded, _depth=depth + 1):
                out.append(
                    EncodedPayload(
                        f"{kind}>{inner.kind}",
                        start,
                        end,
                        inner.depth,
                        inner.decoded_preview,
                        inner.injection_phrases,
                        inner.opaque,
                    )
                )
        return out

    def _as_text(self, raw: bytes) -> str | None:
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
        if _printable_ratio(decoded) < self.printable_ratio:
            return None
        return decoded

    def _looks_like_base64(self, token: str) -> bool:
        body = token.rstrip("=")
        if len(token) % 4 not in (0, 2, 3):  # 1 is never valid
            return False
        if not any(c.isdigit() for c in body) or not any(c.isalpha() for c in body):
            return False
        if "-" not in body and "_" not in body and body.islower():
            return False  # a long lowercase word, not base64
        return shannon_entropy(body) >= 3.5

    @staticmethod
    def _b64decode(token: str) -> bytes | None:
        padded = token + "=" * (-len(token) % 4)
        try:
            return base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            pass
        if "-" in token or "_" in token:
            try:
                return base64.urlsafe_b64decode(padded)
            except (binascii.Error, ValueError):
                return None
        return None

    def _invisible_carriers(self, text: str, depth: int) -> list[EncodedPayload]:
        out: list[EncodedPayload] = []

        tags = sum(1 for c in text if is_tag_char(c))
        if tags >= 4:
            decoded = decode_tag_chars(text)
            phrases = tuple(find_injection_phrases(decoded, self._phrase_res))
            out.append(
                EncodedPayload(
                    "tag_chars", 0, len(text), depth, decoded[:80], phrases, opaque=not decoded.strip()
                )
            )

        zw = sum(1 for c in text if c in ZERO_WIDTH_CHARS)
        if zw >= self.zero_width_min:
            decoded = decode_zero_width_binary(text)
            if decoded is not None:
                phrases = tuple(find_injection_phrases(decoded, self._phrase_res))
                out.append(EncodedPayload("zero_width_binary", 0, len(text), depth, decoded[:80], phrases))
            else:
                out.append(EncodedPayload("zero_width", 0, len(text), depth, "", opaque=True))

        non_space = [c for c in text if not c.isspace()]
        braille = sum(1 for c in non_space if 0x2800 <= ord(c) <= 0x28FF)
        if braille >= self.braille_min_chars and non_space and braille / len(non_space) >= self.braille_ratio:
            decoded = decode_braille(text)
            phrases = tuple(find_injection_phrases(decoded, self._phrase_res))
            out.append(
                EncodedPayload("braille", 0, len(text), depth, decoded[:80], phrases, opaque=not phrases)
            )
        return out

    def _visual_obfuscation(self, text: str, depth: int) -> list[EncodedPayload]:
        """Homoglyph / compatibility-fold / leetspeak paths.

        Only reported when the *folded* text exposes a phrase that the
        raw text did not; plain mixed-script text is left to
        :class:`aisafepy.stream.heuristics.ScriptMixGuard`.
        """
        out: list[EncodedPayload] = []
        raw_phrases = set(find_injection_phrases(text, self._phrase_res))

        norm = Normalizer(casefold=True).normalize(text)
        folds = norm.suspicious.confusable_count + norm.suspicious.nfkc_fold_count
        if folds >= self.homoglyph_min:
            new = tuple(p for p in find_injection_phrases(norm.text, self._phrase_res) if p not in raw_phrases)
            if new:
                kind = "homoglyph" if norm.suspicious.confusable_count else "unicode_fold"
                out.append(EncodedPayload(kind, 0, len(text), depth, norm.text[:80], new))

        if self.check_leetspeak:
            leet = Normalizer(casefold=True, leetspeak=True).normalize(text)
            if leet.changes.get("leetspeak", 0) >= 2:
                new = tuple(p for p in find_injection_phrases(leet.text, self._phrase_res) if p not in raw_phrases)
                if new:
                    out.append(EncodedPayload("leetspeak", 0, len(text), depth, leet.text[:80], new))
        return out
