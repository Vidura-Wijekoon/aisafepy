"""Text normalization for matching and obfuscation detection.

Almost every cheap Tier-1 detector in this package is a pattern match,
and almost every published bypass of a pattern match is an encoding
trick: zero-width characters inside keywords, a Cyrillic er or o standing
in for its Latin twin, fullwidth letters, mathematical bold letters,
or leetspeak. The :class:`Normalizer` folds those tricks away
*before* matching, and at the same time counts them so a downstream
guard can treat heavy obfuscation as a signal in its own right.

Two properties matter for production use:

* **Offset preservation.** ``Normalized.offset_map[i]`` is the index in
  the *original* string that produced normalized character ``i``. A
  match found in normalized text can therefore be reported, redacted
  or highlighted against the text the user actually sent.
* **Idempotence for matching.** ``normalize_for_matching`` is stable
  under repeated application, so patterns can be authored against the
  normalized form once and never drift.

Normalization is intentionally per-character NFKC rather than
whole-string NFKC. Whole-string NFKC can merge or reorder combining
sequences across characters, which would make the offset map lossy.
The cost is that ``e`` + U+0301 is not recomposed into ``\u00e9``; for
keyword matching that is irrelevant.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

# --- invisible / formatting characters -------------------------------------

ZERO_WIDTH_CHARS: frozenset[str] = frozenset(
    {
        "\u200b",  # ZERO WIDTH SPACE
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\u2060",  # WORD JOINER
        "\u2061",  # FUNCTION APPLICATION
        "\u2062",  # INVISIBLE TIMES
        "\u2063",  # INVISIBLE SEPARATOR
        "\u2064",  # INVISIBLE PLUS
        "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
        "\u00ad",  # SOFT HYPHEN
        "\u180e",  # MONGOLIAN VOWEL SEPARATOR
        "\u034f",  # COMBINING GRAPHEME JOINER
        "\u061c",  # ARABIC LETTER MARK
    }
)

BIDI_OVERRIDE_CHARS: frozenset[str] = frozenset(
    {
        "\u202a",  # LRE
        "\u202b",  # RLE
        "\u202c",  # PDF
        "\u202d",  # LRO
        "\u202e",  # RLO
        "\u2066",  # LRI
        "\u2067",  # RLI
        "\u2068",  # FSI
        "\u2069",  # PDI
    }
)


def is_tag_char(ch: str) -> bool:
    """True for the deprecated Unicode TAG block (U+E0000..U+E007F).

    Tag characters are invisible and map one-to-one onto ASCII, which
    makes them a popular carrier for hidden instructions.
    """
    return 0xE0000 <= ord(ch) <= 0xE007F


def is_variation_selector(ch: str) -> bool:
    o = ord(ch)
    return 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF


def is_invisible(ch: str) -> bool:
    """Zero-width, bidi-override, tag or variation-selector character."""
    return (
        ch in ZERO_WIDTH_CHARS
        or ch in BIDI_OVERRIDE_CHARS
        or is_tag_char(ch)
        or is_variation_selector(ch)
    )


# --- confusables -----------------------------------------------------------
#
# A deliberately small table: the characters that actually show up in
# homoglyph attacks against Latin-script keyword filters. Everything that
# NFKC already folds (fullwidth, mathematical alphanumerics, enclosed
# alphanumerics, ligatures, superscripts) is *not* listed here.

_CYRILLIC_TO_LATIN = {
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c", "\u0443": "y",
    "\u0445": "x", "\u0456": "i", "\u0458": "j", "\u0455": "s", "\u04bb": "h", "\u0501": "d",
    "\u051b": "q", "\u051d": "w", "\u04cf": "l", "\u0261": "g", "\u050d": "g", "\u04bd": "e",
    "\u0511": "e", "\u0251": "a", "\u03bf": "o", "\u0410": "A", "\u0412": "B", "\u0415": "E",
    "\u041a": "K", "\u041c": "M", "\u041d": "H", "\u041e": "O", "\u0420": "P", "\u0421": "C",
    "\u0422": "T", "\u0425": "X", "\u0405": "S", "\u0406": "I", "\u0408": "J", "\u0500": "D",
    "\u051a": "Q", "\u051c": "W", "\u04ae": "Y", "\u050c": "G", "\u04c0": "I",
}

_GREEK_TO_LATIN = {
    "\u03b1": "a", "\u03bf": "o", "\u03c1": "p", "\u03bd": "v", "\u03c5": "u", "\u03b9": "i",
    "\u03ba": "k", "\u03c7": "x", "\u03c4": "t", "\u03f2": "c", "\u03f3": "j", "\u03d2": "Y",
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z", "\u0397": "H", "\u0399": "I",
    "\u039a": "K", "\u039c": "M", "\u039d": "N", "\u039f": "O", "\u03a1": "P", "\u03a4": "T",
    "\u03a5": "Y", "\u03a7": "X",
}

_SMALL_CAPS_TO_LATIN = {
    "\u1d00": "a", "\u0299": "b", "\u1d04": "c", "\u1d05": "d", "\u1d07": "e", "\ua730": "f",
    "\u0262": "g", "\u029c": "h", "\u026a": "i", "\u1d0a": "j", "\u1d0b": "k", "\u029f": "l",
    "\u1d0d": "m", "\u0274": "n", "\u1d0f": "o", "\u1d18": "p", "\u01eb": "q", "\u0280": "r",
    "\ua731": "s", "\u1d1b": "t", "\u1d1c": "u", "\u1d20": "v", "\u1d21": "w", "\u028f": "y",
    "\u1d22": "z",
}

_MISC_TO_LATIN = {
    "\u0131": "i", "\u0237": "j", "\u2113": "l", "\u0269": "i", "\u029f": "l", "\ua7b5": "b",
    "\u13aa": "A", "\u13f4": "B", "\u13df": "C", "\u13a0": "D", "\u13ac": "E", "\u13c0": "G",
    "\u13bb": "H", "\u13a5": "v", "\u13ab": "J", "\u13e6": "K", "\u13de": "L", "\u13b7": "M",
    "\u13e2": "P", "\u13a1": "R", "\u13da": "S", "\u13a2": "T", "\u13d9": "V", "\u13b3": "W",
    "\u13c3": "Z",
}

_PUNCTUATION_FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u201c": '"', "\u201d": '"',
    "\u201e": '"', "\u201f": '"', "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-", "\u2024": ".", "\u2044": "/", "\u2215": "/",
    "\u02bc": "'",
}


CONFUSABLES: dict[str, str] = {
    **_CYRILLIC_TO_LATIN,
    **_GREEK_TO_LATIN,
    **_SMALL_CAPS_TO_LATIN,
    **_MISC_TO_LATIN,
    **_PUNCTUATION_FOLD,
}
"""Single-character folds applied after NFKC when ``fold_confusables`` is on."""

LEETSPEAK: dict[str, str] = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
    "@": "a", "$": "s", "!": "i", "|": "l", "+": "t", "(": "c", "\u20ac": "e",
}
"""Leetspeak folds. Only applied to tokens that also contain letters."""


# --- scripts ---------------------------------------------------------------

_SCRIPT_WORDS: dict[str, str] = {
    "LATIN": "Latin",
    "CYRILLIC": "Cyrillic",
    "GREEK": "Greek",
    "CJK": "Han",
    "HIRAGANA": "Hiragana",
    "KATAKANA": "Katakana",
    "HANGUL": "Hangul",
    "ARABIC": "Arabic",
    "HEBREW": "Hebrew",
    "THAI": "Thai",
    "DEVANAGARI": "Devanagari",
    "BENGALI": "Bengali",
    "TAMIL": "Tamil",
    "ARMENIAN": "Armenian",
    "GEORGIAN": "Georgian",
    "CHEROKEE": "Cherokee",
    "ETHIOPIC": "Ethiopic",
}

CONFUSABLE_SCRIPTS: frozenset[str] = frozenset({"Latin", "Cyrillic", "Greek", "Cherokee", "Armenian"})
"""Scripts whose letters have visually identical counterparts in Latin.

Mixing *these* inside a single word is the homoglyph-attack signature.
Mixing Han with Latin (a Japanese sentence with an English acronym) is
ordinary multilingual text and is not treated as suspicious.
"""


def script_of(ch: str) -> str:
    """Return the script family of ``ch``.

    Non-letters (digits, punctuation, whitespace, symbols) are
    ``"Common"``. Letters of scripts not in the table are ``"Other"``.
    """
    if not ch.isalpha():
        return "Common"
    name = unicodedata.name(ch, "")
    for word in name.split(" "):
        script = _SCRIPT_WORDS.get(word)
        if script is not None:
            return script
    return "Other"


def _letter_scripts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ch in text:
        s = script_of(ch)
        if s != "Common":
            counts[s] += 1
    return counts


def mixed_script_ratio(text: str, *, scripts: Iterable[str] | None = None) -> float:
    """Fraction of letters that are not in the dominant script.

    ``0.0`` for monoscript text (and for text with fewer than two
    letters). A Russian paragraph quoting one English word scores low;
    an English sentence with every third letter swapped for a Cyrillic
    lookalike scores high. Pass ``scripts`` to consider only those
    scripts (for example the confusable set), so that Japanese text
    mixing Han, Hiragana and Katakana is not penalised.
    """
    counts = _letter_scripts(text)
    if scripts is not None:
        watched = frozenset(scripts)
        counts = Counter({s: n for s, n in counts.items() if s in watched})
    total = sum(counts.values())
    if total < 2:
        return 0.0
    dominant = counts.most_common(1)[0][1]
    return 1.0 - dominant / total


def mixed_script_words(
    text: str,
    *,
    scripts: Iterable[str] = CONFUSABLE_SCRIPTS,
) -> list[str]:
    """Whitespace-delimited tokens that mix two or more of ``scripts``.

    Word-level mixing is a far stronger homoglyph signal than
    document-level mixing. A bilingual document has many monoscript
    words; an attack has Latin ``igno``, a Cyrillic er and Latin ``e`` inside one.
    """
    watched = frozenset(scripts)
    out: list[str] = []
    for token in text.split():
        seen = {script_of(c) for c in token} & watched
        if len(seen) >= 2:
            out.append(token)
    return out


# --- normalizer ------------------------------------------------------------


@dataclass(frozen=True)
class SuspiciousSignals:
    """Counters gathered during a normalization pass."""

    zero_width_count: int = 0
    bidi_override_count: int = 0
    tag_char_count: int = 0
    confusable_count: int = 0
    nfkc_fold_count: int = 0
    mixed_script_ratio: float = 0.0
    mixed_script_words: tuple[str, ...] = ()

    @property
    def any(self) -> bool:
        return bool(
            self.zero_width_count
            or self.bidi_override_count
            or self.tag_char_count
            or self.confusable_count
            or self.mixed_script_words
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "zero_width_count": self.zero_width_count,
            "bidi_override_count": self.bidi_override_count,
            "tag_char_count": self.tag_char_count,
            "confusable_count": self.confusable_count,
            "nfkc_fold_count": self.nfkc_fold_count,
            "mixed_script_ratio": round(self.mixed_script_ratio, 4),
            "mixed_script_words": list(self.mixed_script_words[:10]),
        }


@dataclass(frozen=True)
class Normalized:
    """Result of :meth:`Normalizer.normalize`."""

    text: str
    original: str
    offset_map: tuple[int, ...]
    """``offset_map[i]`` is the index in ``original`` of normalized char ``i``."""
    changes: dict[str, int] = field(default_factory=dict)
    suspicious: SuspiciousSignals = field(default_factory=SuspiciousSignals)

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a half-open span in normalized text back onto ``original``.

        An empty span maps to an empty span at the corresponding
        position. The end of the original span is one past the original
        index of the last normalized character in the span.
        """
        if not self.offset_map or start >= end:
            pos = self.offset_map[start] if start < len(self.offset_map) else len(self.original)
            return pos, pos
        start = max(0, min(start, len(self.offset_map) - 1))
        end = max(start + 1, min(end, len(self.offset_map)))
        o_start = self.offset_map[start]
        o_end = self.offset_map[end - 1] + 1
        return o_start, o_end

    def original_text(self, start: int, end: int) -> str:
        a, b = self.original_span(start, end)
        return self.original[a:b]


@dataclass
class Normalizer:
    """Configurable, offset-preserving text normalizer.

    Parameters
    ----------
    nfkc:
        Apply per-character NFKC compatibility folding (fullwidth,
        mathematical alphanumerics, enclosed alphanumerics, ligatures).
    strip_zero_width:
        Drop zero-width, bidi-override, tag and variation-selector
        characters. They are always *counted* regardless of this flag.
    fold_confusables:
        Replace Cyrillic / Greek / small-cap lookalikes with Latin.
    collapse_whitespace:
        Replace any run of whitespace with a single ASCII space.
    casefold:
        Apply ``str.casefold``. Off by default so callers that care about
        case (for example secret detection) keep it.
    leetspeak:
        Fold digits and symbols to letters inside tokens that contain
        letters (``1gn0r3`` becomes ``lgnore``; combine with a fuzzy
        matcher or accept that ``1`` is ambiguous between ``l`` and ``i``).
    """

    nfkc: bool = True
    strip_zero_width: bool = True
    fold_confusables: bool = True
    collapse_whitespace: bool = True
    casefold: bool = False
    leetspeak: bool = False

    def normalize(self, text: str) -> Normalized:
        out: list[str] = []
        offsets: list[int] = []
        changes: Counter[str] = Counter()
        zero_width = bidi = tags = confusables = nfkc_folds = 0
        prev_space = False
        leet_indices = self._leet_indices(text) if self.leetspeak else frozenset()

        for i, ch in enumerate(text):
            if is_tag_char(ch):
                tags += 1
                if self.strip_zero_width:
                    changes["invisible"] += 1
                    continue
            elif ch in BIDI_OVERRIDE_CHARS:
                bidi += 1
                if self.strip_zero_width:
                    changes["invisible"] += 1
                    continue
            elif ch in ZERO_WIDTH_CHARS or is_variation_selector(ch):
                zero_width += 1
                if self.strip_zero_width:
                    changes["invisible"] += 1
                    continue

            pieces = unicodedata.normalize("NFKC", ch) if self.nfkc else ch
            if pieces != ch:
                nfkc_folds += 1
                changes["nfkc"] += 1

            for piece in pieces:
                if self.fold_confusables:
                    folded = CONFUSABLES.get(piece)
                    if folded is not None:
                        confusables += 1
                        changes["confusable"] += 1
                        piece = folded
                if i in leet_indices:
                    leet = LEETSPEAK.get(piece)
                    if leet is not None:
                        changes["leetspeak"] += 1
                        piece = leet
                if self.casefold:
                    cf = piece.casefold()
                    if cf != piece:
                        changes["casefold"] += 1
                    piece = cf
                for sub in piece:
                    if sub.isspace():
                        if self.collapse_whitespace:
                            if prev_space:
                                changes["whitespace"] += 1
                                continue
                            sub = " "
                            prev_space = True
                    else:
                        prev_space = False
                    out.append(sub)
                    offsets.append(i)

        normalized = "".join(out)
        signals = SuspiciousSignals(
            zero_width_count=zero_width,
            bidi_override_count=bidi,
            tag_char_count=tags,
            confusable_count=confusables,
            nfkc_fold_count=nfkc_folds,
            mixed_script_ratio=mixed_script_ratio(text),
            mixed_script_words=tuple(mixed_script_words(text)),
        )
        return Normalized(
            text=normalized,
            original=text,
            offset_map=tuple(offsets),
            changes=dict(changes),
            suspicious=signals,
        )

    @staticmethod
    def _leet_indices(text: str) -> frozenset[int]:
        """Indices of leet characters that sit inside letter-bearing tokens."""
        indices: set[int] = set()
        start: int | None = None
        for i, ch in enumerate(text + " "):
            if ch.isspace():
                if start is not None:
                    token = text[start:i]
                    if any(c.isalpha() for c in token) and any(c in LEETSPEAK for c in token):
                        indices.update(start + k for k, c in enumerate(token) if c in LEETSPEAK)
                    start = None
            elif start is None:
                start = i
        return frozenset(indices)


_DEFAULT_MATCHING_NORMALIZER = Normalizer(casefold=True)


def normalize_for_matching(text: str) -> str:
    """NFKC + invisible stripping + confusable folding + casefold + whitespace collapse.

    The one-liner most pattern guards want. Idempotent.
    """
    return _DEFAULT_MATCHING_NORMALIZER.normalize(text).text
