"""Checksum-validated PII detection with country packs.

:class:`aisafepy.stream.deterministic.RegexGuard` ships a small,
conservative PII set. This module is the full-strength version: every
detector that *can* be validated is, so Tier 1 stops paying the
false-positive tax that pure regexes impose on digit-heavy text (order
numbers, invoice ids, timestamps, version strings).

What "validated" means per detector:

===================  =============================================================
Detector             Validation
===================  =============================================================
``iban``             ISO 13616 mod-97 over the rearranged string, plus the
                     per-country length table (60 countries).
``payment_card``     Luhn plus a brand prefix / length table (Visa, Mastercard
                     including the 2-series, Amex, Discover, JCB, Diners,
                     UnionPay). The brand is reported in ``detail``.
``phone``            E.164-style parse: longest-prefix match against a
                     country-calling-code table (50 codes) with per-country
                     national-number length rules; national formats are only
                     accepted for a few countries (US/CA, GB, LK) and only with
                     separators so bare digit runs never match.
``ssn_us``           SSA issuance rules: area not 000 / 666 / 9xx, group not
                     00, serial not 0000.
``nino_gb``          HMRC prefix rules (disallowed letters and prefixes) and
                     an A-D suffix.
``nic_lk``           Old (9 digits + V/X) and new (12 digits) formats with the
                     day-of-year rule (001-366 male, 501-866 female) and a
                     plausible year.
``aadhaar_in``       First digit 2-9 and a Verhoeff check digit.
``pan_in``           AAAAA9999A shape with a valid holder-type fourth letter.
``sin_ca``           Nine digits, Luhn, first digit not 0 or 8.
``tfn_au``           ATO weighted mod-11 (8- and 9-digit forms).
``passport_us``      Nine alphanumerics *and* the word "passport" nearby. Weak,
                     reported with low confidence.
``mrz_td3``          ICAO 9303 TD3 pair of 44-character lines; the 7-3-1 check
                     digits over document number, date of birth and expiry
                     must all agree.
``date_of_birth``    A date within 10 to 120 years of today, within 40
                     characters after a birth-date keyword.
``ipv4``             In-range octets (shared with ``RegexGuard``).
``ipv6``             ``ipaddress.IPv6Address`` parses it.
``mac``              Six hex pairs.
``email``            The same expression ``RegexGuard`` uses.
``street_address``   Number + capitalised words + a street suffix. Heuristic,
                     opt-in, low confidence.
===================  =============================================================

Country packs select detectors by ISO 3166 alpha-2 code. A detector
with no ``countries`` is global and always included. ``PIIGuard(
countries={"GB", "LK"})`` therefore yields email, phone, cards, IBAN
(GB is an IBAN country), NINO and NIC, and skips SSN, Aadhaar, SIN and
TFN.

The guard does not normalise Unicode before scanning. Attackers who
smuggle PII past this guard with fullwidth digits or confusables are
caught upstream by :class:`aisafepy.stream.encoding.EncodingGuard` and
:class:`aisafepy.stream.normalize.Normalizer`; running them first keeps
this module's offsets exact against the original text.
"""

from __future__ import annotations

import datetime as _dt
import ipaddress
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from re import Pattern
from typing import Any, Literal

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.deterministic import luhn_check, valid_ipv4
from aisafepy.stream.pipeline import Context

Scope = Literal["auto", "chunk", "buffer"]
Mode = Literal["block", "redact", "escalate"]
SeverityName = Literal["info", "low", "medium", "high", "critical"]

_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


# ---------------------------------------------------------------------------
# Validation plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Validation:
    """Outcome of a detector's validator.

    ``ok`` accepts or rejects the candidate. ``confidence`` scales the
    detector's base confidence (``1.0`` leaves it alone). ``country`` and
    ``detail`` are surfaced in evidence (the IBAN country, the card
    brand, the phone's region).
    """

    ok: bool
    confidence: float = 1.0
    country: str | None = None
    detail: str | None = None


Validator = Callable[[str, str, int, int], Validation]
"""``(matched_text, full_text, start, end) -> Validation``. The full text
and offsets let context-sensitive validators look around the match."""


def _accept(confidence: float = 1.0, country: str | None = None, detail: str | None = None) -> Validation:
    return Validation(True, confidence, country, detail)


_REJECT = Validation(False)


@dataclass(frozen=True)
class PIIDetector:
    """One detector: a regex, an optional validator, and metadata."""

    name: str
    pattern: Pattern[str]
    validator: Validator | None = None
    severity: SeverityName = "high"
    confidence: float = 0.9
    countries: frozenset[str] = frozenset()
    """ISO 3166 alpha-2 codes this detector belongs to. Empty means global."""
    redact_token: str | None = None
    """Override for the guard-level ``redact_with`` template."""
    enabled_by_default: bool = True
    description: str = ""

    def applies_to(self, countries: frozenset[str] | None) -> bool:
        if countries is None or not self.countries:
            return True
        return bool(self.countries & countries)


@dataclass(frozen=True)
class PIIMatch:
    """A validated hit with offsets into the scanned text."""

    kind: str
    text: str
    start: int
    end: int
    severity: SeverityName
    confidence: float
    country: str | None = None
    detail: str | None = None

    @property
    def masked(self) -> str:
        return mask_value(self.text, self.severity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "severity": self.severity,
            "confidence": round(self.confidence, 3),
            "country": self.country,
            "detail": self.detail,
            "preview": self.masked,
        }


def mask_value(value: str, severity: SeverityName = "high") -> str:
    """Mask a matched value for evidence.

    High and critical kinds keep only the last two characters; lower
    severities keep two at each end. Short values are fully masked.
    """
    n = len(value)
    if n <= 4:
        return "*" * n
    if _SEVERITY_RANK[severity] >= _SEVERITY_RANK["high"]:
        return "*" * (n - 2) + value[-2:]
    return value[:2] + "*" * (n - 4) + value[-2:]


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

# ISO 13616 registry lengths (total characters including country and check).
IBAN_LENGTHS: dict[str, int] = {
    "AL": 28, "AD": 24, "AT": 20, "AZ": 28, "BH": 22, "BE": 16, "BA": 20, "BR": 29,
    "BG": 22, "CR": 22, "HR": 21, "CY": 28, "CZ": 24, "DK": 18, "DO": 28, "EE": 20,
    "EG": 29, "FO": 18, "FI": 18, "FR": 27, "GE": 22, "DE": 22, "GI": 23, "GR": 27,
    "GL": 18, "GT": 28, "HU": 28, "IS": 26, "IQ": 23, "IE": 22, "IL": 23, "IT": 27,
    "JO": 30, "KZ": 20, "XK": 20, "KW": 30, "LV": 21, "LB": 28, "LI": 21, "LT": 20,
    "LU": 20, "MK": 19, "MT": 31, "MR": 27, "MU": 30, "MD": 24, "MC": 27, "ME": 22,
    "NL": 18, "NO": 15, "PK": 24, "PS": 29, "PL": 28, "PT": 25, "QA": 29, "RO": 24,
    "LC": 32, "SM": 27, "ST": 25, "SA": 24, "RS": 22, "SC": 31, "SK": 24, "SI": 19,
    "ES": 24, "SE": 24, "CH": 21, "TL": 23, "TN": 24, "TR": 26, "UA": 29, "AE": 23,
    "GB": 22, "VA": 22, "VG": 24,
}


def validate_iban(candidate: str, *_: Any) -> Validation:
    """ISO 13616: country length table plus mod-97 on the rearranged string."""
    s = re.sub(r"[\s-]", "", candidate).upper()
    if len(s) < 15 or not s[:2].isalpha() or not s[2:4].isdigit():
        return _REJECT
    country = s[:2]
    expected = IBAN_LENGTHS.get(country)
    if expected is None or len(s) != expected:
        return _REJECT
    rearranged = s[4:] + s[:4]
    try:
        numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    except ValueError:
        return _REJECT
    if int(numeric) % 97 != 1:
        return _REJECT
    return _accept(1.0, country=country)


_CARD_BRANDS: tuple[tuple[str, Pattern[str], frozenset[int]], ...] = (
    ("amex", re.compile(r"^3[47]"), frozenset({15})),
    ("diners", re.compile(r"^3(?:0[0-5]|[689])"), frozenset(range(14, 20))),
    ("jcb", re.compile(r"^35(?:2[89]|[3-8]\d)"), frozenset(range(16, 20))),
    ("visa", re.compile(r"^4"), frozenset({13, 16, 19})),
    ("mastercard", re.compile(r"^(?:5[1-5]|2(?:22[1-9]|2[3-9]\d|[3-6]\d\d|7[01]\d|720))"), frozenset({16})),
    ("discover", re.compile(r"^(?:6011|65|64[4-9]|622(?:12[6-9]|1[3-9]\d|[2-8]\d\d|9[01]\d|92[0-5]))"), frozenset(range(16, 20))),
    ("unionpay", re.compile(r"^62"), frozenset(range(16, 20))),
)


def card_brand(digits: str) -> str | None:
    """Return the brand whose prefix and length rules ``digits`` satisfies."""
    for brand, prefix, lengths in _CARD_BRANDS:
        if prefix.match(digits) and len(digits) in lengths:
            return brand
    return None


def validate_card(candidate: str, *_: Any) -> Validation:
    digits = _digits(candidate)
    if not 13 <= len(digits) <= 19:
        return _REJECT
    if len(set(digits)) == 1:
        return _REJECT
    if not luhn_check(digits):
        return _REJECT
    brand = card_brand(digits)
    if brand is None:
        return _REJECT
    return _accept(1.0, detail=brand)


# Calling code -> (region, allowed national-number lengths).
PHONE_COUNTRIES: dict[str, tuple[str, frozenset[int]]] = {
    "1": ("US", frozenset({10})),
    "7": ("RU", frozenset({10})),
    "20": ("EG", frozenset({10})),
    "27": ("ZA", frozenset({9})),
    "30": ("GR", frozenset({10})),
    "31": ("NL", frozenset({9})),
    "32": ("BE", frozenset({8, 9})),
    "33": ("FR", frozenset({9})),
    "34": ("ES", frozenset({9})),
    "36": ("HU", frozenset({8, 9})),
    "39": ("IT", frozenset({9, 10, 11})),
    "41": ("CH", frozenset({9})),
    "43": ("AT", frozenset(range(7, 14))),
    "44": ("GB", frozenset({10})),
    "45": ("DK", frozenset({8})),
    "46": ("SE", frozenset({7, 8, 9})),
    "47": ("NO", frozenset({8})),
    "48": ("PL", frozenset({9})),
    "49": ("DE", frozenset(range(6, 13))),
    "52": ("MX", frozenset({10})),
    "54": ("AR", frozenset({10})),
    "55": ("BR", frozenset({10, 11})),
    "56": ("CL", frozenset({9})),
    "57": ("CO", frozenset({10})),
    "60": ("MY", frozenset({9, 10})),
    "61": ("AU", frozenset({9})),
    "62": ("ID", frozenset(range(9, 12))),
    "63": ("PH", frozenset({10})),
    "64": ("NZ", frozenset({8, 9, 10})),
    "65": ("SG", frozenset({8})),
    "66": ("TH", frozenset({9})),
    "81": ("JP", frozenset({10})),
    "82": ("KR", frozenset({9, 10})),
    "84": ("VN", frozenset({9})),
    "86": ("CN", frozenset({11})),
    "90": ("TR", frozenset({10})),
    "91": ("IN", frozenset({10})),
    "92": ("PK", frozenset({10})),
    "94": ("LK", frozenset({9})),
    "234": ("NG", frozenset({10})),
    "254": ("KE", frozenset({9})),
    "351": ("PT", frozenset({9})),
    "353": ("IE", frozenset({9})),
    "358": ("FI", frozenset(range(6, 11))),
    "420": ("CZ", frozenset({9})),
    "852": ("HK", frozenset({8})),
    "880": ("BD", frozenset({10})),
    "966": ("SA", frozenset({9})),
    "971": ("AE", frozenset({9})),
    "972": ("IL", frozenset({9})),
    "977": ("NP", frozenset({10})),
}

_PHONE_CONTEXT_REJECT = re.compile(
    r"(?i)\b(?:order|invoice|ref(?:erence)?|ticket|case|tracking|po|sku|account|acct|id)\s*(?:no\.?|number|#)?\s*[:#]?\s*$"
)
_PHONE_CONTEXT_ACCEPT = re.compile(r"(?i)\b(?:tel|phone|mobile|cell|call|whatsapp|fax|contact)\b")


def parse_international(candidate: str) -> tuple[str, str] | None:
    """Return ``(region, national_number)`` for a ``+`` prefixed number."""
    s = candidate.strip()
    if not s.startswith("+"):
        return None
    digits = _digits(s)
    for width in (3, 2, 1):
        code = digits[:width]
        entry = PHONE_COUNTRIES.get(code)
        if entry is None:
            continue
        region, lengths = entry
        national = digits[width:]
        if len(national) in lengths and not national.startswith("0"):
            return region, national
    return None


def validate_phone(candidate: str, text: str, start: int, end: int) -> Validation:
    before = text[max(0, start - 24) : start]
    if _PHONE_CONTEXT_REJECT.search(before):
        return _REJECT
    stripped = candidate.strip()
    if stripped.startswith("+"):
        parsed = parse_international(stripped)
        if parsed is None:
            return _REJECT
        region, national = parsed
        if region == "US" and (national[0] in "01" or national[3] in "01"):
            return _REJECT
        return _accept(1.0, country=region, detail="international")

    digits = _digits(stripped)
    has_sep = bool(re.search(r"[\s().-]", stripped))
    if not has_sep:
        return _REJECT
    if len(set(digits)) == 1:
        return _REJECT
    # North American Numbering Plan: NXX-NXX-XXXX.
    if len(digits) == 10 and digits[0] in "23456789" and digits[3] in "23456789":
        conf = 0.85 if _PHONE_CONTEXT_ACCEPT.search(before) else 0.7
        return _accept(conf, country="US", detail="nanp")
    if len(digits) == 11 and digits[0] == "1" and digits[1] in "23456789" and digits[4] in "23456789":
        return _accept(0.8, country="US", detail="nanp")
    # UK national: 0 + 10 digits (mobiles 07xxx xxxxxx, geographic 01/02).
    if len(digits) == 11 and digits[0] == "0" and digits[1] in "127":
        return _accept(0.7, country="GB", detail="national")
    # Sri Lanka national: 0 + 9 digits.
    if len(digits) == 10 and digits[0] == "0" and digits[1] in "1234567890":
        return _accept(0.6, country="LK", detail="national")
    return _REJECT


def validate_ssn(candidate: str, *_: Any) -> Validation:
    digits = _digits(candidate)
    if len(digits) != 9:
        return _REJECT
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in ("000", "666") or area.startswith("9"):
        return _REJECT
    if group == "00" or serial == "0000":
        return _REJECT
    if digits in ("123456789", "111111111", "078051120", "219099999"):
        # Famous placeholders: the advertising sample and the Woolworth card.
        return _REJECT
    return _accept(1.0, country="US")


_NINO_BAD_FIRST = set("DFIQUV")
_NINO_BAD_SECOND = set("DFIOQUV")
_NINO_BAD_PREFIX = {"BG", "GB", "NK", "KN", "TN", "NT", "ZZ"}


def validate_nino(candidate: str, *_: Any) -> Validation:
    s = re.sub(r"\s", "", candidate).upper()
    if len(s) != 9:
        return _REJECT
    prefix, digits, suffix = s[:2], s[2:8], s[8]
    if not (prefix.isalpha() and digits.isdigit() and suffix in "ABCD"):
        return _REJECT
    if prefix[0] in _NINO_BAD_FIRST or prefix[1] in _NINO_BAD_SECOND or prefix in _NINO_BAD_PREFIX:
        return _REJECT
    return _accept(1.0, country="GB")


def _nic_day_ok(day: int) -> tuple[bool, str | None]:
    if 1 <= day <= 366:
        return True, "male"
    if 501 <= day <= 866:
        return True, "female"
    return False, None


def validate_nic_lk(candidate: str, *_: Any) -> Validation:
    s = candidate.strip().upper()
    this_year = _dt.date.today().year
    if re.fullmatch(r"\d{9}[VX]", s):
        year = 1900 + int(s[:2])
        ok, sex = _nic_day_ok(int(s[2:5]))
        if not ok:
            return _REJECT
        return _accept(0.95, country="LK", detail=f"old-format {sex}")
    if re.fullmatch(r"\d{12}", s):
        year = int(s[:4])
        if not 1900 <= year <= this_year:
            return _REJECT
        ok, sex = _nic_day_ok(int(s[4:7]))
        if not ok:
            return _REJECT
        return _accept(0.9, country="LK", detail=f"new-format {sex}")
    return _REJECT


# Verhoeff tables (multiplication, permutation, inverse).
_V_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_V_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_V_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_valid(number: str) -> bool:
    """Return True if ``number`` (digits only) carries a valid Verhoeff check digit."""
    if not number.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _V_D[c][_V_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(number: str) -> str:
    """Compute the Verhoeff check digit to append to ``number``."""
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _V_D[c][_V_P[(i + 1) % 8][int(ch)]]
    return str(_V_INV[c])


def validate_aadhaar(candidate: str, *_: Any) -> Validation:
    digits = _digits(candidate)
    if len(digits) != 12 or digits[0] in "01":
        return _REJECT
    if not verhoeff_valid(digits):
        return _REJECT
    return _accept(1.0, country="IN")


_PAN_TYPES = {
    "A": "association", "B": "body of individuals", "C": "company", "F": "firm",
    "G": "government", "H": "hindu undivided family", "L": "local authority",
    "J": "artificial juridical person", "P": "individual", "T": "trust",
}


def validate_pan(candidate: str, *_: Any) -> Validation:
    s = candidate.upper()
    if not re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", s):
        return _REJECT
    holder = _PAN_TYPES.get(s[3])
    if holder is None:
        return _REJECT
    return _accept(0.9, country="IN", detail=holder)


def _luhn_any_length(digits: str) -> bool:
    """Luhn checksum without the 12-digit floor ``luhn_check`` applies to cards."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def validate_sin(candidate: str, *_: Any) -> Validation:
    digits = _digits(candidate)
    if len(digits) != 9 or digits[0] in "08":
        return _REJECT
    if len(set(digits)) == 1 or not _luhn_any_length(digits):
        return _REJECT
    return _accept(0.9, country="CA")


def tfn_valid(digits: str) -> bool:
    """Australian Tax File Number weighted mod-11 check."""
    if len(digits) == 9:
        weights = (1, 4, 3, 7, 5, 8, 6, 9, 10)
    elif len(digits) == 8:
        weights = (10, 7, 8, 4, 6, 3, 5, 1)
    else:
        return False
    return sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 11 == 0


def validate_tfn(candidate: str, *_: Any) -> Validation:
    digits = _digits(candidate)
    if len(set(digits)) == 1 or not tfn_valid(digits):
        return _REJECT
    return _accept(0.85, country="AU")


def validate_passport_us(candidate: str, text: str, start: int, end: int) -> Validation:
    window = text[max(0, start - 40) : min(len(text), end + 40)].lower()
    if "passport" not in window:
        return _REJECT
    if len(set(candidate)) == 1:
        return _REJECT
    return _accept(0.5, country="US")


def mrz_check_digit(field_text: str) -> int:
    """ICAO 9303 7-3-1 check digit over a field (``<`` counts as zero)."""
    total = 0
    for i, ch in enumerate(field_text):
        if ch == "<":
            value = 0
        elif ch.isdigit():
            value = int(ch)
        else:
            value = ord(ch) - 55
        total += value * (7, 3, 1)[i % 3]
    return total % 10


def validate_mrz_td3(candidate: str, *_: Any) -> Validation:
    lines = [ln.strip() for ln in candidate.strip().splitlines() if ln.strip()]
    if len(lines) != 2 or len(lines[0]) != 44 or len(lines[1]) != 44:
        return _REJECT
    second = lines[1]
    doc_number, doc_check = second[0:9], second[9]
    dob, dob_check = second[13:19], second[19]
    expiry, exp_check = second[21:27], second[27]
    for value, check in ((doc_number, doc_check), (dob, dob_check), (expiry, exp_check)):
        if not check.isdigit() or mrz_check_digit(value) != int(check):
            return _REJECT
    composite = second[0:10] + second[13:20] + second[21:43]
    if second[43].isdigit() and mrz_check_digit(composite) != int(second[43]):
        return _REJECT
    return _accept(1.0, country=lines[0][2:5].replace("<", "") or None, detail="td3")


_MONTHS = {
    m: i
    for i, names in enumerate(
        (
            ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
            ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
            ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for m in names
}


def _parse_date(raw: str) -> _dt.date | None:
    s = raw.strip().rstrip(".,")
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return _safe_date(y, mo, d)
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
    if m:
        a, b, y = (int(x) for x in m.groups())
        # Prefer day-first when unambiguous, else month-first.
        if a > 12 and b <= 12:
            return _safe_date(y, b, a)
        if b > 12 and a <= 12:
            return _safe_date(y, a, b)
        return _safe_date(y, a, b) or _safe_date(y, b, a)
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\.?,?\s+(\d{4})", s)
    if m:
        d, mon, y = m.groups()
        month = _MONTHS.get(mon.lower())
        return _safe_date(int(y), month, int(d)) if month else None
    m = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", s)
    if m:
        mon, d, y = m.groups()
        month = _MONTHS.get(mon.lower())
        return _safe_date(int(y), month, int(d)) if month else None
    return None


def _safe_date(y: int, m: int | None, d: int) -> _dt.date | None:
    if m is None:
        return None
    try:
        return _dt.date(y, m, d)
    except ValueError:
        return None


def validate_dob(candidate: str, text: str, start: int, end: int) -> Validation:
    parsed = _parse_date(candidate)
    if parsed is None:
        return _REJECT
    today = _dt.date.today()
    age_days = (today - parsed).days
    if not 10 * 365 <= age_days <= 120 * 365:
        return _REJECT
    return _accept(0.9, detail=parsed.isoformat())


def validate_ipv4(candidate: str, *_: Any) -> Validation:
    if not valid_ipv4(candidate):
        return _REJECT
    addr = ipaddress.IPv4Address(candidate)
    if addr.is_loopback or addr.is_unspecified or addr.is_multicast:
        return _REJECT
    return _accept(0.8 if addr.is_private else 0.9)


def validate_ipv6(candidate: str, *_: Any) -> Validation:
    if candidate.count(":") < 2:
        return _REJECT
    try:
        addr = ipaddress.IPv6Address(candidate)
    except ValueError:
        return _REJECT
    if addr.is_loopback or addr.is_unspecified or addr.is_multicast:
        return _REJECT
    return _accept(0.85)


def validate_mac(candidate: str, *_: Any) -> Validation:
    pairs = re.split(r"[:-]", candidate)
    if len(pairs) != 6 or len(set(pairs)) == 1:
        return _REJECT
    return _accept(0.8)


# ---------------------------------------------------------------------------
# Default detectors
# ---------------------------------------------------------------------------

_IBAN_COUNTRIES = frozenset(IBAN_LENGTHS)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")

_DATE_TOKEN = (
    r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\.?,?\s+\d{4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})"
)

DEFAULT_DETECTORS: tuple[PIIDetector, ...] = (
    PIIDetector(
        name="email",
        pattern=_EMAIL_RE,
        severity="medium",
        confidence=0.95,
        description="Email address.",
    ),
    PIIDetector(
        name="iban",
        pattern=re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]){11,30}\b"),
        validator=validate_iban,
        severity="high",
        confidence=1.0,
        countries=_IBAN_COUNTRIES,
        description="International Bank Account Number, mod-97 validated.",
    ),
    PIIDetector(
        name="payment_card",
        pattern=re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"),
        validator=validate_card,
        severity="critical",
        confidence=1.0,
        description="Payment card number, Luhn and brand validated.",
    ),
    PIIDetector(
        name="phone",
        pattern=re.compile(
            r"(?<![\w.+-])(?:\+\d{1,3}[\s.-]?(?:\(\d{1,4}\)[\s.-]?)?(?:\d[\s.-]?){6,13}\d"
            r"|\(\d{3}\)\s?\d{3}[\s.-]?\d{4}"
            r"|\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b"
            r"|\b0\d{2,4}[\s-]\d{3}[\s-]?\d{3,4}\b"
            r"|\b0\d{4}\s\d{6}\b)(?![\w-])"
        ),
        validator=validate_phone,
        severity="medium",
        confidence=0.9,
        description="Telephone number with calling-code aware validation.",
    ),
    PIIDetector(
        name="ssn_us",
        pattern=re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"),
        validator=validate_ssn,
        severity="critical",
        confidence=0.95,
        countries=frozenset({"US"}),
        description="US Social Security Number with issuance rules.",
    ),
    PIIDetector(
        name="nino_gb",
        pattern=re.compile(r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"),
        validator=validate_nino,
        severity="high",
        confidence=0.95,
        countries=frozenset({"GB"}),
        description="UK National Insurance number.",
    ),
    PIIDetector(
        name="nic_lk",
        pattern=re.compile(r"\b(?:\d{9}[VvXx]|\d{12})\b"),
        validator=validate_nic_lk,
        severity="high",
        confidence=0.9,
        countries=frozenset({"LK"}),
        description="Sri Lanka National Identity Card number (old and new formats).",
    ),
    PIIDetector(
        name="aadhaar_in",
        pattern=re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"),
        validator=validate_aadhaar,
        severity="critical",
        confidence=0.95,
        countries=frozenset({"IN"}),
        description="India Aadhaar number, Verhoeff validated.",
    ),
    PIIDetector(
        name="pan_in",
        pattern=re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        validator=validate_pan,
        severity="high",
        confidence=0.9,
        countries=frozenset({"IN"}),
        description="India Permanent Account Number.",
    ),
    PIIDetector(
        name="sin_ca",
        pattern=re.compile(r"\b\d{3}[ -]\d{3}[ -]\d{3}\b"),
        validator=validate_sin,
        severity="critical",
        confidence=0.9,
        countries=frozenset({"CA"}),
        description="Canada Social Insurance Number, Luhn validated.",
    ),
    PIIDetector(
        name="tfn_au",
        pattern=re.compile(r"\b\d{3}[ -]?\d{3}[ -]?\d{2,3}\b"),
        validator=validate_tfn,
        severity="high",
        confidence=0.85,
        countries=frozenset({"AU"}),
        description="Australia Tax File Number, weighted mod-11 validated.",
    ),
    PIIDetector(
        name="passport_us",
        pattern=re.compile(r"\b[A-Z]?\d{8,9}\b"),
        validator=validate_passport_us,
        severity="high",
        # The validator reports the (low) confidence; do not discount it twice.
        confidence=1.0,
        countries=frozenset({"US"}),
        description="US passport book number (needs the word passport nearby).",
    ),
    PIIDetector(
        name="mrz_td3",
        pattern=re.compile(r"P<[A-Z0-9<]{42}\s*\n\s*[A-Z0-9<]{44}"),
        validator=validate_mrz_td3,
        severity="critical",
        confidence=1.0,
        description="ICAO 9303 TD3 machine-readable zone with check digits.",
    ),
    PIIDetector(
        name="date_of_birth",
        pattern=re.compile(
            r"(?i)(?:\bd\.?o\.?b\.?|date of birth|birth\s?date|birthday|\bborn)\b"
            r"[^\n\d]{0,40}?(" + _DATE_TOKEN + ")"
        ),
        validator=validate_dob,
        severity="medium",
        confidence=0.85,
        description="Date of birth near a birth-date keyword.",
    ),
    PIIDetector(
        name="ipv4",
        pattern=re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        validator=validate_ipv4,
        severity="low",
        confidence=0.8,
        description="IPv4 address with in-range octets.",
    ),
    PIIDetector(
        name="ipv6",
        pattern=re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,7}(?:[0-9A-Fa-f]{1,4})?(?::[0-9A-Fa-f]{1,4})*(?![\w:])"),
        validator=validate_ipv6,
        severity="low",
        confidence=0.8,
        description="IPv6 address.",
    ),
    PIIDetector(
        name="mac",
        pattern=re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
        validator=validate_mac,
        severity="low",
        confidence=0.8,
        description="MAC address.",
    ),
    PIIDetector(
        name="street_address",
        pattern=re.compile(
            r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s+){1,3}"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|"
            r"Place|Pl|Way|Terrace|Ter|Square|Sq|Crescent|Cres|Parade|Mawatha)\b\.?"
        ),
        severity="medium",
        confidence=0.5,
        enabled_by_default=False,
        description="Street address heuristic (opt-in).",
    ),
)

_EXTRA_DETECTORS: list[PIIDetector] = []


def detectors(*, include_disabled: bool = True) -> tuple[PIIDetector, ...]:
    """Return the built-in detectors plus any registered extensions."""
    all_ = (*DEFAULT_DETECTORS, *_EXTRA_DETECTORS)
    if include_disabled:
        return all_
    return tuple(d for d in all_ if d.enabled_by_default)


def register_detector(detector: PIIDetector, *, replace_existing: bool = False) -> None:
    """Register a user-defined detector for guards built without ``detectors=``.

    Names must be unique unless ``replace_existing`` is set, in which case
    an extension of the same name is replaced (built-ins cannot be
    replaced; pass an explicit ``detectors`` list to the guard instead).
    """
    builtin = {d.name for d in DEFAULT_DETECTORS}
    if detector.name in builtin:
        raise ValueError(f"{detector.name!r} is a built-in detector; pass detectors= explicitly")
    for i, existing in enumerate(_EXTRA_DETECTORS):
        if existing.name == detector.name:
            if not replace_existing:
                raise ValueError(f"detector {detector.name!r} already registered")
            _EXTRA_DETECTORS[i] = detector
            return
    _EXTRA_DETECTORS.append(detector)


def unregister_detector(name: str) -> bool:
    """Remove a registered extension. Returns True if one was removed."""
    for i, existing in enumerate(_EXTRA_DETECTORS):
        if existing.name == name:
            del _EXTRA_DETECTORS[i]
            return True
    return False


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


@dataclass
class PIIGuard:
    """Tier-1 PII guard with validated detectors and country packs.

    Parameters
    ----------
    countries:
        ISO 3166 alpha-2 codes selecting country-specific detectors.
        ``None`` enables every detector; global detectors (email, phone,
        cards, IP addresses) are always enabled.
    only / exclude:
        Detector names to restrict to or drop. Unknown names raise.
    mode:
        ``block`` (default), ``redact`` (TRANSFORM with every occurrence
        masked), or ``escalate``.
    redact_with:
        Template for the replacement token; ``{kind}`` is the detector
        name in upper case. A detector's own ``redact_token`` wins.
    min_confidence:
        Matches below this confidence are dropped from both the decision
        and the evidence.
    include_disabled:
        Also enable opt-in detectors such as ``street_address``.
    """

    name: str = "pii"
    tier: int = 1
    countries: Iterable[str] | None = None
    only: Iterable[str] | None = None
    exclude: Iterable[str] = ()
    mode: Mode = "block"
    redact_with: str = "[{kind}]"
    min_confidence: float = 0.6
    include_disabled: bool = False
    detectors: list[PIIDetector] | None = None
    max_input_chars: int = 50_000
    scope: Scope = "auto"
    max_matches_per_detector: int = 200
    _active: tuple[PIIDetector, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in ("block", "redact", "escalate"):
            raise ValueError(f"mode must be block, redact or escalate (got {self.mode!r})")
        pool = tuple(self.detectors) if self.detectors is not None else detectors()
        known = {d.name for d in pool}
        only = set(self.only) if self.only is not None else None
        exclude = set(self.exclude)
        unknown = ((only or set()) | exclude) - known
        if unknown:
            raise KeyError(f"unknown detector name(s): {sorted(unknown)}; known: {sorted(known)}")
        country_set = frozenset(c.upper() for c in self.countries) if self.countries is not None else None
        active: list[PIIDetector] = []
        for det in pool:
            if only is not None:
                if det.name not in only:
                    continue
            elif not det.enabled_by_default and not self.include_disabled:
                continue
            if det.name in exclude or not det.applies_to(country_set):
                continue
            active.append(det)
        self._active = tuple(active)
        self.countries = sorted(country_set) if country_set is not None else None

    # ---- introspection ---------------------------------------------------

    @property
    def active_detectors(self) -> tuple[PIIDetector, ...]:
        return self._active

    def to_spec(self) -> dict[str, Any]:
        return {
            "kind": "pii",
            "name": self.name,
            "countries": self.countries,
            "only": sorted(self.only) if self.only is not None else None,
            "exclude": sorted(self.exclude),
            "mode": self.mode,
            "redact_with": self.redact_with,
            "min_confidence": self.min_confidence,
            "include_disabled": self.include_disabled,
            "scope": self.scope,
            "max_input_chars": self.max_input_chars,
        }

    # ---- scanning ----------------------------------------------------------

    def _select_text(self, ctx: Context) -> str:
        if self.scope == "buffer":
            return ctx.buffer or ctx.chunk or ""
        if self.scope == "chunk":
            return ctx.chunk or ""
        return ctx.chunk or ctx.buffer or ""

    def scan(self, text: str) -> list[PIIMatch]:
        """Return every validated match in ``text`` ordered by offset."""
        out: list[PIIMatch] = []
        for det in self._active:
            n = 0
            for m in det.pattern.finditer(text):
                # Detectors with a capture group (date_of_birth) report the group.
                if m.lastindex:
                    start, end = m.start(m.lastindex), m.end(m.lastindex)
                else:
                    start, end = m.start(), m.end()
                span = text[start:end]
                if not span:
                    continue
                country: str | None = None
                detail: str | None = None
                confidence = det.confidence
                if det.validator is not None:
                    result = det.validator(span, text, start, end)
                    if not result.ok:
                        continue
                    confidence = min(1.0, det.confidence * result.confidence)
                    country, detail = result.country, result.detail
                if confidence < self.min_confidence:
                    continue
                out.append(
                    PIIMatch(
                        kind=det.name,
                        text=span,
                        start=start,
                        end=end,
                        severity=det.severity,
                        confidence=confidence,
                        country=country,
                        detail=detail,
                    )
                )
                n += 1
                if n >= self.max_matches_per_detector:
                    break
        out.sort(key=lambda h: (h.start, -h.end))
        return _dedupe_nested(out)

    def redact_text(self, text: str, matches: list[PIIMatch] | None = None) -> str:
        """Mask every match; overlapping spans are merged and the most severe kind labels them."""
        matches = self.scan(text) if matches is None else matches
        if not matches:
            return text
        merged: list[tuple[int, int, PIIMatch]] = []
        for m in matches:
            if merged and m.start <= merged[-1][1]:
                s, e, rep = merged[-1]
                keep = m if _SEVERITY_RANK[m.severity] > _SEVERITY_RANK[rep.severity] else rep
                merged[-1] = (s, max(e, m.end), keep)
            else:
                merged.append((m.start, m.end, m))
        parts: list[str] = []
        cursor = 0
        for s, e, rep in merged:
            parts.append(text[cursor:s])
            parts.append(self._token_for(rep))
            cursor = e
        parts.append(text[cursor:])
        return "".join(parts)

    def _token_for(self, match: PIIMatch) -> str:
        det = next((d for d in self._active if d.name == match.kind), None)
        if det is not None and det.redact_token:
            return det.redact_token
        return self.redact_with.format(kind=match.kind.upper())

    async def __call__(self, ctx: Context) -> GuardDecision:
        text = self._select_text(ctx)
        truncated = False
        if len(text) > self.max_input_chars:
            text = text[-self.max_input_chars :]
            truncated = True
        start = time.perf_counter()
        try:
            matches = self.scan(text)
        except Exception as exc:  # pragma: no cover - defensive
            return GuardDecision.allow(
                tier=self.tier,
                rationale=f"pii scan raised; failing open: {exc!r}",
                guard_name=self.name,
                evidence={"error": repr(exc)},
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not matches:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="no PII detected",
                guard_name=self.name,
                latency_ms=latency_ms,
                evidence={"truncated": True} if truncated else None,
            )
        kinds = sorted({m.kind for m in matches})
        worst = max(matches, key=lambda m: _SEVERITY_RANK[m.severity]).severity
        confidence = max(m.confidence for m in matches)
        evidence = {
            "matches": [m.to_dict() for m in matches],
            "kinds": kinds,
            "countries": sorted({m.country for m in matches if m.country}),
            "truncated": truncated,
        }
        if self.mode == "redact":
            return GuardDecision.transform(
                tier=self.tier,
                confidence=confidence,
                rationale=f"PII redacted: {kinds}",
                transformed_content=self.redact_text(text, matches),
                guard_name=self.name,
                evidence=evidence,
                severity=worst,
                latency_ms=latency_ms,
            )
        if self.mode == "escalate":
            return GuardDecision.escalate(
                tier=self.tier,
                confidence=confidence,
                rationale=f"PII detected: {kinds}",
                guard_name=self.name,
                evidence=evidence,
                severity=worst,
                latency_ms=latency_ms,
            )
        return GuardDecision.block(
            tier=self.tier,
            confidence=confidence,
            rationale=f"PII detected: {kinds}",
            guard_name=self.name,
            evidence=evidence,
            severity=worst,
            latency_ms=latency_ms,
        )


def _dedupe_nested(matches: list[PIIMatch]) -> list[PIIMatch]:
    """Drop a match fully contained in another match of the *same* kind.

    Different kinds may legitimately overlap (a SIN inside a longer
    digit run that is also a phone number); redaction merges those.
    """
    kept: list[PIIMatch] = []
    for m in matches:
        if any(k.kind == m.kind and k.start <= m.start and m.end <= k.end and k is not m for k in kept):
            continue
        kept.append(m)
    return kept


def with_detector(guard: PIIGuard, detector: PIIDetector) -> PIIGuard:
    """Return a copy of ``guard`` with one extra detector active."""
    pool = list(guard.detectors) if guard.detectors is not None else list(detectors())
    pool.append(detector)
    return replace(guard, detectors=pool)
