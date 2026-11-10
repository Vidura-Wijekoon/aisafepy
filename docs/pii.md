# PII detection (`aisafepy.stream.pii`)

`PIIGuard` is the full-strength personal-data detector for Tier 1. Every
detector that can be validated is validated, so digit-heavy business text
(order numbers, invoice ids, timestamps, version strings) stops tripping
the guard the way a bare regex does. `RegexGuard.compile_pii()` remains
the lightweight option; use `PIIGuard` when you need country coverage,
checksum validation, or per-kind redaction tokens.

```python
from aisafepy.stream import GuardPipeline
from aisafepy.stream.pii import PIIGuard

guard = PIIGuard(countries={"GB", "LK"}, mode="redact", redact_with="[{kind}]")
pipeline = GuardPipeline(tier1=[guard])
```

## Detectors

| Name | Validation | Severity | Countries |
|---|---|---|---|
| `email` | Address shape (same expression as `RegexGuard`). | medium | global |
| `iban` | ISO 13616 mod-97 over the rearranged string; per-country length table (60 registry entries). | high | IBAN countries |
| `payment_card` | Luhn plus brand prefix and length (Visa, Mastercard incl. 2221-2720, Amex, Discover, JCB, Diners, UnionPay). Brand reported in `detail`. | critical | global |
| `phone` | `+` numbers: longest-prefix match on a 50-entry calling-code table with per-country national lengths. National forms accepted only for US/CA (NXX-NXX-XXXX), GB (0 + 10 digits) and LK (0 + 9 digits), and only with separators. Rejected after words such as `order`, `invoice`, `ref`, `ticket`. | medium | global |
| `ssn_us` | Area not 000, 666 or 9xx; group not 00; serial not 0000; the advertising placeholders are excluded. | critical | US |
| `nino_gb` | Disallowed first letters (D F I Q U V), second letters (D F I O Q U V), prefixes (BG GB NK KN TN NT ZZ); suffix A-D. | high | GB |
| `nic_lk` | Old format `YYDDDSSSC` + V/X and new 12-digit format; day-of-year 001-366 (male) or 501-866 (female); plausible year. Sex reported in `detail`. | high | LK |
| `aadhaar_in` | Twelve digits, first digit 2-9, Verhoeff check digit. | critical | IN |
| `pan_in` | `AAAAA9999A` with a valid fourth-letter holder type (reported in `detail`). | high | IN |
| `sin_ca` | Nine digits, Luhn, first digit not 0 or 8. | critical | CA |
| `tfn_au` | ATO weighted mod-11 for 8- and 9-digit numbers. | high | AU |
| `passport_us` | Nine alphanumerics with the word "passport" within 40 characters. Confidence 0.5, below the default threshold. | high | US |
| `mrz_td3` | ICAO 9303 two 44-character lines; 7-3-1 check digits over document number, date of birth, expiry and the composite field. | critical | global |
| `date_of_birth` | A date in any of six common forms within 40 characters after `DOB`, `date of birth`, `born`, `birth date`, `birthday`; the date must be 10 to 120 years ago. | medium | global |
| `ipv4` | In-range octets; loopback, unspecified and multicast rejected; private ranges at lower confidence. | low | global |
| `ipv6` | `ipaddress.IPv6Address` parses it; loopback and multicast rejected. | low | global |
| `mac` | Six hex pairs, not all identical. | low | global |
| `street_address` | Number + capitalised words + street suffix. Opt-in (`include_disabled=True`), confidence 0.5. | medium | global |

Every validator is a plain function (`validate_iban`, `validate_card`,
...) returning a `Validation(ok, confidence, country, detail)` and can be
reused on its own.

## Configuration

| Parameter | Meaning |
|---|---|
| `countries` | ISO 3166 alpha-2 codes. Country-specific detectors are included only when they intersect; global detectors always run. `None` enables everything. |
| `only`, `exclude` | Select or drop detectors by name. Unknown names raise `KeyError`. |
| `mode` | `block` (default), `redact` (TRANSFORM), or `escalate`. |
| `redact_with` | Template for the replacement token; `{kind}` is the detector name in upper case. A detector's `redact_token` overrides it. |
| `min_confidence` | Matches below this are dropped entirely (default 0.6). Lower it to 0.4 to admit `passport_us` and `street_address`. |
| `include_disabled` | Enable opt-in detectors. |
| `detectors` | Explicit detector list; replaces the built-in table and any registered extensions. |
| `scope` | `auto` (chunk if present, else buffer), `chunk`, or `buffer`, as in `RegexGuard`. |
| `max_input_chars` | Tail cap on the scanned text; `evidence["truncated"]` is set when it applies. |

Confidence is the detector's base confidence scaled by the validator's
factor. Evidence lists each match as `{kind, start, end, severity,
confidence, country, detail, preview}`. The preview is masked
(`mask_value`): high and critical kinds keep only the last two
characters, lower severities keep two at each end. The raw value never
appears in evidence or rationale.

### Extending

```python
import re
from aisafepy.stream.pii import PIIDetector, Validation, register_detector

def valid_badge(text, full_text, start, end):
    return Validation(ok=text[-1] != "0", confidence=0.9, detail="badge")

register_detector(PIIDetector(
    name="employee_badge",
    pattern=re.compile(r"\bEMP-\d{5}\b"),
    validator=valid_badge,
    severity="low",
    redact_token="[BADGE]",
))
```

Registered detectors join the built-in table for any guard constructed
afterwards without an explicit `detectors=` list. Built-in names cannot
be overridden through the registry; pass `detectors=` to replace them.

## Redaction

`mode="redact"` masks every occurrence. Overlapping spans from different
detectors (a Canadian SIN inside a longer run that also parses as a
phone number) are merged into one token labelled with the most severe
kind, so the output never contains a half-masked fragment.
`PIIGuard.redact_text(text)` exposes the same logic for use outside the
pipeline.

## Limits

A regex-plus-checksum detector is precise about *structured*
identifiers and blind to everything else:

* Names, addresses in prose, medical conditions, religion, union
  membership and other free-text special categories are out of scope.
  Put Presidio (`aisafepy.contrib.PresidioPIIGuard`) or an LLM judge
  behind this guard when you need them.
* Identifiers without a checksum (US passport numbers, most national
  phone numbers) are inherently ambiguous; the guard either asks for
  context or caps the confidence, and you should expect to tune
  `min_confidence` per deployment.
* The guard scans the text it is given. Fullwidth digits, confusable
  letters and zero-width padding are not folded here; run
  `EncodingGuard` and the `Normalizer` first so offsets stay exact.
* Country tables (IBAN lengths, calling codes, national number lengths)
  are correct as of writing and are plain dictionaries you can patch
  when a registry changes.
