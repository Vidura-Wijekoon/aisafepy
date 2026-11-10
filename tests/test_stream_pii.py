"""Tests for the validated PII guard.

Fixtures are published test values (ECBS / ISO IBAN examples, the Luhn
test cards, the ICAO 9303 specimen MRZ) or synthetic values built to
satisfy the relevant checksum. None belongs to a real person.
"""

from __future__ import annotations

import datetime as dt

import pytest

from aisafepy.stream.pii import (
    DEFAULT_DETECTORS,
    PIIDetector,
    PIIGuard,
    card_brand,
    detectors,
    mask_value,
    mrz_check_digit,
    parse_international,
    register_detector,
    tfn_valid,
    unregister_detector,
    validate_aadhaar,
    validate_card,
    validate_dob,
    validate_iban,
    validate_ipv4,
    validate_ipv6,
    validate_mac,
    validate_mrz_td3,
    validate_nic_lk,
    validate_nino,
    validate_pan,
    validate_passport_us,
    validate_phone,
    validate_sin,
    validate_ssn,
    validate_tfn,
    verhoeff_check_digit,
    verhoeff_valid,
    with_detector,
)
from aisafepy.stream.pipeline import Context, GuardPipeline

# ---- IBAN -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "iban, country",
    [
        ("GB82 WEST 1234 5698 7654 32", "GB"),
        ("DE89 3704 0044 0532 0130 00", "DE"),
        ("FR14 2004 1010 0505 0001 3M02 606", "FR"),
        ("NL91 ABNA 0417 1643 00", "NL"),
        ("ES91 2100 0418 4502 0005 1332", "ES"),
        ("IT60 X054 2811 1010 0000 0123 456", "IT"),
        ("CH93 0076 2011 6238 5295 7", "CH"),
        ("BE68539007547034", "BE"),
    ],
)
def test_iban_valid(iban, country):
    v = validate_iban(iban)
    assert v.ok and v.country == country


@pytest.mark.parametrize(
    "iban",
    [
        "GB82 WEST 1234 5698 7654 33",  # checksum off by one
        "GB82 WEST 1234 5698 7654",  # wrong length for GB
        "XX82 WEST 1234 5698 7654 32",  # unknown country
        "DE89 3704 0044 0532 0130 0",  # short
        "GBAA WEST 1234 5698 7654 32",  # non-numeric check digits
        "1234 5678 9012 3456 7890",  # not an IBAN at all
    ],
)
def test_iban_invalid(iban):
    assert not validate_iban(iban).ok


# ---- payment cards --------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, brand",
    [
        ("4111 1111 1111 1111", "visa"),
        ("4012888888881881", "visa"),
        ("5500 0000 0000 0004", "mastercard"),
        ("2223000048400011", "mastercard"),  # 2-series test number
        ("378282246310005", "amex"),
        ("6011111111111117", "discover"),
        ("3530111333300000", "jcb"),
        ("30569309025904", "diners"),
    ],
)
def test_card_valid(number, brand):
    v = validate_card(number)
    assert v.ok and v.detail == brand


@pytest.mark.parametrize(
    "number",
    [
        "4111 1111 1111 1112",  # Luhn fails
        "1234 5678 9012 3456",  # Luhn fails
        "0000 0000 0000 0000",  # repeated digit
        "7999999999999999",  # unknown prefix even if Luhn held
        "4111111111111",  # 13-digit visa with bad checksum
        "411111111111111111111",  # too long
    ],
)
def test_card_invalid(number):
    assert not validate_card(number).ok


def test_card_brand_lengths():
    assert card_brand("4" + "0" * 12) == "visa"
    assert card_brand("4" + "0" * 13) is None  # 14 digits is not a visa length
    assert card_brand("62" + "0" * 14) == "unionpay"


# ---- phones -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, region",
    [
        ("+94 77 123 4567", "LK"),
        ("+44 20 7946 0958", "GB"),
        ("+1 (415) 555-2671", "US"),
        ("+91 98765 43210", "IN"),
        ("+61 4 1234 5678", "AU"),
        ("+49 30 123456", "DE"),
        ("+33 6 12 34 56 78", "FR"),
        ("+971 50 123 4567", "AE"),
    ],
)
def test_phone_international_valid(number, region):
    text = f"call {number} today"
    start = text.index(number)
    v = validate_phone(number, text, start, start + len(number))
    assert v.ok and v.country == region
    assert parse_international(number)[0] == region


@pytest.mark.parametrize(
    "text, number",
    [
        ("order 123-456-7890 shipped", "123-456-7890"),  # context says order
        ("ref: 212-555-0199", "212-555-0199"),  # reference number
        ("+999 12 345 6789", "+999 12 345 6789"),  # unknown calling code
        ("+44 123", "+44 123"),  # too short
        ("+1 (015) 555-2671", "+1 (015) 555-2671"),  # NANP area code cannot start with 0
        ("2125550199", "2125550199"),  # bare digits, no separators
        ("111-111-1111", "111-111-1111"),  # repeated digit
    ],
)
def test_phone_invalid(text, number):
    start = text.index(number)
    assert not validate_phone(number, text, start, start + len(number)).ok


def test_phone_national_forms_with_context_confidence():
    text = "phone (415) 555-2671"
    v = validate_phone("(415) 555-2671", text, 6, len(text))
    assert v.ok and v.country == "US" and v.confidence == pytest.approx(0.85)
    text2 = "see 07700 900123"
    v2 = validate_phone("07700 900123", text2, 4, len(text2))
    assert v2.ok and v2.country == "GB"


# ---- US SSN -----------------------------------------------------------------------------------


@pytest.mark.parametrize("ssn", ["212-34-5678", "001-01-0001", "665-99-9999", "772 45 1234", "899-12-3456", "100-50-5000"])
def test_ssn_valid(ssn):
    assert validate_ssn(ssn).ok


@pytest.mark.parametrize(
    "ssn",
    ["000-12-3456", "666-12-3456", "912-34-5678", "212-00-5678", "212-34-0000", "123-45-6789", "219-09-9999"],
)
def test_ssn_invalid(ssn):
    assert not validate_ssn(ssn).ok


# ---- UK NINO ------------------------------------------------------------------------------------


@pytest.mark.parametrize("nino", ["AB 12 34 56 C", "AB123456A", "JG 10 32 25 D", "ZY 99 88 77 B", "AA 00 00 00 A", "PR123456C"])
def test_nino_valid(nino):
    assert validate_nino(nino).ok


@pytest.mark.parametrize(
    "nino",
    ["QQ 12 34 56 C", "DA 12 34 56 C", "AO 12 34 56 C", "BG 12 34 56 C", "GB 12 34 56 A", "AB 12 34 56 E", "AB 12 34 5 C"],
)
def test_nino_invalid(nino):
    assert not validate_nino(nino).ok


# ---- Sri Lanka NIC ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nic, sex",
    [
        ("911234567V", "male"),
        ("915654321X", "female"),  # day 565 = female
        ("200012345678", "male"),
        ("199985612345", "female"),  # day 856 = female
        ("851231234V", "male"),
        ("200536600012", "male"),  # day 366 is allowed
    ],
)
def test_nic_valid(nic, sex):
        v = validate_nic_lk(nic)
        assert v.ok and sex in (v.detail or "")


@pytest.mark.parametrize(
    "nic",
    ["910004567V", "913674567V", "914994567V", "918674567V", "189912345678", "999912345678", "91123456V"],
)
def test_nic_invalid(nic):
    assert not validate_nic_lk(nic).ok


# ---- Aadhaar / Verhoeff --------------------------------------------------------------------------------


@pytest.mark.parametrize("number", ["234567890124", "999999999999", "314159265351", "500000000006"])
def test_aadhaar_valid(number):
    assert verhoeff_valid(number)
    assert validate_aadhaar(number).ok
    assert validate_aadhaar(f"{number[:4]} {number[4:8]} {number[8:]}").ok


@pytest.mark.parametrize("number", ["234567890123", "123456789012", "999999999998", "23456789012", "034567890124", "500000000007"])
def test_aadhaar_invalid(number):
    assert not validate_aadhaar(number).ok


def test_verhoeff_check_digit_round_trip():
    for base in ("23456789012", "87654321098", "55555555555"):
        full = base + verhoeff_check_digit(base)
        assert verhoeff_valid(full)
        assert not verhoeff_valid(base + str((int(full[-1]) + 1) % 10))


# ---- India PAN ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pan, holder", [("ABCPE1234F", "individual"), ("AAACX1234A", "company"), ("ABCFE1234K", "firm"), ("ABCHE1234Z", "hindu undivided family"), ("ABCTE1234Q", "trust"), ("ABCGE0001R", "government")])
def test_pan_valid(pan, holder):
    v = validate_pan(pan)
    assert v.ok and v.detail == holder


@pytest.mark.parametrize("pan", ["ABCXE1234F", "ABCDE12345", "ABCPE123F", "abcpe1234f ", "ABCPE1234", "1BCPE1234F"])
def test_pan_invalid(pan):
    assert not validate_pan(pan).ok


# ---- Canada SIN -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sin", ["130 692 544", "130692544", "130-692-544"])
def test_sin_valid(sin):
    assert validate_sin(sin).ok


@pytest.mark.parametrize("sin", ["130 692 545", "046 454 286", "830 692 544", "111 111 111", "130 692 54", "123 456 789"])
def test_sin_invalid(sin):
    # 046 454 286 passes Luhn but starts with 0, which is not issued.
    assert not validate_sin(sin).ok


# ---- Australia TFN --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tfn", ["123 456 782", "123456782", "876 543 210", "459 599 230", "12345678" if tfn_valid("12345678") else "876543210"])
def test_tfn_valid(tfn):
    assert validate_tfn(tfn).ok


@pytest.mark.parametrize("tfn", ["123 456 783", "111 111 111", "000 000 000", "123 456 78", "987 654 321", "123 456 7890"])
def test_tfn_invalid(tfn):
    assert not validate_tfn(tfn).ok


def test_tfn_eight_digit_form_is_checked():
    # Brute-force one valid 8-digit TFN so the second weight table is exercised.
    valid = next(f"1234567{d}" for d in range(10) if tfn_valid(f"1234567{d}"))
    assert tfn_valid(valid)
    assert not tfn_valid("1234567")


# ---- US passport ----------------------------------------------------------------------------------------------


def test_passport_requires_context_and_is_low_confidence():
    text = "my passport number is 123456789 thanks"
    start = text.index("123456789")
    v = validate_passport_us("123456789", text, start, start + 9)
    assert v.ok and v.confidence == pytest.approx(0.5)
    text2 = "tracking 123456789 shipped"
    assert not validate_passport_us("123456789", text2, 9, 18).ok
    assert not validate_passport_us("111111111", text, start, start + 9).ok


# ---- MRZ ---------------------------------------------------------------------------------------------------------

_MRZ = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10"


def test_mrz_check_digits_match_icao_specimen():
    assert mrz_check_digit("L898902C3") == 6
    assert mrz_check_digit("740812") == 2
    assert mrz_check_digit("120415") == 9
    v = validate_mrz_td3(_MRZ)
    assert v.ok and v.country == "UTO"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("L898902C36", "L898902C37"),
        lambda s: s.replace("7408122", "7408123"),
        lambda s: s.replace("1204159", "1204150"),
        lambda s: s[:-1],
        lambda s: s.splitlines()[0],
        lambda s: s.replace("<<<<<10", "<<<<<11"),
    ],
)
def test_mrz_invalid(mutate):
    assert not validate_mrz_td3(mutate(_MRZ)).ok


# ---- date of birth -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "date",
    ["12 March 1985", "March 12, 1985", "1985-03-12", "03/12/1985", "12th Mar 1985"],
)
def test_dob_valid_forms(date):
    v = validate_dob(date, "", 0, 0)
    assert v.ok and v.detail == "1985-03-12"


def test_dob_rejects_implausible_ages():
    this_year = dt.date.today().year
    assert not validate_dob(f"{this_year - 2}-01-01", "", 0, 0).ok  # too young
    assert not validate_dob("1850-01-01", "", 0, 0).ok  # too old
    assert not validate_dob("1985-13-40", "", 0, 0).ok  # not a date
    assert not validate_dob("not a date", "", 0, 0).ok


def test_dob_requires_keyword_in_guard():
    g = PIIGuard(only=["date_of_birth"])
    assert g.scan("meeting on 12 March 1985 notes") == []
    hits = g.scan("Patient DOB: 12 March 1985, seen today")
    assert [h.kind for h in hits] == ["date_of_birth"]
    assert hits[0].text == "12 March 1985"


# ---- network identifiers --------------------------------------------------------------------------------------------------


def test_ip_and_mac_validators():
    assert validate_ipv4("8.8.8.8").ok
    assert validate_ipv4("10.0.0.5").confidence == pytest.approx(0.8)
    assert not validate_ipv4("127.0.0.1").ok
    assert not validate_ipv4("256.1.1.1").ok
    assert validate_ipv6("2001:db8::1").ok
    assert not validate_ipv6("::1").ok
    assert not validate_ipv6("12:30:45").ok
    assert validate_mac("00:1A:2B:3C:4D:5E").ok
    assert not validate_mac("00:00:00:00:00:00").ok


# ---- guard behaviour -----------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_blocks_with_masked_evidence():
    g = PIIGuard()
    d = await g(Context(buffer="card 4111 1111 1111 1111 on file"))
    assert d.is_blocked
    assert d.severity == "critical"
    m = d.evidence["matches"][0]
    assert m["kind"] == "payment_card"
    assert m["detail"] == "visa"
    assert "4111 1111 1111 1111" not in str(d.evidence)
    assert m["preview"].endswith("11") and m["preview"].startswith("*")
    assert d.evidence["kinds"] == ["payment_card"]


@pytest.mark.asyncio
async def test_guard_redacts_every_occurrence_and_merges_overlaps():
    g = PIIGuard(mode="redact")
    text = "mail a@b.co and c@d.io; IBAN GB82 WEST 1234 5698 7654 32 and GB82 WEST 1234 5698 7654 32"
    d = await g(Context(buffer=text))
    assert d.action.value == "transform"
    assert d.transformed_content == "mail [EMAIL] and [EMAIL]; IBAN [IBAN] and [IBAN]"
    # An email whose domain is an IPv4 literal: overlapping spans collapse to one token.
    g2 = PIIGuard(mode="redact", only=["email", "ipv4"])
    out = g2.redact_text("write to ops@8.8.8.8.example now")
    assert out.count("[") == 1


def test_redact_token_template_and_detector_override():
    g = PIIGuard(mode="redact", redact_with="<{kind}>", only=["email"])
    assert g.redact_text("x a@b.co y") == "x <EMAIL> y"
    custom = PIIDetector(name="ticket", pattern=__import__("re").compile(r"TCK-\d{6}"), severity="low", redact_token="[TICKET]")
    g2 = with_detector(PIIGuard(mode="redact", only=["email"]), custom)
    g2 = PIIGuard(mode="redact", detectors=list(g2.detectors or []), only=["ticket"])
    assert g2.redact_text("see TCK-123456") == "see [TICKET]"


@pytest.mark.asyncio
async def test_escalate_mode_and_allow():
    g = PIIGuard(mode="escalate", only=["email"])
    d = await g(Context(buffer="write to a@b.co"))
    assert d.action.value == "escalate"
    assert (await g(Context(buffer="nothing here"))).action.value == "allow"


def test_country_packs_select_detectors():
    all_names = {d.name for d in PIIGuard().active_detectors}
    assert {"ssn_us", "nino_gb", "nic_lk", "aadhaar_in", "sin_ca", "tfn_au", "iban"} <= all_names
    lk_gb = {d.name for d in PIIGuard(countries={"lk", "GB"}).active_detectors}
    assert {"email", "phone", "payment_card", "iban", "nino_gb", "nic_lk"} <= lk_gb
    assert not {"ssn_us", "aadhaar_in", "sin_ca", "tfn_au"} & lk_gb
    us = {d.name for d in PIIGuard(countries=["US"]).active_detectors}
    assert "ssn_us" in us and "iban" not in us  # the US is not an IBAN country


def test_only_exclude_disabled_and_unknown_names():
    assert [d.name for d in PIIGuard(only=["email", "ipv4"]).active_detectors] == ["email", "ipv4"]
    names = {d.name for d in PIIGuard(exclude=["phone"]).active_detectors}
    assert "phone" not in names and "street_address" not in names
    assert "street_address" in {d.name for d in PIIGuard(include_disabled=True).active_detectors}
    with pytest.raises(KeyError):
        PIIGuard(only=["emial"])
    with pytest.raises(ValueError):
        PIIGuard(mode="explode")


def test_street_address_opt_in():
    g = PIIGuard(only=["street_address"], min_confidence=0.4)
    hits = g.scan("Deliver to 221 Baker Street, London or 45 Galle Road")
    assert [h.text for h in hits] == ["221 Baker Street", "45 Galle Road"]


@pytest.mark.parametrize(
    "benign",
    [
        "order 4500012345 shipped 2024-01-15 at 12:30:45",
        "version 1.2.3 build 20240101 commit 9f8e7d6c",
        "ISBN 978-3-16-148410-0 and ISBN 0-306-40615-2",
        "invoice INV-2024-000123 total 1,234.56",
        "the meeting is at 10:30 in room 4.112",
        "PO 123456789012 approved",
        "temperature 37.5 pressure 101.3",
        "flight BA2490 seat 12A gate B7",
    ],
)
def test_false_positive_discipline(benign):
    assert PIIGuard(min_confidence=0.0).scan(benign) == []


@pytest.mark.asyncio
async def test_min_confidence_and_passport():
    text = "passport number 123456789 on the form"
    assert PIIGuard(only=["passport_us"]).scan(text) == []
    hits = PIIGuard(only=["passport_us"], min_confidence=0.4).scan(text)
    assert hits and hits[0].confidence == pytest.approx(0.5)


def test_scope_and_truncation_and_spec():
    g = PIIGuard(only=["email"])
    g.scope = "chunk"
    assert g.scan("a@b.co")  # scan ignores scope; __call__ uses it
    spec = g.to_spec()
    assert spec["kind"] == "pii" and spec["only"] == ["email"] and spec["scope"] == "chunk"


@pytest.mark.asyncio
async def test_scope_handling_in_call():
    g = PIIGuard(only=["email"])
    g.scope = "buffer"
    d = await g(Context(chunk="co", buffer="write a@b.co"))
    assert d.is_blocked
    g.scope = "chunk"
    assert (await g(Context(chunk="co", buffer="write a@b.co"))).action.value == "allow"
    g.max_input_chars = 20
    g.scope = "auto"
    d2 = await g(Context(buffer="x" * 100))
    assert d2.evidence == {"truncated": True}


def test_register_and_unregister_detector():
    import re

    det = PIIDetector(name="employee_id", pattern=re.compile(r"\bEMP-\d{5}\b"), severity="low")
    with pytest.raises(ValueError):
        register_detector(PIIDetector(name="email", pattern=re.compile("x")))
    register_detector(det)
    try:
        with pytest.raises(ValueError):
            register_detector(det)
        register_detector(det, replace_existing=True)
        assert any(d.name == "employee_id" for d in detectors())
        assert PIIGuard().scan("badge EMP-00042")[0].kind == "employee_id"
    finally:
        assert unregister_detector("employee_id")
    assert not unregister_detector("employee_id")
    assert all(d.name != "employee_id" for d in PIIGuard().active_detectors)


def test_mask_value_rules():
    assert mask_value("1234") == "****"
    assert mask_value("4111111111111111", "critical") == "**************11"
    assert mask_value("alice@example.com", "medium") == "al*************om"


def test_default_detector_table_is_sane():
    names = [d.name for d in DEFAULT_DETECTORS]
    assert len(names) == len(set(names))
    assert all(0 < d.confidence <= 1 for d in DEFAULT_DETECTORS)


@pytest.mark.asyncio
async def test_guard_in_pipeline():
    pipe = GuardPipeline(tier1=[PIIGuard(mode="redact", only=["email", "payment_card"])])
    out = []
    async for x in pipe.guard_stream(["card 5500 0000 0000 0004 for a@b.co"]):
        out.append(x)
    assert out == ["card [PAYMENT_CARD] for [EMAIL]"]
