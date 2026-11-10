"""A transform from an earlier tier must survive later tiers that merely allow.

Regression tests for a cascade bug: with only Tier 1 configured, PII
redaction worked; adding any Tier-2 guard made the pipeline return the
Tier-2 ``allow`` and drop the redacted content on the floor.
"""

from __future__ import annotations

import pytest

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.classifiers import StubClassifierGuard
from aisafepy.stream.deterministic import RegexGuard
from aisafepy.stream.pipeline import Context, GuardPipeline


class _Uncertain:
    """A Tier-2 guard that allows with mid-band confidence so Tier 3 runs."""

    name = "uncertain"
    tier = 2

    async def __call__(self, ctx: Context) -> GuardDecision:
        return GuardDecision.allow(tier=2, confidence=0.5, guard_name=self.name)


class _Tier3Allow:
    name = "t3"
    tier = 3

    async def __call__(self, ctx: Context) -> GuardDecision:
        return GuardDecision.allow(tier=3, guard_name=self.name)


def _redactor() -> RegexGuard:
    return RegexGuard.compile_pii(redact=True, only=["email"])


@pytest.mark.asyncio
async def test_tier1_transform_survives_tier2_allow():
    pipe = GuardPipeline(tier1=[_redactor()], tier2=[StubClassifierGuard(triggers=("HARMFUL",))])
    d = await pipe.evaluate_chunk(Context(buffer="mail bob@example.com"))
    assert d.action.value == "transform"
    assert d.transformed_content == "mail [REDACTED]"
    assert d.tier == 1


@pytest.mark.asyncio
async def test_tier1_transform_survives_tier3_allow():
    pipe = GuardPipeline(tier1=[_redactor()], tier2=[_Uncertain()], tier3=[_Tier3Allow()])
    d = await pipe.evaluate_chunk(Context(buffer="mail bob@example.com"))
    assert d.action.value == "transform"
    assert d.transformed_content == "mail [REDACTED]"


@pytest.mark.asyncio
async def test_later_block_still_wins_over_earlier_transform():
    pipe = GuardPipeline(tier1=[_redactor()], tier2=[StubClassifierGuard(triggers=("HARMFUL",))])
    d = await pipe.evaluate_chunk(Context(buffer="HARMFUL mail bob@example.com"))
    assert d.is_blocked
    assert d.tier == 2


@pytest.mark.asyncio
async def test_transform_survives_budget_exhaustion():
    pipe = GuardPipeline(
        tier1=[_redactor()],
        tier2=[StubClassifierGuard(triggers=("HARMFUL",))],
        budget_ms_p95=0.0,
    )
    d = await pipe.evaluate_chunk(Context(buffer="mail bob@example.com"))
    assert d.action.value == "transform"


@pytest.mark.asyncio
async def test_stream_yields_redacted_text_with_tier2_present():
    pipe = GuardPipeline(tier1=[_redactor()], tier2=[StubClassifierGuard(triggers=("HARMFUL",))])
    out = [x async for x in pipe.guard_stream(["write to bob@example.com please"])]
    assert out == ["write to [REDACTED] please"]


@pytest.mark.asyncio
async def test_plain_allow_is_unchanged_without_transform():
    pipe = GuardPipeline(tier1=[_redactor()], tier2=[StubClassifierGuard(triggers=("HARMFUL",))])
    d = await pipe.evaluate_chunk(Context(buffer="nothing personal"))
    assert d.action.value == "allow"
    assert d.tier == 2
