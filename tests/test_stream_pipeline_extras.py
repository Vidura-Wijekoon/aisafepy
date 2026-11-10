from __future__ import annotations

import asyncio

import pytest

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.classifiers import StubClassifierGuard
from aisafepy.stream.deterministic import AhoCorasickGuard, RegexGuard
from aisafepy.stream.metrics import PipelineMetrics
from aisafepy.stream.pipeline import Context, GuardPipeline


class Recording:
    """Guard that records how often it was called and returns a fixed decision."""

    def __init__(self, name: str, tier: int, action: str = "allow", delay: float = 0.0):
        self.name = name
        self.tier = tier
        self.action = action
        self.delay = delay
        self.calls: list[str] = []

    async def __call__(self, ctx: Context) -> GuardDecision:
        self.calls.append(ctx.buffer)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.action == "block":
            return GuardDecision.block(tier=self.tier, confidence=0.95, rationale="rec", guard_name=self.name)
        if self.action == "escalate":
            return GuardDecision.escalate(tier=self.tier, confidence=0.5, rationale="rec", guard_name=self.name)
        return GuardDecision.allow(tier=self.tier, guard_name=self.name, rationale="rec")


# ---- shadow guards ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_guard_never_changes_outcome_but_is_reported():
    shadow = Recording("shadow-block", tier=1, action="block")
    seen: list[GuardDecision] = []
    metrics = PipelineMetrics()
    pipe = GuardPipeline(
        tier1=[Recording("t1", 1)],
        shadow_guards=[shadow],
        on_decision=lambda d, ctx: seen.append(d),
        metrics=metrics,
    )
    d = await pipe.evaluate_chunk(Context(buffer="hello"))
    assert d.action.value == "allow"
    assert shadow.calls == ["hello"]
    shadows = [x for x in seen if x.evidence.get("shadow")]
    finals = [x for x in seen if not x.evidence.get("shadow")]
    assert len(shadows) == 1 and shadows[0].is_blocked and shadows[0].guard_name == "shadow-block"
    assert len(finals) == 1 and finals[0].action.value == "allow"
    snap = metrics.snapshot()
    assert snap["shadow"]["shadow-block"]["actions"]["block"] == 1
    assert snap["final"]["actions"]["allow"] == 1


@pytest.mark.asyncio
async def test_shadow_runs_even_when_tier1_blocks():
    shadow = Recording("shadow", tier=1)
    pipe = GuardPipeline(tier1=[AhoCorasickGuard(name="bl", terms=["STOP"])], shadow_guards=[shadow])
    d = await pipe.evaluate_chunk(Context(buffer="STOP now"))
    assert d.is_blocked
    assert shadow.calls == ["STOP now"]


# ---- on_decision ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_decision_sees_every_final_decision_and_survives_errors():
    seen: list[str] = []

    async def hook(d: GuardDecision, ctx: Context) -> None:
        seen.append(f"{d.action.value}:{ctx.buffer}")
        if "boom" in ctx.buffer:
            raise RuntimeError("hook failure")

    pipe = GuardPipeline(tier1=[AhoCorasickGuard(name="bl", terms=["BAD"])], on_decision=hook)
    assert (await pipe.evaluate_chunk(Context(buffer="ok"))).action.value == "allow"
    assert (await pipe.evaluate_chunk(Context(buffer="BAD"))).is_blocked
    assert (await pipe.evaluate_chunk(Context(buffer="boom"))).action.value == "allow"
    assert seen == ["allow:ok", "block:BAD", "allow:boom"]


# ---- metrics ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_record_guards_tiers_and_finals():
    metrics = PipelineMetrics()
    pipe = GuardPipeline(
        tier1=[Recording("a", 1), Recording("b", 1)],
        tier2=[StubClassifierGuard(name="stub", triggers=("X",))],
        metrics=metrics,
    )
    await pipe.evaluate_chunk(Context(buffer="clean"))
    await pipe.evaluate_chunk(Context(buffer="X marks"))
    snap = metrics.snapshot()
    assert snap["guards"]["a"]["count"] == 2
    assert snap["guards"]["stub"]["actions"]["block"] == 1
    assert snap["tiers"]["1"]["count"] == 2 and snap["tiers"]["2"]["count"] == 2
    assert snap["final"]["count"] == 2 and snap["final"]["actions"]["block"] == 1
    assert "aisafepy_guard_evaluations_total" in metrics.to_prometheus()


# ---- timeouts -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tier_timeout_fails_open_with_evidence():
    slow = Recording("slow", 2, action="block", delay=0.3)
    metrics = PipelineMetrics()
    pipe = GuardPipeline(tier1=[Recording("t1", 1)], tier2=[slow], tier_timeouts_ms={2: 20}, metrics=metrics)
    d = await pipe.evaluate_chunk(Context(buffer="x"))
    assert d.action.value == "allow"
    assert d.evidence.get("timeout") is True
    assert d.guard_name == "slow"
    assert metrics.guard("slow").timeouts == 1
    # Tier 1 has no timeout configured and is unaffected.
    assert metrics.guard("t1").timeouts == 0


@pytest.mark.asyncio
async def test_tier_timeout_not_triggered_when_fast_enough():
    fast = Recording("fast", 2, action="block", delay=0.001)
    pipe = GuardPipeline(tier1=[Recording("t1", 1)], tier2=[fast], tier_timeouts_ms={2: 500})
    d = await pipe.evaluate_chunk(Context(buffer="x"))
    assert d.is_blocked


# ---- streaming cadence ---------------------------------------------------------------------


async def _collect(pipe: GuardPipeline, chunks, **kw):
    out = []
    async for x in pipe.guard_stream(chunks, **kw):
        out.append(x)
    return out


@pytest.mark.asyncio
async def test_evaluate_every_batches_but_always_checks_last_chunk():
    rec = Recording("rec", 1)
    pipe = GuardPipeline(tier1=[rec])
    chunks = ["a", "b", "c", "d", "e"]
    out = await _collect(pipe, chunks, evaluate_every=2)
    assert out == chunks  # every chunk still yielded, in order
    # Evaluated after chunk 2, 4 and the final chunk 5.
    assert rec.calls == ["ab", "abcd", "abcde"]


@pytest.mark.asyncio
async def test_evaluate_every_blocks_at_batch_boundary_and_withholds_pending():
    pipe = GuardPipeline(tier1=[AhoCorasickGuard(name="bl", terms=["BAD"])])
    out = await _collect(pipe, ["ok ", "BAD", " more", "tail"], evaluate_every=2)
    # First batch ("ok BAD") is blocked: nothing from it leaks.
    assert out[0].is_blocked if hasattr(out[0], "is_blocked") else False
    assert len(out) == 1


@pytest.mark.asyncio
async def test_min_buffer_chars_delays_evaluation():
    rec = Recording("rec", 1)
    pipe = GuardPipeline(tier1=[rec])
    out = await _collect(pipe, ["ab", "cd", "ef", "gh"], min_buffer_chars=5)
    assert out == ["ab", "cd", "ef", "gh"]
    assert rec.calls == ["abcdef", "abcdefgh"]


@pytest.mark.asyncio
async def test_default_cadence_matches_legacy_behaviour():
    rec = Recording("rec", 1)
    pipe = GuardPipeline(tier1=[rec])
    out = await _collect(pipe, ["x", "y", "z"])
    assert out == ["x", "y", "z"]
    assert rec.calls == ["x", "xy", "xyz"]
    with pytest.raises(ValueError):
        await _collect(pipe, ["x"], evaluate_every=0)


@pytest.mark.asyncio
async def test_transform_with_batching_yields_transformed_batch():
    pipe = GuardPipeline(tier1=[RegexGuard.compile_pii(redact=True)])
    out = await _collect(pipe, ["mail: alice", "@example.com", " ok"], evaluate_every=2)
    assert any("[REDACTED]" in str(c) for c in out)
    assert out[-1] == " ok"


# ---- convenience ------------------------------------------------------------------------------


def test_evaluate_text_and_describe():
    pipe = GuardPipeline(
        name="demo",
        tier1=[AhoCorasickGuard(name="bl", terms=["nope"])],
        tier2=[StubClassifierGuard()],
        shadow_guards=[StubClassifierGuard(name="sh")],
        tier_timeouts_ms={2: 50},
    )
    d = pipe.evaluate_text("this is nope", prompt="p", role="user", metadata={"k": 1})
    assert d.is_blocked
    text = pipe.describe()
    assert "GuardPipeline 'demo'" in text
    assert "tier 1: 1 guard(s)" in text and "bl (AhoCorasickGuard)" in text
    assert "shadow: 1 guard(s)" in text and "tier_timeouts_ms={2: 50}" in text
    assert "metrics=none" in text


def test_from_dict_builds_pipeline():
    pipe = GuardPipeline.from_dict(
        {"name": "cfg", "tiers": {"1": [{"kind": "blocklist", "terms": ["zz"]}]}}
    )
    assert pipe.name == "cfg"
    assert pipe.evaluate_text("has zz").is_blocked
    assert pipe.to_dict()["tiers"][1][0]["kind"] == "blocklist"
