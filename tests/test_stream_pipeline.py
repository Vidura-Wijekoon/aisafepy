from __future__ import annotations

import pytest

from aisafepy.stream.classifiers import StubClassifierGuard
from aisafepy.stream.deterministic import AhoCorasickGuard, RegexGuard
from aisafepy.stream.pipeline import Context, GuardPipeline


@pytest.mark.asyncio
async def test_empty_pipeline_allows():
    pipe = GuardPipeline()
    d = await pipe.evaluate_chunk(Context(buffer="hello world"))
    assert d.action.value == "allow"


@pytest.mark.asyncio
async def test_tier1_block_short_circuits_cascade():
    pipe = GuardPipeline(
        tier1=[AhoCorasickGuard(name="bl", terms=["BAD"])],
        tier2=[StubClassifierGuard(triggers=("SHOULD_NOT_REACH",))],
    )
    d = await pipe.evaluate_chunk(Context(buffer="this is BAD"))
    assert d.is_blocked
    assert d.tier == 1


@pytest.mark.asyncio
async def test_tier2_runs_when_tier1_allows():
    pipe = GuardPipeline(
        tier1=[AhoCorasickGuard(name="bl", terms=["NOT_THERE"])],
        tier2=[StubClassifierGuard(triggers=("UNSAFE",))],
    )
    d = await pipe.evaluate_chunk(Context(buffer="please be UNSAFE here"))
    assert d.is_blocked
    assert d.tier == 2


@pytest.mark.asyncio
async def test_streaming_yields_chunks_until_block():
    pipe = GuardPipeline(
        tier1=[AhoCorasickGuard(name="bl", terms=["STOP"])],
    )
    chunks = ["safe ", "still safe ", "now STOP and explode"]
    out = []
    async for x in pipe.guard_stream(chunks):
        out.append(x)
    # The first two chunks pass through, then a GuardDecision is yielded.
    assert out[0] == "safe "
    assert out[1] == "still safe "
    assert hasattr(out[-1], "action")
    assert out[-1].is_blocked


@pytest.mark.asyncio
async def test_regex_transform_is_passed_through_stream():
    pipe = GuardPipeline(
        tier1=[RegexGuard.compile_pii(redact=True)],
    )
    chunks = ["My email is alice@example.com today."]
    out = []
    async for x in pipe.guard_stream(chunks):
        out.append(x)
    assert any("[REDACTED]" in str(c) for c in out)


@pytest.mark.asyncio
async def test_run_sync_returns_decision():
    pipe = GuardPipeline(tier1=[AhoCorasickGuard(name="bl", terms=["x"])])
    d = pipe.run_sync(Context(buffer="hello x"))
    assert d.is_blocked


@pytest.mark.asyncio
async def test_failing_guard_fails_open_with_evidence():
    class Broken:
        name = "broken"
        tier = 1

        async def __call__(self, _ctx):
            raise RuntimeError("boom")

    pipe = GuardPipeline(tier1=[Broken()])
    d = await pipe.evaluate_chunk(Context(buffer="hi"))
    assert d.action.value == "allow"
    assert "error" in d.evidence
