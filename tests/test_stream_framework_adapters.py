"""Tests for the ``aisafepy.stream.adapters`` shims.

The shims are duck-typed against LangChain Runnables, LlamaIndex
post-processors, and OpenAI Agents SDK guardrail callables. We verify
the three observable behaviours each one promises: allow passes the
text through unchanged, transform substitutes the redacted text, and
block substitutes the decision's fallback (or raises a Tripwire).
"""

from __future__ import annotations

import pytest

from aisafepy.core.decisions import Tripwire
from aisafepy.stream import as_openai_output_guardrail
from aisafepy.stream.adapters.langchain import GuardrailRunnable, as_langchain_runnable
from aisafepy.stream.adapters.llamaindex import (
    GuardrailPostprocessor,
    as_llamaindex_postprocessor,
)
from aisafepy.stream.adapters.openai_agents import as_input_guardrail, as_output_guardrail
from aisafepy.stream.deterministic import AhoCorasickGuard, RegexGuard
from aisafepy.stream.pipeline import GuardPipeline


def _block_pipeline() -> GuardPipeline:
    return GuardPipeline(tier1=[AhoCorasickGuard(name="bl", terms=["FORBIDDEN"])])


def _redact_pipeline() -> GuardPipeline:
    return GuardPipeline(tier1=[RegexGuard.compile_pii(redact=True)])


# ---- LangChain -------------------------------------------------------------


def test_langchain_runnable_allows_clean_text():
    runnable = as_langchain_runnable(_block_pipeline())
    assert isinstance(runnable, GuardrailRunnable)
    assert runnable.invoke("all good") == "all good"


def test_langchain_runnable_accepts_dict_input():
    runnable = as_langchain_runnable(_block_pipeline())
    assert runnable.invoke({"output": "all good"}) == "all good"


def test_langchain_runnable_substitutes_fallback_on_block():
    runnable = as_langchain_runnable(_block_pipeline())
    out = runnable.invoke("this is FORBIDDEN")
    assert out != "this is FORBIDDEN"
    assert isinstance(out, str) and out


def test_langchain_runnable_returns_transformed_content():
    runnable = as_langchain_runnable(_redact_pipeline())
    out = runnable.invoke("mail me at bob@example.com")
    assert "bob@example.com" not in out
    assert "[REDACTED]" in out


@pytest.mark.asyncio
async def test_langchain_runnable_async_path():
    runnable = as_langchain_runnable(_block_pipeline())
    assert await runnable.ainvoke("fine") == "fine"
    blocked = await runnable.ainvoke("FORBIDDEN")
    assert blocked != "FORBIDDEN"


# ---- LlamaIndex ------------------------------------------------------------


class _Response:
    def __init__(self, text):
        self.response = text


def test_llamaindex_postprocessor_allows_and_preserves_object():
    pp = as_llamaindex_postprocessor(_block_pipeline())
    assert isinstance(pp, GuardrailPostprocessor)
    resp = _Response("clean")
    assert pp.postprocess_response(resp) is resp
    assert resp.response == "clean"


def test_llamaindex_postprocessor_mutates_response_on_block():
    pp = as_llamaindex_postprocessor(_block_pipeline())
    resp = _Response("FORBIDDEN content")
    out = pp.postprocess_response(resp)
    assert out is resp
    assert "FORBIDDEN" not in resp.response


def test_llamaindex_postprocessor_redacts_in_place():
    pp = as_llamaindex_postprocessor(_redact_pipeline())
    resp = _Response("ssn 123-45-6789")
    pp.postprocess_response(resp)
    assert "123-45-6789" not in resp.response


def test_llamaindex_postprocessor_handles_plain_strings():
    pp = as_llamaindex_postprocessor(_block_pipeline())
    assert pp.postprocess_response("clean") == "clean"
    assert pp.postprocess_response("FORBIDDEN") != "FORBIDDEN"


# ---- OpenAI Agents SDK -----------------------------------------------------


@pytest.mark.asyncio
async def test_openai_output_guardrail_returns_none_when_allowed():
    guardrail = as_output_guardrail(_block_pipeline())
    assert await guardrail(None, None, "clean output") is None


@pytest.mark.asyncio
async def test_openai_output_guardrail_returns_tripwire_on_block():
    guardrail = as_output_guardrail(_block_pipeline())
    result = await guardrail(None, None, "FORBIDDEN output")
    assert isinstance(result, Tripwire)
    assert result.is_blocked
    assert result.tier == 1


@pytest.mark.asyncio
async def test_openai_output_guardrail_reads_output_text_attribute():
    class Out:
        output_text = "FORBIDDEN"

    guardrail = as_output_guardrail(_block_pipeline())
    assert isinstance(await guardrail(None, None, Out()), Tripwire)


@pytest.mark.asyncio
async def test_openai_input_guardrail_uses_user_role():
    seen_roles = []

    class RoleSpy:
        name = "spy"
        tier = 1

        async def __call__(self, ctx):
            seen_roles.append(ctx.role)
            from aisafepy.core.decisions import GuardDecision

            return GuardDecision.allow(tier=1, guard_name=self.name)

    guardrail = as_input_guardrail(GuardPipeline(tier1=[RoleSpy()]))
    assert await guardrail(None, None, "hello") is None
    assert seen_roles == ["user"]


@pytest.mark.asyncio
async def test_package_level_helper_matches_adapter():
    guardrail = as_openai_output_guardrail(_block_pipeline())
    assert isinstance(await guardrail(None, None, "FORBIDDEN"), Tripwire)
