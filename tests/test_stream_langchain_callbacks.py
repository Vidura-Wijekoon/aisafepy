"""Tests for the LangChain callback handlers in ``aisafepy.stream.adapters``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from aisafepy.stream.adapters.langchain_callbacks import (
    AsyncGuardrailCallbackHandler,
    GuardrailCallbackHandler,
    GuardrailTripped,
)
from aisafepy.stream.deterministic import AhoCorasickGuard, RegexGuard
from aisafepy.stream.pipeline import Context, GuardPipeline
from aisafepy.stream.toolcalls import ToolCallGuard


def _pipeline() -> GuardPipeline:
    return GuardPipeline(tier1=[AhoCorasickGuard(name="bl", terms=["FORBIDDEN"], scope="buffer")])


def _response(text: str):
    return SimpleNamespace(generations=[[SimpleNamespace(text=text)]])


# ---- sync handler -----------------------------------------------------------------


def test_tokens_accumulate_and_block_mid_stream_raises():
    h = GuardrailCallbackHandler(_pipeline())
    h.on_llm_start({}, ["what is the policy?"], run_id="r1")
    h.on_llm_new_token("the ", run_id="r1")
    h.on_llm_new_token("policy ", run_id="r1")
    with pytest.raises(GuardrailTripped) as exc_info:
        h.on_llm_new_token("is FORBID" + "DEN here", run_id="r1")
    assert exc_info.value.decision.is_blocked
    assert exc_info.value.run_id == "r1"
    assert exc_info.value.stage == "token"
    assert "r1" in h.blocked_runs
    # Further tokens for a blocked run are ignored instead of raising again.
    h.on_llm_new_token("more", run_id="r1")
    assert len(h.decisions) == 3


def test_record_mode_collects_instead_of_raising():
    h = GuardrailCallbackHandler(_pipeline(), on_block="record")
    h.on_llm_new_token("FORBIDDEN", run_id="r2")
    assert h.blocked_runs == {"r2"}
    assert h.decisions[-1].is_blocked
    h.on_llm_end(_response("FORBIDDEN"), run_id="r2")  # no re-evaluation, no raise
    assert len(h.decisions) == 1


def test_on_llm_end_evaluates_final_text_and_clears_state():
    h = GuardrailCallbackHandler(_pipeline())
    h.on_llm_new_token("fine", run_id="r3")
    with pytest.raises(GuardrailTripped) as exc_info:
        h.on_llm_end(_response("final text is FORBIDDEN"), run_id="r3")
    assert exc_info.value.stage == "end"
    assert h.buffer("r3") == ""  # buffer released


def test_prompt_is_threaded_into_context():
    seen = []

    class Spy:
        name = "spy"
        tier = 1

        async def __call__(self, ctx: Context):
            from aisafepy.core.decisions import GuardDecision

            seen.append((ctx.prompt, ctx.role, ctx.metadata.get("run_id")))
            return GuardDecision.allow(tier=1, guard_name="spy")

    h = GuardrailCallbackHandler(GuardPipeline(tier1=[Spy()]))
    h.on_chat_model_start({}, [[{"role": "user", "content": "hello there"}]], run_id="r4")
    h.on_llm_new_token("tok", run_id="r4")
    assert seen == [("hello there", "assistant", "r4")]


def test_chat_model_start_handles_content_parts_and_opt_out():
    h = GuardrailCallbackHandler(_pipeline())
    msg = SimpleNamespace(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    h.on_chat_model_start({}, [[msg]], run_id="r5")
    assert h._prompts["r5"] == "ab"
    h2 = GuardrailCallbackHandler(_pipeline(), prompt_from_messages=False)
    h2.on_chat_model_start({}, [[msg]], run_id="r5")
    assert "r5" not in h2._prompts


def test_evaluate_every_batches_tokens():
    h = GuardrailCallbackHandler(_pipeline(), evaluate_every=3)
    for tok in ("a", "b"):
        h.on_llm_new_token(tok, run_id="r6")
    assert h.decisions == []
    h.on_llm_new_token("c", run_id="r6")
    assert len(h.decisions) == 1


def test_tool_start_runs_tool_guard():
    guard = ToolCallGuard(name="tools", denied_tools=frozenset({"shell"}))
    h = GuardrailCallbackHandler(_pipeline(), tool_guard=guard)
    assert h.ignore_tool is False
    h.on_tool_start({"name": "search"}, '{"query": "weather"}', run_id="t1")
    assert h.decisions[-1].action.value == "allow"
    with pytest.raises(GuardrailTripped) as exc_info:
        h.on_tool_start({"name": "shell"}, "ls -la", run_id="t2")
    assert exc_info.value.stage == "tool"
    assert GuardrailCallbackHandler(_pipeline()).ignore_tool is True


def test_tool_guard_accepts_plain_guard_and_inputs_dict():
    class Spy:
        name = "spy"
        tier = 1

        def __init__(self):
            self.seen = []

        async def __call__(self, ctx: Context):
            from aisafepy.core.decisions import GuardDecision

            self.seen.append(ctx.metadata["tool_call"])
            return GuardDecision.allow(tier=1)

    spy = Spy()
    h = GuardrailCallbackHandler(_pipeline(), tool_guard=spy)
    h.on_tool_start({"name": "calc"}, "ignored", inputs={"a": 1}, run_id="t3")
    assert spy.seen == [{"name": "calc", "arguments": {"a": 1}}]


def test_transform_decisions_are_recorded_not_raised():
    pipe = GuardPipeline(tier1=[RegexGuard.compile_pii(redact=True, only=["email"])])
    h = GuardrailCallbackHandler(pipe)
    h.on_llm_new_token("mail bob@example.com", run_id="r7")
    assert h.decisions[-1].action.value == "transform"
    assert h.blocked_runs == set()


def test_reset_and_error_cleanup_and_flags():
    h = GuardrailCallbackHandler(_pipeline(), on_block="record")
    h.on_llm_new_token("FORBIDDEN", run_id="r8")
    h.on_llm_error(RuntimeError("x"), run_id="r8")
    assert h.buffer("r8") == ""
    h.reset()
    assert h.decisions == [] and h.blocked_runs == set()
    assert GuardrailCallbackHandler.raise_error is True
    assert GuardrailCallbackHandler.ignore_llm is False
    assert GuardrailCallbackHandler.ignore_chain is True
    with pytest.raises(ValueError):
        GuardrailCallbackHandler(_pipeline(), on_block="explode")


def test_sync_handler_works_inside_running_loop():
    async def drive():
        h = GuardrailCallbackHandler(_pipeline(), on_block="record")
        h.on_llm_new_token("FORBIDDEN", run_id="loop")
        return h.blocked_runs

    assert asyncio.run(drive()) == {"loop"}


# ---- async handler ---------------------------------------------------------------


def test_async_handler_blocks_and_records():
    async def drive():
        h = AsyncGuardrailCallbackHandler(_pipeline())
        await h.on_llm_start({}, ["p"], run_id="a1")
        await h.on_llm_new_token("ok ", run_id="a1")
        with pytest.raises(GuardrailTripped):
            await h.on_llm_new_token("FORBIDDEN", run_id="a1")
        rec = AsyncGuardrailCallbackHandler(_pipeline(), on_block="record")
        await rec.on_llm_end(_response("FORBIDDEN"), run_id="a2")
        await rec.on_llm_error(RuntimeError(), run_id="a2")
        return h.blocked_runs, rec.blocked_runs, rec.decisions[-1].is_blocked

    blocked, recorded, is_blocked = asyncio.run(drive())
    assert blocked == {"a1"} and recorded == {"a2"} and is_blocked


def test_async_tool_start_and_chat_start():
    async def drive():
        guard = ToolCallGuard(name="tools", denied_tools=frozenset({"rm"}))
        h = AsyncGuardrailCallbackHandler(_pipeline(), tool_guard=guard, on_block="record")
        await h.on_chat_model_start({}, [[{"content": "hi"}]], run_id="c1")
        await h.on_tool_start({"name": "rm"}, "-rf /", run_id="c1")
        return h._prompts.get("c1"), h.blocked_runs

    prompt, blocked = asyncio.run(drive())
    assert prompt == "hi" and blocked == {"c1"}
