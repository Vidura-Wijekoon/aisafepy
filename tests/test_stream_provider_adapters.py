"""Tests for the OpenAI Chat Completions and Anthropic Messages adapters.

Both adapters are duck-typed; the fakes below mirror the wire shapes
closely enough (dicts and SimpleNamespace objects) to exercise the
attribute and mapping code paths.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from aisafepy.stream.adapters.anthropic import (
    BLOCKED_STOP_REASON,
    aguard_messages_stream,
    guard_message,
    guard_messages_stream,
)
from aisafepy.stream.adapters.openai_chat import (
    StreamOutcome,
    aguard_chat_completion_stream,
    guard_chat_completion,
    guard_chat_completion_stream,
)
from aisafepy.stream.deterministic import AhoCorasickGuard, RegexGuard
from aisafepy.stream.pipeline import GuardPipeline


def _block_pipeline() -> GuardPipeline:
    return GuardPipeline(tier1=[AhoCorasickGuard(name="bl", terms=["FORBIDDEN"], scope="buffer")])


def _redact_pipeline() -> GuardPipeline:
    # Default scope ("auto") scans the chunk while streaming and the whole
    # buffer for non-streaming payloads, which is what the adapters need.
    return GuardPipeline(tier1=[RegexGuard.compile_pii(redact=True, only=["email"])])


# ---- OpenAI: streaming ---------------------------------------------------------------------


def _oa_chunk(text, *, finish=None, as_obj=False):
    chunk = {
        "id": "chatcmpl-1",
        "model": "gpt-test",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
    }
    if as_obj:
        return SimpleNamespace(
            id=chunk["id"],
            model=chunk["model"],
            choices=[
                SimpleNamespace(
                    index=0, delta=SimpleNamespace(content=text), finish_reason=finish
                )
            ],
        )
    return chunk


def test_openai_stream_passes_clean_chunks_and_records_outcome():
    outcome = StreamOutcome()
    chunks = [_oa_chunk("Hello"), _oa_chunk(", world"), _oa_chunk(None, finish="stop")]
    out = list(guard_chat_completion_stream(_block_pipeline(), chunks, outcome=outcome))
    assert out == chunks
    assert outcome.blocked is False
    assert outcome.text == "Hello, world"
    assert outcome.evaluations == 2
    assert outcome.chunks_forwarded == 3


def test_openai_stream_blocks_mid_stream_with_content_filter_chunk():
    outcome = StreamOutcome()
    chunks = [_oa_chunk("safe start "), _oa_chunk("then FORBIDDEN"), _oa_chunk(" never seen")]
    out = list(guard_chat_completion_stream(_block_pipeline(), chunks, outcome=outcome))
    assert out[0] is chunks[0]
    final = out[-1]
    assert len(out) == 2
    assert final["choices"][0]["finish_reason"] == "content_filter"
    assert final["choices"][0]["delta"]["content"] == "I can't help with that."
    assert final["aisafepy"]["action"] == "block"
    assert final["id"] == "chatcmpl-1"
    assert outcome.blocked is True
    assert outcome.decision is not None and outcome.decision.is_blocked


def test_openai_stream_handles_sdk_style_objects():
    chunks = [_oa_chunk("ok", as_obj=True), _oa_chunk("FORBIDDEN", as_obj=True)]
    out = list(guard_chat_completion_stream(_block_pipeline(), chunks))
    assert out[0] is chunks[0]
    assert out[1]["model"] == "gpt-test"
    assert out[1]["choices"][0]["finish_reason"] == "content_filter"


def test_openai_stream_transform_rewrites_delta():
    chunks = [_oa_chunk("mail me at bob@example.com please")]
    out = list(guard_chat_completion_stream(_redact_pipeline(), chunks))
    assert out[0]["choices"][0]["delta"]["content"] == "mail me at [REDACTED] please"
    assert out[0]["choices"][0]["finish_reason"] is None


def test_openai_stream_evaluate_every_batches_but_checks_tail():
    outcome = StreamOutcome()
    chunks = [_oa_chunk("a"), _oa_chunk("b"), _oa_chunk("c")]
    list(guard_chat_completion_stream(_block_pipeline(), chunks, evaluate_every=2, outcome=outcome))
    # One evaluation after chunk 2, one for the dangling tail.
    assert outcome.evaluations == 2
    assert outcome.text == "abc"


def test_openai_stream_tail_block_emits_filter_chunk():
    chunks = [_oa_chunk("x"), _oa_chunk("FORBIDDEN")]
    out = list(guard_chat_completion_stream(_block_pipeline(), chunks, evaluate_every=5))
    assert out[-1]["choices"][0]["finish_reason"] == "content_filter"
    assert len(out) == 3  # both originals forwarded, then the filter chunk


def test_openai_stream_async_variant():
    async def gen():
        for c in [_oa_chunk("fine"), _oa_chunk("FORBIDDEN")]:
            yield c

    async def run():
        outcome = StreamOutcome()
        out = [c async for c in aguard_chat_completion_stream(_block_pipeline(), gen(), outcome=outcome)]
        return out, outcome

    out, outcome = asyncio.run(run())
    assert outcome.blocked is True
    assert out[-1]["choices"][0]["finish_reason"] == "content_filter"


def test_openai_stream_async_accepts_sync_iterables():
    async def run():
        return [c async for c in aguard_chat_completion_stream(_block_pipeline(), [_oa_chunk("ok")])]

    assert len(asyncio.run(run())) == 1


# ---- OpenAI: non-streaming ----------------------------------------------------------------------


def test_openai_completion_rewrites_blocked_choice():
    response = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "FORBIDDEN"}, "finish_reason": "stop"},
            {"index": 1, "message": {"role": "assistant", "content": "fine"}, "finish_reason": "stop"},
        ]
    }
    decisions = guard_chat_completion(_block_pipeline(), response)
    assert decisions[0].is_blocked and decisions[1].action.value == "allow"
    assert response["choices"][0]["message"]["content"] == "I can't help with that."
    assert response["choices"][0]["finish_reason"] == "content_filter"
    assert response["choices"][1]["message"]["content"] == "fine"


def test_openai_completion_transforms_and_handles_objects():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="write to bob@example.com"), finish_reason="stop"
            ),
            SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="tool_calls"),
        ]
    )
    decisions = guard_chat_completion(_redact_pipeline(), response)
    assert response.choices[0].message.content == "write to [REDACTED]"
    assert decisions[1].rationale == "no text content"


# ---- Anthropic: streaming ---------------------------------------------------------------------


def _anth_events(*texts, index=0):
    events = [
        {"type": "message_start", "message": {"id": "msg_1"}},
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
    ]
    for t in texts:
        events.append({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": t}})
    events += [
        {"type": "content_block_stop", "index": index},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    return events


def test_anthropic_stream_passes_control_events_and_clean_text():
    events = _anth_events("Hello", " there")
    outcome = StreamOutcome()
    out = list(guard_messages_stream(_block_pipeline(), events, outcome=outcome))
    assert out == events
    assert outcome.text == "Hello there"
    assert outcome.evaluations == 2


def test_anthropic_stream_blocks_with_well_formed_tail():
    events = _anth_events("Hello", " FORBIDDEN", " more", index=2)
    outcome = StreamOutcome()
    out = list(guard_messages_stream(_block_pipeline(), events, outcome=outcome))
    types = [e["type"] for e in out]
    # message_start, content_block_start, first delta, then the synthetic tail.
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert out[3]["delta"]["text"] == "I can't help with that."
    assert out[3]["index"] == 2
    assert out[5]["delta"]["stop_reason"] == BLOCKED_STOP_REASON
    assert outcome.blocked is True


def test_anthropic_stream_ignores_non_text_deltas():
    events = [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"FORBIDDEN\""}},
        {"type": "message_stop"},
    ]
    out = list(guard_messages_stream(_block_pipeline(), events))
    assert out == events


def test_anthropic_stream_transform_rewrites_delta_and_objects():
    delta = SimpleNamespace(type="text_delta", text="bob@example.com here")
    event = SimpleNamespace(type="content_block_delta", index=0, delta=delta)
    out = list(guard_messages_stream(_redact_pipeline(), [event]))
    assert out[0]["delta"]["text"] == "[REDACTED] here"


def test_anthropic_stream_async():
    async def gen():
        for e in _anth_events("ok", "FORBIDDEN"):
            yield e

    async def run():
        return [e async for e in aguard_messages_stream(_block_pipeline(), gen())]

    out = asyncio.run(run())
    assert out[-1]["type"] == "message_stop"
    assert out[-2]["delta"]["stop_reason"] == BLOCKED_STOP_REASON


def test_anthropic_stream_tail_block_with_batching():
    out = list(guard_messages_stream(_block_pipeline(), _anth_events("FORBIDDEN"), evaluate_every=3))
    assert out[-2]["delta"]["stop_reason"] == BLOCKED_STOP_REASON


# ---- Anthropic: non-streaming ----------------------------------------------------------------------


def test_anthropic_message_rewrites_blocked_text_block_only():
    message = {
        "content": [
            {"type": "text", "text": "FORBIDDEN"},
            {"type": "tool_use", "name": "x", "input": {"FORBIDDEN": 1}},
            {"type": "text", "text": "fine"},
        ],
        "stop_reason": "end_turn",
    }
    decisions = guard_message(_block_pipeline(), message)
    assert len(decisions) == 2
    assert message["content"][0]["text"] == "I can't help with that."
    assert message["content"][1]["input"] == {"FORBIDDEN": 1}
    assert message["content"][2]["text"] == "fine"
    assert message["stop_reason"] == BLOCKED_STOP_REASON


def test_anthropic_message_transform_on_objects():
    block = SimpleNamespace(type="text", text="bob@example.com")
    message = SimpleNamespace(content=[block], stop_reason="end_turn")
    guard_message(_redact_pipeline(), message)
    assert block.text == "[REDACTED]"
    assert message.stop_reason == "end_turn"


@pytest.mark.parametrize("payload", [{}, {"content": None}, SimpleNamespace(content=[])])
def test_anthropic_message_tolerates_empty_payloads(payload):
    assert guard_message(_block_pipeline(), payload) == []
