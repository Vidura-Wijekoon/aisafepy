"""Anthropic Messages API adapter (streaming and non-streaming).

Anthropic streams a sequence of typed events. The assistant text lives
in ``content_block_delta`` events whose ``delta.type`` is
``"text_delta"``; everything else (``message_start``,
``content_block_start``, ``ping``, ``message_delta``, ``message_stop``)
is control traffic that should pass through untouched.

:func:`guard_messages_stream` forwards events and evaluates the
accumulated text after each text delta. On a terminal decision it
emits a synthetic ``content_block_delta`` carrying the fallback text, a
``content_block_stop``, a ``message_delta`` with
``stop_reason="aisafepy_blocked"``, and a ``message_stop``, so a
client written against the normal event grammar terminates cleanly.

:func:`guard_message` handles the non-streaming ``Message`` object by
evaluating each text block and rewriting blocked or transformed blocks
in place.

The module is duck-typed and never imports ``anthropic``; events may be
SDK objects or plain dicts.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

from aisafepy.core.decisions import Action, GuardDecision
from aisafepy.stream.adapters.openai_chat import StreamOutcome
from aisafepy.stream.pipeline import Context, GuardPipeline

BLOCKED_STOP_REASON = "aisafepy_blocked"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        with contextlib.suppress(Exception):  # frozen SDK models
            setattr(obj, key, value)


def _text_delta(event: Any) -> str | None:
    """Return the text of a ``text_delta`` event, else ``None``."""
    if _get(event, "type") != "content_block_delta":
        return None
    delta = _get(event, "delta")
    if delta is None or _get(delta, "type") != "text_delta":
        return None
    text = _get(delta, "text")
    return text if isinstance(text, str) else None


def _block_events(index: int, decision: GuardDecision) -> list[dict[str, Any]]:
    meta = {
        "decision_id": decision.decision_id,
        "action": decision.action.value,
        "rationale": decision.rationale,
        "guard": decision.guard_name,
        "tier": decision.tier,
    }
    return [
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": decision.fallback or ""},
            "aisafepy": meta,
        },
        {"type": "content_block_stop", "index": index},
        {
            "type": "message_delta",
            "delta": {"stop_reason": BLOCKED_STOP_REASON, "stop_sequence": None},
            "usage": {"output_tokens": 0},
            "aisafepy": meta,
        },
        {"type": "message_stop"},
    ]


def guard_messages_stream(
    pipeline: GuardPipeline,
    events: Iterable[Any],
    *,
    prompt: str | None = None,
    evaluate_every: int = 1,
    outcome: StreamOutcome | None = None,
) -> Iterator[Any]:
    """Synchronous guarded pass-through of an Anthropic event stream."""
    outcome = outcome if outcome is not None else StreamOutcome()
    buffer: list[str] = []
    pending = 0
    current_index = 0
    for event in events:
        idx = _get(event, "index")
        if isinstance(idx, int):
            current_index = idx
        text = _text_delta(event)
        if text:
            buffer.append(text)
            pending += 1
            if pending >= max(1, evaluate_every):
                pending = 0
                decision = pipeline.run_sync(
                    Context(chunk=text, buffer="".join(buffer), prompt=prompt)
                )
                outcome.evaluations += 1
                outcome.history.append(decision)
                outcome.decision = decision
                if decision.is_terminal:
                    outcome.blocked = decision.action == Action.BLOCK
                    outcome.text = "".join(buffer)
                    yield from _block_events(current_index, decision)
                    return
                if decision.action == Action.TRANSFORM and decision.transformed_content is not None:
                    buffer[-1] = decision.transformed_content
                    event = {
                        "type": "content_block_delta",
                        "index": current_index,
                        "delta": {"type": "text_delta", "text": decision.transformed_content},
                    }
        outcome.chunks_forwarded += 1
        yield event
    if pending and buffer:
        decision = pipeline.run_sync(Context(chunk=buffer[-1], buffer="".join(buffer), prompt=prompt))
        outcome.evaluations += 1
        outcome.history.append(decision)
        outcome.decision = decision
        if decision.is_terminal:
            outcome.blocked = decision.action == Action.BLOCK
            yield from _block_events(current_index, decision)
    outcome.text = "".join(buffer)


async def aguard_messages_stream(
    pipeline: GuardPipeline,
    events: AsyncIterator[Any] | Iterable[Any],
    *,
    prompt: str | None = None,
    evaluate_every: int = 1,
    outcome: StreamOutcome | None = None,
) -> AsyncIterator[Any]:
    """Async variant of :func:`guard_messages_stream`."""
    outcome = outcome if outcome is not None else StreamOutcome()
    buffer: list[str] = []
    pending = 0
    current_index = 0
    async for event in _aiter(events):
        idx = _get(event, "index")
        if isinstance(idx, int):
            current_index = idx
        text = _text_delta(event)
        if text:
            buffer.append(text)
            pending += 1
            if pending >= max(1, evaluate_every):
                pending = 0
                decision = await pipeline.evaluate_chunk(
                    Context(chunk=text, buffer="".join(buffer), prompt=prompt)
                )
                outcome.evaluations += 1
                outcome.history.append(decision)
                outcome.decision = decision
                if decision.is_terminal:
                    outcome.blocked = decision.action == Action.BLOCK
                    outcome.text = "".join(buffer)
                    for ev in _block_events(current_index, decision):
                        yield ev
                    return
                if decision.action == Action.TRANSFORM and decision.transformed_content is not None:
                    buffer[-1] = decision.transformed_content
                    event = {
                        "type": "content_block_delta",
                        "index": current_index,
                        "delta": {"type": "text_delta", "text": decision.transformed_content},
                    }
        outcome.chunks_forwarded += 1
        yield event
    if pending and buffer:
        decision = await pipeline.evaluate_chunk(
            Context(chunk=buffer[-1], buffer="".join(buffer), prompt=prompt)
        )
        outcome.evaluations += 1
        outcome.history.append(decision)
        outcome.decision = decision
        if decision.is_terminal:
            outcome.blocked = decision.action == Action.BLOCK
            for ev in _block_events(current_index, decision):
                yield ev
    outcome.text = "".join(buffer)


def guard_message(
    pipeline: GuardPipeline,
    message: Any,
    *,
    prompt: str | None = None,
) -> list[GuardDecision]:
    """Evaluate each text block of a non-streaming ``Message`` in place.

    Blocked blocks are replaced with the fallback text and the message's
    ``stop_reason`` is set to :data:`BLOCKED_STOP_REASON`. Tool-use and
    other non-text blocks are left alone. Returns one decision per text
    block.
    """
    decisions: list[GuardDecision] = []
    for block in _get(message, "content") or []:
        if _get(block, "type") != "text":
            continue
        text = _get(block, "text")
        if not isinstance(text, str):
            continue
        decision = pipeline.run_sync(Context(buffer=text, prompt=prompt))
        decisions.append(decision)
        if decision.is_terminal:
            _set(block, "text", decision.fallback or "")
            _set(message, "stop_reason", BLOCKED_STOP_REASON)
        elif decision.action == Action.TRANSFORM and decision.transformed_content is not None:
            _set(block, "text", decision.transformed_content)
    return decisions


async def _aiter(source: AsyncIterator[Any] | Iterable[Any]) -> AsyncIterator[Any]:
    if hasattr(source, "__aiter__"):
        async for item in source:  # type: ignore[union-attr]
            yield item
    else:
        for item in source:  # type: ignore[union-attr]
            yield item
            await asyncio.sleep(0)
