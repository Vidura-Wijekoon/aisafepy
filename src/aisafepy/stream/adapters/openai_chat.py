"""OpenAI Chat Completions adapter (streaming and non-streaming).

The OpenAI Agents SDK adapter in :mod:`aisafepy.stream.adapters.openai_agents`
covers agent *guardrails*. This module covers the far more common plain
``client.chat.completions.create(...)`` call, in both shapes:

* **Streaming.** :func:`guard_chat_completion_stream` consumes the
  iterator of ``ChatCompletionChunk`` objects, runs the cascade on the
  accumulated assistant text after every chunk, and forwards chunks
  until a terminal decision. On block it stops the upstream stream,
  emits one synthetic chunk carrying the decision's ``fallback`` text
  (so the UI shows a clean message instead of a half sentence) and a
  ``finish_reason`` of ``"content_filter"``, then ends.
* **Non-streaming.** :func:`guard_chat_completion` evaluates every
  choice's ``message.content`` and rewrites blocked or transformed
  choices in place.

Everything is duck-typed. Chunks may be SDK objects (attribute access)
or plain dicts (the shape proxies and tests use); both are handled by
:func:`_delta_text`. The adapter never imports ``openai``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from aisafepy.core.decisions import Action, GuardDecision
from aisafepy.stream.pipeline import Context, GuardPipeline


@dataclass
class StreamOutcome:
    """Summary of one guarded stream, available after iteration ends."""

    chunks_forwarded: int = 0
    text: str = ""
    decision: GuardDecision | None = None
    """The terminal decision, or the last allow when the stream ended cleanly."""
    blocked: bool = False
    evaluations: int = 0
    history: list[GuardDecision] = field(default_factory=list)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _delta_text(chunk: Any) -> str:
    """Extract the assistant text delta from a chunk, or ``""``."""
    choices = _get(chunk, "choices") or []
    if not choices:
        return ""
    delta = _get(choices[0], "delta")
    if delta is None:
        return ""
    content = _get(delta, "content")
    return content if isinstance(content, str) else ""


def _synthetic_chunk(template: Any, text: str, *, finish_reason: str, decision: GuardDecision) -> dict[str, Any]:
    """Build a dict chunk in the OpenAI wire shape carrying the fallback text."""
    return {
        "id": _get(template, "id", "aisafepy"),
        "object": "chat.completion.chunk",
        "model": _get(template, "model", None),
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "aisafepy": {
            "decision_id": decision.decision_id,
            "action": decision.action.value,
            "rationale": decision.rationale,
            "guard": decision.guard_name,
            "tier": decision.tier,
        },
    }


def guard_chat_completion_stream(
    pipeline: GuardPipeline,
    stream: Iterable[Any],
    *,
    prompt: str | None = None,
    evaluate_every: int = 1,
    outcome: StreamOutcome | None = None,
) -> Iterator[Any]:
    """Synchronous wrapper around a streaming chat completion.

    ``evaluate_every`` lets expensive cascades evaluate every N text
    chunks instead of every one; the final text is always evaluated.
    Pass an :class:`StreamOutcome` to read the terminal decision after
    the generator is exhausted.
    """
    outcome = outcome if outcome is not None else StreamOutcome()
    buffer: list[str] = []
    pending = 0
    last_chunk: Any = None
    for chunk in stream:
        last_chunk = chunk
        text = _delta_text(chunk)
        if text:
            buffer.append(text)
            pending += 1
        if text and pending >= max(1, evaluate_every):
            pending = 0
            decision = pipeline.run_sync(Context(chunk=text, buffer="".join(buffer), prompt=prompt))
            outcome.evaluations += 1
            outcome.history.append(decision)
            outcome.decision = decision
            if decision.is_terminal:
                outcome.blocked = decision.action == Action.BLOCK
                outcome.text = "".join(buffer)
                yield _synthetic_chunk(
                    chunk,
                    decision.fallback or "",
                    finish_reason="content_filter",
                    decision=decision,
                )
                return
            if decision.action == Action.TRANSFORM and decision.transformed_content is not None:
                # Replace the just-emitted delta with the transformed tail.
                chunk = _synthetic_chunk(
                    chunk, decision.transformed_content, finish_reason=None, decision=decision
                )
                buffer[-1] = decision.transformed_content
        outcome.chunks_forwarded += 1
        yield chunk
    if pending and buffer:
        decision = pipeline.run_sync(Context(chunk=buffer[-1], buffer="".join(buffer), prompt=prompt))
        outcome.evaluations += 1
        outcome.history.append(decision)
        outcome.decision = decision
        if decision.is_terminal:
            outcome.blocked = decision.action == Action.BLOCK
            yield _synthetic_chunk(
                last_chunk, decision.fallback or "", finish_reason="content_filter", decision=decision
            )
    outcome.text = "".join(buffer)


async def aguard_chat_completion_stream(
    pipeline: GuardPipeline,
    stream: AsyncIterator[Any] | Iterable[Any],
    *,
    prompt: str | None = None,
    evaluate_every: int = 1,
    outcome: StreamOutcome | None = None,
) -> AsyncIterator[Any]:
    """Async variant of :func:`guard_chat_completion_stream`."""
    outcome = outcome if outcome is not None else StreamOutcome()
    buffer: list[str] = []
    pending = 0
    last_chunk: Any = None
    async for chunk in _aiter(stream):
        last_chunk = chunk
        text = _delta_text(chunk)
        if text:
            buffer.append(text)
            pending += 1
        if text and pending >= max(1, evaluate_every):
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
                yield _synthetic_chunk(
                    chunk, decision.fallback or "", finish_reason="content_filter", decision=decision
                )
                return
            if decision.action == Action.TRANSFORM and decision.transformed_content is not None:
                chunk = _synthetic_chunk(
                    chunk, decision.transformed_content, finish_reason=None, decision=decision
                )
                buffer[-1] = decision.transformed_content
        outcome.chunks_forwarded += 1
        yield chunk
    if pending and buffer:
        decision = await pipeline.evaluate_chunk(
            Context(chunk=buffer[-1], buffer="".join(buffer), prompt=prompt)
        )
        outcome.evaluations += 1
        outcome.history.append(decision)
        outcome.decision = decision
        if decision.is_terminal:
            outcome.blocked = decision.action == Action.BLOCK
            yield _synthetic_chunk(
                last_chunk, decision.fallback or "", finish_reason="content_filter", decision=decision
            )
    outcome.text = "".join(buffer)


def guard_chat_completion(
    pipeline: GuardPipeline,
    response: Any,
    *,
    prompt: str | None = None,
) -> list[GuardDecision]:
    """Evaluate every choice of a non-streaming completion in place.

    Blocked choices get their ``message.content`` replaced with the
    decision's fallback and ``finish_reason`` set to ``"content_filter"``;
    transformed choices get the transformed text. Returns one decision
    per choice so callers can log them.
    """
    decisions: list[GuardDecision] = []
    for choice in _get(response, "choices") or []:
        message = _get(choice, "message")
        content = _get(message, "content") if message is not None else None
        if not isinstance(content, str):
            decisions.append(GuardDecision.allow(tier=0, rationale="no text content"))
            continue
        decision = pipeline.run_sync(Context(buffer=content, prompt=prompt))
        decisions.append(decision)
        if decision.is_terminal:
            _set(message, "content", decision.fallback or "")
            _set(choice, "finish_reason", "content_filter")
        elif decision.action == Action.TRANSFORM and decision.transformed_content is not None:
            _set(message, "content", decision.transformed_content)
    return decisions


def _set(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        with contextlib.suppress(Exception):  # frozen SDK models
            setattr(obj, key, value)


async def _aiter(source: AsyncIterator[Any] | Iterable[Any]) -> AsyncIterator[Any]:
    if hasattr(source, "__aiter__"):
        async for item in source:  # type: ignore[union-attr]
            yield item
    else:
        for item in source:  # type: ignore[union-attr]
            yield item
            await asyncio.sleep(0)
