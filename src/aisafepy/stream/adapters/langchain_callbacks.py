"""LangChain callback handlers that run a :class:`GuardPipeline` on streamed tokens.

LangChain emits ``on_llm_new_token`` for every streamed token and
``on_llm_end`` with the final generation; chat models announce their
input through ``on_chat_model_start`` and tools through
``on_tool_start``. :class:`GuardrailCallbackHandler` turns those
events into cascade evaluations:

* tokens are accumulated per ``run_id`` and the growing buffer is
  evaluated after each token (``evaluate_every`` batches when the
  cascade is expensive);
* the final generation is always evaluated once more in ``on_llm_end``;
* the user prompt captured at ``on_llm_start`` / ``on_chat_model_start``
  is passed to guards through ``Context.prompt``;
* tool invocations are screened with an optional ``tool_guard`` (for
  example :class:`aisafepy.stream.ToolCallGuard`) over
  ``{"name": ..., "arguments": ...}``.

A terminal decision either raises :class:`GuardrailTripped`
(``on_block="raise"``, which aborts the chain because ``raise_error`` is
set) or is recorded on ``.decisions`` and ``.blocked_runs``
(``on_block="record"``) for callers that prefer to inspect afterwards.

The handler is duck-typed against ``BaseCallbackHandler``: it defines
the same method names and the ``ignore_*`` / ``raise_error`` class
attributes LangChain inspects, and imports nothing from LangChain.
:class:`AsyncGuardrailCallbackHandler` is the ``AsyncCallbackHandler``
counterpart with coroutine methods.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any, Literal

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context, GuardPipeline

OnBlock = Literal["raise", "record"]


class GuardrailTripped(Exception):
    """Raised inside a LangChain run when the cascade blocks or escalates."""

    def __init__(self, decision: GuardDecision, *, run_id: str | None = None, stage: str = "llm"):
        super().__init__(decision.rationale)
        self.decision = decision
        self.run_id = run_id
        self.stage = stage


def _run_coroutine(coro: Any) -> Any:
    """Run ``coro`` to completion from sync code, inside or outside a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _message_text(message: Any) -> str | None:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts) if parts else None
    return None


def _generation_text(response: Any) -> str | None:
    generations = getattr(response, "generations", None)
    if generations is None and isinstance(response, dict):
        generations = response.get("generations")
    if not generations:
        return None
    first = generations[0]
    if isinstance(first, list | tuple):
        first = first[0] if first else None
    if first is None:
        return None
    text = first.get("text") if isinstance(first, dict) else getattr(first, "text", None)
    if isinstance(text, str):
        return text
    message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
    return _message_text(message) if message is not None else None


def _tool_arguments(input_str: Any, inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return inputs
    if isinstance(input_str, dict):
        return input_str
    if isinstance(input_str, str):
        stripped = input_str.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"input": input_str}
    return {"input": input_str}


class _HandlerState:
    """Shared bookkeeping for the sync and async handlers."""

    # Attributes LangChain reads to decide which events to deliver.
    raise_error = True
    run_inline = True
    ignore_llm = False
    ignore_chat_model = False
    ignore_chain = True
    ignore_agent = True
    ignore_retriever = True
    ignore_retry = True
    ignore_custom_event = True

    def __init__(
        self,
        pipeline: GuardPipeline,
        *,
        on_block: OnBlock = "raise",
        prompt_from_messages: bool = True,
        tool_guard: Any | None = None,
        evaluate_every: int = 1,
        role: Literal["user", "assistant", "tool"] = "assistant",
    ) -> None:
        if on_block not in ("raise", "record"):
            raise ValueError("on_block must be 'raise' or 'record'")
        self.pipeline = pipeline
        self.on_block: OnBlock = on_block
        self.prompt_from_messages = prompt_from_messages
        self.tool_guard = tool_guard
        self.evaluate_every = max(1, int(evaluate_every))
        self.role = role
        self.decisions: list[GuardDecision] = []
        self.blocked_runs: set[str] = set()
        self._buffers: dict[str, list[str]] = {}
        self._pending: dict[str, int] = {}
        self._prompts: dict[str, str] = {}

    # ---- helpers ----------------------------------------------------------

    @property
    def ignore_tool(self) -> bool:
        return self.tool_guard is None

    def reset(self) -> None:
        self.decisions.clear()
        self.blocked_runs.clear()
        self._buffers.clear()
        self._pending.clear()
        self._prompts.clear()

    def buffer(self, run_id: Any) -> str:
        return "".join(self._buffers.get(_key(run_id), []))

    def _remember_prompt(self, run_id: Any, prompt: str | None) -> None:
        if prompt is not None:
            self._prompts[_key(run_id)] = prompt

    def _append(self, run_id: Any, token: str) -> bool:
        key = _key(run_id)
        self._buffers.setdefault(key, []).append(token)
        self._pending[key] = self._pending.get(key, 0) + 1
        if self._pending[key] >= self.evaluate_every:
            self._pending[key] = 0
            return True
        return False

    def _context(self, run_id: Any, *, chunk: str | None, buffer: str) -> Context:
        return Context(
            chunk=chunk,
            buffer=buffer,
            prompt=self._prompts.get(_key(run_id)),
            role=self.role,
            metadata={"run_id": _key(run_id)},
        )

    def _tool_context(self, serialized: Any, input_str: Any, inputs: Any, run_id: Any) -> Context:
        name = serialized.get("name") if isinstance(serialized, dict) else getattr(serialized, "name", None)
        call = {"name": name or "tool", "arguments": _tool_arguments(input_str, inputs)}
        return Context(
            chunk=json.dumps(call, default=str),
            buffer=json.dumps(call, default=str),
            role="tool",
            metadata={"tool_call": call, "run_id": _key(run_id)},
        )

    def _settle(self, decision: GuardDecision, run_id: Any, stage: str) -> GuardDecision:
        self.decisions.append(decision)
        if decision.is_terminal:
            self.blocked_runs.add(_key(run_id))
            if self.on_block == "raise":
                raise GuardrailTripped(decision, run_id=_key(run_id), stage=stage)
        return decision

    def _finish(self, run_id: Any) -> None:
        key = _key(run_id)
        self._buffers.pop(key, None)
        self._pending.pop(key, None)
        self._prompts.pop(key, None)


def _key(run_id: Any) -> str:
    return str(run_id) if run_id is not None else "default"


class GuardrailCallbackHandler(_HandlerState):
    """Synchronous LangChain callback handler. See the module docstring."""

    def on_llm_start(self, serialized: Any, prompts: Sequence[str], *, run_id: Any = None, **kw: Any) -> None:
        if prompts:
            self._remember_prompt(run_id, str(prompts[-1]))

    def on_chat_model_start(
        self, serialized: Any, messages: Sequence[Sequence[Any]], *, run_id: Any = None, **kw: Any
    ) -> None:
        if not self.prompt_from_messages or not messages:
            return
        last_batch = messages[-1]
        if last_batch:
            self._remember_prompt(run_id, _message_text(last_batch[-1]))

    def on_llm_new_token(self, token: str, *, run_id: Any = None, **kw: Any) -> None:
        if _key(run_id) in self.blocked_runs or not token:
            return
        if not self._append(run_id, token):
            return
        ctx = self._context(run_id, chunk=token, buffer=self.buffer(run_id))
        decision = self.pipeline.run_sync(ctx)
        self._settle(decision, run_id, "token")

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kw: Any) -> None:
        try:
            if _key(run_id) in self.blocked_runs:
                return
            text = _generation_text(response)
            if text is None:
                text = self.buffer(run_id)
            if not text:
                return
            ctx = self._context(run_id, chunk=None, buffer=text)
            self._settle(self.pipeline.run_sync(ctx), run_id, "end")
        finally:
            self._finish(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: Any = None, **kw: Any) -> None:
        self._finish(run_id)

    def on_tool_start(
        self,
        serialized: Any,
        input_str: Any,
        *,
        run_id: Any = None,
        inputs: Any = None,
        **kw: Any,
    ) -> None:
        if self.tool_guard is None:
            return
        ctx = self._tool_context(serialized, input_str, inputs, run_id)
        decision = _evaluate_sync(self.tool_guard, ctx)
        self._settle(decision, run_id, "tool")


class AsyncGuardrailCallbackHandler(_HandlerState):
    """Asynchronous twin of :class:`GuardrailCallbackHandler`."""

    async def on_llm_start(
        self, serialized: Any, prompts: Sequence[str], *, run_id: Any = None, **kw: Any
    ) -> None:
        if prompts:
            self._remember_prompt(run_id, str(prompts[-1]))

    async def on_chat_model_start(
        self, serialized: Any, messages: Sequence[Sequence[Any]], *, run_id: Any = None, **kw: Any
    ) -> None:
        if not self.prompt_from_messages or not messages:
            return
        last_batch = messages[-1]
        if last_batch:
            self._remember_prompt(run_id, _message_text(last_batch[-1]))

    async def on_llm_new_token(self, token: str, *, run_id: Any = None, **kw: Any) -> None:
        if _key(run_id) in self.blocked_runs or not token:
            return
        if not self._append(run_id, token):
            return
        ctx = self._context(run_id, chunk=token, buffer=self.buffer(run_id))
        decision = await self.pipeline.evaluate_chunk(ctx)
        self._settle(decision, run_id, "token")

    async def on_llm_end(self, response: Any, *, run_id: Any = None, **kw: Any) -> None:
        try:
            if _key(run_id) in self.blocked_runs:
                return
            text = _generation_text(response)
            if text is None:
                text = self.buffer(run_id)
            if not text:
                return
            ctx = self._context(run_id, chunk=None, buffer=text)
            self._settle(await self.pipeline.evaluate_chunk(ctx), run_id, "end")
        finally:
            self._finish(run_id)

    async def on_llm_error(self, error: BaseException, *, run_id: Any = None, **kw: Any) -> None:
        self._finish(run_id)

    async def on_tool_start(
        self,
        serialized: Any,
        input_str: Any,
        *,
        run_id: Any = None,
        inputs: Any = None,
        **kw: Any,
    ) -> None:
        if self.tool_guard is None:
            return
        ctx = self._tool_context(serialized, input_str, inputs, run_id)
        decision = await _evaluate_async(self.tool_guard, ctx)
        self._settle(decision, run_id, "tool")


# ---- evaluating an arbitrary guard or pipeline ----------------------------------


async def _evaluate_async(guard: Any, ctx: Context) -> GuardDecision:
    if isinstance(guard, GuardPipeline):
        return await guard.evaluate_chunk(ctx)
    result = guard(ctx)
    if asyncio.iscoroutine(result):
        result = await result
    return result


def _evaluate_sync(guard: Any, ctx: Context) -> GuardDecision:
    if isinstance(guard, GuardPipeline):
        return guard.run_sync(ctx)
    return _run_coroutine(_evaluate_async(guard, ctx))
