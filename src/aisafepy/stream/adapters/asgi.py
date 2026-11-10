"""ASGI middleware that runs a :class:`GuardPipeline` at the HTTP boundary.

This is the "reverse proxy" deployment shape: the application behind
the middleware does not change, and every matching request and
response is screened. It works with any ASGI 3 framework (FastAPI,
Starlette, Quart, Litestar, Django ASGI) without importing any of them.

Request path
    The JSON body is buffered (up to ``max_body_bytes``), the prompt
    text is extracted with ``extract_request_text`` (default: the last
    ``messages[].content``, else ``prompt`` / ``input`` / ``text``), and
    ``pipeline_in`` is evaluated with ``role="user"``. A terminal
    decision short-circuits with ``reject_status`` and a JSON error body
    that carries the decision id and rationale, so support staff can
    find the trace. Otherwise the body is replayed to the app unchanged.

Response path
    Non-streaming bodies (JSON or text) are buffered and evaluated with
    ``pipeline_out``; blocked bodies are replaced by a JSON error with
    ``block_status`` (default 200 so streaming clients render the
    fallback rather than a transport error), transformed bodies are
    substituted and ``content-length`` is recomputed. Server-sent event
    and chunked text responses are screened incrementally: each body
    chunk is evaluated against the accumulated text, and on block the
    middleware emits one final chunk with the fallback text and closes.

Bodies larger than ``max_body_bytes``, non-text content types, and
paths outside ``paths`` are passed through untouched; the pipeline is
only as useful as the text it sees, and silently truncating a request
would be worse than not screening it. Every pass-through reason is
recorded in ``scope["aisafepy"]`` for the application to log.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from aisafepy.core.decisions import Action, GuardDecision
from aisafepy.stream.pipeline import Context, GuardPipeline

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_TEXT_TYPES = ("application/json", "text/", "application/x-ndjson", "application/problem+json")


def default_extract_request_text(payload: Any) -> str | None:
    """Pull the user prompt out of common chat-style request bodies."""
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return None
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") not in (None, "user"):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # OpenAI / Anthropic content parts
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return "".join(parts) or None
    for key in ("prompt", "input", "text", "query", "question"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def default_extract_response_text(payload: Any) -> str | None:
    """Pull the assistant text out of common chat-style response bodies."""
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message") or first.get("delta") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    content = payload.get("content")
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        if parts:
            return "".join(parts)
    for key in ("output", "response", "answer", "text", "completion"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


class GuardrailMiddleware:
    """ASGI 3 middleware. See the module docstring for behaviour."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        pipeline_in: GuardPipeline | None = None,
        pipeline_out: GuardPipeline | None = None,
        paths: Iterable[str] = ("/",),
        methods: Iterable[str] = ("POST",),
        extract_request_text: Callable[[Any], str | None] = default_extract_request_text,
        extract_response_text: Callable[[Any], str | None] = default_extract_response_text,
        reject_status: int = 400,
        block_status: int = 200,
        max_body_bytes: int = 1_000_000,
        on_decision: Callable[[GuardDecision, Scope], Awaitable[None] | None] | None = None,
    ) -> None:
        if pipeline_in is None and pipeline_out is None:
            raise ValueError("GuardrailMiddleware needs pipeline_in and/or pipeline_out")
        self.app = app
        self.pipeline_in = pipeline_in
        self.pipeline_out = pipeline_out
        self.paths = tuple(paths)
        self.methods = tuple(m.upper() for m in methods)
        self.extract_request_text = extract_request_text
        self.extract_response_text = extract_response_text
        self.reject_status = reject_status
        self.block_status = block_status
        self.max_body_bytes = max_body_bytes
        self.on_decision = on_decision

    # ---- ASGI entry point -------------------------------------------------

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or not self._matches(scope):
            await self.app(scope, receive, send)
            return
        info: dict[str, Any] = scope.setdefault("aisafepy", {})

        body = b""
        if self.pipeline_in is not None:
            body, too_large = await self._read_body(receive)
            if too_large:
                info["request"] = "skipped:too_large"
                receive = _replay(body, more=True, upstream=receive)
            else:
                decision = await self._screen_request(body, scope)
                if decision is not None:
                    info["request_decision"] = decision.decision_id
                    await self._notify(decision, scope)
                    if decision.is_terminal:
                        await _send_json(
                            send,
                            self.reject_status,
                            {
                                "error": "request blocked by guardrail",
                                "decision_id": decision.decision_id,
                                "rationale": decision.rationale,
                                "fallback": decision.fallback,
                            },
                        )
                        return
                receive = _replay(body)

        if self.pipeline_out is None:
            await self.app(scope, receive, send)
            return
        await self._run_with_response_guard(scope, receive, send)

    # ---- request side -----------------------------------------------------

    def _matches(self, scope: Scope) -> bool:
        path = scope.get("path", "/")
        method = str(scope.get("method", "GET")).upper()
        if method not in self.methods:
            return False
        return any(path == p or path.startswith(p.rstrip("/") + "/") or p == "/" for p in self.paths)

    async def _read_body(self, receive: Receive) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            chunk = message.get("body", b"") or b""
            size += len(chunk)
            chunks.append(chunk)
            if size > self.max_body_bytes:
                return b"".join(chunks), True
            if not message.get("more_body", False):
                break
        return b"".join(chunks), False

    async def _screen_request(self, body: bytes, scope: Scope) -> GuardDecision | None:
        assert self.pipeline_in is not None
        text = _extract(body, self.extract_request_text)
        if text is None:
            scope["aisafepy"]["request"] = "skipped:no_text"
            return None
        ctx = Context(buffer=text, role="user", metadata={"path": scope.get("path")})
        return await self.pipeline_in.evaluate_chunk(ctx)

    # ---- response side ----------------------------------------------------

    async def _run_with_response_guard(self, scope: Scope, receive: Receive, send: Send) -> None:
        assert self.pipeline_out is not None
        pipeline = self.pipeline_out
        state: dict[str, Any] = {
            "start": None,
            "streaming": False,
            "textual": False,
            "buffer": bytearray(),
            "text_so_far": [],
            "blocked": False,
            "too_large": False,
        }

        async def guarded_send(message: dict[str, Any]) -> None:
            if state["blocked"]:
                return
            mtype = message["type"]
            if mtype == "http.response.start":
                headers = {k.decode().lower(): v.decode() for k, v in message.get("headers", [])}
                ctype = headers.get("content-type", "")
                state["textual"] = any(ctype.startswith(t) for t in _TEXT_TYPES)
                state["streaming"] = ctype.startswith("text/event-stream") or "chunked" in headers.get(
                    "transfer-encoding", ""
                )
                state["start"] = message
                if not state["textual"]:
                    scope["aisafepy"]["response"] = "skipped:content_type"
                    await send(message)
                return
            if mtype != "http.response.body":
                await send(message)
                return
            if not state["textual"]:
                await send(message)
                return

            body = message.get("body", b"") or b""
            more = message.get("more_body", False)

            if state["streaming"]:
                await self._handle_stream_chunk(pipeline, scope, send, state, body, more)
                return

            state["buffer"].extend(body)
            if len(state["buffer"]) > self.max_body_bytes and not state["too_large"]:
                state["too_large"] = True
                scope["aisafepy"]["response"] = "skipped:too_large"
                await send(state["start"])
            if state["too_large"]:
                await send({"type": "http.response.body", "body": body, "more_body": more})
                return
            if more:
                return
            await self._finish_buffered(pipeline, scope, send, state)

        await self.app(scope, receive, guarded_send)

    async def _finish_buffered(
        self, pipeline: GuardPipeline, scope: Scope, send: Send, state: dict[str, Any]
    ) -> None:
        raw = bytes(state["buffer"])
        text = _extract(raw, self.extract_response_text)
        start = state["start"]
        if text is None:
            scope["aisafepy"]["response"] = "skipped:no_text"
            await send(start)
            await send({"type": "http.response.body", "body": raw, "more_body": False})
            return
        decision = await pipeline.evaluate_chunk(
            Context(buffer=text, role="assistant", metadata={"path": scope.get("path")})
        )
        scope["aisafepy"]["response_decision"] = decision.decision_id
        await self._notify(decision, scope)
        if decision.is_terminal:
            await _send_json(
                send,
                self.block_status,
                {
                    "error": "response blocked by guardrail",
                    "decision_id": decision.decision_id,
                    "rationale": decision.rationale,
                    "fallback": decision.fallback,
                },
            )
            return
        if decision.action == Action.TRANSFORM and decision.transformed_content is not None:
            raw = _substitute(raw, text, decision.transformed_content)
        await send(_with_content_length(start, len(raw)))
        await send({"type": "http.response.body", "body": raw, "more_body": False})

    async def _handle_stream_chunk(
        self,
        pipeline: GuardPipeline,
        scope: Scope,
        send: Send,
        state: dict[str, Any],
        body: bytes,
        more: bool,
    ) -> None:
        if state["start"] is not None:
            await send(state["start"])
            state["start"] = None
        text = body.decode("utf-8", errors="replace")
        if text.strip():
            state["text_so_far"].append(text)
            decision = await pipeline.evaluate_chunk(
                Context(chunk=text, buffer="".join(state["text_so_far"]), role="assistant")
            )
            await self._notify(decision, scope)
            if decision.is_terminal:
                state["blocked"] = True
                scope["aisafepy"]["response_decision"] = decision.decision_id
                payload = json.dumps(
                    {
                        "aisafepy": {
                            "blocked": True,
                            "decision_id": decision.decision_id,
                            "rationale": decision.rationale,
                        },
                        "fallback": decision.fallback,
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": f"data: {payload}\n\n".encode(),
                        "more_body": False,
                    }
                )
                return
            if decision.action == Action.TRANSFORM and decision.transformed_content is not None:
                body = decision.transformed_content.encode("utf-8")
                state["text_so_far"][-1] = decision.transformed_content
        await send({"type": "http.response.body", "body": body, "more_body": more})

    async def _notify(self, decision: GuardDecision, scope: Scope) -> None:
        if self.on_decision is None:
            return
        try:
            result = self.on_decision(decision, scope)
            if result is not None and hasattr(result, "__await__"):
                await result
        except Exception:  # pragma: no cover - observer must not break traffic
            pass


# ---- helpers ------------------------------------------------------------


def _extract(raw: bytes, extractor: Callable[[Any], str | None]) -> str | None:
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        payload = text
    try:
        return extractor(payload)
    except Exception:
        return None


def _substitute(raw: bytes, original: str, replacement: str) -> bytes:
    text = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text.replace(original, replacement).encode("utf-8")
    # JSON bodies: re-serialise so escaping stays valid.
    encoded_original = json.dumps(original)[1:-1]
    encoded_replacement = json.dumps(replacement)[1:-1]
    if encoded_original and encoded_original in text:
        return text.replace(encoded_original, encoded_replacement).encode("utf-8")
    return json.dumps(payload).replace(encoded_original, encoded_replacement).encode("utf-8")


def _with_content_length(start: dict[str, Any], length: int) -> dict[str, Any]:
    headers = [
        (k, v) for k, v in start.get("headers", []) if k.decode().lower() != "content-length"
    ]
    headers.append((b"content-length", str(length).encode()))
    return {**start, "headers": headers}


async def _send_json(send: Send, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"x-aisafepy-decision", str(payload.get("decision_id", "")).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


def _replay(body: bytes, *, more: bool = False, upstream: Receive | None = None) -> Receive:
    """Return a ``receive`` that yields ``body`` once, then defers upstream."""
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": more}
        if more and upstream is not None:
            return await upstream()
        return {"type": "http.disconnect"}

    return receive
