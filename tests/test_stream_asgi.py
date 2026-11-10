"""Tests for the ASGI guardrail middleware using a hand-rolled ASGI harness."""

from __future__ import annotations

import json

import pytest

from aisafepy.stream.adapters.asgi import (
    GuardrailMiddleware,
    default_extract_request_text,
    default_extract_response_text,
)
from aisafepy.stream.deterministic import AhoCorasickGuard, RegexGuard
from aisafepy.stream.pipeline import GuardPipeline

# ---- harness -------------------------------------------------------------------


async def _call(app, *, method="POST", path="/v1/chat", body=b"", scope_extra=None):
    """Drive an ASGI app with a single request; return (status, headers, body, scope)."""
    scope = {"type": "http", "method": method, "path": path, "headers": []}
    scope.update(scope_extra or {})
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    status = None
    headers = {}
    out = bytearray()
    for m in sent:
        if m["type"] == "http.response.start":
            status = m["status"]
            headers = {k.decode(): v.decode() for k, v in m.get("headers", [])}
        elif m["type"] == "http.response.body":
            out.extend(m.get("body", b""))
    return status, headers, bytes(out), scope, sent


def _json_app(reply):
    """An app that echoes a JSON chat response with ``reply`` as content."""

    async def app(scope, receive, send):
        body = b""
        while True:
            m = await receive()
            body += m.get("body", b"")
            if not m.get("more_body"):
                break
        payload = {"choices": [{"message": {"role": "assistant", "content": reply}}], "echo": body.decode()}
        raw = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": raw, "more_body": False})

    return app


def _sse_app(chunks):
    async def app(scope, receive, send):
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
        )
        for i, c in enumerate(chunks):
            await send(
                {"type": "http.response.body", "body": f"data: {c}\n\n".encode(), "more_body": i < len(chunks) - 1}
            )

    return app


def _blocklist(*terms):
    return GuardPipeline(tier1=[AhoCorasickGuard(name="bl", terms=list(terms), scope="buffer")])


def _redactor():
    g = RegexGuard.compile_pii(redact=True, only=["email"])
    g.scope = "buffer"
    return GuardPipeline(tier1=[g])


# ---- extractors --------------------------------------------------------------------------


def test_default_request_extractor_prefers_last_user_message():
    payload = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": [{"type": "text", "text": "parts "}, {"type": "text", "text": "joined"}]},
        ]
    }
    assert default_extract_request_text(payload) == "parts joined"
    assert default_extract_request_text({"prompt": "p"}) == "p"
    assert default_extract_request_text("raw") == "raw"
    assert default_extract_request_text({"unrelated": 1}) is None


def test_default_response_extractor_covers_common_shapes():
    assert default_extract_response_text({"choices": [{"message": {"content": "a"}}]}) == "a"
    assert default_extract_response_text({"choices": [{"text": "b"}]}) == "b"
    assert default_extract_response_text({"content": [{"type": "text", "text": "c"}, {"type": "tool_use"}]}) == "c"
    assert default_extract_response_text({"output": "d"}) == "d"
    assert default_extract_response_text(42) is None


def test_middleware_requires_a_pipeline():
    with pytest.raises(ValueError):
        GuardrailMiddleware(_json_app("x"))


# ---- request side ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocked_request_is_rejected_before_reaching_app():
    reached = []

    async def app(scope, receive, send):
        reached.append(True)

    mw = GuardrailMiddleware(app, pipeline_in=_blocklist("FORBIDDEN"))
    status, headers, body, scope, _ = await _call(
        mw, body=json.dumps({"messages": [{"role": "user", "content": "say FORBIDDEN"}]}).encode()
    )
    assert status == 400
    assert not reached
    payload = json.loads(body)
    assert payload["error"] == "request blocked by guardrail"
    assert headers["x-aisafepy-decision"] == payload["decision_id"]
    assert scope["aisafepy"]["request_decision"] == payload["decision_id"]


@pytest.mark.asyncio
async def test_clean_request_body_is_replayed_to_app():
    mw = GuardrailMiddleware(_json_app("hi"), pipeline_in=_blocklist("FORBIDDEN"))
    raw = json.dumps({"prompt": "hello"}).encode()
    status, _, body, _, _ = await _call(mw, body=raw)
    assert status == 200
    assert json.loads(body)["echo"] == raw.decode()


@pytest.mark.asyncio
async def test_request_without_text_is_passed_through_with_reason():
    mw = GuardrailMiddleware(_json_app("hi"), pipeline_in=_blocklist("FORBIDDEN"))
    status, _, _, scope, _ = await _call(mw, body=b'{"unrelated": "FORBIDDEN"}')
    assert status == 200
    assert scope["aisafepy"]["request"] == "skipped:no_text"


@pytest.mark.asyncio
async def test_oversized_request_is_not_screened():
    mw = GuardrailMiddleware(_json_app("hi"), pipeline_in=_blocklist("FORBIDDEN"), max_body_bytes=10)
    status, _, _, scope, _ = await _call(mw, body=b'{"prompt": "FORBIDDEN FORBIDDEN"}')
    assert status == 200
    assert scope["aisafepy"]["request"] == "skipped:too_large"


@pytest.mark.asyncio
async def test_non_matching_method_and_path_bypass():
    mw = GuardrailMiddleware(_json_app("hi"), pipeline_in=_blocklist("FORBIDDEN"), paths=("/v1/chat",))
    status, _, _, scope, _ = await _call(mw, method="GET", body=b'{"prompt": "FORBIDDEN"}')
    assert status == 200 and "aisafepy" not in scope
    status, _, _, scope, _ = await _call(mw, path="/health", body=b'{"prompt": "FORBIDDEN"}')
    assert status == 200 and "aisafepy" not in scope
    status, _, _, _, _ = await _call(mw, path="/v1/chat/completions", body=b'{"prompt": "FORBIDDEN"}')
    assert status == 400


@pytest.mark.asyncio
async def test_non_http_scopes_pass_through():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    mw = GuardrailMiddleware(app, pipeline_in=_blocklist("x"))
    await mw({"type": "lifespan"}, None, None)
    assert seen == ["lifespan"]


# ---- response side ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocked_response_is_replaced():
    notified = []
    mw = GuardrailMiddleware(
        _json_app("this is FORBIDDEN"),
        pipeline_out=_blocklist("FORBIDDEN"),
        on_decision=lambda d, s: notified.append(d.action.value),
    )
    status, headers, body, scope, _ = await _call(mw, body=b'{"prompt": "x"}')
    assert status == 200
    payload = json.loads(body)
    assert payload["error"] == "response blocked by guardrail"
    assert payload["fallback"] == "I can't help with that."
    assert headers["content-length"] == str(len(body))
    assert notified == ["block"]
    assert scope["aisafepy"]["response_decision"] == payload["decision_id"]


@pytest.mark.asyncio
async def test_transformed_response_keeps_json_valid_and_fixes_length():
    mw = GuardrailMiddleware(_json_app('write "bob@example.com"'), pipeline_out=_redactor())
    status, headers, body, _, _ = await _call(mw, body=b'{"prompt": "x"}')
    assert status == 200
    payload = json.loads(body)
    assert payload["choices"][0]["message"]["content"] == 'write "[REDACTED]"'
    assert headers["content-length"] == str(len(body))


@pytest.mark.asyncio
async def test_clean_response_is_forwarded_unchanged():
    mw = GuardrailMiddleware(_json_app("all good"), pipeline_out=_blocklist("FORBIDDEN"))
    status, _, body, _, _ = await _call(mw, body=b'{"prompt": "x"}')
    assert status == 200
    assert json.loads(body)["choices"][0]["message"]["content"] == "all good"


@pytest.mark.asyncio
async def test_binary_response_is_not_buffered():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"image/png")]})
        await send({"type": "http.response.body", "body": b"FORBIDDEN-bytes", "more_body": False})

    mw = GuardrailMiddleware(app, pipeline_out=_blocklist("FORBIDDEN"))
    status, _, body, scope, _ = await _call(mw, body=b"{}")
    assert status == 200 and body == b"FORBIDDEN-bytes"
    assert scope["aisafepy"]["response"] == "skipped:content_type"


@pytest.mark.asyncio
async def test_oversized_response_streams_through_unscreened():
    big = "x" * 50
    mw = GuardrailMiddleware(_json_app(big + " FORBIDDEN"), pipeline_out=_blocklist("FORBIDDEN"), max_body_bytes=20)
    status, _, body, scope, _ = await _call(mw, body=b"{}")
    assert status == 200
    assert b"FORBIDDEN" in body
    assert scope["aisafepy"]["response"] == "skipped:too_large"


@pytest.mark.asyncio
async def test_sse_stream_is_cut_at_first_block():
    mw = GuardrailMiddleware(_sse_app(["one", "two", "FORBIDDEN", "four"]), pipeline_out=_blocklist("FORBIDDEN"))
    status, _, body, _, sent = await _call(mw, body=b"{}")
    text = body.decode()
    assert status == 200
    assert "data: one" in text and "data: two" in text
    assert "data: four" not in text
    last = json.loads(text.strip().split("data: ")[-1])
    assert last["aisafepy"]["blocked"] is True
    assert sent[-1]["more_body"] is False


@pytest.mark.asyncio
async def test_sse_stream_clean_passes_all_chunks():
    mw = GuardrailMiddleware(_sse_app(["a", "b"]), pipeline_out=_blocklist("FORBIDDEN"))
    _, _, body, _, _ = await _call(mw, body=b"{}")
    assert body.decode() == "data: a\n\ndata: b\n\n"


@pytest.mark.asyncio
async def test_both_pipelines_compose():
    mw = GuardrailMiddleware(
        _json_app("reply FORBIDDEN"), pipeline_in=_blocklist("BADREQ"), pipeline_out=_blocklist("FORBIDDEN")
    )
    status, _, _, _, _ = await _call(mw, body=b'{"prompt": "BADREQ"}')
    assert status == 400
    status, _, body, _, _ = await _call(mw, body=b'{"prompt": "fine"}')
    assert status == 200 and json.loads(body)["error"] == "response blocked by guardrail"
