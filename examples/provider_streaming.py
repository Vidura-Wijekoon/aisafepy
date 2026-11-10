"""Guard a streaming completion from OpenAI or Anthropic without changing the client.

The script runs fully offline: it fabricates the chunk / event objects
each SDK would yield and pushes them through the adapters so you can see
exactly what the consumer receives when the cascade blocks mid-stream.
Replace the fake generators with the real SDK calls::

    stream = client.chat.completions.create(model=..., messages=..., stream=True)
    for chunk in guard_chat_completion_stream(pipeline, stream):
        ...

    with client.messages.stream(model=..., messages=...) as events:
        for event in guard_messages_stream(pipeline, events):
            ...

Run with ``python examples/provider_streaming.py``.
"""

from __future__ import annotations

from aisafepy.stream.adapters.anthropic import guard_messages_stream
from aisafepy.stream.adapters.openai_chat import StreamOutcome, guard_chat_completion_stream
from aisafepy.stream.deterministic import AhoCorasickGuard, RegexGuard
from aisafepy.stream.pipeline import GuardPipeline

PIPELINE = GuardPipeline(
    tier1=[
        RegexGuard.compile_pii(redact=True, only=["email", "ssn"]),
        AhoCorasickGuard(name="exfil-markers", terms=["BEGIN PRIVATE KEY", "api_key="], scope="buffer"),
    ]
)

ASSISTANT_TOKENS = [
    "Sure, ",
    "the on-call ",
    "engineer is ",
    "reachable at ",
    "oncall@example.com. ",
    "Their deploy key is ",
    "-----BEGIN PRIVATE" + " KEY-----",
    "MIIEvQ...",
]


def openai_chunks():
    for token in ASSISTANT_TOKENS:
        yield {
            "id": "chatcmpl-demo",
            "model": "gpt-demo",
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
    yield {"id": "chatcmpl-demo", "model": "gpt-demo", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


def anthropic_events():
    yield {"type": "message_start", "message": {"id": "msg_demo"}}
    yield {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    for token in ASSISTANT_TOKENS:
        yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": token}}
    yield {"type": "content_block_stop", "index": 0}
    yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
    yield {"type": "message_stop"}


def demo_openai() -> None:
    print("== OpenAI chat.completions stream ==")
    outcome = StreamOutcome()
    for chunk in guard_chat_completion_stream(PIPELINE, openai_chunks(), outcome=outcome):
        choice = chunk["choices"][0]
        text = choice["delta"].get("content")
        marker = f"  <- finish_reason={choice['finish_reason']}" if choice["finish_reason"] else ""
        print(f"  {text!r}{marker}")
    print(f"  forwarded={outcome.chunks_forwarded} evaluations={outcome.evaluations} blocked={outcome.blocked}")
    if outcome.decision is not None:
        print(f"  terminal decision: {outcome.decision.guard_name}: {outcome.decision.rationale}")
    print()


def demo_anthropic() -> None:
    print("== Anthropic messages stream ==")
    outcome = StreamOutcome()
    for event in guard_messages_stream(PIPELINE, anthropic_events(), outcome=outcome):
        kind = event["type"]
        if kind == "content_block_delta":
            print(f"  delta {event['delta']['text']!r}")
        elif kind == "message_delta":
            print(f"  message_delta stop_reason={event['delta']['stop_reason']}")
        else:
            print(f"  {kind}")
    print(f"  forwarded={outcome.chunks_forwarded} evaluations={outcome.evaluations} blocked={outcome.blocked}")
    print()


if __name__ == "__main__":
    demo_openai()
    demo_anthropic()
