# Integrations

AIsafePy ships two families of adapters. **Flow adapters** put
information-flow control around tool calls inside an agent framework.
**Stream adapters** put the guard cascade in front of model output (or
user input) at whatever boundary you own: the SDK call, the agent
framework's guardrail hook, or the HTTP edge.

Every adapter is duck-typed. None of them import the framework they
target, so installing AIsafePy never drags in `openai`, `anthropic`,
`langgraph`, `llama-index` or a web framework. The optional extras in
`pyproject.toml` exist only to pin compatible versions when you *do*
install the upstream package.

## Choosing a boundary

| You control | Use | Blocks how |
|---|---|---|
| The SDK call (`client.chat.completions.create`) | `stream.adapters.openai_chat` | Cuts the stream, emits a `content_filter` chunk with the fallback text |
| The SDK call (`client.messages.stream`) | `stream.adapters.anthropic` | Emits a well-formed `content_block_delta` / `message_delta(stop_reason=aisafepy_blocked)` / `message_stop` tail |
| An OpenAI Agents SDK agent | `stream.as_openai_output_guardrail`, `flow.secure_agent` | Raises the SDK's tripwire with a structured `Tripwire` |
| A LangChain chain | `stream.adapters.langchain.as_langchain_runnable` | Returns the fallback string in place of the output |
| A LlamaIndex query engine | `stream.adapters.llamaindex.as_llamaindex_postprocessor` | Rewrites `response.response` |
| The HTTP service (FastAPI, Starlette, Quart, Litestar, Django ASGI) | `stream.adapters.asgi.GuardrailMiddleware` | Rejects requests with 400, replaces responses, cuts SSE streams |
| Plain Python tool functions | `flow.secure_callables` | Raises `IFCViolationError` before the function body runs |
| An MCP server | `flow.ifc_mcp_server` | Rejects poisoned tool descriptions at registration, denies calls at dispatch |

Several boundaries can be layered. The usual production shape is
`flow` inside the agent (architectural control over tool calls) plus
`stream` at the HTTP edge (detective control over text), which is the
defense-in-depth story in the README.

## Stream adapters

### OpenAI Chat Completions

```python
from aisafepy.stream import GuardPipeline, RegexGuard
from aisafepy.stream.adapters.openai_chat import (
    StreamOutcome, guard_chat_completion_stream, guard_chat_completion,
)

pipeline = GuardPipeline(tier1=[RegexGuard.compile_pii(redact=True)])

stream = client.chat.completions.create(model="gpt-4.1", messages=msgs, stream=True)
outcome = StreamOutcome()
for chunk in guard_chat_completion_stream(pipeline, stream, outcome=outcome):
    print(chunk["choices"][0]["delta"].get("content") if isinstance(chunk, dict)
          else chunk.choices[0].delta.content, end="")
if outcome.blocked:
    log(outcome.decision)
```

Behaviour:

* The cascade runs on the accumulated assistant text after every text
  chunk (`evaluate_every=N` batches; the final text is always checked).
* On BLOCK or ESCALATE the upstream iterator is abandoned and one
  synthetic dict chunk is yielded with the decision's `fallback` as
  content, `finish_reason="content_filter"`, and an `aisafepy` key
  holding the decision id, action, rationale, guard and tier.
* On TRANSFORM the offending delta is replaced by the transformed text.
* `guard_chat_completion(pipeline, response)` handles the non-streaming
  response: blocked choices get the fallback and `content_filter`,
  transformed choices get the new text, and one decision per choice is
  returned.
* `aguard_chat_completion_stream` is the `async for` variant and also
  accepts a synchronous iterable.

Chunks may be SDK objects or dicts; forwarded chunks are passed through
by identity, only synthetic chunks are dicts.

### Anthropic Messages

```python
from aisafepy.stream.adapters.anthropic import guard_messages_stream, guard_message

with client.messages.stream(model="claude-sonnet-4-5", messages=msgs, max_tokens=800) as events:
    for event in guard_messages_stream(pipeline, events):
        if event["type"] == "content_block_delta":
            print(event["delta"]["text"], end="")
```

Only `content_block_delta` events whose `delta.type == "text_delta"`
are evaluated; `message_start`, `content_block_start`, `ping`,
`input_json_delta` and friends pass through untouched. On a terminal
decision the adapter emits a tail that satisfies the normal event
grammar, so a client that renders deltas and stops on `message_stop`
terminates cleanly:

```
content_block_delta   (fallback text, with an "aisafepy" key)
content_block_stop
message_delta         stop_reason = "aisafepy_blocked"
message_stop
```

`guard_message(pipeline, message)` handles the non-streaming object,
rewriting text blocks only and leaving `tool_use` blocks alone.

### ASGI middleware

```python
from aisafepy.stream.adapters.asgi import GuardrailMiddleware

app = FastAPI()
...
app = GuardrailMiddleware(
    app,
    pipeline_in=inbound,       # screens the user prompt
    pipeline_out=outbound,     # screens the completion
    paths=("/v1/chat",),
    reject_status=400,
    block_status=200,
    max_body_bytes=1_000_000,
    on_decision=lambda decision, scope: metrics.record(decision),
)
```

Request side: the JSON body is buffered, the prompt is extracted (last
`messages[].content`, else `prompt` / `input` / `text` / `query`), and
`pipeline_in` runs with `role="user"`. A terminal decision returns
`reject_status` with a JSON body carrying `decision_id`, `rationale`
and `fallback`, plus an `x-aisafepy-decision` header. Otherwise the
body is replayed to the app unchanged.

Response side: JSON and text bodies are buffered and evaluated;
blocked bodies are replaced (`block_status` defaults to 200 so chat
UIs render the fallback instead of an error), transformed bodies are
substituted and `content-length` is recomputed. `text/event-stream`
and chunked text are screened chunk by chunk; on block the middleware
emits one final `data:` frame with the fallback and closes.

Pass-through cases are recorded in `scope["aisafepy"]` so the app can
log them: `skipped:too_large`, `skipped:no_text`,
`skipped:content_type`. Non-matching methods and paths, and non-HTTP
scopes (lifespan, websockets), are not touched.

Custom extractors: pass `extract_request_text` / `extract_response_text`
callables taking the decoded JSON (or raw string) and returning the
text to screen, or `None` to skip.

### OpenAI Agents SDK, LangChain, LlamaIndex

```python
from aisafepy.stream import as_openai_output_guardrail
agent = Agent(name="support", output_guardrails=[as_openai_output_guardrail(pipeline)])

from aisafepy.stream.adapters.langchain import as_langchain_runnable
chain = prompt | model | StrOutputParser() | as_langchain_runnable(pipeline)

from aisafepy.stream.adapters.llamaindex import as_llamaindex_postprocessor
engine = index.as_query_engine(node_postprocessors=[as_llamaindex_postprocessor(pipeline)])
```

These three are thin: one evaluation over the whole output, allow
passes the text through, transform substitutes, block substitutes the
decision's `fallback` (or raises the SDK's tripwire for OpenAI Agents).

## Flow adapters

### Plain callables (no framework)

```python
from aisafepy.flow import Policy, Capability, secure_callables, IFCViolationError

policy = (
    Policy()
    .label_source("gmail.read", integrity="UNTRUSTED", caps={Capability.READ_USER})
    .require("send_email", arg_max_integrity="TRUSTED", caps={Capability.WRITE_EXTERNAL})
)
tools = secure_callables({"gmail.read": read_mail, "send_email": send_email}, policy)

mail = tools["gmail.read"](query="latest")      # output lifted with gmail.read's label
try:
    tools["send_email"](to="x@y", body=mail)    # denied: UNTRUSTED argument into a TRUSTED sink
except IFCViolationError as exc:
    print(exc.violation.reason)
```

Plain string arguments are lifted as UNTRUSTED `llm.output`, matching
the framework adapters; outputs are lifted with the policy's label for
the tool (or UNTRUSTED `tool:<name>`); decorated `@secure_tool`
metadata is honoured. `ToolRegistry` collects tools with a decorator
and derives JSON-schema descriptions from their signatures, which
feeds MCP manifest pinning.

### OpenAI Agents, LangGraph, LlamaIndex, Anthropic tool use

```python
from aisafepy.flow import secure_agent
safe = secure_agent(agent, policy)                     # framework inferred from the object
safe = secure_agent(graph, policy, framework="langgraph")
handlers = secure_agent({"send_email": send_email}, policy, framework="anthropic")
```

Each adapter wraps the framework's tool callable in place and raises a
framework-specific violation (`_TripwireException` carrying a
`Tripwire`, `LangGraphIFCViolation`, `LlamaIndexIFCViolation`,
`AnthropicIFCViolation`) that wraps the structured `IFCViolation`.
`control_flow_integrity="UNTRUSTED"` can be passed to any of them when
the adapter cannot compute CFI and should be pessimistic.

### MCP servers

```python
from aisafepy.flow import ifc_mcp_server
from aisafepy.flow.adapters.mcp import build_manifest, manifest_from_json

pinned = manifest_from_json(Path("tools.manifest.json").read_text())
server = ifc_mcp_server(server, policy, manifest=pinned, screen_parameters=True)
```

At registration every tool description, and every prose field inside
its input schema (parameter descriptions, titles, examples), is
screened for tool-poisoning patterns, and the schema hash is compared
against the pinned manifest. At dispatch each `tools/call` runs through
`evaluate_call` with string arguments lifted as UNTRUSTED `mcp.input`.
Pin a manifest in CI with `aisafepy manifest pin` and verify it on
start with `aisafepy manifest verify` (see `docs/cli.md`).

## Writing your own adapter

Both families reduce to one call each:

* Stream: build a `Context(buffer=..., chunk=..., role=...)` and
  `await pipeline.evaluate_chunk(ctx)` (or `pipeline.run_sync(ctx)`).
  Act on `decision.is_terminal`, `decision.action`, and
  `decision.transformed_content`.
* Flow: lift untrusted inputs with `lift(value, source=..., integrity=...)`
  and call `evaluate_call(tool=..., args=..., kwargs=..., policy=...,
  metadata=getattr(fn, "__aisafepy_tool__", None), context=ctx)`.

Keep the conventions the shipped adapters follow: never raise from the
observer hooks, fail open with evidence when the guard itself errors,
and carry the `decision_id` through to whatever the user sees so a
support engineer can find the trace.
