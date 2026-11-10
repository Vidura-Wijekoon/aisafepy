"""Integration shims that put a :class:`aisafepy.stream.GuardPipeline` in front
of a model provider, an agent framework, or an HTTP boundary.

Every module here is duck-typed and imports nothing from the framework
it targets, so ``aisafepy.stream.adapters.openai_chat`` works whether
the chunks come from the ``openai`` SDK, a proxy, or a test fixture.

================  ==========================================================
Module            Targets
================  ==========================================================
``openai_chat``   ``chat.completions`` streaming chunks and responses
``anthropic``     Messages API event streams and ``Message`` objects
``openai_agents`` OpenAI Agents SDK input / output guardrail callables
``langchain``     a ``Runnable``-shaped wrapper for LangChain chains
``langchain_callbacks`` a streaming callback handler for LangChain models
``llamaindex``    a response post-processor for LlamaIndex
``asgi``          ASGI 3 middleware (FastAPI, Starlette, Quart, Litestar)
================  ==========================================================
"""

from aisafepy.stream.adapters.anthropic import (
    aguard_messages_stream,
    guard_message,
    guard_messages_stream,
)
from aisafepy.stream.adapters.asgi import GuardrailMiddleware
from aisafepy.stream.adapters.langchain import GuardrailRunnable, as_langchain_runnable
from aisafepy.stream.adapters.langchain_callbacks import (
    AsyncGuardrailCallbackHandler,
    GuardrailCallbackHandler,
    GuardrailTripped,
)
from aisafepy.stream.adapters.llamaindex import (
    GuardrailPostprocessor,
    as_llamaindex_postprocessor,
)
from aisafepy.stream.adapters.openai_agents import as_input_guardrail, as_output_guardrail
from aisafepy.stream.adapters.openai_chat import (
    StreamOutcome,
    aguard_chat_completion_stream,
    guard_chat_completion,
    guard_chat_completion_stream,
)

__all__ = [
    "AsyncGuardrailCallbackHandler",
    "GuardrailCallbackHandler",
    "GuardrailMiddleware",
    "GuardrailPostprocessor",
    "GuardrailRunnable",
    "GuardrailTripped",
    "StreamOutcome",
    "aguard_chat_completion_stream",
    "aguard_messages_stream",
    "as_input_guardrail",
    "as_langchain_runnable",
    "as_llamaindex_postprocessor",
    "as_output_guardrail",
    "guard_chat_completion",
    "guard_chat_completion_stream",
    "guard_message",
    "guard_messages_stream",
]
