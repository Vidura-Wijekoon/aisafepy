"""Duck-typed tests for the framework adapters in ``aisafepy.flow.adapters``.

None of the upstream SDKs (openai-agents, LangGraph, LlamaIndex, the
Anthropic SDK) are imported here. Each adapter is written against the
*shape* of the upstream object, so these tests build minimal fakes with
the same shape and verify that:

* allowed calls pass straight through to the user's function,
* denied calls raise the adapter-specific violation type carrying the
  structured ``IFCViolation``,
* plain string arguments are auto-lifted as UNTRUSTED ``llm.output``,
* ``@secure_tool`` metadata is honoured,
* both sync and async callables are supported where the adapter claims to.
"""

from __future__ import annotations

import asyncio

import pytest

from aisafepy.flow import Capability, Policy, secure_agent, secure_tool
from aisafepy.flow.adapters.anthropic_tools import (
    AnthropicIFCViolation,
    secure_anthropic_tools,
)
from aisafepy.flow.adapters.langgraph import LangGraphIFCViolation, secure_langgraph
from aisafepy.flow.adapters.llamaindex import (
    LlamaIndexIFCViolation,
    secure_llamaindex_agent,
)
from aisafepy.flow.adapters.openai_agents import (
    _TripwireException,
    secure_openai_agent,
)
from aisafepy.flow.taint import lift


def _sink_policy() -> Policy:
    """A policy where ``send_email`` only accepts TRUSTED arguments."""
    return (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source("llm.output", integrity="UNTRUSTED")
        .require(
            "send_email",
            control_flow_integrity="TRUSTED",
            arg_max_integrity="TRUSTED",
            caps={Capability.WRITE_EXTERNAL},
        )
        .require("lookup", control_flow_integrity="UNTRUSTED")
    )


# ---------------------------------------------------------------------------
# OpenAI Agents SDK
# ---------------------------------------------------------------------------


class _FakeFunctionTool:
    def __init__(self, name, fn):
        self.name = name
        self.on_invoke_tool = fn


class _FakeAgent:
    def __init__(self, tools):
        self.tools = tools


def test_openai_adapter_allows_trusted_call():
    calls = []

    def send_email(to, body):
        calls.append((to, body))
        return "sent"

    agent = _FakeAgent([_FakeFunctionTool("send_email", send_email)])
    secure_openai_agent(agent, _sink_policy())
    to = lift("a@example.com", source="user_prompt", integrity="TRUSTED")
    body = lift("hello", source="user_prompt", integrity="TRUSTED")
    assert agent.tools[0].on_invoke_tool(to, body) == "sent"
    assert calls == [(to, body)]


def test_openai_adapter_auto_lifts_plain_strings_as_untrusted():
    def send_email(to, body):  # pragma: no cover - must not run
        raise AssertionError("tool body executed despite IFC denial")

    agent = _FakeAgent([_FakeFunctionTool("send_email", send_email)])
    secure_openai_agent(agent, _sink_policy())
    with pytest.raises(_TripwireException) as exc_info:
        agent.tools[0].on_invoke_tool("a@example.com", "hello")
    tripwire = exc_info.value.tripwire
    assert tripwire.action.value == "block"
    assert tripwire.evidence["kind"] == "ifc_violation"
    assert "llm.output" in tripwire.evidence["provenance"]


def test_openai_adapter_wraps_async_tools():
    async def lookup(query):
        return f"result for {query}"

    agent = _FakeAgent([_FakeFunctionTool("lookup", lookup)])
    secure_openai_agent(agent, _sink_policy())
    out = asyncio.run(agent.tools[0].on_invoke_tool("weather"))
    assert out == "result for weather"


def test_openai_adapter_leaves_unrecognised_tools_alone():
    class Opaque:
        name = "opaque"

    opaque = Opaque()
    agent = _FakeAgent([opaque])
    secure_openai_agent(agent, _sink_policy())
    assert agent.tools[0] is opaque


def test_openai_adapter_rejects_objects_without_tools():
    with pytest.raises(TypeError):
        secure_openai_agent(object(), _sink_policy())


def test_openai_adapter_honours_control_flow_integrity_kwarg():
    def send_email(to, body):  # pragma: no cover - must not run
        raise AssertionError("tool body executed despite IFC denial")

    agent = _FakeAgent([_FakeFunctionTool("send_email", send_email)])
    secure_openai_agent(agent, _sink_policy(), control_flow_integrity="UNTRUSTED")
    to = lift("a@example.com", source="user_prompt", integrity="TRUSTED")
    body = lift("hello", source="user_prompt", integrity="TRUSTED")
    with pytest.raises(_TripwireException) as exc_info:
        agent.tools[0].on_invoke_tool(to, body)
    assert "control flow integrity" in exc_info.value.tripwire.rationale


def test_secure_agent_infers_openai_framework_from_module_name():
    # ``_infer_framework`` keys on the top-level module being ``agents``.
    fake_cls = type("Agent", (), {"__module__": "agents.agent"})
    agent = fake_cls()
    agent.tools = [_FakeFunctionTool("lookup", lambda q: q)]
    secured = secure_agent(agent, _sink_policy())
    assert secured is agent
    assert getattr(agent, "__aisafepy_policy__", None) is not None


def test_secure_agent_unknown_framework_raises():
    with pytest.raises(ValueError, match="Unknown framework"):
        secure_agent(object(), _sink_policy())


# ---------------------------------------------------------------------------
# LangGraph
# ---------------------------------------------------------------------------


class _FakeCompiledGraph:
    def __init__(self, nodes):
        self.nodes = nodes


class _FakeGraphBuilder:
    def __init__(self):
        self.added = {}

    def add_node(self, name, action, *a, **kw):
        self.added[name] = action
        return self


def test_langgraph_post_compile_nodes_are_wrapped_and_enforced():
    def send_email(state):  # pragma: no cover - must not run
        raise AssertionError("node executed despite IFC denial")

    graph = _FakeCompiledGraph({"send_email": send_email})
    secure_langgraph(graph, _sink_policy())
    # Every string value in the state is lifted as UNTRUSTED ``state``.
    with pytest.raises(LangGraphIFCViolation) as exc_info:
        graph.nodes["send_email"]({"body": "wire the money"})
    assert exc_info.value.violation.tool == "send_email"
    assert "state" in exc_info.value.violation.provenance


def test_langgraph_allows_uncovered_nodes_and_threads_context():
    seen = {}

    def lookup(state):
        seen.update(state)
        return {"ok": True}

    graph = _FakeCompiledGraph({"lookup": lookup})
    secure_langgraph(graph, _sink_policy())
    state = {"query": "hi"}
    assert graph.nodes["lookup"](state) == {"ok": True}
    # The adapter stores the run context on the state for downstream nodes.
    assert "__aisafepy_taint__" in state
    assert state["__aisafepy_taint__"] is graph.__aisafepy_context__


def test_langgraph_builder_path_wraps_on_add_node():
    builder = _FakeGraphBuilder()
    secure_langgraph(builder, _sink_policy())

    def send_email(state):  # pragma: no cover - must not run
        raise AssertionError("node executed despite IFC denial")

    builder.add_node("send_email", send_email)
    wrapped = builder.added["send_email"]
    assert wrapped is not send_email
    with pytest.raises(LangGraphIFCViolation):
        wrapped({"body": "x"})


def test_langgraph_rejects_unknown_shapes():
    with pytest.raises(TypeError):
        secure_langgraph(42, _sink_policy())


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


class _FakeToolMetadata:
    def __init__(self, name):
        self.name = name


class _FakeLITool:
    def __init__(self, name, fn):
        self.metadata = _FakeToolMetadata(name)
        self.fn = fn


class _FakeLIAgent:
    def __init__(self, tools):
        self.tools = tools


def test_llamaindex_adapter_enforces_policy_on_fn():
    def send_email(to, body):  # pragma: no cover - must not run
        raise AssertionError("tool executed despite IFC denial")

    tool = _FakeLITool("send_email", send_email)
    agent = _FakeLIAgent([tool])
    secure_llamaindex_agent(agent, _sink_policy())
    with pytest.raises(LlamaIndexIFCViolation):
        tool.fn("a@example.com", "hi")


def test_llamaindex_adapter_passes_trusted_args_and_async():
    async def lookup(q):
        return q.value.upper() if hasattr(q, "value") else q.upper()

    tool = _FakeLITool("lookup", lookup)
    agent = _FakeLIAgent([tool])
    secure_llamaindex_agent(agent, _sink_policy())
    assert asyncio.run(tool.fn("abc")) == "ABC"


def test_llamaindex_adapter_accepts_dict_of_tools_and_private_fn():
    class PrivateTool:
        def __init__(self):
            self.name = "lookup"
            self._fn = lambda q: "ok"

    tool = PrivateTool()

    class Agent:
        _tools = {"lookup": tool}

    secure_llamaindex_agent(Agent(), _sink_policy())
    assert tool._fn("anything") == "ok"


def test_llamaindex_adapter_rejects_agents_without_tools():
    with pytest.raises(TypeError):
        secure_llamaindex_agent(object(), _sink_policy())


# ---------------------------------------------------------------------------
# Anthropic tool use
# ---------------------------------------------------------------------------


def test_anthropic_mapping_form_wraps_each_tool():
    def send_email(to, body):  # pragma: no cover - must not run
        raise AssertionError("tool executed despite IFC denial")

    def lookup(q):
        return "found"

    wrapped = secure_anthropic_tools({"send_email": send_email, "lookup": lookup}, _sink_policy())
    assert wrapped["lookup"]("x") == "found"
    with pytest.raises(AnthropicIFCViolation):
        wrapped["send_email"]("a@example.com", "hi")


def test_anthropic_dispatch_form_uses_tool_name_argument():
    def dispatch(tool_name, **kw):
        return (tool_name, kw)

    wrapped = secure_anthropic_tools(dispatch, _sink_policy())
    assert wrapped("lookup", q="x") == ("lookup", {"q": "x"})
    with pytest.raises(AnthropicIFCViolation) as exc_info:
        wrapped("send_email", to="a@example.com", body="hi")
    assert exc_info.value.violation.tool == "send_email"


def test_anthropic_async_dispatch_is_supported():
    async def dispatch(tool_name, **kw):
        return tool_name

    wrapped = secure_anthropic_tools(dispatch, _sink_policy())
    assert asyncio.run(wrapped("lookup", q="x")) == "lookup"


def test_anthropic_secure_tool_metadata_is_enforced():
    @secure_tool(capabilities={Capability.WRITE_EXTERNAL}, required_integrity="TRUSTED")
    def post(payload):  # pragma: no cover - must not run
        raise AssertionError("tool executed despite IFC denial")

    # No policy requirement for ``post``; only the decorator's metadata applies.
    wrapped = secure_anthropic_tools({"post": post}, Policy())
    with pytest.raises(AnthropicIFCViolation) as exc_info:
        wrapped["post"]("untrusted payload")
    assert "tool-declared" in exc_info.value.violation.reason
