"""Tests for the LangChain tool adapter in ``aisafepy.flow.adapters.langchain_tools``."""

from __future__ import annotations

import asyncio

import pytest

from aisafepy.flow import Capability, Policy, secure_tool
from aisafepy.flow.adapters.langchain_tools import (
    LangChainIFCViolation,
    secure_langchain_tools,
    secure_tool_node,
)
from aisafepy.flow.interpreter import IFCContext
from aisafepy.flow.taint import Tainted, lift


class FakeTool:
    """Mimics the parts of ``langchain_core.tools.BaseTool`` the adapter touches."""

    def __init__(self, name, fn=None, afn=None):
        self.name = name
        self.func = fn
        self.coroutine = afn
        self.calls = []
        if fn is not None:
            self._run = self._make_run(fn)
        if afn is not None:
            self._arun = self._make_arun(afn)

    def _make_run(self, fn):
        def _run(*args, run_manager=None, **kwargs):
            self.calls.append((args, kwargs))
            return fn(*args, **kwargs)

        return _run

    def _make_arun(self, afn):
        async def _arun(*args, run_manager=None, **kwargs):
            self.calls.append((args, kwargs))
            return await afn(*args, **kwargs)

        return _arun

    def invoke(self, input, config=None):
        if isinstance(input, dict):
            return self._run(**input)
        return self._run(input)

    async def ainvoke(self, input, config=None):
        if isinstance(input, dict):
            return await self._arun(**input)
        return await self._arun(input)


def _policy() -> Policy:
    return (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source("gmail.read", integrity="UNTRUSTED", caps={Capability.READ_USER})
        .require("send_email", arg_max_integrity="TRUSTED", caps={Capability.WRITE_EXTERNAL})
        .require("lookup", control_flow_integrity="UNTRUSTED")
    )


def test_untrusted_string_args_are_denied_for_trusted_sink():
    tool = FakeTool("send_email", lambda to, body: "sent")
    secure_langchain_tools([tool], _policy())
    with pytest.raises(LangChainIFCViolation) as exc_info:
        tool.invoke({"to": "a@b.c", "body": "hi"})
    assert exc_info.value.violation.tool == "send_email"
    assert "llm.output" in exc_info.value.violation.provenance
    assert tool.calls == []  # body never ran


def test_trusted_tainted_args_pass_and_tool_sees_originals():
    tool = FakeTool("send_email", lambda to, body: f"sent:{to}")
    secure_langchain_tools([tool], _policy(), lift_outputs=False)
    to = lift("a@b.c", source="user_prompt", integrity="TRUSTED", capabilities=("write.external",))
    body = lift("hi", source="user_prompt", integrity="TRUSTED")
    result = tool.invoke({"to": to, "body": body})
    # The tool body received the Tainted objects untouched (no unwrapping).
    assert tool.calls[0][1]["to"] is to
    assert isinstance(result, Tainted) is False
    assert str(result).startswith("sent:")


def test_outputs_are_lifted_with_policy_label():
    tool = FakeTool("gmail.read", lambda query: "mail body")
    secure_langchain_tools([tool], _policy())
    out = tool.invoke({"query": "latest"})
    assert isinstance(out, Tainted)
    assert out.integrity == "UNTRUSTED"
    assert "gmail.read" in out.provenance
    assert "read.user" in out.capabilities
    assert str(out) == "mail body"  # ToolMessage rendering lowers transparently


def test_unknown_tool_output_gets_default_label_and_passes():
    tool = FakeTool("lookup", lambda q: {"ok": True})
    secure_langchain_tools([tool], _policy())
    out = tool.invoke("x")
    assert isinstance(out, Tainted) and out.integrity == "UNTRUSTED"
    assert out.value == {"ok": True}


def test_async_tools_are_wrapped():
    async def send(to, body):
        return "sent"

    tool = FakeTool("send_email", afn=send)
    secure_langchain_tools([tool], _policy())
    with pytest.raises(LangChainIFCViolation):
        asyncio.run(tool.ainvoke({"to": "a@b.c", "body": "hi"}))

    async def lookup(q):
        return "found"

    ok = FakeTool("lookup", afn=lookup)
    secure_langchain_tools([ok], _policy(), lift_outputs=False)
    assert asyncio.run(ok.ainvoke("q")) == "found"


def test_secure_tool_metadata_on_func_is_honoured():
    @secure_tool(required_integrity="TRUSTED")
    def delete_all(confirm):
        return "deleted"

    tool = FakeTool("delete_all", delete_all)
    secure_langchain_tools([tool], Policy())
    with pytest.raises(LangChainIFCViolation, match="tool-declared"):
        tool.invoke({"confirm": "yes"})


def test_wrapping_is_idempotent_and_shares_context():
    tool = FakeTool("lookup", lambda q: q)
    ctx = IFCContext()
    first = secure_langchain_tools([tool], _policy(), context=ctx)
    second = secure_langchain_tools([tool], _policy(), context=ctx)
    assert first == second == [tool]
    assert tool.__aisafepy_context__ is ctx
    run = tool._run
    secure_langchain_tools([tool], _policy())
    assert tool._run is run  # not re-wrapped


def test_mediated_mode_consults_mediator():
    approvals = []

    def mediator(v):
        approvals.append(v.tool)
        return True

    tool = FakeTool("send_email", lambda to, body: "sent")
    secure_langchain_tools([tool], _policy().with_mode("mediated"), mediator=mediator, lift_outputs=False)
    assert tool.invoke({"to": "a@b.c", "body": "hi"}) == "sent"
    assert approvals == ["send_email"]


def test_invoke_only_objects_are_wrapped_via_invoke():
    class InvokeOnly:
        name = "lookup"

        def invoke(self, input, config=None):
            return f"got {input}"

    obj = InvokeOnly()
    secure_langchain_tools([obj], _policy(), lift_outputs=False)
    assert obj.invoke("x") == "got x"

    class Nothing:
        name = "n"

    with pytest.raises(TypeError):
        secure_langchain_tools([Nothing()], _policy())


def test_secure_tool_node_wraps_registry():
    class Node:
        def __init__(self):
            self.tools_by_name = {
                "send_email": FakeTool("send_email", lambda to, body: "sent"),
                "lookup": FakeTool("lookup", lambda q: "ok"),
            }

    node = secure_tool_node(Node(), _policy())
    assert node.__aisafepy_policy__ is not None
    with pytest.raises(LangChainIFCViolation):
        node.tools_by_name["send_email"].invoke({"to": "x", "body": "y"})
    assert str(node.tools_by_name["lookup"].invoke("q")) == "ok"
    with pytest.raises(TypeError):
        secure_tool_node(object(), _policy())


def test_plumbing_kwargs_are_ignored_and_lists_are_lifted():
    seen = {}

    def send_email(to, body, cc=None, **_plumbing):
        seen["ok"] = True
        return "sent"

    policy = (
        Policy()
        .require("send_email", arg_max_integrity="UNTRUSTED")
        .deny_if(
            "send_email",
            when=lambda **kw: any("evil" in str(getattr(c, "value", c)) for c in (kw.get("cc") or [])),
            reason="evil cc",
        )
    )
    tool = FakeTool("send_email", send_email)
    secure_langchain_tools([tool], policy, lift_outputs=False)
    assert tool._run(to="a", body="b", cc=["fine"], run_manager=object(), config={"x": 1}) == "sent"
    with pytest.raises(LangChainIFCViolation, match="evil cc"):
        tool._run(to="a", body="b", cc=["evil@x"])
