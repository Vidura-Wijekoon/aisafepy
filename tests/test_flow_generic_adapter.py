from __future__ import annotations

import asyncio
from typing import Literal

import pytest

from aisafepy.flow.adapters.generic import (
    ToolRegistry,
    lift_tool_output,
    secure_callable,
    secure_callables,
    tool_schema,
)
from aisafepy.flow.adapters.mcp import schema_hash
from aisafepy.flow.interpreter import IFCContext, IFCViolationError
from aisafepy.flow.policy import Capability, Policy, secure_tool
from aisafepy.flow.taint import Tainted, lift


def _policy() -> Policy:
    return (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source("gmail.read", integrity="UNTRUSTED", caps={Capability.READ_USER})
        .label_source("vault", caps={Capability.READ_SECRETS})
        .require("send_email", control_flow_integrity="TRUSTED", arg_max_integrity="TRUSTED")
        .require("gmail.read")
        .declassify("summarize", by="quarantined model, structured output")
    )


def _tools():
    sent: list[tuple[str, str]] = []

    def gmail_read(query: str) -> str:
        return f"body for {query}"

    def send_email(to: str, body: str) -> str:
        sent.append((to, body))
        return "ok"

    def summarize(text: str) -> str:
        return text[:5]

    return {"gmail.read": gmail_read, "send_email": send_email, "summarize": summarize}, sent


def test_plain_strings_are_untrusted_and_blocked_by_arg_integrity():
    tools, sent = _tools()
    secured = secure_callables(tools, _policy())
    with pytest.raises(IFCViolationError, match="argument integrity"):
        secured["send_email"]("a@b", "hi")
    assert sent == []


def test_trusted_tainted_args_pass_and_inner_receives_originals():
    tools, sent = _tools()
    secured = secure_callables(tools, _policy())
    to = lift("a@b", source="user_prompt", integrity="TRUSTED")
    body = lift("hi", source="user_prompt", integrity="TRUSTED")
    result = secured["send_email"](to, body)
    assert sent == [(to, body)]  # the inner function sees what the caller passed
    assert isinstance(result, Tainted) and result.value == "ok"
    assert result.provenance == frozenset({"tool:send_email"})
    assert result.integrity == "UNTRUSTED"


def test_outputs_get_policy_source_labels_and_declassifier_output_is_trusted():
    tools, _ = _tools()
    secured = secure_callables(tools, _policy())
    mail = secured["gmail.read"]("latest")
    assert mail.provenance == frozenset({"gmail.read"})
    assert mail.capabilities == frozenset({"read.user"})
    assert mail.integrity == "UNTRUSTED"
    # summarize slices its Tainted input, so it returns a Tainted value; the
    # declassifier upgrades it to TRUSTED and records why.
    summary = secured["summarize"](mail)
    assert summary.integrity == "TRUSTED"
    assert summary.provenance == frozenset({"gmail.read"})
    assert summary.annotations["declassified_by"] == "summarize"
    plain = secured["summarize"]("plain text")
    assert plain.integrity == "TRUSTED" and plain.provenance == frozenset({"tool:summarize"})


def test_lift_outputs_can_be_disabled_and_tainted_results_pass_through():
    tools, _ = _tools()
    secured = secure_callables(tools, _policy(), lift_outputs=False)
    assert secured["gmail.read"]("x") == "body for x"
    policy = _policy()
    pre = lift("v", source="custom")
    assert lift_tool_output(policy, "gmail.read", pre) is pre
    assert lift_tool_output(policy, "unknown", "v").provenance == frozenset({"tool:unknown"})
    assert lift_tool_output(policy, "x", "v", source="vault").capabilities == frozenset(
        {"read.secrets"}
    )


def test_secure_tool_metadata_is_honoured():
    @secure_tool(required_integrity="TRUSTED")
    def strict_tool(x: str) -> str:
        return x

    secured = secure_callable(strict_tool, Policy())
    assert secured.__aisafepy_tool__.required_integrity == "TRUSTED"
    with pytest.raises(IFCViolationError, match="tool-declared"):
        secured("plain string is untrusted")
    assert secured(lift("ok", source="user", integrity="TRUSTED")).value == "ok"


def test_shared_context_and_mediator():
    tools, sent = _tools()
    ctx = IFCContext(control_flow_integrity="UNTRUSTED")
    secured = secure_callables(tools, _policy(), context=ctx)
    to = lift("a@b", source="user_prompt", integrity="TRUSTED")
    with pytest.raises(IFCViolationError, match="control flow integrity"):
        secured["send_email"](to, to)
    approvals = []
    mediated = secure_callables(
        tools, _policy().with_mode("mediated"), context=ctx,
        mediator=lambda v: approvals.append(v) or True,
    )
    assert mediated["send_email"](to, to).value == "ok"
    assert len(approvals) == 1 and sent[-1][0] is to


def test_async_callables_are_wrapped():
    async def fetch(url: str) -> str:
        return "<html>"

    secured = secure_callable(fetch, Policy().label_source("fetch"), name="fetch")
    assert asyncio.iscoroutinefunction(secured)
    out = asyncio.run(secured("http://x"))
    assert out.value == "<html>" and out.provenance == frozenset({"fetch"})
    assert secured.__name__ == "fetch" and secured.__aisafepy_wrapped__ is True


def test_lift_inputs_can_be_disabled():
    def t(x: str) -> str:
        return x

    policy = Policy().require("t", arg_max_integrity="TRUSTED")
    assert secure_callable(t, policy, lift_inputs=False)("plain").value == "plain"
    with pytest.raises(IFCViolationError):
        secure_callable(t, policy)("plain")


# ---- registry ---------------------------------------------------------------------


def test_registry_register_variants_and_duplicates():
    reg = ToolRegistry()

    @reg.register
    def bare(x: str) -> str:
        """Bare registration."""
        return x

    @reg.register(name="send_email", capabilities={Capability.WRITE_EXTERNAL},
                  required_integrity="TRUSTED")
    def send(to: str, body: str = "") -> bool:
        """Send an email.

        Longer description that should not appear in the schema summary.
        """
        return True

    reg.add("extra", lambda: 1)
    assert reg.names() == ["bare", "send_email", "extra"]
    assert "send_email" in reg and len(reg) == 3 and list(reg) == reg.names()
    assert reg.metadata("send_email").capabilities == frozenset({"write.external"})
    assert reg.metadata("extra") is None
    assert reg.get("bare") is bare
    with pytest.raises(ValueError, match="already registered"):
        reg.register(name="bare")(lambda: None)
    with pytest.raises(ValueError):
        reg.add("extra", lambda: 2)
    assert set(reg.as_dict()) == set(reg.names())


def test_registry_secured_enforces_policy():
    reg = ToolRegistry()

    @reg.register(name="send_email", required_integrity="TRUSTED")
    def send(to: str) -> str:
        return "sent"

    tools = reg.secured(Policy())
    with pytest.raises(IFCViolationError):
        tools["send_email"]("x@y")
    assert tools["send_email"](lift("x@y", source="u", integrity="TRUSTED")).value == "sent"


def test_schemas_are_deterministic_and_hashable():
    reg = ToolRegistry()

    @reg.register(name="search", capabilities={"network"})
    def search(
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        mode: Literal["fast", "deep"] = "fast",
        body: Tainted[str] | None = None,
        meta: dict[str, int] | None = None,
    ) -> list[str]:
        """Search the web."""
        return []

    schemas = reg.schemas()
    s = schemas["search"]
    assert s["name"] == "search" and s["description"] == "Search the web."
    props = s["parameters"]["properties"]
    assert props["query"] == {"type": "string"}
    assert props["limit"] == {"type": "integer", "default": 10}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}, "nullable": True,
                             "default": None}
    assert props["mode"] == {"enum": ["fast", "deep"], "default": "fast"}
    assert props["body"]["x-tainted"] is True and props["body"]["type"] == "string"
    assert props["meta"]["additionalProperties"] == {"type": "integer"}
    assert s["parameters"]["required"] == ["query"]
    assert s["returns"] == {"type": "array", "items": {"type": "string"}}
    assert s["aisafepy"] == {"capabilities": ["network"], "required_integrity": "UNTRUSTED"}
    assert schema_hash(s) == schema_hash(reg.schemas()["search"])


def test_tool_schema_handles_odd_callables():
    class Weird:
        pass

    def f(a, b: Weird, *args, c: "Unresolvable", **kwargs) -> Weird:  # noqa: F821, UP037
        pass

    s = tool_schema("f", f)
    props = s["parameters"]["properties"]
    assert props["a"] == {}
    assert props["b"] == {"description": "type: Weird"}
    assert props["c"] == {"description": "type: Unresolvable"}
    assert "args" not in props and "kwargs" not in props
    assert s["parameters"]["required"] == ["a", "b", "c"]
    assert s["description"] == ""
    # Builtins without inspectable signatures still produce a schema.
    assert tool_schema("len", len)["parameters"]["properties"] in ({}, {"obj": {}})
