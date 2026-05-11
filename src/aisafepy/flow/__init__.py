"""Capability-based information-flow control for tool-using LLM agents.

The ``aisafepy.flow`` module implements a developer-friendly version of
the CaMeL / FIDES / RTBAS / SAFEFLOW research line. Untrusted data
(emails, web pages, retrieval hits, MCP tool outputs) is wrapped in
``Tainted[T]`` values that carry a provenance set, a capability set,
and an integrity label. Tool calls declare what they *require* via
``@secure_tool``; the policy declares what is *forbidden*. Operations
on ``Tainted`` values propagate labels (CaMeL-style) so the policy
engine can deny by construction rather than by classification.

Quickstart::

    from aisafepy.flow import Policy, Capability, Tainted, secure_tool, secure_agent

    policy = (
        Policy()
        .label_source("user_prompt", integrity="TRUSTED")
        .label_source("gmail.read", integrity="UNTRUSTED", caps={Capability.READ_USER})
        .require("send_email", control_flow_integrity="TRUSTED",
                 caps={Capability.WRITE_EXTERNAL})
        .deny_if("send_email",
                 when=lambda to, body: "read.secrets" in body.provenance,
                 reason="secret-to-external-sink")
    )

    @secure_tool(capabilities={Capability.WRITE_EXTERNAL})
    def send_email(to: Tainted[str], body: Tainted[str]) -> None:
        ...
"""

from aisafepy.flow.policy import Capability, Integrity, Policy, secure_tool
from aisafepy.flow.taint import Tainted, lift, lower
from aisafepy.flow.interpreter import IFCContext, evaluate_call

__all__ = [
    "Capability",
    "IFCContext",
    "Integrity",
    "Policy",
    "Tainted",
    "evaluate_call",
    "lift",
    "lower",
    "secure_agent",
    "secure_tool",
    "ifc_mcp_server",
]


# Adapters are imported lazily so the top-level package doesn't pull in
# openai-agents / langgraph / llama-index just because someone wrote
# ``from aisafepy.flow import Policy``.


def secure_agent(agent, policy, *, framework: str | None = None, **kwargs):
    """Wrap an agent from any supported framework with IFC enforcement.

    The framework is inferred from the agent object's class name when
    ``framework`` is not supplied. Override via the ``framework`` arg
    to be explicit.
    """
    fw = framework or _infer_framework(agent)
    if fw == "openai-agents":
        from aisafepy.flow.adapters.openai_agents import secure_openai_agent

        return secure_openai_agent(agent, policy, **kwargs)
    if fw == "langgraph":
        from aisafepy.flow.adapters.langgraph import secure_langgraph

        return secure_langgraph(agent, policy, **kwargs)
    if fw == "llamaindex":
        from aisafepy.flow.adapters.llamaindex import secure_llamaindex_agent

        return secure_llamaindex_agent(agent, policy, **kwargs)
    if fw == "anthropic":
        from aisafepy.flow.adapters.anthropic_tools import secure_anthropic_tools

        return secure_anthropic_tools(agent, policy, **kwargs)
    raise ValueError(
        f"Unknown framework {fw!r}. Pass an explicit framework= argument "
        "(one of 'openai-agents', 'langgraph', 'llamaindex', 'anthropic')."
    )


def ifc_mcp_server(server, policy, **kwargs):
    """Wrap an MCP server with the IFC middleware.

    Adds runtime checks on ``tools/call`` JSON-RPC frames and static
    metadata analysis on tool descriptions at registration time
    (the MCP-38 / arXiv 2603.22489 recommendation).
    """
    from aisafepy.flow.adapters.mcp import ifc_mcp_server as _impl

    return _impl(server, policy, **kwargs)


def _infer_framework(agent: object) -> str:
    qual = type(agent).__module__ + "." + type(agent).__qualname__
    if "agents" in qual.split(".")[0]:  # openai-agents
        return "openai-agents"
    if "langgraph" in qual:
        return "langgraph"
    if "llama_index" in qual or "llamaindex" in qual:
        return "llamaindex"
    if "anthropic" in qual:
        return "anthropic"
    return "unknown"
