"""Framework-specific IFC adapters.

Each adapter wraps an agent / graph / tool collection from one of the
dominant frameworks (OpenAI Agents SDK, LangGraph, LlamaIndex,
Anthropic SDK tools, MCP server) with an IFC enforcement layer.

The adapters share a common contract:
* They accept an *agent-like* object plus a ``Policy``.
* They return a structurally compatible wrapper (you can hand it back
  to the framework's runner / executor).
* They register interception points so that ``aisafepy.flow.interpreter``
  is called *before* every tool execution.
"""
