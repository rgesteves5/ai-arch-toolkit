"""Server-side tool types (web search, code execution, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ServerTool:
    """A server-side tool managed by the LLM provider.

    Unlike function tools, server tools are executed by the provider's
    infrastructure (e.g. web search, code interpreter).

    .. todo:: ``config`` is spread into the wire dict by ``prepare_tools()``
       but currently ignored by all provider implementations. Add
       provider-specific config forwarding (e.g. Anthropic ``max_uses``,
       OpenAI ``allowed_domains``) or remove config until the shapes are known.
    """

    type: str
    config: dict[str, Any] = field(default_factory=dict)


def web_search(**config: Any) -> ServerTool:
    """Create a web search server tool."""
    return ServerTool(type="web_search", config=config)


def code_execution(**config: Any) -> ServerTool:
    """Create a code execution/interpreter server tool."""
    return ServerTool(type="code_execution", config=config)
