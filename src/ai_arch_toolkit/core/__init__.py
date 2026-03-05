"""Core primitives — zero opinion, maximum flexibility."""

from __future__ import annotations

import logging

from ai_arch_toolkit.core._batch import BatchRequest, BatchResult
from ai_arch_toolkit.core._content import (
    CachePart,
    Content,
    ContentPart,
    DocumentPart,
    ImagePart,
    assistant,
    cache,
    document,
    image,
    system,
    tool_result,
    user,
)
from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._middleware import Middleware, Request
from ai_arch_toolkit.core._pricing import pricing
from ai_arch_toolkit.core._rate_limit import RateLimitMiddleware
from ai_arch_toolkit.core._response import (
    Attempt,
    Citation,
    OutputSchema,
    Response,
    RichStreamResponse,
    StreamEvent,
    SyncRichStreamResponse,
    ThinkingBlock,
    ToolCall,
    Usage,
)
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.core._server_tools import ServerTool, code_execution, web_search
from ai_arch_toolkit.core._sync import configure_sync_timeouts
from ai_arch_toolkit.core._telemetry import TracingMiddleware
from ai_arch_toolkit.core._tools import (
    ToolGroup,
    async_execute_tool,
    execute_tool,
    infer_schema,
    prepare_tools,
    tool,
)
from ai_arch_toolkit.core.graph import (
    Edge as GraphEdge,
)
from ai_arch_toolkit.core.graph import (
    Graph,
    NodeID,
    NodeType,
)
from ai_arch_toolkit.core.graph import (
    GraphAlgorithms as GraphAlgorithmsProto,
)
from ai_arch_toolkit.core.graph import (
    GraphBackend as GraphBackendProto,
)
from ai_arch_toolkit.core.graph import (
    Node as GraphNode,
)

# Keep package logger configured on import.
_pkg_logger = logging.getLogger("ai_arch_toolkit")
if not any(isinstance(h, logging.NullHandler) for h in _pkg_logger.handlers):
    _pkg_logger.addHandler(logging.NullHandler())

__all__ = [
    "LLM",
    "APIError",
    "Attempt",
    "BatchRequest",
    "BatchResult",
    "CachePart",
    "Citation",
    "Content",
    "ContentPart",
    "DocumentPart",
    "Graph",
    "GraphAlgorithmsProto",
    "GraphBackendProto",
    "GraphEdge",
    "GraphNode",
    "ImagePart",
    "Middleware",
    "NodeID",
    "NodeType",
    "OutputSchema",
    "RateLimitError",
    "RateLimitMiddleware",
    "Request",
    "Response",
    "RetryConfig",
    "RichStreamResponse",
    "ServerTool",
    "StreamEvent",
    "SyncRichStreamResponse",
    "ThinkingBlock",
    "ToolCall",
    "ToolGroup",
    "TracingMiddleware",
    "Usage",
    "assistant",
    "async_execute_tool",
    "cache",
    "code_execution",
    "configure_sync_timeouts",
    "document",
    "execute_tool",
    "image",
    "infer_schema",
    "prepare_tools",
    "pricing",
    "system",
    "tool",
    "tool_result",
    "user",
    "web_search",
]
