"""LLM client — unified interface for multiple LLM providers."""

from ai_arch_toolkit.llm._async_batch import AsyncBatchClient
from ai_arch_toolkit.llm._async_client import AsyncClient
from ai_arch_toolkit.llm._batch import BatchClient, BatchJob, BatchRequest, BatchResult
from ai_arch_toolkit.llm._client import Client
from ai_arch_toolkit.llm._exceptions import APIError, RateLimitError
from ai_arch_toolkit.llm._http import RetryConfig
from ai_arch_toolkit.llm._types import (
    AudioPart,
    DocumentPart,
    ImagePart,
    JsonSchema,
    Message,
    Response,
    ServerTool,
    StreamEvent,
    TextPart,
    ThinkingBlock,
    ThinkingConfig,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
)

__all__ = [
    "APIError",
    "AsyncBatchClient",
    "AsyncClient",
    "AudioPart",
    "BatchClient",
    "BatchJob",
    "BatchRequest",
    "BatchResult",
    "Client",
    "DocumentPart",
    "ImagePart",
    "JsonSchema",
    "Message",
    "RateLimitError",
    "Response",
    "RetryConfig",
    "ServerTool",
    "StreamEvent",
    "TextPart",
    "ThinkingBlock",
    "ThinkingConfig",
    "Tool",
    "ToolCall",
    "ToolResult",
    "Usage",
]
