"""Core primitives — zero opinion, maximum flexibility."""

from __future__ import annotations

import logging

from ai_arch_toolkit.core._content import assistant, system, tool_result, user
from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._pricing import pricing
from ai_arch_toolkit.core._response import (
    OutputSchema,
    Response,
    ThinkingBlock,
    ToolCall,
    Usage,
)
from ai_arch_toolkit.core._tools import (
    ToolGroup,
    async_execute_tool,
    execute_tool,
    infer_schema,
    prepare_tools,
    tool,
)

# Keep package logger configured on import.
_pkg_logger = logging.getLogger("ai_arch_toolkit")
if not any(isinstance(h, logging.NullHandler) for h in _pkg_logger.handlers):
    _pkg_logger.addHandler(logging.NullHandler())

__all__ = [
    "LLM",
    "APIError",
    "OutputSchema",
    "RateLimitError",
    "Response",
    "ThinkingBlock",
    "ToolCall",
    "ToolGroup",
    "Usage",
    "assistant",
    "async_execute_tool",
    "execute_tool",
    "infer_schema",
    "prepare_tools",
    "pricing",
    "system",
    "tool",
    "tool_result",
    "user",
]
