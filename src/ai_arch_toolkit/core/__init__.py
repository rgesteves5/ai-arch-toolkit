"""Core primitives — zero opinion, maximum flexibility."""

from __future__ import annotations

from ai_arch_toolkit.core import _logging as _package_logging
from ai_arch_toolkit.core._content import assistant, system, tool_result, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._pricing import pricing
from ai_arch_toolkit.core._tools import (
    ToolGroup,
    async_execute_tool,
    execute_tool,
)

# Keep package logger configured on import without exporting internal symbols.
del _package_logging

__all__ = [
    "LLM",
    "ToolGroup",
    "assistant",
    "async_execute_tool",
    "execute_tool",
    "pricing",
    "system",
    "tool_result",
    "user",
]
