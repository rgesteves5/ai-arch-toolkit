"""Shared fixtures for agent tests."""

from __future__ import annotations

from typing import Any

from ai_arch_toolkit.core._response import Response, ToolCall, Usage


def make_tool_call(
    name: str = "test_tool",
    input: dict[str, Any] | None = None,
    id: str = "tc_1",
) -> ToolCall:
    """Build a ToolCall for testing."""
    return ToolCall(id=id, name=name, input=input or {})


def make_response(
    text: str = "",
    tool_calls: tuple[ToolCall, ...] = (),
    usage: Usage | None = None,
    cost: float | None = None,
) -> Response:
    """Build a Response for testing."""
    return Response(
        text=text,
        tool_calls=tool_calls,
        usage=usage or Usage(input_tokens=10, output_tokens=5),
        cost=cost or 0.001,
    )
