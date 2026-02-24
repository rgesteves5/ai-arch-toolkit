"""Message constructor helpers — input is dicts (flexible)."""

from __future__ import annotations

from typing import Any


def system(content: str) -> dict[str, Any]:
    """Create a system message dict."""
    return {"role": "system", "content": content}


def user(content: str | list[Any]) -> dict[str, Any]:
    """Create a user message dict."""
    return {"role": "user", "content": content}


def assistant(content: str) -> dict[str, Any]:
    """Create an assistant message dict."""
    return {"role": "assistant", "content": content}


def tool_result(content: Any, *, tool_use_id: str) -> dict[str, Any]:
    """Create a tool_result message dict."""
    return {"role": "tool", "content": content, "tool_use_id": tool_use_id}
