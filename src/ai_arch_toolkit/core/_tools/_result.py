"""Structured results for tool execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolError:
    """Structured information about a tool execution failure."""

    type: str
    message: str
    retryable: bool = False
    safe_to_show: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "type": self.type,
            "message": self.message,
            "retryable": self.retryable,
            "safe_to_show": self.safe_to_show,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured result produced by the tool runtime."""

    ok: bool
    value: Any | None = None
    error: ToolError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, value: Any, metadata: dict[str, Any] | None = None) -> ToolResult:
        """Create a successful tool result."""
        return cls(ok=True, value=value, metadata=metadata or {})

    @classmethod
    def failure(
        cls,
        error_type: str,
        message: str,
        *,
        retryable: bool = False,
        safe_to_show: bool = True,
        details: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Create a failed tool result."""
        return cls(
            ok=False,
            error=ToolError(
                type=error_type,
                message=message,
                retryable=retryable,
                safe_to_show=safe_to_show,
                details=details or {},
            ),
            metadata=metadata or {},
        )

    def to_model_text(self) -> str:
        """Convert this result to text for an LLM tool-result message."""
        if self.ok:
            return _format_value(self.value)

        if self.error is None:
            return "Tool error: unknown failure"

        message = self.error.message if self.error.safe_to_show else "Tool execution failed"
        return f"Tool error [{self.error.type}]: {message}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "ok": self.ok,
            "value": self.value,
            "error": self.error.to_dict() if self.error else None,
            "metadata": self.metadata,
        }


def _format_value(value: Any) -> str:
    """Convert a successful tool value to provider-facing text."""
    if isinstance(value, str):
        return value
    return json.dumps(value)
