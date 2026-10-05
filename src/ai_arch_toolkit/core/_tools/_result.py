"""Structured results for tool execution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

type ToolFailureType = Literal["not_found", "validation_error", "upstream", "rate_limited"]
"""What a tool that could not answer says happened (D37, D42): what it was asked for does not
exist, an argument is wrong, the source failed, or the source is rate limiting."""


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


class ToolFailure(Exception):
    """Raised by a tool that could not answer; the executor returns it as a failed result.

    ``message`` says why, in the source's words, and what to do next ("... does not exist; search
    with wiki_search"). Zero results is not a failure: a tool says so in a successful answer.

    Attributes:
        error: The :class:`ToolError` the executor puts in ``ToolResult.error`` (its message
            redacted).
    """

    def __init__(
        self,
        type: ToolFailureType,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error = ToolError(
            type=type, message=message, retryable=retryable, details=dict(details or {})
        )

    def __reduce__(self) -> tuple[Any, ...]:
        # Pickled and copied by its error and attributes, whatever a subclass's signature.
        return (_restored_failure, (type(self), self.error), self.__dict__)


def _restored_failure(cls: type[ToolFailure], error: ToolError) -> ToolFailure:
    failure = cls.__new__(cls)
    Exception.__init__(failure, error.message)
    failure.error = error
    return failure


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


def line_cut(text: str, start: int, end: int) -> int:
    """Where to end ``text[start:end]`` so a cut part keeps whole lines.

    Just after the last line break in the second half of the span, so a table row or a paragraph
    is not split; at ``end`` when the span reaches the end of ``text`` or has no break there (one
    long line is cut where the limit falls). The executor's output bound and the toolkit's text
    windows both cut here.
    """
    if end >= len(text):
        return len(text)
    brk = text.rfind("\n", start + (end - start) // 2, end)
    return brk + 1 if brk >= 0 else end
