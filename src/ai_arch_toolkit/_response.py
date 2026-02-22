"""Response types — output is typed (safe)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation returned by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage counters."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Response:
    """Immutable LLM response with string-like convenience."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    cost_estimated: bool = False
    stop_reason: str = ""
    model: str = ""
    raw: Any = None

    # --- shortcut properties ---

    @property
    def tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.usage.input_tokens + self.usage.output_tokens

    @property
    def input_tokens(self) -> int:
        return self.usage.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.usage.output_tokens

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    # --- string-like behaviour ---

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        if self.tool_calls:
            tools = ", ".join(tc.name for tc in self.tool_calls)
            return f"Response(text={self.text!r}, tool_calls=[{tools}])"
        return f"Response(text={self.text!r})"

    def __bool__(self) -> bool:
        return bool(self.text) or bool(self.tool_calls)

    def __contains__(self, item: str) -> bool:
        return item in self.text

    def __add__(self, other: str) -> str:
        return self.text + other

    def __radd__(self, other: str) -> str:
        return other + self.text
