"""Unified types for LLM API responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- Multimodal content parts ---


@dataclass(frozen=True, slots=True)
class TextPart:
    """A plain text content part."""

    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    """An image content part (URL, data URI, or raw base64)."""

    url: str = ""
    media_type: str = ""
    data: str = ""
    detail: str = "auto"


@dataclass(frozen=True, slots=True)
class AudioPart:
    """An audio content part (base64-encoded)."""

    data: str = ""
    media_type: str = ""
    transcript: str = ""


@dataclass(frozen=True, slots=True)
class DocumentPart:
    """A document content part (PDF, etc.)."""

    data: str = ""
    media_type: str = "application/pdf"
    uri: str = ""


type ContentPart = TextPart | ImagePart | AudioPart | DocumentPart
type Content = str | tuple[ContentPart, ...]


# --- Thinking / reasoning ---


@dataclass(frozen=True, slots=True)
class ThinkingConfig:
    """Thinking/reasoning configuration passed to providers."""

    effort: str = "medium"
    budget_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """A single thinking/reasoning block from a response."""

    text: str


# --- Streaming events ---


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """A rich streaming event (text, tool_call, thinking, usage, done)."""

    type: str
    text: str = ""
    tool_call: ToolCall | None = None
    thinking: str = ""
    usage: Usage | None = None


# --- Server tools ---


@dataclass(frozen=True, slots=True)
class ServerTool:
    """Built-in server-side tool (web_search, code_interpreter, etc.)."""

    type: str
    config: dict[str, Any] = field(default_factory=dict)


# --- Core types ---


@dataclass(frozen=True, slots=True)
class Message:
    """A chat message."""

    role: str
    content: Content = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A tool result sent back to the model after executing a tool call."""

    tool_call_id: str
    name: str
    content: str


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool/function definition passed to the model."""

    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call returned by the model."""

    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage counts."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass(frozen=True, slots=True)
class JsonSchema:
    """Schema for structured JSON output."""

    name: str
    schema: dict[str, Any]
    strict: bool = True


@dataclass(frozen=True, slots=True)
class Response:
    """Unified response from any LLM provider."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""
    thinking: str = ""
    thinking_blocks: tuple[ThinkingBlock, ...] = ()
    raw: dict[str, object] = field(default_factory=dict)

    def to_message(self) -> Message:
        """Convert this response to a Message suitable for multi-turn conversations."""
        return Message(role="assistant", content=self.text, tool_calls=self.tool_calls)


type ConversationItem = Message | ToolResult
