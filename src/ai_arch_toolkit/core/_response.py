"""Response types — output is typed (safe)."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation returned by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """Token usage counters."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class Attempt:
    """Record of a single LLM call attempt (successful or failed)."""

    model: str
    status: Literal["ok", "failed"]
    error: str | None = None
    error_type: str | None = None
    status_code: int | None = None
    usage: Usage | None = None
    duration: float = 0.0
    timestamp: float = 0.0
    retry_number: int = 0


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """A thinking/reasoning block from the model."""

    text: str


@dataclass(frozen=True, slots=True)
class OutputSchema:
    """Structured output schema definition.

    Use directly or pass a Pydantic model class to ``_resolve_output_schema()``.
    """

    name: str
    schema: dict[str, Any]
    strict: bool = True
    model_class: type | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("OutputSchema.name must be a non-empty string")
        if not self.schema:
            raise ValueError("OutputSchema.schema must be a non-empty dict")


def _resolve_output_schema(schema: OutputSchema | type) -> OutputSchema:
    """Accept ``OutputSchema`` or a Pydantic model class, return ``OutputSchema``."""
    if isinstance(schema, OutputSchema):
        return schema
    try:
        from pydantic import BaseModel

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return OutputSchema(
                name=schema.__name__,
                schema=schema.model_json_schema(),
                model_class=schema,
            )
    except ImportError:
        pass
    raise TypeError(f"Expected OutputSchema or Pydantic model, got {type(schema)}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Citation:
    """A citation from a web search or grounding result."""

    text: str
    url: str = ""
    title: str = ""
    start_index: int | None = None
    end_index: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Response:
    """Immutable LLM response with string-like convenience."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    thinking: tuple[ThinkingBlock, ...] = ()
    parsed: Any = None  # populated only when output_schema was requested
    usage: Usage = field(default_factory=Usage)
    cost: float | None = None
    stop_reason: str = ""
    model: str = ""
    raw: Any = None
    response_id: str = ""
    logprobs: Any = None
    citations: tuple[Citation, ...] = ()
    attempts: tuple[Attempt, ...] = ()

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

    def to_message(self) -> dict[str, Any]:
        """Convert this response to a provider-agnostic assistant message dict.

        Returns a dict suitable for appending to a conversation history and
        passing back to ``LLM.complete()`` or provider ``complete()``.

        If ``raw`` contains a provider SDK response with additional metadata
        (e.g. Gemini thought signatures), it is preserved under ``_raw`` so
        provider-specific message formatters can use the original parts.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "input": dict(tc.input)} for tc in self.tool_calls
            ]
        if self.parsed is not None:
            msg["parsed"] = self.parsed
        if self.raw is not None:
            msg["_raw"] = self.raw
        return msg

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


# ---------------------------------------------------------------------------
# Stream wrappers
# ---------------------------------------------------------------------------


class StreamResponse:
    """Async-iterable stream that accumulates a final ``Response``.

    Usage::

        stream = llm.stream("Hello")
        async for chunk in stream:
            print(chunk, end="")
        print(stream.response.cost)
    """

    __slots__ = ("_aiter", "_chunks", "_finalizer", "_partial_parsed", "_response")

    def __init__(
        self,
        aiter: AsyncIterator[str],
        finalizer: Callable[[str], Response],
    ) -> None:
        self._aiter = aiter
        self._finalizer = finalizer
        self._response: Response | None = None
        self._chunks: list[str] = []
        self._partial_parsed: Any = None

    def __aiter__(self) -> StreamResponse:
        return self

    async def __anext__(self) -> str:
        try:
            chunk = await self._aiter.__anext__()
            self._chunks.append(chunk)
            # Attempt incremental JSON parsing
            text = "".join(self._chunks)
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                self._partial_parsed = json.loads(text)
            return chunk
        except StopAsyncIteration:
            self._response = self._finalizer("".join(self._chunks))
            raise

    @property
    def response(self) -> Response | None:
        """Available after stream is fully consumed. ``None`` during iteration."""
        return self._response

    @property
    def partial_parsed(self) -> Any:
        """Latest successfully parsed partial JSON, or None."""
        return self._partial_parsed

    async def __aenter__(self) -> StreamResponse:
        return self

    async def __aexit__(self, *args: Any) -> None:
        # Unwinding on an exception means the stream was interrupted mid-way — do NOT finalize it
        # as a clean success (the metered finalizer would settle it, under-recording spend and
        # mislabelling a failed call). Leave the op STARTED so scope.close() marks it INCOMPLETE.
        if args and args[0] is not None:
            return
        # Normal exit without full drain — finalize with what we have so far.
        if self._response is None:
            self._response = self._finalizer("".join(self._chunks))


class SyncStreamResponse:
    """Sync-iterable stream that accumulates a final ``Response``.

    Usage::

        stream = llm.stream_sync("Hello")
        for chunk in stream:
            print(chunk, end="")
        print(stream.response.cost)
    """

    __slots__ = ("_chunks", "_finalizer", "_iter", "_response")

    def __init__(
        self,
        sync_iter: Iterator[str],
        finalizer: Callable[[str], Response],
    ) -> None:
        self._iter = sync_iter
        self._finalizer = finalizer
        self._response: Response | None = None
        self._chunks: list[str] = []

    @property
    def response(self) -> Response | None:
        """Available after stream is fully consumed. ``None`` during iteration."""
        return self._response

    def __iter__(self) -> Iterator[str]:
        for chunk in self._iter:
            self._chunks.append(chunk)
            yield chunk
        self._response = self._finalizer("".join(self._chunks))

    def __enter__(self) -> SyncStreamResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        # See StreamResponse.__aexit__: don't settle an exception-interrupted stream as a success.
        if args and args[0] is not None:
            return
        if self._response is None:
            self._response = self._finalizer("".join(self._chunks))


# ---------------------------------------------------------------------------
# Rich streaming event wrappers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Structured streaming event (text chunk, thinking block, or tool call).

    ``partial`` marks an incremental fragment rather than a finished unit.
    Providers that stream reasoning token-by-token (OpenAI-compatible servers)
    set ``partial=True`` on each ``thinking`` event; concatenate consecutive
    partial thinking events for the full trace. Providers that emit complete
    thinking blocks (Anthropic) leave it ``False``. The finalized
    ``Response.thinking`` always holds complete blocks regardless.
    """

    kind: Literal["text", "thinking", "tool_call"]
    text: str = ""
    thinking: ThinkingBlock | None = None
    tool_call: ToolCall | None = None
    partial: bool = False


class RichStreamResponse:
    """Async-iterable stream of ``StreamEvent`` with finalized ``Response``.

    Usage::

        stream = llm.stream_events("Hello")
        async for event in stream:
            if event.kind == "text":
                print(event.text, end="")
        print(stream.response.cost)
    """

    __slots__ = ("_aiter", "_finalizer", "_response", "_text_chunks")

    def __init__(
        self,
        aiter: AsyncIterator[StreamEvent],
        finalizer: Callable[[str], Response],
    ) -> None:
        self._aiter = aiter
        self._finalizer = finalizer
        self._response: Response | None = None
        self._text_chunks: list[str] = []

    def __aiter__(self) -> RichStreamResponse:
        return self

    async def __anext__(self) -> StreamEvent:
        try:
            event = await self._aiter.__anext__()
            if event.kind == "text":
                self._text_chunks.append(event.text)
            return event
        except StopAsyncIteration:
            self._response = self._finalizer("".join(self._text_chunks))
            raise

    @property
    def response(self) -> Response | None:
        """Available after stream is fully consumed. ``None`` during iteration."""
        return self._response

    async def __aenter__(self) -> RichStreamResponse:
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._response is None:
            self._response = self._finalizer("".join(self._text_chunks))


class SyncRichStreamResponse:
    """Sync-iterable stream of ``StreamEvent`` with finalized ``Response``."""

    __slots__ = ("_finalizer", "_iter", "_response", "_text_chunks")

    def __init__(
        self,
        sync_iter: Iterator[StreamEvent],
        finalizer: Callable[[str], Response],
    ) -> None:
        self._iter = sync_iter
        self._finalizer = finalizer
        self._response: Response | None = None
        self._text_chunks: list[str] = []

    @property
    def response(self) -> Response | None:
        """Available after stream is fully consumed. ``None`` during iteration."""
        return self._response

    def __iter__(self) -> Iterator[StreamEvent]:
        for event in self._iter:
            if event.kind == "text":
                self._text_chunks.append(event.text)
            yield event
        self._response = self._finalizer("".join(self._text_chunks))

    def __enter__(self) -> SyncRichStreamResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        if self._response is None:
            self._response = self._finalizer("".join(self._text_chunks))
