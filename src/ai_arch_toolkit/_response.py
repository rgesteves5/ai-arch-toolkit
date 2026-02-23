"""Response types — output is typed (safe)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
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

    __slots__ = ("_aiter", "_chunks", "_finalizer", "_response")

    def __init__(
        self,
        aiter: AsyncIterator[str],
        finalizer: Callable[[str], Response],
    ) -> None:
        self._aiter = aiter
        self._finalizer = finalizer
        self._response: Response | None = None
        self._chunks: list[str] = []

    def __aiter__(self) -> StreamResponse:
        return self

    async def __anext__(self) -> str:
        try:
            chunk = await self._aiter.__anext__()
            self._chunks.append(chunk)
            return chunk
        except StopAsyncIteration:
            self._response = self._finalizer("".join(self._chunks))
            raise

    @property
    def response(self) -> Response | None:
        """Available after stream is fully consumed. ``None`` during iteration."""
        return self._response

    async def __aenter__(self) -> StreamResponse:
        return self

    async def __aexit__(self, *args: Any) -> None:
        # Don't drain — just finalize with what we have so far.
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
