"""Abstract base for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ai_arch_toolkit.llm._http import RetryConfig
from ai_arch_toolkit.llm._types import ConversationItem, JsonSchema, Response, StreamEvent, Tool


class BaseProvider(ABC):
    """Interface that every provider must implement."""

    def __init__(self, *, retry: RetryConfig | None = None) -> None:
        self._retry = retry

    @abstractmethod
    def complete(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        json_schema: JsonSchema | None = None,
        **kwargs: Any,
    ) -> Response: ...

    @abstractmethod
    def stream(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]: ...

    @abstractmethod
    def stream_events(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]: ...

    @abstractmethod
    async def acomplete(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        json_schema: JsonSchema | None = None,
        **kwargs: Any,
    ) -> Response: ...

    @abstractmethod
    def astream(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]: ...

    @abstractmethod
    def astream_events(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]: ...
