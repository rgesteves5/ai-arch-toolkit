"""Client — the main user-facing entry point."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from ai_arch_toolkit.llm._http import RetryConfig
from ai_arch_toolkit.llm._providers import create_provider
from ai_arch_toolkit.llm._types import (
    ConversationItem,
    JsonSchema,
    Message,
    Response,
    StreamEvent,
    Tool,
    ToolResult,
)


class Client:
    """Unified LLM client that delegates to provider-specific implementations.

    Usage::

        from ai_arch_toolkit import Client

        client = Client("openai", model="gpt-4o")
        response = client.chat("Hello!")
        print(response.text)
    """

    def __init__(
        self,
        provider: str,
        *,
        model: str,
        api_key: str | None = None,
        retry: RetryConfig | None = None,
    ) -> None:
        self._provider = create_provider(provider, model, api_key, retry=retry)

    @staticmethod
    def _normalize_input(
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
    ) -> list[ConversationItem]:
        if isinstance(prompt_or_messages, str):
            return [Message(role="user", content=prompt_or_messages)]
        items: list[ConversationItem] = []
        for m in prompt_or_messages:
            if isinstance(m, (Message, ToolResult)):
                items.append(m)
            else:
                items.append(Message(role=m["role"], content=m["content"]))
        return items

    def chat(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        json_schema: JsonSchema | None = None,
        **kwargs: Any,
    ) -> Response:
        """Send a chat request and return a unified Response."""
        messages = self._normalize_input(prompt_or_messages)
        return self._provider.complete(
            messages, system=system, tools=tools, json_schema=json_schema, **kwargs
        )

    def stream(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text chunks from the model."""
        messages = self._normalize_input(prompt_or_messages)
        yield from self._provider.stream(messages, system=system, **kwargs)

    def stream_events(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Stream rich events (text, tool_call, thinking, usage, done) from the model."""
        messages = self._normalize_input(prompt_or_messages)
        yield from self._provider.stream_events(messages, system=system, tools=tools, **kwargs)
