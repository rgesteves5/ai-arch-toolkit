"""AsyncClient — the async user-facing entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from ai_arch_toolkit._legacy.llm._http import RetryConfig
from ai_arch_toolkit._legacy.llm._middleware import Middleware, Request
from ai_arch_toolkit._legacy.llm._providers import create_provider
from ai_arch_toolkit._legacy.llm._types import (
    ConversationItem,
    JsonSchema,
    Message,
    Response,
    StreamEvent,
    Tool,
    ToolResult,
)

logger = logging.getLogger(__name__)
_SHORT_CIRCUIT_RESULT_KEY = "middleware.short_circuit_result"


class AsyncClient:
    """Async unified LLM client that delegates to provider-specific implementations.

    Usage::

        from ai_arch_toolkit import AsyncClient

        client = AsyncClient("openai", model="gpt-4o")
        response = await client.chat("Hello!")
        print(response.text)
    """

    def __init__(
        self,
        provider: str,
        *,
        model: str,
        api_key: str | None = None,
        retry: RetryConfig | None = None,
        middleware: Sequence[Middleware] | None = None,
    ) -> None:
        logger.debug("Initializing async client for provider=%s model=%s", provider, model)
        self._provider_name = provider
        self._model = model
        self._middleware = list(middleware or ())
        self._provider = create_provider(provider, model, api_key, retry=retry)

    async def _run_before(self, request: Request) -> Request:
        for m in self._middleware:
            if hasattr(m, "abefore"):
                request = await m.abefore(request)
            else:
                request = m.before(request)
        return request

    async def _run_after(self, request: Request, result: Any) -> Any:
        for m in reversed(self._middleware):
            if hasattr(m, "aafter"):
                result = await m.aafter(request, result)
            else:
                result = m.after(request, result)
        return result

    @staticmethod
    def _normalize_input(
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
    ) -> list[ConversationItem]:
        if isinstance(prompt_or_messages, str):
            logger.debug("Normalizing string prompt into one user message")
            return [Message(role="user", content=prompt_or_messages)]
        items: list[ConversationItem] = []
        for m in prompt_or_messages:
            if isinstance(m, (Message, ToolResult)):
                items.append(m)
            else:
                items.append(Message(role=m["role"], content=m["content"]))
        if not items:
            logger.warning("Received empty prompt_or_messages sequence")
        logger.debug("Normalized %s conversation item(s)", len(items))
        return items

    async def chat(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        json_schema: JsonSchema | None = None,
        timeout: int | float | None = None,
        **kwargs: Any,
    ) -> Response:
        """Send an async chat request and return a unified Response."""
        messages = self._normalize_input(prompt_or_messages)
        logger.debug(
            "Running async chat with %s message(s), has_system=%s, has_tools=%s, has_schema=%s",
            len(messages),
            system is not None,
            tools is not None,
            json_schema is not None,
        )
        if timeout is not None:
            logger.debug("Using explicit timeout=%s for async chat request", timeout)
            kwargs["timeout"] = timeout
        request = await self._run_before(
            Request(
                operation="chat",
                provider=self._provider_name,
                model=self._model,
                messages=messages,
                system=system,
                tools=tools,
                json_schema=json_schema,
                kwargs=dict(kwargs),
            )
        )
        if _SHORT_CIRCUIT_RESULT_KEY in request.context:
            logger.debug("Async chat request short-circuited by middleware")
            return await self._run_after(request, request.context[_SHORT_CIRCUIT_RESULT_KEY])
        result = await self._provider.acomplete(
            request.messages,
            system=request.system,
            tools=request.tools,
            json_schema=request.json_schema,
            **request.kwargs,
        )
        return await self._run_after(request, result)

    async def stream(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        *,
        system: str | None = None,
        timeout: int | float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text chunks from the model asynchronously."""
        messages = self._normalize_input(prompt_or_messages)
        logger.debug("Starting async text stream with %s message(s)", len(messages))
        if timeout is not None:
            logger.debug("Using explicit timeout=%s for async stream request", timeout)
            kwargs["timeout"] = timeout
        request = await self._run_before(
            Request(
                operation="stream",
                provider=self._provider_name,
                model=self._model,
                messages=messages,
                system=system,
                kwargs=dict(kwargs),
            )
        )
        stream = self._provider.astream(request.messages, system=request.system, **request.kwargs)
        transformed_stream = await self._run_after(request, stream)
        async for chunk in transformed_stream:
            yield chunk

    async def stream_events(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        timeout: int | float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream rich events (text, tool_call, thinking, usage, done) asynchronously."""
        messages = self._normalize_input(prompt_or_messages)
        logger.debug(
            "Starting async event stream with %s message(s), has_tools=%s",
            len(messages),
            tools is not None,
        )
        if timeout is not None:
            logger.debug("Using explicit timeout=%s for async stream_events request", timeout)
            kwargs["timeout"] = timeout
        request = await self._run_before(
            Request(
                operation="stream_events",
                provider=self._provider_name,
                model=self._model,
                messages=messages,
                system=system,
                tools=tools,
                kwargs=dict(kwargs),
            )
        )
        events = self._provider.astream_events(
            request.messages,
            system=request.system,
            tools=request.tools,
            **request.kwargs,
        )
        transformed_events = await self._run_after(request, events)
        async for event in transformed_events:
            yield event
