"""Guardrail middleware for input/output validation."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from ai_arch_toolkit.llm._middleware import Request
from ai_arch_toolkit.llm._types import ConversationItem, Response, StreamEvent, ToolResult


class GuardrailViolation(ValueError):
    """Raised when guardrail checks fail on input or output content."""


def _item_text(item: ConversationItem) -> str:
    if isinstance(item, ToolResult):
        return f"{item.name}\n{item.content}"
    if isinstance(item.content, str):
        return item.content
    text_parts: list[str] = []
    for part in item.content:
        if hasattr(part, "text"):
            text_parts.append(part.text)
    return "\n".join(text_parts)


class GuardrailMiddleware:
    """Middleware to block disallowed patterns and run custom validators."""

    def __init__(
        self,
        *,
        blocked_patterns: list[str] | None = None,
        input_validator: Callable[[Request], None] | None = None,
        output_validator: Callable[[Response], None] | None = None,
    ) -> None:
        self._input_validator = input_validator
        self._output_validator = output_validator
        self._blocked = [re.compile(pat, flags=re.IGNORECASE) for pat in (blocked_patterns or [])]

    def before(self, request: Request) -> Request:
        if self._input_validator is not None:
            self._input_validator(request)
        for item in request.messages:
            self._check_text(_item_text(item), stage="input")
        return request

    def after(self, request: Request, result: Any) -> Any:
        if isinstance(result, Response):
            self._check_response(result)
            return result
        if request.operation == "stream" and isinstance(result, Iterator):
            return self._wrap_stream(result)
        if request.operation == "stream_events" and isinstance(result, Iterator):
            return self._wrap_stream_events(result)
        return result

    async def abefore(self, request: Request) -> Request:
        return self.before(request)

    async def aafter(self, request: Request, result: Any) -> Any:
        if isinstance(result, Response):
            self._check_response(result)
            return result
        if request.operation == "stream" and isinstance(result, AsyncIterator):
            return self._awrap_stream(result)
        if request.operation == "stream_events" and isinstance(result, AsyncIterator):
            return self._awrap_stream_events(result)
        return result

    def _check_response(self, response: Response) -> None:
        self._check_text(response.text, stage="output")
        if self._output_validator is not None:
            self._output_validator(response)

    def _check_text(self, text: str, *, stage: str) -> None:
        if not text:
            return
        for pattern in self._blocked:
            if pattern.search(text):
                raise GuardrailViolation(
                    f"Blocked {stage} content matched pattern: {pattern.pattern!r}"
                )

    def _wrap_stream(self, stream: Iterator[str]) -> Iterator[str]:
        for chunk in stream:
            self._check_text(chunk, stage="output")
            yield chunk

    def _wrap_stream_events(self, stream: Iterator[StreamEvent]) -> Iterator[StreamEvent]:
        for event in stream:
            if event.type == "text":
                self._check_text(event.text, stage="output")
            yield event

    async def _awrap_stream(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        async for chunk in stream:
            self._check_text(chunk, stage="output")
            yield chunk

    async def _awrap_stream_events(
        self, stream: AsyncIterator[StreamEvent]
    ) -> AsyncIterator[StreamEvent]:
        async for event in stream:
            if event.type == "text":
                self._check_text(event.text, stage="output")
            yield event

