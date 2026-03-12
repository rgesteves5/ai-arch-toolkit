"""ModerationMiddleware — adapter that plugs a Moderator into the LLM middleware chain."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._moderation import ModerationError
from ai_arch_toolkit.core._response import Response

if TYPE_CHECKING:
    from ai_arch_toolkit.core._moderation import Moderator

logger = logging.getLogger(__name__)


class ModerationMiddleware:
    """Middleware that moderates LLM input, output, or both.

    Runs moderation checks via the ``Moderator`` protocol. Input moderation
    runs in ``abefore`` (before the LLM call); output moderation runs in
    ``aafter`` (after the response is finalized).

    Note: for streaming, output moderation fires after stream finalization,
    so streamed text is seen by the user before moderation completes.

    Example::

        mod = OpenAIModerator()
        mw = ModerationMiddleware(input=mod, on_flagged="raise")
        llm = LLM("claude-sonnet-4-20250514", middleware=[mw])
    """

    __slots__ = ("_input", "_on_flagged", "_output")

    def __init__(
        self,
        *,
        input: Moderator | None = None,
        output: Moderator | None = None,
        on_flagged: Literal["raise", "warn"] = "raise",
    ) -> None:
        if input is None and output is None:
            raise ValueError("At least one of 'input' or 'output' must be provided")
        self._input = input
        self._output = output
        self._on_flagged = on_flagged

    def before(self, request: Request) -> Request:
        """Sync no-op (protocol conformance)."""
        return request

    def after(self, request: Request, response: Response) -> Response:
        """Sync no-op (protocol conformance)."""
        return response

    async def abefore(self, request: Request) -> Request:
        """Moderate user input before the LLM call."""
        if self._input is None:
            return request
        text = _extract_text(request.messages)
        if not text:
            return request
        result = await self._input.moderate(text)
        if result.flagged:
            if self._on_flagged == "raise":
                raise ModerationError(result.categories, result.explanation)
            logger.warning(
                "Input flagged: categories=%s explanation=%s",
                result.categories,
                result.explanation,
            )
        return request

    async def aafter(self, request: Request, response: Response) -> Response:
        """Moderate LLM output after the response is finalized."""
        if self._output is None:
            return response
        text = response.text
        if not text:
            return response
        result = await self._output.moderate(text)
        if result.flagged:
            if self._on_flagged == "raise":
                raise ModerationError(result.categories, result.explanation)
            logger.warning(
                "Output flagged: categories=%s explanation=%s",
                result.categories,
                result.explanation,
            )
        return response


def _extract_text(messages: list[dict]) -> str:  # type: ignore[type-arg]
    """Extract the latest user message text."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return " ".join(parts)
    return ""
