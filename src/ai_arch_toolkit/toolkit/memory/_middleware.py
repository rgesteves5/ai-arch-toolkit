"""MemoryMiddleware — auto inject relevant memories and record interactions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._response import Response
from ai_arch_toolkit.toolkit.memory._types import SearchResult

type FindFn = Callable[..., Awaitable[Sequence[SearchResult]]]
type RecordFn = Callable[[dict[str, Any]], Awaitable[Any]]


class MemoryMiddleware:
    """Middleware that auto-injects relevant memories and records interactions.

    Wiring example::

        mw = MemoryMiddleware(
            find=similarity_view.find,
            record=temporal_view.append,
        )
        llm = LLM("claude-sonnet-4-20250514", middleware=[mw])
    """

    __slots__ = ("_find", "_header", "_k", "_record")

    def __init__(
        self,
        find: FindFn,
        record: RecordFn,
        *,
        k: int = 3,
        header: str = "Relevant memories:",
    ) -> None:
        self._find = find
        self._record = record
        self._k = k
        self._header = header

    def before(self, request: Request) -> Request:
        """Sync no-op (protocol conformance)."""
        return request

    def after(self, request: Request, response: Response) -> Response:
        """Sync no-op (protocol conformance)."""
        return response

    async def abefore(self, request: Request) -> Request:
        """Inject relevant memories into the system prompt."""
        query = _extract_query(request)
        if not query:
            return request
        results = await self._find(query, k=self._k)
        if not results:
            return request
        memory_text = self._header + "\n"
        for r in results:
            text_parts = [str(v) for v in r.node.content.values() if isinstance(v, str)]
            text = " ".join(text_parts)
            if text:
                memory_text += f"- {text}\n"
        # Prepend to system prompt
        system = request.system or ""
        system = memory_text + "\n" + system if system else memory_text.rstrip()
        return Request(
            messages=request.messages,
            system=system,
            tools=request.tools,
            model=request.model,
            kwargs=request.kwargs,
        )

    async def aafter(self, request: Request, response: Response) -> Response:
        """Record the interaction as a memory node."""
        query = _extract_query(request)
        summary = response.text[:200] if response.text else ""
        if query or summary:
            await self._record(
                {
                    "query": query,
                    "response_summary": summary,
                }
            )
        return response


def _extract_query(request: Request) -> str:
    """Extract the latest user message text as a search query."""
    for msg in reversed(request.messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return " ".join(parts)
    return ""
