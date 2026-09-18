"""The suite's one provider double: it keeps the provider contract (R02, costura B).

A real ``LLM`` with a ``FakeProvider`` runs the whole call pipeline — preparation, admission,
metering, retry, fallback, middleware — and only the provider's I/O is scripted.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.core import LLM, RequestError, Response, StreamEvent, Usage
from ai_arch_toolkit.core._exceptions import ProviderError
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._providers._base import BaseProvider, Done

MODEL = "claude-sonnet-4-6"  # priced in the default table
USAGE = Usage(input_tokens=10, output_tokens=5)


def _ok() -> Response:
    return Response(text="ok", usage=USAGE, model=MODEL)


@dataclass(frozen=True, slots=True, kw_only=True)
class Reply:
    """One scripted answer.

    A call returns ``response``. A stream sends ``chunks`` as text events (by default the
    response's text in one chunk), then the response. ``error`` is raised instead of answering:
    before anything, or after ``error_after`` chunks. ``hold`` makes the call wait for an event
    first (after ``hold_after`` chunks in a stream), to line up cancellations. ``usage_reported``
    is ``False`` for a provider that sends no usage.
    """

    response: Response = field(default_factory=_ok)
    chunks: Sequence[str] | None = None
    error: BaseException | None = None
    error_after: int | None = None
    hold: asyncio.Event | None = None
    hold_after: int = 0
    delay: float = 0.0
    usage_reported: bool = True

    def stream_chunks(self) -> Sequence[str]:
        if self.chunks is not None:
            return self.chunks
        return [self.response.text] if self.response.text else []


type Script = Reply | Response | BaseException


def _reply(item: Script) -> Reply:
    if isinstance(item, Reply):
        return item
    if isinstance(item, Response):
        return Reply(response=item)
    return Reply(error=item)


class FakeProvider(BaseProvider[Request, Reply]):
    """Answers from a script — one item per call, the last one repeating — and records calls.

    ``refuse`` lists ``RequestError``s that the next preparations raise, in order.
    """

    def __init__(
        self,
        *script: Script,
        model: str = MODEL,
        refuse: Sequence[RequestError] = (),
    ) -> None:
        self._model = model
        self.script = [_reply(item) for item in script] or [Reply()]
        self.refusals = list(refuse)
        self.requests: list[Request] = []
        self.entered = asyncio.Event()
        self.live = 0
        self.peak = 0
        self.closed = False
        self.batches: list[list[dict[str, Any]]] = []

    @property
    def calls(self) -> int:
        """How many requests were sent."""
        return len(self.requests)

    @property
    def last(self) -> Request:
        """The last request sent."""
        return self.requests[-1]

    def prepare(self, request: Request) -> Request:
        if self.refusals:
            raise self.refusals.pop(0)
        return request

    async def _begin(self, request: Request) -> Reply:
        reply = self.script[min(len(self.requests), len(self.script) - 1)]
        self.requests.append(request)
        self.entered.set()
        self.live += 1
        self.peak = max(self.peak, self.live)
        if reply.delay:
            await asyncio.sleep(reply.delay)
        return reply

    async def send(self, prepared: Request) -> Reply:
        reply = await self._begin(prepared)
        try:
            if reply.hold is not None:
                await reply.hold.wait()
            if reply.error is not None:
                raise reply.error
            return reply
        finally:
            self.live -= 1

    async def open_stream(self, prepared: Request) -> AsyncIterator[StreamEvent | Done[Reply]]:
        reply = await self._begin(prepared)
        try:
            chunks = reply.stream_chunks()
            for index in range(len(chunks) + 1):
                if reply.hold is not None and index == reply.hold_after:
                    await reply.hold.wait()
                if reply.error is not None and index == (reply.error_after or 0):
                    raise reply.error
                if index < len(chunks):
                    yield StreamEvent(kind="text", text=chunks[index])
            yield Done(reply)
        finally:
            self.live -= 1

    def assemble(self, final: Reply, prepared: Request) -> Response:
        return final.response

    def usage(self, final: Reply) -> Usage | None:
        return final.response.usage if final.usage_reported else None

    def map_error(self, exc: Exception, *, sent: bool) -> ProviderError | None:
        return None  # the script raises normalized errors itself

    async def batch_submit(self, requests: list[dict[str, Any]], **kwargs: Any) -> str:
        self.batches.append(requests)
        return f"batch-{len(self.batches)}"

    async def close(self) -> None:
        self.closed = True


def fake_llm(
    *script: Script,
    model: str = MODEL,
    **llm_kwargs: Any,
) -> tuple[LLM, FakeProvider]:
    """A real ``LLM`` whose provider answers from ``script``."""
    llm = LLM(model, api_key="test", **llm_kwargs)
    provider = FakeProvider(*script, model=model)
    llm._provider = provider
    return llm, provider
