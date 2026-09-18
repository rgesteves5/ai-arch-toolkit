"""Async middleware runs around streams exactly as around complete()."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from ai_arch_toolkit.core._exceptions import APIError, TransportError
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._admission import (
    AdmissionDecision,
    AdmissionDenied,
    Reservation,
)
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._moderation import ModerationError, ModerationResult
from ai_arch_toolkit.core._providers._base import StreamState
from ai_arch_toolkit.core._response import Response, StreamEvent, Usage
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.core._server_tools import web_search
from ai_arch_toolkit.core._tools import prepare_tools
from ai_arch_toolkit.toolkit.memory import MemoryMiddleware
from ai_arch_toolkit.toolkit.moderation import ModerationMiddleware

_MODEL = "claude-sonnet-4-6"
_USAGE = Usage(input_tokens=30, output_tokens=10)


class _Provider:
    """Streams two chunks; can fail the first attempts before any chunk; records every call."""

    def __init__(self, *, fail_first: int = 0, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail_first = fail_first
        self._error = error or APIError(500, "try again")

    def _stream(self, messages: Any, system: Any, as_events: bool, tools: Any, kwargs: Any):
        self.calls.append(
            {"messages": list(messages), "system": system, "tools": tools, "kwargs": kwargs}
        )
        state = StreamState()
        state.usage = _USAGE
        failing = len(self.calls) <= self._fail_first
        error = self._error

        async def chunks() -> AsyncIterator[Any]:
            if failing:
                raise error
            for text in ("hel", "lo"):
                yield StreamEvent(kind="text", text=text) if as_events else text

        return chunks(), state

    def stream(self, messages, *, system=None, tools=None, **kwargs):
        return self._stream(messages, system, False, tools, kwargs)

    def stream_events(self, messages, *, system=None, tools=None, **kwargs):
        return self._stream(messages, system, True, tools, kwargs)

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.calls.append(
            {"messages": list(messages), "system": system, "tools": tools, "kwargs": kwargs}
        )
        if len(self.calls) <= self._fail_first:
            raise self._error
        return Response(text="hello", usage=_USAGE, model=_MODEL)


class _Spy:
    def __init__(self, inject: str | None = None) -> None:
        self.calls: list[str] = []
        self.after_response: Response | None = None
        self._inject = inject

    def before(self, request: Request) -> Request:
        self.calls.append("before")
        return request

    def after(self, request: Request, response: Response) -> Response:
        self.calls.append("after")
        return response

    async def abefore(self, request: Request) -> Request:
        self.calls.append("abefore")
        if self._inject is None:
            return request
        return Request(
            messages=[*request.messages, {"role": "user", "content": self._inject}],
            system=self._inject,
            tools=request.tools,
            model=request.model,
            kwargs=request.kwargs,
        )

    async def aafter(self, request: Request, response: Response) -> Response:
        self.calls.append("aafter")
        self.after_response = response
        return response


class _AlwaysFlag:
    async def moderate(self, text: str) -> ModerationResult:
        return ModerationResult(flagged=True, categories=["x"], scores={}, explanation="blocked")


def _llm(provider: _Provider, **kwargs: Any) -> LLM:
    llm = LLM(_MODEL, api_key="test", **kwargs)
    llm._provider = provider  # type: ignore[assignment]
    return llm


@pytest.mark.parametrize("method", ["stream", "stream_events"])
async def test_async_hooks_run_once_around_an_async_stream(method: str) -> None:
    spy = _Spy()
    stream = getattr(_llm(_Provider(), middleware=[spy]), method)("hi")

    async for _ in stream:
        pass

    assert spy.calls == ["abefore", "aafter"]
    assert spy.after_response is not None and spy.after_response.usage == _USAGE
    assert stream.response is spy.after_response


@pytest.mark.parametrize("method", ["stream_sync", "stream_events_sync"])
def test_async_hooks_run_once_around_a_sync_stream(method: str) -> None:
    spy = _Spy()
    stream = getattr(_llm(_Provider(), middleware=[spy]), method)("hi")

    list(stream)

    assert spy.calls == ["abefore", "aafter"]
    assert stream.response is spy.after_response


async def test_input_moderation_blocks_a_stream_before_the_provider_is_called() -> None:
    provider = _Provider()
    llm = _llm(provider, middleware=[ModerationMiddleware(input=_AlwaysFlag())])

    with MeterScope() as scope:
        stream = llm.stream_events("bad input")
        with pytest.raises(ModerationError):
            await stream.__anext__()

    assert provider.calls == []
    snap = scope.snapshot()
    assert snap.llm_calls == 0
    assert snap.out_llm_calls == 0
    assert snap.unknown_cost_count == 0


async def test_output_moderation_runs_once_the_stream_is_consumed() -> None:
    llm = _llm(_Provider(), middleware=[ModerationMiddleware(output=_AlwaysFlag())])
    received: list[str] = []

    with pytest.raises(ModerationError):
        async for chunk in llm.stream("hi"):
            received.append(chunk)

    assert received == ["hel", "lo"]  # already delivered: output screening cannot un-show text


async def test_memory_middleware_injects_and_records_around_a_stream() -> None:
    queries: list[str] = []
    recorded: list[dict[str, Any]] = []

    async def find(query: str, k: int = 3) -> list[Any]:
        queries.append(query)
        return []

    async def record(item: dict[str, Any]) -> None:
        recorded.append(item)

    llm = _llm(_Provider(), middleware=[MemoryMiddleware(find=find, record=record)])

    async for _ in llm.stream("remember this"):
        pass

    assert queries == ["remember this"]
    assert recorded == [{"query": "remember this", "response_summary": "hello"}]


async def test_parent_middleware_wraps_a_fallback_stream_once() -> None:
    spy = _Spy(inject="INJECTED")
    failing = _Provider(fail_first=1, error=TransportError("down"))
    fallback_provider = _Provider()
    fallback = _llm(fallback_provider)
    llm = _llm(failing, middleware=[spy], fallback=fallback)

    stream = llm.stream("hi")
    received = [chunk async for chunk in stream]

    assert received == ["hel", "lo"]
    assert spy.calls == ["abefore", "aafter"]
    sent = fallback_provider.calls[0]
    assert sent["messages"][-1]["content"] == "INJECTED"
    assert sent["system"] == "INJECTED"


async def test_a_retry_before_the_first_chunk_does_not_rerun_abefore() -> None:
    spy = _Spy()
    provider = _Provider(fail_first=1)
    llm = _llm(provider, middleware=[spy], retry=RetryConfig(max_retries=1, base_delay=0.001))

    received = [chunk async for chunk in llm.stream("hi")]

    assert received == ["hel", "lo"]
    assert len(provider.calls) == 2
    assert spy.calls == ["abefore", "aafter"]


async def test_an_abandoned_stream_does_not_run_aafter() -> None:
    spy = _Spy()
    stream = _llm(_Provider(), middleware=[spy]).stream("hi")

    await stream.__anext__()
    await stream.aclose()

    assert spy.calls == ["abefore"]


async def test_complete_fallback_receives_the_messages_after_middleware() -> None:
    spy = _Spy(inject="INJECTED")
    fallback_provider = _Provider()
    llm = _llm(
        _Provider(fail_first=1),
        middleware=[spy],
        fallback=_llm(fallback_provider),
    )

    response = await llm.complete("hi")

    assert response.text == "hello"
    sent = fallback_provider.calls[0]
    assert sent["messages"][-1]["content"] == "INJECTED"
    assert sent["system"] == "INJECTED"


async def test_a_stream_that_is_never_iterated_releases_its_reservation() -> None:
    llm = _llm(_Provider())

    with MeterScope() as scope:
        llm.stream("hi")
        assert scope.snapshot().out_llm_calls == 1  # reserved at creation (admission stays eager)

    snap = scope.snapshot()
    assert snap.llm_calls == 0
    assert snap.out_llm_calls == 0
    assert snap.unknown_cost_count == 0


_ORIGINAL_TOOL = {"name": "original_tool", "input_schema": {"type": "object", "properties": {}}}
_ONLY_TOOL = {"name": "only_tool", "input_schema": {"type": "object", "properties": {}}}


class _Rewrite:
    """abefore rewrites tools and kwargs, as a guard that swaps tools or caps output would."""

    def __init__(
        self, *, tools: list[dict[str, Any]], set_kwargs: dict[str, Any], drop: str
    ) -> None:
        self._tools = tools
        self._set = set_kwargs
        self._drop = drop

    async def abefore(self, request: Request) -> Request:
        kwargs = {k: v for k, v in request.kwargs.items() if k != self._drop} | self._set
        return Request(
            messages=request.messages,
            system=request.system,
            tools=self._tools,
            model=request.model,
            kwargs=kwargs,
        )

    async def aafter(self, request: Request, response: Response) -> Response:
        return response


@pytest.mark.parametrize("method", ["complete", "stream"])
async def test_a_fallback_receives_the_tools_and_kwargs_after_middleware(method: str) -> None:
    rewrite = _Rewrite(
        tools=[_ONLY_TOOL, *prepare_tools([web_search()])],
        set_kwargs={"max_tokens": 7, "json_mode": True},
        drop="temperature",
    )
    fallback_provider = _Provider()
    llm = _llm(
        _Provider(fail_first=1, error=TransportError("down")),
        middleware=[rewrite],
        fallback=_llm(fallback_provider),
    )
    call = {"tools": [_ORIGINAL_TOOL], "max_tokens": 4096, "temperature": 0.3, "top_p": 0.9}

    if method == "complete":
        await llm.complete("hi", **call)
    else:
        async for _ in llm.stream("hi", **call):
            pass

    sent = fallback_provider.calls[0]
    assert [tool.get("name", tool.get("type")) for tool in sent["tools"]] == [
        "only_tool",
        "web_search",
    ]
    assert sent["kwargs"]["max_tokens"] == 7
    assert sent["kwargs"]["json_mode"] is True
    assert sent["kwargs"]["top_p"] == 0.9
    assert sent["kwargs"].get("temperature") != 0.3  # dropped by middleware; the fallback's own


class _Admissions:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def admit(self, snapshot: Any, request: Any) -> AdmissionDecision:
        self.requests.append(request)
        return AdmissionDecision(admitted=True, reservation=Reservation(), limits=None)

    def wants_request_size(self) -> bool:
        return True


async def test_a_stream_is_admitted_and_priced_on_the_request_after_middleware() -> None:
    rewrite = _Rewrite(
        tools=prepare_tools([web_search()]) or [], set_kwargs={"max_tokens": 50}, drop=""
    )
    admissions = _Admissions()
    llm = _llm(_Provider(), middleware=[rewrite])

    with MeterScope(RunConfig(controller=admissions, retain_meter_events=True)) as scope:
        async for _ in llm.stream("hi", max_tokens=4096):
            pass

    admitted = admissions.requests[-1]
    assert admitted.declared_max_output_tokens == 50
    assert admitted.has_server_tools
    settled = [event for event in scope.events() if event.status == "settled"]
    assert len(settled) == 1
    assert settled[0].cost.kind == "unknown"  # a server tool's cost cannot be metered
    assert scope.snapshot().llm_calls == 1


async def test_a_stream_rejected_by_middleware_records_no_attempt() -> None:
    provider = _Provider()
    llm = _llm(provider, middleware=[ModerationMiddleware(input=_AlwaysFlag())])

    with MeterScope() as scope:
        stream = llm.stream("bad input")
        with pytest.raises(ModerationError):
            await stream.__anext__()
        await stream.aclose()

    assert provider.calls == []
    assert stream.response is not None
    assert stream.response.attempts == ()
    assert scope.snapshot().llm_calls == 0


class _CapOutputTokens:
    """Admits requests declaring at most ``cap`` output tokens."""

    def __init__(self, cap: int) -> None:
        self._cap = cap

    def admit(self, snapshot: Any, request: Any) -> AdmissionDecision:
        declared = request.declared_max_output_tokens or 0
        if declared > self._cap:
            denial = AdmissionDenied(dimension="max_tokens", limit=self._cap, attempted=declared)
            return AdmissionDecision.deny(denial)
        return AdmissionDecision.allow()


async def test_a_stream_whose_rewritten_request_is_denied_never_calls_the_provider() -> None:
    rewrite = _Rewrite(tools=[], set_kwargs={"max_tokens": 5_000}, drop="")
    provider = _Provider()
    llm = _llm(provider, middleware=[rewrite])

    with MeterScope(RunConfig(controller=_CapOutputTokens(cap=1_000))) as scope:
        stream = llm.stream("hi", max_tokens=100)  # admitted as created
        with pytest.raises(AdmissionDenied):
            await stream.__anext__()  # middleware raised max_tokens past the cap
        await stream.aclose()

    assert provider.calls == []
    assert stream.response is not None and stream.response.attempts == ()
    snap = scope.snapshot()
    assert snap.llm_calls == 0 and snap.out_llm_calls == 0
