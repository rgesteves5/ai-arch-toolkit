"""The provider contract (costura B): three phases, one assembly, one error mapper."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from ai_arch_toolkit.core import (
    LLM,
    APIError,
    MeterScope,
    RequestError,
    Response,
    RunConfig,
    StreamEvent,
    ToolCall,
    Usage,
)
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._providers._base import Answer, BaseProvider, Done

MODEL = "claude-sonnet-4-6"


class SDKError(Exception):
    """Stands in for an SDK's own exception class."""


@dataclass(frozen=True, slots=True)
class Final:
    """Stands in for an SDK's final object."""

    text: str
    calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = field(default_factory=lambda: Usage(input_tokens=3, output_tokens=2))


class MinimalProvider(BaseProvider[Request, Final]):
    """The smallest provider that keeps the contract; chunks are the final text in two halves."""

    def __init__(self, final: Final, *, fail: BaseException | None = None, at: int = 0) -> None:
        self._model = MODEL
        self.final = final
        self.fail, self.at = fail, at
        self.prepared: list[Request] = []
        self.sent = 0

    def prepare(self, request: Request) -> Request:
        if request.kwargs.get("bad"):
            raise RequestError("the model cannot take this request")
        self.prepared.append(request)
        return request

    async def send(self, prepared: Request) -> Final:
        self.sent += 1
        if self.fail is not None:
            raise self.fail
        return self.final

    async def open_stream(self, prepared: Request) -> AsyncIterator[StreamEvent | Done[Final]]:
        self.sent += 1
        half = len(self.final.text) // 2
        for index, chunk in enumerate((self.final.text[:half], self.final.text[half:])):
            if self.fail is not None and index == self.at:
                raise self.fail
            yield StreamEvent(kind="text", text=chunk)
        yield Done(self.final)

    def assemble(self, final: Final, prepared: Request) -> Response:
        return Response(text=final.text, tool_calls=final.calls, model=MODEL, response_id="r1")

    def usage(self, final: Final) -> Usage | None:
        return final.usage

    def map_error(self, exc: Exception, *, sent: bool) -> APIError | None:
        return APIError(502, str(exc)) if isinstance(exc, SDKError) else None


def request(**kwargs: object) -> Request:
    return Request(
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=None,
        model=MODEL,
        kwargs=dict(kwargs),
    )


async def drain(provider: MinimalProvider) -> tuple[list[StreamEvent], Answer]:
    events: list[StreamEvent] = []
    answer: Answer | None = None
    async for item in provider.stream(provider.prepare(request())):
        if isinstance(item, Answer):
            answer = item
        else:
            events.append(item)
    assert answer is not None
    return events, answer


async def test_complete_and_stream_assemble_the_same_response() -> None:
    call = ToolCall(id="c1", name="lookup", input={"q": "x"})
    provider = MinimalProvider(Final("hello world", calls=(call,)))
    completed = await provider.complete(provider.prepare(request()))
    events, streamed = await drain(provider)
    assert streamed == completed
    assert completed.response.text == "hello world"
    assert completed.response.usage == Usage(input_tokens=3, output_tokens=2)
    assert completed.response.cost is not None and completed.usage_reported
    # Tool-call events come from the assembled response, after the text.
    assert [e.kind for e in events] == ["text", "text", "tool_call"]
    assert events[-1].tool_call == call


async def test_a_final_without_usage_has_no_cost() -> None:
    answer = await MinimalProvider(Final("hi", usage=None)).complete(request())
    assert not answer.usage_reported
    assert answer.response.usage == Usage()
    assert answer.response.cost is None


@pytest.mark.parametrize("at", [0, 1])
async def test_sdk_errors_leave_through_the_mapper_before_and_during_a_stream(at: int) -> None:
    provider = MinimalProvider(Final("hello world"), fail=SDKError("boom"), at=at)
    with pytest.raises(APIError) as raised:
        await drain(provider)
    assert raised.value.status_code == 502
    assert isinstance(raised.value.__cause__, SDKError)
    with pytest.raises(APIError):
        await provider.complete(request())


async def test_errors_the_mapper_does_not_know_pass_unchanged() -> None:
    provider = MinimalProvider(Final("hello world"), fail=KeyError("bug"))
    with pytest.raises(KeyError):
        await provider.complete(request())


def configured(provider: MinimalProvider, *, middleware: list[object] | None = None) -> LLM:
    llm = LLM(MODEL, api_key="test", middleware=middleware)
    llm._provider = provider
    return llm


@pytest.mark.parametrize("path", ["complete", "stream", "stream_events"])
async def test_a_request_the_adapter_refuses_never_opens_an_operation(path: str) -> None:
    provider = MinimalProvider(Final("ok"))
    llm = configured(provider)
    with MeterScope(RunConfig(retain_meter_events=True)) as scope:
        with pytest.raises(RequestError):
            if path == "complete":
                await llm.complete("hi", bad=True)
            else:
                getattr(llm, path)("hi", bad=True)
        assert scope.snapshot().out_llm_calls == 0
    assert provider.sent == 0
    assert scope.snapshot().llm_calls == 0
    assert scope.events() == ()


class Breaker:
    """Middleware that turns a good request into one the adapter refuses."""

    def before(self, request: Request) -> Request:
        return Request(
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            model=request.model,
            kwargs={**request.kwargs, "bad": True},
        )

    def after(self, request: Request, response: Response) -> Response:
        return response


async def test_a_middleware_rewrite_the_adapter_refuses_releases_the_stream_reservation() -> None:
    provider = MinimalProvider(Final("ok"))
    llm = configured(provider, middleware=[Breaker()])
    with MeterScope(RunConfig(retain_meter_events=True)) as scope:
        stream = llm.stream("hi")
        assert scope.snapshot().out_llm_calls == 1  # reserved at creation (D8)
        with pytest.raises(RequestError):
            async for _ in stream:
                pass
        assert scope.snapshot().out_llm_calls == 0
    assert provider.sent == 0
    assert scope.snapshot().llm_calls == 0
    assert [event.status for event in scope.events()] == ["aborted"]


@pytest.mark.parametrize("path", ["complete", "stream", "stream_events"])
async def test_a_response_without_usage_settles_an_unknown_cost(path: str) -> None:
    llm = configured(MinimalProvider(Final("hello", usage=None)))
    with MeterScope() as scope:
        if path == "complete":
            response = await llm.complete("hi")
        else:
            stream = getattr(llm, path)("hi")
            async for _ in stream:
                pass
            response = stream.response
    assert response.text == "hello"
    assert response.cost is None
    assert scope.snapshot().llm_calls == 1
    assert scope.snapshot().unknown_cost_count == 1


@pytest.mark.parametrize("path", ["stream", "stream_events"])
async def test_streams_finalize_with_the_assembled_response(path: str) -> None:
    call = ToolCall(id="c1", name="lookup", input={})
    llm = configured(MinimalProvider(Final("hello world", calls=(call,))))
    stream = getattr(llm, path)("hi")
    items = [item async for item in stream]
    if path == "stream":
        assert items == ["hello", " world"]
    else:
        assert [event.kind for event in items] == ["text", "text", "tool_call"]
    assert stream.response is not None
    assert stream.response.response_id == "r1"  # assembled, not rebuilt from chunks
    assert stream.response.tool_calls == (call,)
