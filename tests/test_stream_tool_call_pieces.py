"""The pieces of a tool call as the model writes it: ``tool_call_delta`` events (G-37).

Every adapter is fed its SDK's own stream types. Each call's pieces carry its place among the
answer's calls, its id and its name; joined, their input is the finished call's, which still ends
the stream as one ``tool_call`` event.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic import types as anthropic_types
from google.genai import types as genai_types
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
)
from xai_sdk.proto import sample_pb2

from ai_arch_toolkit.core import (
    LLM,
    Result,
    RetryConfig,
    State,
    StateSnapshot,
    Step,
    StreamEvent,
    ToolCall,
    ToolCallDelta,
    llm_events_to,
)
from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._base import CallPieces
from ai_arch_toolkit.core._providers._gemini import GeminiProvider
from ai_arch_toolkit.core._providers._meta import MetaProvider
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from ai_arch_toolkit.core._providers._openai_compatible import OpenAICompatibleProvider
from ai_arch_toolkit.toolkit.flow import Flow
from tests.integration import fakegrpc
from tests.provider_calls import stream
from tests.sdk_streams import (
    AnthropicStream,
    OpenAIStream,
    anthropic_block_start,
    anthropic_block_stop,
    anthropic_json,
    anthropic_message_end,
    anthropic_message_start,
    anthropic_text,
    openai_chunk,
    openai_usage_chunk,
)

HI = [{"role": "user", "content": "Hi"}]
META = "muse-spark-1.3"
WEATHER = {
    "name": "get_weather",
    "description": "The weather in a city.",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def pieces(events: list[StreamEvent]) -> list[ToolCallDelta]:
    found = [event.tool_call_delta for event in events if event.kind == "tool_call_delta"]
    assert all(delta is not None for delta in found)
    assert all(event.partial for event in events if event.kind == "tool_call_delta")
    return [delta for delta in found if delta is not None]


def finished(events: list[StreamEvent]) -> list[ToolCall]:
    return [event.tool_call for event in events if event.tool_call is not None]


def joined(deltas: list[ToolCallDelta]) -> dict[int, tuple[str, str, Any]]:
    """Each call's identity and its pieces' input, joined and parsed, by its place."""
    calls: dict[int, tuple[str, str, str]] = {}
    for delta in deltas:
        _, _, text = calls.get(delta.index, (delta.id, delta.name, ""))
        calls[delta.index] = (delta.id, delta.name, text + delta.input_json)
    return {
        index: (id_, name, json.loads(text) if text else {})
        for index, (id_, name, text) in calls.items()
    }


def assert_pieces_make_the_calls(events: list[StreamEvent]) -> None:
    """Every call arrived in pieces before it finished, and the pieces are the call."""
    calls = finished(events)
    first_finished = next(i for i, event in enumerate(events) if event.kind == "tool_call")
    assert all(event.kind != "tool_call_delta" for event in events[first_finished:])
    made = joined(pieces(events))
    assert sorted(made) == list(range(len(calls)))
    for index, call in enumerate(calls):
        id_, name, input_ = made[index]
        assert (name, input_) == (call.name, call.input)
        assert id_ in ("", call.id)


# ---------------------------------------------------------------------------
# Anthropic: a tool_use block starts with its id and name, its input comes as input_json_delta
# ---------------------------------------------------------------------------


def _anthropic(events: list[Any]) -> AnthropicProvider:
    provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
    provider._client = MagicMock()
    provider._client.messages.stream.return_value = AnthropicStream(events)
    return provider


TWO_CALLS_AFTER_TEXT = [
    anthropic_message_start(input_tokens=10),
    anthropic_block_start(0, "text"),
    anthropic_text(0, "Checking both."),
    anthropic_block_stop(0),
    anthropic_block_start(1, "tool_use", id="toolu_1", name="get_weather"),
    anthropic_json(1, '{"ci'),
    anthropic_json(1, 'ty": "Lis'),
    anthropic_json(1, 'bon"}'),
    anthropic_block_stop(1),
    anthropic_block_start(2, "tool_use", id="toolu_2", name="get_weather"),
    anthropic_json(2, ""),
    anthropic_json(2, '{"city": "Porto"}'),
    anthropic_block_stop(2),
    *anthropic_message_end("tool_use", output_tokens=30),
]


class TestAnthropic:
    async def test_a_server_tool_takes_no_place_among_the_calls(self):
        server_tool = anthropic_types.RawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=anthropic_types.ServerToolUseBlock(
                type="server_tool_use", id="srvtoolu_1", name="web_search", input={}
            ),
        )
        events, response = await stream(
            _anthropic(
                [
                    anthropic_message_start(),
                    server_tool,
                    anthropic_json(0, '{"query": "weather"}'),
                    anthropic_block_stop(0),
                    anthropic_block_start(1, "tool_use", id="toolu_1", name="get_weather"),
                    anthropic_json(1, '{"city": "Lisbon"}'),
                    anthropic_block_stop(1),
                    *anthropic_message_end("tool_use"),
                ]
            ),
            HI,
            max_tokens=1024,
        )

        assert [(d.index, d.id, d.input_json) for d in pieces(events)] == [
            (0, "toolu_1", ""),
            (0, "toolu_1", '{"city": "Lisbon"}'),
        ]
        assert [call.id for call in response.tool_calls] == ["toolu_1"]

    async def test_a_call_arrives_in_pieces_as_the_model_writes_it(self):
        events, response = await stream(
            _anthropic(TWO_CALLS_AFTER_TEXT), HI, max_tokens=1024, tools=[WEATHER]
        )

        assert [(d.index, d.id, d.name, d.input_json) for d in pieces(events)] == [
            (0, "toolu_1", "get_weather", ""),
            (0, "toolu_1", "get_weather", '{"ci'),
            (0, "toolu_1", "get_weather", 'ty": "Lis'),
            (0, "toolu_1", "get_weather", 'bon"}'),
            (1, "toolu_2", "get_weather", ""),
            (1, "toolu_2", "get_weather", '{"city": "Porto"}'),
        ]
        assert [e.kind for e in events][:2] == ["text", "tool_call_delta"]
        assert_pieces_make_the_calls(events)
        assert finished(events) == list(response.tool_calls)

    async def test_a_call_with_no_input_still_announces_its_name(self):
        events, _ = await stream(
            _anthropic(
                [
                    anthropic_message_start(),
                    anthropic_block_start(0, "tool_use", id="toolu_1", name="get_status"),
                    anthropic_block_stop(0),
                    *anthropic_message_end("tool_use"),
                ]
            ),
            HI,
            max_tokens=1024,
        )

        assert [(d.index, d.name, d.input_json) for d in pieces(events)] == [(0, "get_status", "")]
        assert_pieces_make_the_calls(events)


# ---------------------------------------------------------------------------
# The Responses API (OpenAI's host and Meta): a function_call item, then its arguments' deltas
# ---------------------------------------------------------------------------


_RESPONSES_EVENTS: dict[str, type[Any]] = {
    "response.output_item.added": ResponseOutputItemAddedEvent,
    "response.function_call_arguments.delta": ResponseFunctionCallArgumentsDeltaEvent,
    "response.function_call_arguments.done": ResponseFunctionCallArgumentsDoneEvent,
    "response.completed": ResponseCompletedEvent,
}


def _responses_events(*events: dict[str, Any]) -> list[Any]:
    """The SDK's own event types, each checked against its schema."""
    return [
        _RESPONSES_EVENTS[event["type"]].model_validate({"sequence_number": n, **event})
        for n, event in enumerate(events)
    ]


def _function_call(call_id: str, arguments: str, status: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": "get_weather",
        "arguments": arguments,
        "status": status,
    }


def _added(index: int, call_id: str) -> dict[str, Any]:
    return {
        "type": "response.output_item.added",
        "output_index": index,
        "item": _function_call(call_id, "", "in_progress"),
    }


def _arguments(index: int, call_id: str, delta: str) -> dict[str, Any]:
    return {
        "type": "response.function_call_arguments.delta",
        "output_index": index,
        "item_id": f"fc_{call_id}",
        "delta": delta,
    }


def _completed(*calls: tuple[str, str]) -> dict[str, Any]:
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": []}
    return _finished(reasoning, *(_function_call(id_, args, "completed") for id_, args in calls))


def _message_item(status: str) -> dict[str, Any]:
    text = [{"type": "output_text", "text": "Checking.", "annotations": []}]
    content = text if status == "completed" else []
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": status,
        "content": content,
    }


def _finished(*output: dict[str, Any], model: str = "gpt-6-luna") -> dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": model,
            "status": "completed",
            "output": list(output),
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 20,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 10,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 30,
            },
        },
    }


def _responses_provider(cls: type[Any], model: str, events: list[Any]) -> Any:
    client = AsyncMock()
    client.responses.create = AsyncMock(return_value=OpenAIStream(events))
    provider = cls(model, "test-key")
    provider._client = client
    return provider


class TestResponses:
    async def test_each_function_call_streams_by_its_output_index(self):
        # The reasoning item is output 0: the calls are outputs 1 and 2, and the answer's 0 and 1.
        events = _responses_events(
            _added(1, "call_a"),
            _arguments(1, "call_a", '{"city": '),
            _added(2, "call_b"),
            _arguments(2, "call_b", '{"city": "Porto"}'),
            _arguments(1, "call_a", '"Lisbon"}'),
            _completed(("call_a", '{"city": "Lisbon"}'), ("call_b", '{"city": "Porto"}')),
        )
        provider = _responses_provider(OpenAIProvider, "gpt-6-luna", events)

        streamed, response = await stream(provider, HI, tools=[WEATHER])

        assert [(d.index, d.id, d.input_json) for d in pieces(streamed)] == [
            (0, "call_a", ""),
            (0, "call_a", '{"city": '),
            (1, "call_b", ""),
            (1, "call_b", '{"city": "Porto"}'),
            (0, "call_a", '"Lisbon"}'),
        ]
        assert_pieces_make_the_calls(streamed)
        assert [call.id for call in response.tool_calls] == ["call_a", "call_b"]

    async def test_items_between_the_calls_take_no_place_among_them(self):
        message = {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": _message_item("in_progress"),
        }
        events = _responses_events(
            _added(0, "call_a"),
            _arguments(0, "call_a", '{"city": "Lisbon"}'),
            message,
            _added(2, "call_b"),
            _arguments(2, "call_b", '{"city": "Porto"}'),
            _finished(
                _function_call("call_a", '{"city": "Lisbon"}', "completed"),
                _message_item("completed"),
                _function_call("call_b", '{"city": "Porto"}', "completed"),
            ),
        )
        provider = _responses_provider(OpenAIProvider, "gpt-6-luna", events)

        streamed, _ = await stream(provider, HI, tools=[WEATHER])

        assert [(d.index, d.id) for d in pieces(streamed) if d.input_json] == [
            (0, "call_a"),
            (1, "call_b"),
        ]
        assert_pieces_make_the_calls(streamed)

    async def test_a_server_that_sends_no_deltas_gives_the_arguments_when_done(self):
        # Meta shares the Responses core; nothing shows it streams deltas, so the done event's
        # arguments complete the pieces.
        done = {
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": "fc_call_a",
            "arguments": '{"city": "Lisbon"}',
        }
        events = _responses_events(
            _added(0, "call_a"),
            done,
            _finished(_function_call("call_a", '{"city": "Lisbon"}', "completed"), model=META),
        )
        provider = _responses_provider(MetaProvider, META, events)

        streamed, _ = await stream(provider, HI, tools=[WEATHER])

        assert [d.input_json for d in pieces(streamed)] == ["", '{"city": "Lisbon"}']
        assert_pieces_make_the_calls(streamed)

    async def test_the_done_event_adds_nothing_the_deltas_wrote(self):
        done = {
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": "fc_call_a",
            "arguments": '{"city": "Lisbon"}',
        }
        events = _responses_events(
            _added(0, "call_a"),
            _arguments(0, "call_a", '{"city": '),
            _arguments(0, "call_a", '"Lisbon"}'),
            done,
            _finished(_function_call("call_a", '{"city": "Lisbon"}', "completed")),
        )
        provider = _responses_provider(OpenAIProvider, "gpt-6-luna", events)

        streamed, _ = await stream(provider, HI, tools=[WEATHER])

        assert [d.input_json for d in pieces(streamed)] == ["", '{"city": ', '"Lisbon"}']


# ---------------------------------------------------------------------------
# Chat Completions (OpenAI-compatible servers): deltas by index, the id and name in the first
# ---------------------------------------------------------------------------


def _compatible(chunks: list[Any]) -> OpenAICompatibleProvider:
    client = AsyncMock()
    client.chat.completions.create.return_value = OpenAIStream(chunks)
    provider = OpenAICompatibleProvider("qwen3", "test-key", base_url="http://localhost:8000/v1")
    provider._client = client
    return provider


class TestChatCompletions:
    async def test_the_pieces_follow_each_calls_index(self):
        events, _ = await stream(
            _compatible(
                [
                    openai_chunk(tool_calls=[{"index": 0, "id": "tc_1", "name": "get_weather"}]),
                    openai_chunk(tool_calls=[{"index": 0, "arguments": '{"city"'}]),
                    openai_chunk(tool_calls=[{"index": 0, "arguments": ""}]),
                    openai_chunk(tool_calls=[{"index": 0, "arguments": ': "Lisbon"}'}]),
                    openai_chunk(
                        tool_calls=[
                            {
                                "index": 1,
                                "id": "tc_2",
                                "name": "get_weather",
                                "arguments": '{"city": "Porto"}',
                            }
                        ],
                        finish_reason="tool_calls",
                    ),
                    openai_usage_chunk(20, 10),
                ]
            ),
            HI,
        )

        assert [(d.index, d.id, d.input_json) for d in pieces(events)] == [
            (0, "tc_1", ""),
            (0, "tc_1", '{"city"'),
            (0, "tc_1", ': "Lisbon"}'),
            (1, "tc_2", '{"city": "Porto"}'),
        ]
        assert_pieces_make_the_calls(events)

    async def test_a_server_that_sends_no_id_gets_one_on_the_finished_call(self):
        events, response = await stream(
            _compatible(
                [
                    openai_chunk(
                        tool_calls=[
                            {"index": 0, "id": "", "name": "get_weather", "arguments": "{}"}
                        ],
                        finish_reason="tool_calls",
                    )
                ]
            ),
            HI,
        )

        assert [d.id for d in pieces(events)] == [""]
        assert response.tool_calls[0].id.startswith("call_")
        assert_pieces_make_the_calls(events)

    async def test_a_call_with_empty_arguments_has_an_empty_input(self):
        events, response = await stream(
            _compatible(
                [
                    openai_chunk(
                        tool_calls=[{"index": 0, "id": "tc_1", "name": "now", "arguments": ""}],
                        finish_reason="tool_calls",
                    )
                ]
            ),
            HI,
        )

        assert response.tool_calls == (ToolCall(id="tc_1", name="now", input={}),)
        assert_pieces_make_the_calls(events)


# ---------------------------------------------------------------------------
# Gemini and xAI send a call whole: one piece, as soon as it arrives
# ---------------------------------------------------------------------------


def _gemini_chunk(*parts: genai_types.Part, finish: str | None = None) -> Any:
    content = genai_types.Content(role="model", parts=list(parts))
    return genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(content=content, finish_reason=finish)]
    )


def _gemini_call(name: str, call_id: str | None = None, **args: Any) -> genai_types.Part:
    return genai_types.Part(
        function_call=genai_types.FunctionCall(id=call_id, name=name, args=args)
    )


def _gemini(*chunks: Any) -> GeminiProvider:
    async def chunks_of(**kwargs: Any) -> Any:
        for chunk in chunks:
            yield chunk

    async def generate_content_stream(**kwargs: Any) -> Any:
        return chunks_of(**kwargs)

    provider = GeminiProvider("gemini-3.8-flash", "test-key")
    provider._client = MagicMock()
    provider._client.aio.models.generate_content_stream = generate_content_stream
    return provider


class TestWholeCalls:
    async def test_gemini_sends_each_call_whole_before_the_stream_ends(self):
        events, response = await stream(
            _gemini(
                _gemini_chunk(genai_types.Part(text="Checking.")),
                _gemini_chunk(_gemini_call("get_weather", city="Lisbon")),
                _gemini_chunk(_gemini_call("get_weather", "fc-2", city="Porto"), finish="STOP"),
            ),
            HI,
        )

        assert [(d.index, d.id, json.loads(d.input_json)) for d in pieces(events)] == [
            (0, "", {"city": "Lisbon"}),
            (1, "fc-2", {"city": "Porto"}),
        ]
        assert [e.kind for e in events] == [
            "text",
            "tool_call_delta",
            "tool_call_delta",
            "tool_call",
            "tool_call",
        ]
        assert_pieces_make_the_calls(events)
        assert response.tool_calls[0].id  # the base names a call Gemini sent without one

    async def test_xai_sends_each_call_whole_in_its_chunk(self):
        script = fakegrpc.Script(
            chunks=[
                fakegrpc.chunk("Checking."),
                fakegrpc.chunk(calls=[("tc_1", "get_weather", '{"city": "Lisbon"}')]),
                fakegrpc.chunk(calls=[("tc_2", "get_weather", '{"city": "Porto"}')]),
                fakegrpc.chunk(
                    finish=sample_pb2.FinishReason.REASON_TOOL_CALLS, usage=fakegrpc.USAGE
                ),
            ]
        )
        async with fakegrpc.serving(script) as (_, port):
            provider = await fakegrpc.provider("grok-4.6", port)
            events, _ = await stream(provider, HI, tools=[WEATHER])
            await provider.close()

        assert [(d.index, d.id, d.input_json) for d in pieces(events)] == [
            (0, "tc_1", '{"city": "Lisbon"}'),
            (1, "tc_2", '{"city": "Porto"}'),
        ]
        assert_pieces_make_the_calls(events)


# ---------------------------------------------------------------------------
# Through the LLM: stream_events, the core's event channel, and a stream left mid-call
# ---------------------------------------------------------------------------


ONE_CALL = [
    anthropic_message_start(input_tokens=10),
    anthropic_block_start(0, "tool_use", id="toolu_1", name="get_weather"),
    anthropic_json(0, '{"city": "Lisbon"}'),
    anthropic_block_stop(0),
    *anthropic_message_end("tool_use", output_tokens=5),
]


def _anthropic_llm(events: list[Any]) -> LLM:
    client = MagicMock()
    client.messages.stream.return_value = AnthropicStream(events)
    client.close = AsyncMock()
    llm = LLM("claude-sonnet-4-6", api_key="test-key")
    llm._provider._client = client  # type: ignore[attr-defined]
    return llm


class TestThroughTheLLM:
    async def test_stream_events_delivers_the_pieces_then_the_call(self):
        async with _anthropic_llm(TWO_CALLS_AFTER_TEXT) as llm:
            stream_ = llm.stream_events("Weather in Lisbon and Porto?", tools=[WEATHER])
            events = [event async for event in stream_]

        assert_pieces_make_the_calls(events)
        assert stream_.response.tool_calls == tuple(finished(events))

    async def test_a_complete_streamed_to_the_channel_carries_the_pieces(self):
        seen: list[StreamEvent] = []
        async with _anthropic_llm(TWO_CALLS_AFTER_TEXT) as llm:
            with llm_events_to(lambda event, call: seen.append(event)):
                response = await llm.complete("Weather?", tools=[WEATHER])

        assert_pieces_make_the_calls(seen)
        assert [call.input["city"] for call in response.tool_calls] == ["Lisbon", "Porto"]

    async def test_a_stream_left_mid_call_has_no_half_written_call(self):
        async with _anthropic_llm(TWO_CALLS_AFTER_TEXT) as llm:
            stream_ = llm.stream_events("Weather?", tools=[WEATHER])
            async for event in stream_:
                if event.kind == "tool_call_delta":
                    break
            await stream_.aclose()

        assert stream_.response.tool_calls == ()
        assert stream_.response.text == "Checking both."

    def test_the_sync_stream_delivers_them_too(self):
        llm = _anthropic_llm(TWO_CALLS_AFTER_TEXT)
        stream_ = llm.stream_events_sync("Weather?", tools=[WEATHER])
        events = list(stream_)

        assert_pieces_make_the_calls(events)

    async def test_an_iterated_flow_carries_them_as_llm_events(self):
        llm = _anthropic_llm(TWO_CALLS_AFTER_TEXT)

        async def ask(snap: StateSnapshot) -> Result:
            response = await llm.complete("Weather?", tools=[WEATHER])
            return Result(value=len(response.tool_calls))

        execution = Flow(Step(name="ask", fn=ask), name="weather").iter(State())
        seen = [
            event.llm_event
            async for event in execution
            if event.type == "llm_event" and event.llm_event is not None
        ]

        assert_pieces_make_the_calls(seen)
        assert execution.result is not None
        step = execution.result.trace.step("ask")
        assert step is not None and step.error is None


class TestRetries:
    """D60: once a piece has reached the consumer, as text does (D54), the call is not retried."""

    @staticmethod
    def _llm() -> tuple[LLM, MagicMock]:
        class FailsAfterThePieces(AnthropicStream):
            async def __aiter__(self) -> Any:
                async for event in super().__aiter__():
                    yield event
                    if event.type == "content_block_delta":
                        raise APIError(529, "overloaded mid-stream")

        client = MagicMock()
        client.messages.stream.side_effect = [
            FailsAfterThePieces(ONE_CALL),
            AnthropicStream(ONE_CALL),
        ]
        client.close = AsyncMock()
        llm = LLM("claude-sonnet-4-6", api_key="test-key", retry=RetryConfig(base_delay=0.001))
        llm._provider._client = client  # type: ignore[attr-defined]
        return llm, client

    async def test_a_stream_that_showed_a_piece_is_not_retried(self):
        llm, client = self._llm()
        seen: list[StreamEvent] = []

        with pytest.raises(APIError), llm_events_to(lambda event, call: seen.append(event)):
            await llm.complete("Weather?", tools=[WEATHER])

        assert client.messages.stream.call_count == 1
        assert [e.kind for e in seen] == ["tool_call_delta", "tool_call_delta"]

    async def test_the_same_failure_unseen_is_retried(self):
        # stream() shows only text: the pieces never reach its consumer.
        llm, client = self._llm()

        stream_ = llm.stream("Weather?", tools=[WEATHER])
        assert [chunk async for chunk in stream_] == []
        response = stream_.response

        assert client.messages.stream.call_count == 2
        assert response.tool_calls[0].input == {"city": "Lisbon"}


def test_pieces_of_an_unknown_call_or_empty_are_dropped():
    calls = CallPieces()
    start = calls.start("a", call_id="c1", name="f")
    assert start.tool_call_delta == ToolCallDelta(index=0, id="c1", name="f")
    assert calls.piece("a", "") is None
    assert calls.piece("b", "{}") is None
    assert "a" in calls and "b" not in calls and len(calls) == 1


def test_finish_gives_what_the_pieces_have_not_written():
    calls = CallPieces()
    calls.start(0, call_id="c1", name="f", input_json='{"a"')
    rest = calls.finish(0, '{"a": 1}')
    assert rest is not None and rest.tool_call_delta is not None
    assert rest.tool_call_delta.input_json == ": 1}"
    assert calls.finish(0, '{"a": 1}') is None  # all written
    assert calls.finish(0, '{"b": 2}') is None  # not what they began
    assert calls.finish(1, "{}") is None
