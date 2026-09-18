"""The same logical answer, received whole or streamed in chunks, gives the same Response.

Parity holds by construction (one assembly from the SDK's final object); these tests hold each
adapter to it with the SDK's own types, so a stream accumulator that drops a field shows here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from anthropic import types as claude
from google.genai import types as gemini
from openai.types.chat import ChatCompletion
from openai.types.responses import Response as MetaResponse
from xai_sdk.proto import sample_pb2

from ai_arch_toolkit.core import OutputSchema, Response
from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._gemini import GeminiProvider
from ai_arch_toolkit.core._providers._meta import MetaProvider
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from tests.integration import fakegrpc
from tests.provider_calls import complete, stream
from tests.sdk_streams import OpenAIStream, openai_chunk, openai_usage_chunk

HI = [{"role": "user", "content": "Weather and time in Lisbon?"}]


def _same(completed: Response, streamed: Response) -> None:
    fields = (
        "text",
        "tool_calls",
        "thinking",
        "parsed",
        "citations",
        "response_id",
        "usage",
        "cost",
        "stop_reason",
        "model",
    )
    assert {f: getattr(streamed, f) for f in fields} == {f: getattr(completed, f) for f in fields}
    replay = completed.to_message()
    streamed_replay = streamed.to_message()
    replay.pop("_raw")
    streamed_replay.pop("_raw")
    assert streamed_replay == replay  # what the history sends back


def _openai(completion: ChatCompletion, chunks: list[Any]) -> OpenAIProvider:
    client = AsyncMock()

    async def create(**kwargs: Any) -> Any:
        return OpenAIStream(chunks) if kwargs.get("stream") else completion

    client.chat.completions.create = create
    provider = OpenAIProvider("gpt-4o", "test-key")
    provider._client = client
    return provider


def _completion(message: dict[str, Any], finish: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        }
    )


async def test_openai_text_and_parallel_calls() -> None:
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "get_time", "arguments": '{"tz": "UTC"}'},
        },
    ]
    completion = _completion(
        {"role": "assistant", "content": "Checking both.", "tool_calls": calls}, "tool_calls"
    )
    chunks = [
        openai_chunk(content="Checking "),
        openai_chunk(content="both."),
        openai_chunk(tool_calls=[{"index": 0, "id": "call_1", "name": "get_weather"}]),
        openai_chunk(tool_calls=[{"index": 0, "arguments": '{"city": "Lisbon"}'}]),
        openai_chunk(tool_calls=[{"index": 1, "id": "call_2", "name": "get_time"}]),
        openai_chunk(tool_calls=[{"index": 1, "arguments": '{"tz": "UTC"}'}]),
        openai_chunk(finish_reason="tool_calls"),
        openai_usage_chunk(30, 12),
    ]
    provider = _openai(completion, chunks)
    _same(await complete(provider, HI), (await stream(provider, HI))[1])


async def test_openai_structured_output() -> None:
    schema = OutputSchema(name="Answer", schema={"type": "object"})
    completion = _completion({"role": "assistant", "content": '{"answer": 42}'}, "stop")
    chunks = [
        openai_chunk(content='{"answer"'),
        openai_chunk(content=": 42}"),
        openai_chunk(finish_reason="stop"),
        openai_usage_chunk(30, 12),
    ]
    provider = _openai(completion, chunks)
    completed = await complete(provider, HI, output_schema=schema)
    assert completed.parsed == {"answer": 42}
    _same(completed, (await stream(provider, HI, output_schema=schema))[1])


# ---------------------------------------------------------------------------
# xAI: the SDK's sample() and its stream accumulator, over a loopback gRPC server
# ---------------------------------------------------------------------------

CALLS = [("call_1", "get_weather", '{"city": "Lisbon"}'), ("call_2", "get_time", '{"tz": "UTC"}')]
TOOL_CALLS = sample_pb2.FinishReason.REASON_TOOL_CALLS


async def _xai(script: fakegrpc.Script, **kwargs: Any) -> tuple[Response, Response]:
    async with fakegrpc.serving(script) as (_, port):
        provider = await fakegrpc.provider("grok-4.6", port)
        completed = await complete(provider, HI, **kwargs)
        streamed = (await stream(provider, HI, **kwargs))[1]
        await provider.close()
    return completed, streamed


async def test_xai_reasoning_text_and_parallel_calls() -> None:
    script = fakegrpc.Script(
        response=fakegrpc.answer(
            "Checking both.", reasoning="Two lookups.", calls=CALLS, finish=TOOL_CALLS
        ),
        chunks=[
            fakegrpc.chunk(reasoning="Two "),
            fakegrpc.chunk(reasoning="lookups."),
            fakegrpc.chunk("Checking "),
            fakegrpc.chunk("both."),
            fakegrpc.chunk(calls=CALLS[:1]),
            fakegrpc.chunk(calls=CALLS[1:]),
            fakegrpc.chunk(finish=TOOL_CALLS, usage=fakegrpc.USAGE),
        ],
    )
    completed, streamed = await _xai(script)
    assert [call.id for call in completed.tool_calls] == ["call_1", "call_2"]
    _same(completed, streamed)


async def test_xai_structured_output() -> None:
    schema = OutputSchema(name="Answer", schema={"type": "object"})
    script = fakegrpc.Script(
        response=fakegrpc.answer('{"answer": 42}'),
        chunks=[
            fakegrpc.chunk('{"answer"'),
            fakegrpc.chunk(
                ": 42}", finish=sample_pb2.FinishReason.REASON_STOP, usage=fakegrpc.USAGE
            ),
        ],
    )
    completed, streamed = await _xai(script, output_schema=schema)
    assert completed.parsed == {"answer": 42}
    _same(completed, streamed)


# ---------------------------------------------------------------------------
# Gemini: the SDK's response types; the adapter joins the stream's chunks itself
# ---------------------------------------------------------------------------


def _gemini(response: gemini.GenerateContentResponse, chunks: list[Any]) -> GeminiProvider:
    async def chunked() -> Any:
        for chunk in chunks:
            yield chunk

    async def generate_content_stream(**kwargs: Any) -> Any:
        return chunked()

    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    client.aio.models.generate_content_stream = generate_content_stream
    provider = GeminiProvider("gemini-3.8-flash", "test-key")
    provider._client = client
    return provider


def _gemini_answer(*parts: gemini.Part, **fields: Any) -> gemini.GenerateContentResponse:
    return gemini.GenerateContentResponse(
        candidates=[
            gemini.Candidate(
                content=gemini.Content(role="model", parts=list(parts)),
                finish_reason=fields.get("finish"),
            )
        ],
        usage_metadata=fields.get("usage"),
        model_version="gemini-3.8-flash",
        response_id="resp-1",
    )


GEMINI_USAGE = gemini.GenerateContentResponseUsageMetadata(
    prompt_token_count=30, candidates_token_count=12, thoughts_token_count=5
)
THOUGHT = gemini.Part(text="Two lookups.", thought=True)
WEATHER_CALL = gemini.Part(
    function_call=gemini.FunctionCall(id="fc-1", name="get_weather", args={"city": "Lisbon"}),
    thought_signature=b"sig-1",
)
TIME_CALL = gemini.Part(
    function_call=gemini.FunctionCall(id="fc-2", name="get_time", args={"tz": "UTC"})
)


async def test_gemini_thinking_text_and_parallel_calls() -> None:
    text = gemini.Part(text="Checking both.")
    whole = _gemini_answer(
        THOUGHT, text, WEATHER_CALL, TIME_CALL, finish="STOP", usage=GEMINI_USAGE
    )
    chunks = [
        _gemini_answer(THOUGHT),
        _gemini_answer(gemini.Part(text="Checking ")),
        _gemini_answer(gemini.Part(text="both.")),
        _gemini_answer(WEATHER_CALL),
        _gemini_answer(TIME_CALL, finish="STOP", usage=GEMINI_USAGE),
    ]
    provider = _gemini(whole, chunks)
    completed = await complete(provider, HI)
    assert [call.id for call in completed.tool_calls] == ["fc-1", "fc-2"]
    _same(completed, (await stream(provider, HI))[1])


async def test_gemini_structured_output() -> None:
    schema = OutputSchema(name="Answer", schema={"type": "object"})
    whole = _gemini_answer(gemini.Part(text='{"answer": 42}'), finish="STOP", usage=GEMINI_USAGE)
    chunks = [
        _gemini_answer(gemini.Part(text='{"answer"')),
        _gemini_answer(gemini.Part(text=": 42}"), finish="STOP", usage=GEMINI_USAGE),
    ]
    provider = _gemini(whole, chunks)
    completed = await complete(provider, HI, output_schema=schema)
    assert completed.parsed == {"answer": 42}
    _same(completed, (await stream(provider, HI, output_schema=schema))[1])


# ---------------------------------------------------------------------------
# Meta: the Responses API's terminal event carries the whole response
# ---------------------------------------------------------------------------


def _meta_response(*output: dict[str, Any]) -> MetaResponse:
    return MetaResponse.construct(
        id="resp_1",
        object="response",
        created_at=0.0,
        model="muse-spark-1.3",
        status="completed",
        output=list(output),
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        usage={
            "input_tokens": 30,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 12,
            "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 42,
        },
    )


def _meta_message(text: str, item_id: str, phase: str | None = None) -> dict[str, Any]:
    content = [{"type": "output_text", "text": text, "annotations": []}]
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": phase,
        "content": content,
    }


def _meta(final: MetaResponse, events: list[dict[str, Any]]) -> MetaProvider:
    from tests.sdk_streams import OpenAIStream

    async def create(**kwargs: Any) -> Any:
        if not kwargs.get("stream"):
            return final
        stream = [*events, {"type": "response.completed", "response": final}]
        return OpenAIStream([SimpleNamespace(**event) for event in stream])

    client = MagicMock()
    client.responses.create = create
    provider = MetaProvider("muse-spark-1.3", "test-key")
    provider._client = client
    return provider


async def test_meta_reasoning_commentary_and_parallel_calls() -> None:
    final = _meta_response(
        {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Two lookups."}],
            "encrypted_content": "enc",
        },
        _meta_message("Checking both.", "m1", phase="commentary"),
        {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "Lisbon"}',
        },
        {
            "id": "fc_2",
            "type": "function_call",
            "call_id": "call_2",
            "name": "get_time",
            "arguments": '{"tz": "UTC"}',
        },
    )
    events = [
        {"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": "Two "},
        {"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": "lookups."},
        {"type": "response.output_text.delta", "item_id": "m1", "delta": "Checking "},
        {"type": "response.output_text.delta", "item_id": "m1", "delta": "both."},
    ]
    provider = _meta(final, events)
    completed = await complete(provider, HI)
    assert [call.id for call in completed.tool_calls] == ["call_1", "call_2"]
    _same(completed, (await stream(provider, HI))[1])


async def test_meta_structured_output() -> None:
    schema = OutputSchema(name="Answer", schema={"type": "object"})
    final = _meta_response(_meta_message('{"answer": 42}', "m1"))
    events = [
        {"type": "response.output_text.delta", "item_id": "m1", "delta": '{"answer"'},
        {"type": "response.output_text.delta", "item_id": "m1", "delta": ": 42}"},
    ]
    provider = _meta(final, events)
    completed = await complete(provider, HI, output_schema=schema)
    assert completed.parsed == {"answer": 42}
    _same(completed, (await stream(provider, HI, output_schema=schema))[1])


# ---------------------------------------------------------------------------
# Anthropic: the SDK's own stream accumulator (accumulate_event) over its raw event types
# ---------------------------------------------------------------------------


def _claude(final: claude.Message, events: list[Any]) -> AnthropicProvider:
    from tests.sdk_streams import AnthropicStream

    client = MagicMock()
    client.messages.create = AsyncMock(return_value=final)
    client.messages.stream = MagicMock(side_effect=lambda **_: AnthropicStream(events))
    provider = AnthropicProvider("claude-opus-5", "test-key")
    provider._client = client
    return provider


def _claude_message(content: list[Any], stop: str) -> claude.Message:
    return claude.Message(
        id="msg_test",  # the id anthropic_message_start gives the streamed message
        type="message",
        role="assistant",
        model="claude-opus-5",
        content=content,
        stop_reason=stop,
        usage=claude.Usage(input_tokens=30, output_tokens=12),
    )


async def test_claude_thinking_text_and_parallel_calls() -> None:
    from tests.sdk_streams import (
        anthropic_block_start,
        anthropic_block_stop,
        anthropic_json,
        anthropic_message_end,
        anthropic_message_start,
        anthropic_text,
        anthropic_thinking,
    )

    final = _claude_message(
        [
            claude.ThinkingBlock(type="thinking", thinking="Two lookups.", signature="sig"),
            claude.TextBlock(type="text", text="Checking both."),
            claude.ToolUseBlock(
                type="tool_use", id="call_1", name="get_weather", input={"city": "Lisbon"}
            ),
            claude.ToolUseBlock(
                type="tool_use", id="call_2", name="get_time", input={"tz": "UTC"}
            ),
        ],
        "tool_use",
    )
    signature = claude.RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=claude.SignatureDelta(type="signature_delta", signature="sig"),
    )
    events = [
        anthropic_message_start(model="claude-opus-5", input_tokens=30),
        anthropic_block_start(0, "thinking"),
        anthropic_thinking(0, "Two lookups."),
        signature,
        anthropic_block_stop(0),
        anthropic_block_start(1, "text"),
        anthropic_text(1, "Checking "),
        anthropic_text(1, "both."),
        anthropic_block_stop(1),
        anthropic_block_start(2, "tool_use", id="call_1", name="get_weather"),
        anthropic_json(2, '{"city": "Lisbon"}'),
        anthropic_block_stop(2),
        anthropic_block_start(3, "tool_use", id="call_2", name="get_time"),
        anthropic_json(3, '{"tz": "UTC"}'),
        anthropic_block_stop(3),
        *anthropic_message_end("tool_use", output_tokens=12),
    ]
    provider = _claude(final, events)
    completed = await complete(provider, HI, max_tokens=1024)
    assert [call.id for call in completed.tool_calls] == ["call_1", "call_2"]
    _same(completed, (await stream(provider, HI, max_tokens=1024))[1])
