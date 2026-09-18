"""Transport failures through each adapter and its real SDK, against a loopback server only.

Each case must leave as the right typed error with the right delivery, the fact the meter's
disposition (R01) and the retry and fallback rules read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import grpc
import pytest
from google import genai

from ai_arch_toolkit.core._exceptions import (
    APIError,
    Delivery,
    ProviderError,
    ProviderTimeout,
    RateLimitError,
    RequestError,
    ResponseError,
    TransportError,
)
from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._gemini import GeminiProvider, _http_options
from ai_arch_toolkit.core._providers._meta import MetaProvider
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from tests.integration import fakegrpc, fakeserver
from tests.provider_calls import complete, stream

pytestmark = pytest.mark.integration

HI = [{"role": "user", "content": "hi"}]


@dataclass(frozen=True, slots=True)
class Outcome:
    error: type[ProviderError] | None
    delivery: Delivery | None = None


def _openai_chunk(**delta: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


OPENAI_ERROR = {"error": {"message": "busy", "type": "server_error", "code": None}}
OPENAI_OK = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
OPENAI_EVENTS = [
    fakeserver.sse(_openai_chunk(role="assistant", content="o")),
    fakeserver.sse(_openai_chunk(content="k")),
]


def _openai(port: int, timeout: float = 2.0) -> OpenAIProvider:
    return OpenAIProvider(
        "gpt-4o", "local-test", base_url=f"http://127.0.0.1:{port}/v1", timeout=timeout
    )


async def _outcome(call: Any) -> Outcome:
    try:
        await call
    except ProviderError as error:
        return Outcome(type(error), error.delivery)
    return Outcome(None)


# A case: how the server behaves, and what complete() and a stream must raise.
CASES: dict[str, tuple[dict[str, Any], Outcome, Outcome]] = {
    "429": (
        {"behaviour": "status", "status": 429, "body": OPENAI_ERROR},
        Outcome(RateLimitError, "unbilled"),
        Outcome(RateLimitError, "unbilled"),
    ),
    "500": (
        {"behaviour": "status", "status": 500, "body": OPENAI_ERROR},
        Outcome(APIError, "indeterminate"),
        Outcome(APIError, "indeterminate"),
    ),
    "529": (
        {"behaviour": "status", "status": 529, "body": OPENAI_ERROR},
        Outcome(APIError, "indeterminate"),
        Outcome(APIError, "indeterminate"),
    ),
    "closed without an answer": (
        {"behaviour": "close"},
        Outcome(TransportError, "indeterminate"),
        Outcome(TransportError, "indeterminate"),
    ),
    "reset": (
        {"behaviour": "reset"},
        Outcome(TransportError, "indeterminate"),
        Outcome(TransportError, "indeterminate"),
    ),
    "silent": (
        {"behaviour": "hang", "seconds": 1.0},
        Outcome(ProviderTimeout, "indeterminate"),
        Outcome(ProviderTimeout, "indeterminate"),
    ),
    "200 that is not JSON": (
        {"behaviour": "raw", "raw": b"<html>not json</html>"},
        Outcome(ResponseError, "indeterminate"),
        Outcome(ResponseError, "indeterminate"),  # no event: the stream never finishes
    ),
    "error inside the stream": (
        {
            "behaviour": "sse",
            "events": [*OPENAI_EVENTS, fakeserver.sse(OPENAI_ERROR)],
        },
        Outcome(None),
        Outcome(ResponseError, "indeterminate"),
    ),
    "cut mid-stream": (
        {"behaviour": "sse", "events": OPENAI_EVENTS, "ending": "drop"},
        Outcome(None),
        Outcome(TransportError, "indeterminate"),
    ),
    "success": (
        {"behaviour": "sse", "events": [*OPENAI_EVENTS, fakeserver.sse("[DONE]")]},
        Outcome(None),
        Outcome(None),
    ),
}


@pytest.mark.parametrize("case", CASES)
async def test_openai_transport_failures_are_typed_with_their_delivery(case: str) -> None:
    behaviour, on_complete, on_stream = CASES[case]
    kwargs = dict(behaviour)
    server, port, _ = await fakeserver.start(kwargs.pop("behaviour"), **kwargs)
    async with server:
        provider = _openai(port, timeout=0.3 if case == "silent" else 2.0)
        if case not in ("error inside the stream", "cut mid-stream", "success"):
            assert await _outcome(complete(provider, HI)) == on_complete
        assert await _outcome(stream(provider, HI)) == on_stream
        await provider.close()


async def test_openai_a_refused_connection_was_never_sent() -> None:
    provider = _openai(fakeserver.closed_port())
    assert await _outcome(complete(provider, HI)) == Outcome(TransportError, "not_sent")
    assert await _outcome(stream(provider, HI)) == Outcome(TransportError, "not_sent")
    await provider.close()


async def test_openai_a_request_the_sdk_cannot_serialize_was_never_sent() -> None:
    # The SDK serializes the body before handing it to the transport: a set is not JSON.
    server, port, stats = await fakeserver.start("status", body=OPENAI_OK)
    async with server:
        provider = _openai(port)
        outcome = await _outcome(complete(provider, HI, stop={"never"}))
        await provider.close()
    assert outcome == Outcome(RequestError, "not_sent")
    assert stats.requests == 0


async def test_openai_success_through_the_real_sdk() -> None:
    server, port, _ = await fakeserver.start("status", body=OPENAI_OK)
    async with server:
        provider = _openai(port)
        response = await complete(provider, HI)
        await provider.close()
    assert response.text == "ok"
    assert response.usage.input_tokens == 3


# ---------------------------------------------------------------------------
# xAI: gRPC through the real xai-sdk, against a loopback gRPC server (or a TCP server that does not
# speak it). gRPC has no HTTP status: UNAVAILABLE is a transport failure, DEADLINE_EXCEEDED a
# timeout, and neither says whether the request left, so both stay indeterminate.
# ---------------------------------------------------------------------------

UNBILLED_429 = Outcome(RateLimitError, "unbilled")
API_ERROR = Outcome(APIError, "indeterminate")
TRANSPORT = Outcome(TransportError, "indeterminate")
TIMEOUT = Outcome(ProviderTimeout, "indeterminate")

# A case: the server's script, and what complete() and a stream must raise.
XAI_CASES: dict[str, tuple[fakegrpc.Script, Outcome, Outcome]] = {
    "429": (fakegrpc.Script(code=grpc.StatusCode.RESOURCE_EXHAUSTED), UNBILLED_429, UNBILLED_429),
    "500": (fakegrpc.Script(code=grpc.StatusCode.INTERNAL), API_ERROR, API_ERROR),
    "unavailable": (fakegrpc.Script(code=grpc.StatusCode.UNAVAILABLE), TRANSPORT, TRANSPORT),
    "silent": (fakegrpc.Script(hang=5.0), TIMEOUT, TIMEOUT),
    "error inside the stream": (
        fakegrpc.Script(code=grpc.StatusCode.INTERNAL, abort_after=1),
        API_ERROR,
        API_ERROR,
    ),
    "cut mid-stream": (fakegrpc.Script(drop=True, abort_after=1), Outcome(None), TRANSPORT),
    "success": (fakegrpc.Script(), Outcome(None), Outcome(None)),
}


@pytest.mark.parametrize("case", XAI_CASES)
async def test_xai_transport_failures_are_typed_with_their_delivery(case: str) -> None:
    script, on_complete, on_stream = XAI_CASES[case]
    async with fakegrpc.serving(script) as (_, port):
        provider = await fakegrpc.provider(
            "grok-4.6", port, timeout=0.3 if case == "silent" else 2.0
        )
        assert await _outcome(complete(provider, HI)) == on_complete
        assert await _outcome(stream(provider, HI)) == on_stream
        await provider.close()


@pytest.mark.parametrize("behaviour", ["close", "reset"])
async def test_xai_a_server_that_drops_the_connection_is_a_transport_failure(
    behaviour: fakeserver.Behaviour,
) -> None:
    server, port, _ = await fakeserver.start(behaviour)
    async with server:
        provider = await fakegrpc.provider("grok-4.6", port)
        assert await _outcome(complete(provider, HI)) == TRANSPORT
        assert await _outcome(stream(provider, HI)) == TRANSPORT
        await provider.close()


async def test_xai_a_refused_connection_stays_indeterminate() -> None:
    # gRPC reports it as UNAVAILABLE, as it does a failure after the request left.
    provider = await fakegrpc.provider("grok-4.6", fakeserver.closed_port())
    assert await _outcome(complete(provider, HI)) == TRANSPORT
    assert await _outcome(stream(provider, HI)) == TRANSPORT
    await provider.close()


async def test_xai_a_request_the_sdk_refuses_is_never_sent() -> None:
    # The SDK's chat.create checks the proto field types while the adapter prepares the request.
    async with fakegrpc.serving() as (script, port):
        provider = await fakegrpc.provider("grok-4.6", port)
        outcome = await _outcome(complete(provider, HI, temperature="hot"))
        await provider.close()
    assert outcome == Outcome(RequestError, "not_sent")
    assert script.requests == []


# ---------------------------------------------------------------------------
# Gemini: the real google-genai SDK over its httpx stack, against the loopback HTTP server.
# ---------------------------------------------------------------------------

GEMINI_ERROR = {"error": {"code": 500, "message": "busy", "status": "INTERNAL"}}
GEMINI_OK = {
    "candidates": [
        {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
    ],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4},
    "modelVersion": "gemini-3.8-flash",
    "responseId": "resp-1",
}
GEMINI_EVENTS = [
    fakeserver.sse({"candidates": [{"content": {"role": "model", "parts": [{"text": "o"}]}}]}),
    fakeserver.sse({"candidates": [{"content": {"role": "model", "parts": [{"text": "k"}]}}]}),
]


async def _gemini(port: int, timeout: float = 2.0) -> GeminiProvider:
    """The adapter, with its own HTTP options pointed at 127.0.0.1."""
    provider = GeminiProvider("gemini-3.8-flash", "local-test", timeout=timeout)
    await provider.close()  # its client for Google never connected
    options = _http_options(timeout).model_copy(update={"base_url": f"http://127.0.0.1:{port}"})
    provider._install_client(lambda: genai.Client(api_key="local-test", http_options=options))
    return provider


def _gemini_status(status: int) -> dict[str, Any]:
    body = {"error": {"code": status, "message": "scripted", "status": "X"}}
    return {"behaviour": "status", "status": status, "body": body}


UNBILLED = Outcome(APIError, "unbilled")

# A case: how the server behaves, and what complete() and a stream must raise.
GEMINI_CASES: dict[str, tuple[dict[str, Any], Outcome, Outcome]] = {
    "429": (_gemini_status(429), UNBILLED_429, UNBILLED_429),
    "400": (_gemini_status(400), UNBILLED, UNBILLED),
    "500": (_gemini_status(500), UNBILLED, UNBILLED),
    "503": (_gemini_status(503), API_ERROR, API_ERROR),
    "closed without an answer": ({"behaviour": "close"}, TRANSPORT, TRANSPORT),
    "reset": ({"behaviour": "reset"}, TRANSPORT, TRANSPORT),
    "silent": ({"behaviour": "hang", "seconds": 1.0}, TIMEOUT, TIMEOUT),
    "200 that is not JSON": (
        {"behaviour": "raw", "raw": b"<html>not json</html>"},
        Outcome(ResponseError, "indeterminate"),
        Outcome(ResponseError, "indeterminate"),
    ),
    "error inside the stream": (
        {"behaviour": "sse", "events": [*GEMINI_EVENTS, fakeserver.sse(GEMINI_ERROR)]},
        Outcome(None),
        API_ERROR,  # tokens may already be billed: not the 500 of the billing page
    ),
    "cut mid-stream": (
        {"behaviour": "sse", "events": GEMINI_EVENTS, "ending": "drop"},
        Outcome(None),
        TRANSPORT,
    ),
    "success": ({"behaviour": "sse", "events": GEMINI_EVENTS}, Outcome(None), Outcome(None)),
}
STREAM_ONLY = ("error inside the stream", "cut mid-stream", "success")


@pytest.mark.parametrize("case", GEMINI_CASES)
async def test_gemini_transport_failures_are_typed_with_their_delivery(case: str) -> None:
    behaviour, on_complete, on_stream = GEMINI_CASES[case]
    kwargs = dict(behaviour)
    server, port, _ = await fakeserver.start(kwargs.pop("behaviour"), **kwargs)
    async with server:
        provider = await _gemini(port, timeout=0.3 if case == "silent" else 2.0)
        if case not in STREAM_ONLY:
            assert await _outcome(complete(provider, HI)) == on_complete
        assert await _outcome(stream(provider, HI)) == on_stream
        await provider.close()


async def test_gemini_a_dropped_connection_is_sent_once() -> None:
    # The SDK's aiohttp path re-sent it after a pause of 1 to 10 s, outside any retry option.
    server, port, stats = await fakeserver.start("close")
    async with server:
        provider = await _gemini(port)
        assert await _outcome(complete(provider, HI)) == TRANSPORT
        await provider.close()
    assert stats.requests == 1


async def test_gemini_a_refused_connection_was_never_sent() -> None:
    provider = await _gemini(fakeserver.closed_port())
    assert await _outcome(complete(provider, HI)) == Outcome(TransportError, "not_sent")
    assert await _outcome(stream(provider, HI)) == Outcome(TransportError, "not_sent")
    await provider.close()


async def test_gemini_a_request_the_sdk_refuses_was_never_sent() -> None:
    # The SDK checks the contents before handing the request to the transport.
    server, port, stats = await fakeserver.start("status", body=GEMINI_OK)
    async with server:
        provider = await _gemini(port)
        outcome = await _outcome(complete(provider, []))
        await provider.close()
    assert outcome == Outcome(RequestError, "not_sent")
    assert stats.requests == 0


async def test_gemini_success_through_the_real_sdk() -> None:
    server, port, stats = await fakeserver.start("status", body=GEMINI_OK)
    async with server:
        provider = await _gemini(port)
        response = await complete(provider, HI, thinking_effort="low")
        await provider.close()
    assert response.text == "ok"
    assert response.usage.input_tokens == 3
    sent = json.loads(stats.bodies[0])
    # The SDK passes the thinking config with its proto field names, which proto3 JSON accepts.
    assert sent["generationConfig"]["thinkingConfig"] == {"thinking_level": "LOW"}


# ---------------------------------------------------------------------------
# Meta: the real openai SDK's Responses API, against the loopback HTTP server.
# ---------------------------------------------------------------------------

META_USAGE = {
    "input_tokens": 30,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 12,
    "output_tokens_details": {"reasoning_tokens": 5},
    "total_tokens": 42,
}
META_OK = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "muse-spark-1.3",
    "status": "completed",
    "output": [
        {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": META_USAGE,
}
META_FAILED = {
    **META_OK,
    "status": "failed",
    "output": [],
    "error": {"code": "service_overloaded", "message": "busy"},
}
META_ERROR = {"error": {"message": "busy", "type": "server_error", "code": None}}


def _meta_event(kind: str, **fields: Any) -> bytes:
    return fakeserver.sse({"type": kind, "sequence_number": 0, **fields}, event=kind)


def _meta_delta(text: str) -> bytes:
    return _meta_event(
        "response.output_text.delta",
        item_id="m1",
        output_index=0,
        content_index=0,
        delta=text,
        logprobs=[],
    )


META_DELTAS = [_meta_delta("o"), _meta_delta("k")]
META_DONE = _meta_event("response.completed", response=META_OK)


def _meta(port: int, timeout: float = 2.0) -> MetaProvider:
    return MetaProvider(
        "muse-spark-1.3", "local-test", base_url=f"http://127.0.0.1:{port}/v1", timeout=timeout
    )


# A case: how the server behaves, and what complete() and a stream must raise.
META_CASES: dict[str, tuple[dict[str, Any], Outcome, Outcome]] = {
    "429": (
        {"behaviour": "status", "status": 429, "body": META_ERROR},
        UNBILLED_429,
        UNBILLED_429,
    ),
    "500": ({"behaviour": "status", "status": 500, "body": META_ERROR}, API_ERROR, API_ERROR),
    "503": ({"behaviour": "status", "status": 503, "body": META_ERROR}, API_ERROR, API_ERROR),
    "closed without an answer": ({"behaviour": "close"}, TRANSPORT, TRANSPORT),
    "reset": ({"behaviour": "reset"}, TRANSPORT, TRANSPORT),
    "silent": ({"behaviour": "hang", "seconds": 1.0}, TIMEOUT, TIMEOUT),
    "200 that is not JSON": (
        {"behaviour": "raw", "raw": b"<html>not json</html>"},
        Outcome(ResponseError, "indeterminate"),
        Outcome(ResponseError, "indeterminate"),  # no event: the stream never finishes
    ),
    "response.failed inside the stream": (
        {
            "behaviour": "sse",
            "events": [*META_DELTAS, _meta_event("response.failed", response=META_FAILED)],
        },
        Outcome(None),
        API_ERROR,  # service_overloaded is a 503 in Meta's table
    ),
    "error event without a code": (
        {
            "behaviour": "sse",
            "events": [*META_DELTAS, _meta_event("error", code=None, message="x", param=None)],
        },
        Outcome(None),
        Outcome(ResponseError, "indeterminate"),  # a 400 and a 500 share the null code
    ),
    "error payload inside the stream": (
        {
            "behaviour": "sse",
            "events": [
                *META_DELTAS,
                fakeserver.sse({"error": {"code": "service_overloaded", "message": "x"}}),
            ],
        },
        Outcome(None),
        API_ERROR,
    ),
    "cut mid-stream": (
        {"behaviour": "sse", "events": META_DELTAS, "ending": "drop"},
        Outcome(None),
        TRANSPORT,
    ),
    "success": (
        {"behaviour": "sse", "events": [*META_DELTAS, META_DONE]},
        Outcome(None),
        Outcome(None),
    ),
}
META_STREAM_ONLY = (
    "response.failed inside the stream",
    "error event without a code",
    "error payload inside the stream",
    "cut mid-stream",
    "success",
)


@pytest.mark.parametrize("case", META_CASES)
async def test_meta_transport_failures_are_typed_with_their_delivery(case: str) -> None:
    behaviour, on_complete, on_stream = META_CASES[case]
    kwargs = dict(behaviour)
    server, port, _ = await fakeserver.start(kwargs.pop("behaviour"), **kwargs)
    async with server:
        provider = _meta(port, timeout=0.3 if case == "silent" else 2.0)
        if case not in META_STREAM_ONLY:
            assert await _outcome(complete(provider, HI)) == on_complete
        assert await _outcome(stream(provider, HI)) == on_stream
        await provider.close()


async def test_meta_a_failed_response_keeps_the_usage_it_reported() -> None:
    server, port, _ = await fakeserver.start("status", body=META_FAILED)
    async with server:
        provider = _meta(port)
        with pytest.raises(APIError) as failed:
            await complete(provider, HI)
        await provider.close()
    assert (failed.value.status_code, failed.value.delivery) == (503, "indeterminate")
    assert failed.value.usage is not None and failed.value.usage.output_tokens == 12


async def test_meta_a_refused_connection_was_never_sent() -> None:
    provider = _meta(fakeserver.closed_port())
    assert await _outcome(complete(provider, HI)) == Outcome(TransportError, "not_sent")
    assert await _outcome(stream(provider, HI)) == Outcome(TransportError, "not_sent")
    await provider.close()


async def test_meta_a_request_the_sdk_cannot_serialize_was_never_sent() -> None:
    server, port, stats = await fakeserver.start("status", body=META_OK)
    async with server:
        provider = _meta(port)
        outcome = await _outcome(complete(provider, HI, prompt_cache_key={"not", "json"}))
        await provider.close()
    assert outcome == Outcome(RequestError, "not_sent")
    assert stats.requests == 0


async def test_meta_success_through_the_real_sdk() -> None:
    server, port, stats = await fakeserver.start("status", body=META_OK)
    async with server:
        provider = _meta(port)
        response = await complete(provider, HI, thinking_effort="low")
        await provider.close()
    assert response.text == "ok"
    assert response.usage.input_tokens == 30
    sent = json.loads(stats.bodies[0])
    assert (sent["store"], sent["reasoning"]) == (False, {"effort": "low"})


# ---------------------------------------------------------------------------
# Anthropic: the real anthropic SDK against the loopback HTTP server. Failed requests are not
# charged, a client timeout is (https://support.claude.com/en/articles/8977456).
# ---------------------------------------------------------------------------

CLAUDE_USAGE = {"input_tokens": 3, "output_tokens": 1}
CLAUDE_OK = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": CLAUDE_USAGE,
}


def _claude_event(kind: str, **fields: Any) -> bytes:
    return fakeserver.sse({"type": kind, **fields}, event=kind)


def _claude_error(status: int, kind: str) -> dict[str, Any]:
    body = {"type": "error", "error": {"type": kind, "message": "scripted"}}
    return {"behaviour": "status", "status": status, "body": body}


CLAUDE_START = [
    _claude_event("message_start", message={**CLAUDE_OK, "content": [], "stop_reason": None}),
    _claude_event("content_block_start", index=0, content_block={"type": "text", "text": ""}),
    _claude_event("content_block_delta", index=0, delta={"type": "text_delta", "text": "o"}),
    _claude_event("content_block_delta", index=0, delta={"type": "text_delta", "text": "k"}),
]
CLAUDE_END = [
    _claude_event("content_block_stop", index=0),
    _claude_event(
        "message_delta",
        delta={"stop_reason": "end_turn", "stop_sequence": None},
        usage={"output_tokens": 2},
    ),
    _claude_event("message_stop"),
]
CLAUDE_BAD_TOOL = [
    _claude_event("message_start", message={**CLAUDE_OK, "content": [], "stop_reason": None}),
    _claude_event(
        "content_block_start",
        index=0,
        content_block={"type": "tool_use", "id": "tc_1", "name": "fn", "input": {}},
    ),
    _claude_event(
        "content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": "{no"}
    ),
    _claude_event("content_block_stop", index=0),
]


def _claude(port: int, timeout: float | None = 2.0) -> AnthropicProvider:
    return AnthropicProvider(
        "claude-opus-5", "local-test", base_url=f"http://127.0.0.1:{port}", timeout=timeout
    )


# A case: how the server behaves, and what complete() and a stream must raise.
CLAUDE_CASES: dict[str, tuple[dict[str, Any], Outcome, Outcome]] = {
    "429": (_claude_error(429, "rate_limit_error"), UNBILLED_429, UNBILLED_429),
    "400": (_claude_error(400, "invalid_request_error"), UNBILLED, UNBILLED),
    "500": (_claude_error(500, "api_error"), UNBILLED, UNBILLED),
    "529": (_claude_error(529, "overloaded_error"), UNBILLED, UNBILLED),
    "closed without an answer": ({"behaviour": "close"}, TRANSPORT, TRANSPORT),
    "reset": ({"behaviour": "reset"}, TRANSPORT, TRANSPORT),
    "silent": ({"behaviour": "hang", "seconds": 1.0}, TIMEOUT, TIMEOUT),
    "200 that is not JSON": (
        {"behaviour": "raw", "raw": b"<html>not json</html>"},
        Outcome(ResponseError, "indeterminate"),
        Outcome(ResponseError, "indeterminate"),
    ),
    "error inside the stream": (
        {
            "behaviour": "sse",
            "events": [
                *CLAUDE_START,
                fakeserver.sse(
                    {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
                    event="error",
                ),
            ],
        },
        Outcome(None),
        API_ERROR,  # overloaded_error is a 529; after a 200, tokens may already be billed
    ),
    "malformed tool input in the stream": (
        {"behaviour": "sse", "events": CLAUDE_BAD_TOOL},
        Outcome(None),
        Outcome(ResponseError, "indeterminate"),
    ),
    "cut mid-stream": (
        {"behaviour": "sse", "events": CLAUDE_START, "ending": "drop"},
        Outcome(None),
        TRANSPORT,
    ),
    "success": (
        {"behaviour": "sse", "events": [*CLAUDE_START, *CLAUDE_END]},
        Outcome(None),
        Outcome(None),
    ),
}
CLAUDE_STREAM_ONLY = (
    "error inside the stream",
    "malformed tool input in the stream",
    "cut mid-stream",
    "success",
)


@pytest.mark.parametrize("case", CLAUDE_CASES)
async def test_claude_transport_failures_are_typed_with_their_delivery(case: str) -> None:
    behaviour, on_complete, on_stream = CLAUDE_CASES[case]
    kwargs = dict(behaviour)
    server, port, _ = await fakeserver.start(kwargs.pop("behaviour"), **kwargs)
    async with server:
        provider = _claude(port, timeout=0.3 if case == "silent" else 2.0)
        if case not in CLAUDE_STREAM_ONLY:
            assert await _outcome(complete(provider, HI, max_tokens=64)) == on_complete
        assert await _outcome(stream(provider, HI, max_tokens=64)) == on_stream
        await provider.close()


async def test_claude_a_refused_connection_was_never_sent() -> None:
    provider = _claude(fakeserver.closed_port())
    assert await _outcome(complete(provider, HI, max_tokens=64)) == Outcome(
        TransportError, "not_sent"
    )
    assert await _outcome(stream(provider, HI, max_tokens=64)) == Outcome(
        TransportError, "not_sent"
    )
    await provider.close()


async def test_claude_a_request_the_sdk_refuses_was_never_sent() -> None:
    # Without a timeout of its own, the SDK refuses a non-streaming call that may take over
    # 10 minutes, before sending it.
    server, port, stats = await fakeserver.start("status", body=CLAUDE_OK)
    async with server:
        provider = _claude(port, timeout=None)
        outcome = await _outcome(complete(provider, HI, max_tokens=128_000))
        await provider.close()
    assert outcome == Outcome(RequestError, "not_sent")
    assert stats.requests == 0


async def test_claude_success_through_the_real_sdk() -> None:
    server, port, stats = await fakeserver.start("status", body=CLAUDE_OK)
    async with server:
        provider = _claude(port)
        response = await complete(provider, HI, max_tokens=64, thinking=True)
        await provider.close()
    assert response.text == "ok"
    assert response.usage.input_tokens == 3
    sent = json.loads(stats.bodies[0])
    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}
