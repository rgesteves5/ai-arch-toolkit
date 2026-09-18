"""Tests for _providers/_xai.py — the xAI adapter over the real ``xai-sdk`` (gRPC).

A request is read from the SDK's own request proto (``chat.create`` builds it without an RPC);
answers come from the SDK's response type, or from a loopback gRPC server
(``tests/integration/fakegrpc.py``). The adapter's client needs a running event loop, so the tests
that build one are async.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import grpc
import pytest
from xai_sdk import chat as xai_chat
from xai_sdk.proto import chat_pb2, sample_pb2, usage_pb2

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
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._providers._base import Prepared
from ai_arch_toolkit.core._providers._xai import (
    XAIProvider,
    _build_response_format,
    _extract_usage,
    _messages_to_sdk,
    _parse_sdk_response,
    _tool_to_sdk,
)
from ai_arch_toolkit.core._response import OutputSchema, Response
from tests.integration import fakegrpc
from tests.provider_calls import assembled, complete, prepare, stream

HI = [{"role": "user", "content": "Hi"}]
WEATHER = {
    "name": "get_weather",
    "description": "Get weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}
ROLE = chat_pb2.MessageRole
EFFORT = chat_pb2.ReasoningEffort


@pytest.fixture
async def grok() -> AsyncIterator[XAIProvider]:
    """An adapter for the current generation (its gRPC client needs a running loop)."""
    async with XAIProvider("grok-4.6", "test-key") as provider:
        yield provider


def _response(proto: chat_pb2.GetChatCompletionResponse | None = None) -> xai_chat.Response:
    """What ``chat.sample()`` returns: the SDK's wrapper around the response proto."""
    return xai_chat.Response(proto or fakegrpc.answer(), 0)


def _usage(**tokens: int) -> usage_pb2.SamplingUsage:
    return usage_pb2.SamplingUsage(**tokens)


async def _wire(model: str, messages: list[dict[str, Any]] = HI, **kwargs: Any) -> Any:
    """The request the SDK would send for this call."""
    async with XAIProvider(model, "test-key") as provider:
        return prepare(provider, messages, **kwargs).params.proto


async def _refusal(model: str, **kwargs: Any) -> str:
    with pytest.raises(RequestError) as refused:
        await _wire(model, **kwargs)
    return str(refused.value)


# ---------------------------------------------------------------------------
# Neutral messages → SDK messages
# ---------------------------------------------------------------------------


class TestMessagesToSdk:
    def test_simple_user_message(self):
        msgs, sys = _messages_to_sdk([{"role": "user", "content": "Hello"}])
        assert sys is None
        assert len(msgs) == 1

    def test_system_extracted(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        msgs, sys = _messages_to_sdk(messages)
        assert sys == "Be helpful."
        assert len(msgs) == 1  # only user message

    def test_multiple_system_joined(self):
        messages = [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "Hi"},
        ]
        _, sys = _messages_to_sdk(messages)
        assert sys == "First.\n\nSecond."

    def test_assistant_message(self):
        msgs, _ = _messages_to_sdk([{"role": "assistant", "content": "Hi there"}])
        assert len(msgs) == 1

    def test_tool_result(self):
        messages = [
            {"role": "tool", "tool_use_id": "tc_1", "content": "Sunny in NYC"},
        ]
        msgs, _ = _messages_to_sdk(messages)
        assert len(msgs) == 1
        assert msgs[0].role == ROLE.ROLE_TOOL
        assert msgs[0].tool_call_id == "tc_1"

    def test_assistant_with_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"id": "tc_1", "name": "get_weather", "input": {"city": "NYC"}},
                ],
            }
        ]
        msgs, _ = _messages_to_sdk(messages)
        assert len(msgs) == 1
        # Should be a proto Message with tool_calls
        assert len(msgs[0].tool_calls) == 1
        assert msgs[0].tool_calls[0].function.name == "get_weather"


class TestToolToSdk:
    def test_basic(self):
        sdk_tool = _tool_to_sdk(WEATHER)
        assert sdk_tool.function.name == "get_weather"

    def test_parameters_key_fallback(self):
        tool = {"name": "fn", "description": "Do something", "parameters": {"type": "object"}}
        sdk_tool = _tool_to_sdk(tool)
        assert sdk_tool.function.name == "fn"


class TestBuildResponseFormat:
    def test_creates_json_schema_format(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        rf = _build_response_format(schema)
        assert rf.format_type == chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA
        assert json.loads(rf.schema) == {"type": "object"}


# ---------------------------------------------------------------------------
# Usage, cost and assembly, from the SDK's own response type
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_basic(self):
        usage = _extract_usage(_usage(prompt_tokens=100, completion_tokens=50))
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_cached_tokens(self):
        usage = _extract_usage(_usage(prompt_tokens=100, cached_prompt_text_tokens=20))
        assert usage.input_tokens == 80
        assert usage.cache_read_tokens == 20
        assert usage.input_tokens + usage.cache_read_tokens == 100

    def test_reasoning_tokens_are_included_in_billable_output(self):
        usage = _extract_usage(
            _usage(prompt_tokens=100, completion_tokens=50, reasoning_tokens=200)
        )
        assert usage.output_tokens == 250


class TestParseSdkResponse:
    def test_text_response(self):
        r = _parse_sdk_response(_response(fakegrpc.answer("Hello world")), "grok-4.6")
        assert isinstance(r, Response)
        assert r.text == "Hello world"
        assert r.model == "grok-4.6"
        assert r.response_id == "grok-1"

    def test_tool_calls(self):
        proto = fakegrpc.answer("", calls=[("tc_1", "get_weather", '{"city": "NYC"}')])
        r = _parse_sdk_response(_response(proto), "grok-4.6")
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].id == "tc_1"
        assert r.tool_calls[0].name == "get_weather"
        assert r.tool_calls[0].input == {"city": "NYC"}

    def test_reasoning_content(self):
        proto = fakegrpc.answer("42", reasoning="Let me think step by step...")
        r = _parse_sdk_response(_response(proto), "grok-4.6")
        assert [block.text for block in r.thinking] == ["Let me think step by step..."]
        assert r.text == "42"

    def test_no_reasoning(self):
        r = _parse_sdk_response(_response(fakegrpc.answer("Hello")), "grok-4.6")
        assert r.thinking == ()

    def test_structured_output_parsed(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        proto = fakegrpc.answer('{"name": "Alice"}')
        r = _parse_sdk_response(_response(proto), "grok-4.6", output_schema=schema)
        assert r.parsed == {"name": "Alice"}

    def test_raw_is_preserved(self):
        resp = _response()
        r = _parse_sdk_response(resp, "grok-4.6")
        assert r.raw is resp

    def test_finish_reason(self):
        r = _parse_sdk_response(_response(), "grok-4.6")
        assert r.stop_reason == "REASON_STOP"

    async def test_provider_reported_cost_is_preferred(self, grok: XAIProvider):
        usage = _usage(prompt_tokens=10, completion_tokens=5, cost_in_usd_ticks=123_450_000)
        r = assembled(grok, _response(fakegrpc.answer(usage=usage)))
        assert r.provider_cost == pytest.approx(0.012345)
        assert r.cost == pytest.approx(0.012345)

    async def test_raw_provider_ticks_match_reasoning_aware_local_calculation(
        self, grok: XAIProvider
    ):
        usage = _usage(
            prompt_tokens=100,
            completion_tokens=50,
            reasoning_tokens=200,
            cost_in_usd_ticks=17_000_000,
        )
        r = assembled(grok, _response(fakegrpc.answer(usage=usage)))
        assert r.provider_cost == pytest.approx(0.0017)
        assert _estimate_response_cost("grok-4.6", r.usage) == pytest.approx(r.provider_cost)

    async def test_missing_provider_cost_falls_back_to_reasoning_aware_estimate(
        self, grok: XAIProvider
    ):
        usage = _usage(prompt_tokens=100, completion_tokens=50, reasoning_tokens=200)
        r = assembled(grok, _response(fakegrpc.answer(usage=usage)))
        assert r.provider_cost is None
        assert r.cost == pytest.approx(0.0017)

    async def test_a_negative_reported_cost_is_ignored(self, grok: XAIProvider):
        usage = _usage(prompt_tokens=100, completion_tokens=50, cost_in_usd_ticks=-5)
        r = assembled(grok, _response(fakegrpc.answer(usage=usage)))
        assert r.provider_cost is None
        assert r.cost == _estimate_response_cost("grok-4.6", r.usage)

    async def test_an_answer_without_usage_has_an_unknown_cost(self, grok: XAIProvider):
        answer = grok._answer(_response(fakegrpc.answer(usage=None)), Prepared(None))
        assert not answer.usage_reported
        assert answer.response.cost is None


# ---------------------------------------------------------------------------
# The request: built by the SDK's own chat.create, from each model's profile
# ---------------------------------------------------------------------------


class TestRequest:
    async def test_system_prompts_merge_explicit_first(self):
        messages = [{"role": "system", "content": "A"}, {"role": "user", "content": "x"}]
        wire = await _wire("grok-4.6", messages, system="B")
        assert wire.messages[0].role == ROLE.ROLE_SYSTEM
        assert [part.text for part in wire.messages[0].content] == ["B\n\nA"]
        assert [m.role for m in wire.messages] == [ROLE.ROLE_SYSTEM, ROLE.ROLE_USER]

    async def test_tools_and_forwarded_parameters_reach_the_wire(self):
        wire = await _wire(
            "grok-4.6", tools=[WEATHER], temperature=0.5, top_p=0.9, max_tokens=64, seed=7
        )
        assert [tool.function.name for tool in wire.tools] == ["get_weather"]
        assert (wire.temperature, wire.top_p, wire.max_tokens, wire.seed) == (
            pytest.approx(0.5),
            pytest.approx(0.9),
            64,
            7,
        )

    async def test_a_named_tool_choice_uses_the_sdks_form(self):
        wire = await _wire("grok-4.6", tools=[WEATHER], tool_choice="get_weather")
        assert wire.tool_choice.function_name == "get_weather"

    @pytest.mark.parametrize(
        ("choice", "mode"),
        [
            ("auto", chat_pb2.ToolMode.TOOL_MODE_AUTO),
            ("required", chat_pb2.ToolMode.TOOL_MODE_REQUIRED),
            ("none", chat_pb2.ToolMode.TOOL_MODE_NONE),
        ],
    )
    async def test_tool_choice_modes(self, choice: str, mode: int):
        wire = await _wire("grok-4.6", tools=[WEATHER], tool_choice=choice)
        assert wire.tool_choice.mode == mode

    async def test_server_tools_are_refused(self):
        server_tool = {"_server_tool": True, "type": "web_search", "name": "web_search"}
        assert "web_search" in await _refusal("grok-4.6", tools=[server_tool])

    async def test_output_schema_and_json_mode(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        wire = await _wire("grok-4.6", output_schema=schema)
        assert wire.response_format.format_type == chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA
        assert json.loads(wire.response_format.schema) == {"type": "object"}
        wire = await _wire("grok-4.6", json_mode=True)
        assert wire.response_format.format_type == chat_pb2.FormatType.FORMAT_TYPE_JSON_OBJECT

    async def test_a_request_the_sdk_refuses_is_a_request_error_before_any_call(self):
        # The SDK's own validation in chat.create: a proto field's type, an agent count.
        await _refusal("grok-4.6", temperature="hot")
        assert "agent count" in await _refusal("grok-4.20-multi-agent", agent_count=8)

    async def test_unknown_kwargs_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await _wire("grok-4.6", typo_param=True)
        assert [str(x.message) for x in w if "typo_param" in str(x.message)]

    async def test_known_kwargs_no_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await _wire("grok-4.6", temperature=0.5, top_p=0.9)
        assert [str(x.message) for x in w] == []

    @patch("ai_arch_toolkit.core._providers._xai.xai_sdk.AsyncClient")
    def test_disables_transparent_grpc_retries(self, client_cls):
        XAIProvider("grok-4.6", "test-key")
        client_cls.assert_called_once_with(
            api_key="test-key", channel_options=[("grpc.enable_retries", 0)], timeout=None
        )

    @patch("ai_arch_toolkit.core._providers._xai.xai_sdk.AsyncClient")
    def test_forwards_timeout(self, client_cls):
        XAIProvider("grok-4.6", "test-key", timeout=2.5)
        client_cls.assert_called_once_with(
            api_key="test-key", channel_options=[("grpc.enable_retries", 0)], timeout=2.5
        )


# Documented efforts per model (https://docs.x.ai/developers/models/<id>, 2026-09-18).
CURRENT = ("grok-4.6", "grok-4.5", "grok-4.7")  # grok-4.7: a model newer than the table
GROK_4_3 = ("grok-4.3", "grok-4.3-latest", "grok-4-1-fast-reasoning", "grok-4-0709")
AUTO = ("grok-4.20-reasoning", "grok-4.20-0309-reasoning", "grok-build-0.1", "grok-code-fast-1")
PLAIN = ("grok-4.20-non-reasoning", "grok-4.20-0309-non-reasoning", "grok-3")


class TestProfiles:
    @pytest.mark.parametrize("model", CURRENT)
    @pytest.mark.parametrize(
        ("effort", "sent"),
        [
            ("low", EFFORT.EFFORT_LOW),
            ("medium", EFFORT.EFFORT_MEDIUM),
            ("high", EFFORT.EFFORT_HIGH),
            ("xhigh", EFFORT.EFFORT_XHIGH),
        ],
    )
    async def test_the_current_generation_takes_every_effort(
        self, model: str, effort: str, sent: int
    ):
        assert (await _wire(model, thinking_effort=effort)).reasoning_effort == sent

    async def test_the_effort_applies_without_thinking_and_thinking_alone_sends_none(self):
        assert (
            await _wire("grok-4.6", thinking_effort="low")
        ).reasoning_effort == EFFORT.EFFORT_LOW
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wire = await _wire("grok-4.6", thinking=True)
        assert not wire.HasField("reasoning_effort")

    @pytest.mark.parametrize("effort", ["none", "max", "minimal"])
    async def test_an_effort_the_model_does_not_take_is_refused(self, effort: str):
        assert effort in await _refusal("grok-4.6", thinking_effort=effort)

    @pytest.mark.parametrize("model", GROK_4_3)
    async def test_grok_4_3_and_the_ids_it_serves_take_none(self, model: str):
        wire = await _wire(model, thinking_effort="none")
        assert wire.reasoning_effort == EFFORT.EFFORT_NONE

    @pytest.mark.parametrize("model", AUTO)
    async def test_models_that_reason_on_their_own_refuse_an_effort(self, model: str):
        await _refusal(model, thinking_effort="high")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wire = await _wire(model, thinking=True)
        assert not wire.HasField("reasoning_effort")

    @pytest.mark.parametrize("model", PLAIN)
    async def test_models_that_do_not_reason_refuse_thinking(self, model: str):
        assert "does not reason" in await _refusal(model, thinking=True)
        await _refusal(model, thinking_effort="low")

    @pytest.mark.parametrize("model", [*CURRENT, *GROK_4_3, *AUTO])
    @pytest.mark.parametrize("parameter", ["stop", "presence_penalty", "frequency_penalty"])
    async def test_reasoning_models_refuse_stop_and_penalties(self, model: str, parameter: str):
        value: Any = ["END"] if parameter == "stop" else 0.5
        assert parameter in await _refusal(model, **{parameter: value})

    @pytest.mark.parametrize("model", PLAIN)
    async def test_models_that_do_not_reason_take_stop_and_penalties(self, model: str):
        wire = await _wire(model, stop=["END"], presence_penalty=0.5, frequency_penalty=0.5)
        assert list(wire.stop) == ["END"]
        assert wire.presence_penalty == pytest.approx(0.5)

    async def test_thinking_budget_only_warns(self):
        with pytest.warns(UserWarning, match="thinking_budget"):
            wire = await _wire("grok-4.6", thinking_budget=1000)
        assert not wire.HasField("reasoning_effort")

    @pytest.mark.parametrize("model", ["grok-4.20-multi-agent", "grok-4.20-multi-agent-0309"])
    @pytest.mark.parametrize(
        ("options", "agents"),
        [
            ({"thinking_effort": "low"}, chat_pb2.AgentCount.AGENT_COUNT_4),
            ({"thinking_effort": "medium"}, chat_pb2.AgentCount.AGENT_COUNT_4),
            ({"thinking_effort": "high"}, chat_pb2.AgentCount.AGENT_COUNT_16),
            ({"thinking_effort": "xhigh"}, chat_pb2.AgentCount.AGENT_COUNT_16),
            ({"thinking": True}, chat_pb2.AgentCount.AGENT_COUNT_4),
            ({"agent_count": 16, "thinking_effort": "low"}, chat_pb2.AgentCount.AGENT_COUNT_16),
        ],
    )
    async def test_multi_agent_counts_agents_from_the_effort_and_takes_no_max_tokens(
        self, model: str, options: dict[str, Any], agents: int
    ):
        wire = await _wire(model, max_tokens=64, **options)
        assert wire.agent_count == agents
        assert not wire.HasField("max_tokens")
        assert not wire.HasField("reasoning_effort")

    async def test_multi_agent_refuses_client_tools(self):
        assert "tools" in await _refusal("grok-4.20-multi-agent", tools=[WEATHER])


# ---------------------------------------------------------------------------
# Errors: the one mapper
# ---------------------------------------------------------------------------


def _rpc_error(code: grpc.StatusCode) -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(
        code=code,
        initial_metadata=grpc.aio.Metadata(),
        trailing_metadata=grpc.aio.Metadata(),
        details="scripted",
    )


# gRPC status → (error, HTTP status of the google.rpc.Code mapping, delivery).
# https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto
STATUSES: dict[grpc.StatusCode, tuple[type[ProviderError], int | None, Delivery]] = {
    grpc.StatusCode.RESOURCE_EXHAUSTED: (RateLimitError, 429, "unbilled"),
    grpc.StatusCode.UNAVAILABLE: (TransportError, None, "indeterminate"),
    grpc.StatusCode.DEADLINE_EXCEEDED: (ProviderTimeout, None, "indeterminate"),
    grpc.StatusCode.CANCELLED: (APIError, 499, "indeterminate"),
    grpc.StatusCode.UNKNOWN: (APIError, 500, "indeterminate"),
    grpc.StatusCode.INVALID_ARGUMENT: (APIError, 400, "indeterminate"),
    grpc.StatusCode.NOT_FOUND: (APIError, 404, "indeterminate"),
    grpc.StatusCode.ALREADY_EXISTS: (APIError, 409, "indeterminate"),
    grpc.StatusCode.PERMISSION_DENIED: (APIError, 403, "indeterminate"),
    grpc.StatusCode.UNAUTHENTICATED: (APIError, 401, "indeterminate"),
    grpc.StatusCode.FAILED_PRECONDITION: (APIError, 400, "indeterminate"),
    grpc.StatusCode.ABORTED: (APIError, 409, "indeterminate"),
    grpc.StatusCode.OUT_OF_RANGE: (APIError, 400, "indeterminate"),
    grpc.StatusCode.UNIMPLEMENTED: (APIError, 501, "indeterminate"),
    grpc.StatusCode.INTERNAL: (APIError, 500, "indeterminate"),
    grpc.StatusCode.DATA_LOSS: (APIError, 500, "indeterminate"),
}


class TestErrors:
    @pytest.mark.parametrize("code", STATUSES, ids=lambda code: code.name)
    async def test_each_grpc_status_is_typed_with_its_delivery(
        self, grok: XAIProvider, code: grpc.StatusCode
    ):
        error_type, status, delivery = STATUSES[code]
        error = grok.map_error(_rpc_error(code), sent=True)
        assert type(error) is error_type
        assert error.delivery == delivery
        assert getattr(error, "status_code", None) == status

    async def test_an_unknown_failure_after_dispatch_is_an_unreadable_response(
        self, grok: XAIProvider
    ):
        assert isinstance(grok.map_error(ValueError("x"), sent=True), ResponseError)
        assert isinstance(grok.map_error(ValueError("x"), sent=False), RequestError)


# ---------------------------------------------------------------------------
# Calls through the real SDK, against a loopback gRPC server
# ---------------------------------------------------------------------------


def _texts(events) -> list[str]:
    return [event.text for event in events if event.kind == "text"]


class TestCalls:
    async def test_complete_sends_the_prepared_request(self):
        async with fakegrpc.serving() as (script, port):
            provider = await fakegrpc.provider("grok-4.6", port)
            result = await complete(provider, HI, system="Be brief.", thinking_effort="low")
            await provider.close()
        assert result.text == "ok"
        assert result.usage.input_tokens == 30
        assert result.usage.output_tokens == 17  # completion and reasoning tokens
        (request,) = script.requests
        assert request.model == "grok-4.6"
        assert request.reasoning_effort == EFFORT.EFFORT_LOW
        assert [m.role for m in request.messages] == [ROLE.ROLE_SYSTEM, ROLE.ROLE_USER]

    async def test_stream_text_reasoning_and_tool_calls(self):
        script = fakegrpc.Script(
            chunks=[
                fakegrpc.chunk(reasoning="Two lookups."),
                fakegrpc.chunk("Checking "),
                fakegrpc.chunk("both."),
                fakegrpc.chunk(calls=[("tc_1", "get_weather", '{"city": "Lisbon"}')]),
                fakegrpc.chunk(
                    finish=sample_pb2.FinishReason.REASON_TOOL_CALLS, usage=fakegrpc.USAGE
                ),
            ]
        )
        async with fakegrpc.serving(script) as (_, port):
            provider = await fakegrpc.provider("grok-4.6", port)
            events, final = await stream(provider, HI, tools=[WEATHER])
            await provider.close()
        assert [e.kind for e in events] == ["thinking", "text", "text", "tool_call"]
        assert _texts(events) == ["Checking ", "both."]
        assert final.text == "Checking both."
        assert [block.text for block in final.thinking] == ["Two lookups."]
        assert [(c.id, c.name, c.input) for c in final.tool_calls] == [
            ("tc_1", "get_weather", {"city": "Lisbon"})
        ]
        assert final.usage.input_tokens == 30

    async def test_a_stream_without_usage_has_an_unknown_cost(self):
        script = fakegrpc.Script(chunks=[fakegrpc.chunk("o"), fakegrpc.chunk("k")])
        async with fakegrpc.serving(script) as (_, port):
            provider = await fakegrpc.provider("grok-4.6", port)
            items = [item async for item in provider.stream(prepare(provider, HI))]
            await provider.close()
        answer = items[-1]
        assert answer.response.text == "ok"
        assert not answer.usage_reported
        assert answer.response.cost is None

    async def test_provider_cost_and_reasoning_usage_through_a_stream(self):
        usage = _usage(
            prompt_tokens=25,
            completion_tokens=10,
            reasoning_tokens=30,
            cost_in_usd_ticks=42_000_000,
        )
        script = fakegrpc.Script(chunks=[fakegrpc.chunk("Hi"), fakegrpc.chunk(usage=usage)])
        async with fakegrpc.serving(script) as (_, port):
            provider = await fakegrpc.provider("grok-4.6", port)
            _, final = await stream(provider, HI)
            await provider.close()
        assert final.usage.output_tokens == 40
        assert final.provider_cost == pytest.approx(0.0042)
        assert final.cost == pytest.approx(0.0042)

    async def test_a_rate_limit_is_unbilled(self):
        script = fakegrpc.Script(code=grpc.StatusCode.RESOURCE_EXHAUSTED)
        async with fakegrpc.serving(script) as (_, port):
            provider = await fakegrpc.provider("grok-4.6", port)
            with pytest.raises(RateLimitError) as limited:
                await complete(provider, HI)
            with pytest.raises(RateLimitError):
                await stream(provider, HI)
            await provider.close()
        assert limited.value.status_code == 429
        assert limited.value.delivery == "unbilled"
