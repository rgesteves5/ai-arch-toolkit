"""Tests for _providers/_gemini.py — the Gemini adapter over the ``google-genai`` SDK.

Requests are read from what ``prepare`` builds (the SDK's own pydantic types); answers are the
SDK's ``GenerateContentResponse``. Calls through the real SDK and HTTP are in
``tests/integration/test_provider_transport.py``.
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from ai_arch_toolkit.core import ServerTool
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
from ai_arch_toolkit.core._providers._base import on_request
from ai_arch_toolkit.core._providers._gemini import (
    GeminiProvider,
    _extract_usage,
    _messages_to_sdk,
    _parse_sdk_response,
    _tool_to_sdk,
)
from ai_arch_toolkit.core._response import OutputSchema, Response
from ai_arch_toolkit.core._tools import prepare_tools, tool
from tests.provider_calls import assembled, complete, prepare, stream

HI = [{"role": "user", "content": "Hi"}]
WEATHER = {
    "name": "get_weather",
    "description": "Get weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}


@tool
def _locate(point: tuple[float, float]) -> str:
    """Describe a coordinate pair."""
    return str(point)


@tool
def _lookup(query: str, limit: int = 5) -> str:
    """Search for something."""
    return f"{query}:{limit}"


# ---------------------------------------------------------------------------
# SDK objects
# ---------------------------------------------------------------------------


def _usage(
    prompt: int = 10, candidates: int = 5, cached: int = 0, thoughts: int = 0, tool_use: int = 0
) -> types.GenerateContentResponseUsageMetadata:
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        cached_content_token_count=cached,
        thoughts_token_count=thoughts,
        tool_use_prompt_token_count=tool_use,
    )


def _response(
    *parts: types.Part,
    finish: str | None = "STOP",
    usage: types.GenerateContentResponseUsageMetadata | None = None,
) -> types.GenerateContentResponse:
    """A ``GenerateContentResponse`` with one candidate (a text part "Hello" by default)."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model", parts=list(parts) or [types.Part(text="Hello")]
                ),
                finish_reason=finish,
            )
        ],
        usage_metadata=usage or _usage(),
        response_id="resp-1",
    )


def _call(name: str, call_id: str | None = None, signature: bytes | None = None, **args: Any):
    return types.Part(
        function_call=types.FunctionCall(id=call_id, name=name, args=args),
        thought_signature=signature,
    )


def _chunk(*parts: types.Part, finish: str | None = None, usage=None):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)), finish_reason=finish
            )
        ],
        usage_metadata=usage,
    )


def _config(model: str, **kwargs: Any) -> types.GenerateContentConfig:
    """The config the adapter would send for this call."""
    return prepare(GeminiProvider(model, "test-key"), HI, **kwargs).params["config"]


def _refusal(model: str, messages: list[dict[str, Any]] = HI, **kwargs: Any) -> str:
    with pytest.raises(RequestError) as refused:
        prepare(GeminiProvider(model, "test-key"), messages, **kwargs)
    return str(refused.value)


def _mocked(model: str = "gemini-3.8-flash", **methods: Any) -> GeminiProvider:
    provider = GeminiProvider(model, "test-key")
    client = MagicMock()
    for name, value in methods.items():
        setattr(client.aio.models, name, value)
    provider._client = client
    return provider


def _streaming(*chunks: types.GenerateContentResponse, model: str = "gemini-3.8-flash"):
    async def _stream(**kwargs: Any):
        for chunk in chunks:
            yield chunk

    async def generate_content_stream(**kwargs: Any):  # the SDK returns an async iterator
        return _stream(**kwargs)

    return _mocked(model, generate_content_stream=generate_content_stream)


# ---------------------------------------------------------------------------
# Neutral messages → contents
# ---------------------------------------------------------------------------


class TestMessagesToSdk:
    def test_simple_user_message(self):
        sys, contents = _messages_to_sdk([{"role": "user", "content": "Hello"}])
        assert sys is None
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hello"

    def test_system_extracted(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        sys, contents = _messages_to_sdk(msgs)
        assert sys == "Be helpful."
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_multiple_system_joined(self):
        msgs = [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "Hi"},
        ]
        sys, _contents = _messages_to_sdk(msgs)
        assert sys == "First.\n\nSecond."

    def test_assistant_mapped_to_model(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        _, contents = _messages_to_sdk(msgs)
        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "Hello"

    def test_assistant_with_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{"name": "get_weather", "input": {"city": "NYC"}}],
            }
        ]
        _, contents = _messages_to_sdk(msgs)
        assert contents[0].role == "model"
        assert contents[0].parts[0].text == "Let me check."
        assert contents[0].parts[1].function_call.name == "get_weather"
        assert contents[0].parts[1].function_call.args == {"city": "NYC"}

    def test_tool_result_as_function_response(self):
        msgs = [
            {
                "role": "tool",
                "tool_use_id": "tc_1",
                "name": "get_weather",
                "content": '{"temp": 72}',
            }
        ]
        _, contents = _messages_to_sdk(msgs)
        assert contents[0].role == "user"
        fr = contents[0].parts[0].function_response
        assert fr.name == "get_weather"
        assert fr.response == {"temp": 72}

    def test_tool_result_non_json_wrapped(self):
        msgs = [{"role": "tool", "tool_use_id": "tc_1", "name": "fn", "content": "plain text"}]
        _, contents = _messages_to_sdk(msgs)
        fr = contents[0].parts[0].function_response
        assert fr.response == {"result": "plain text"}

    def test_a_turns_results_go_back_together_with_the_ids_gemini_gave(self):
        # https://ai.google.dev/gemini-api/docs/generate-content/function-calling
        raw = _response(
            _call("get_weather", "fc-1", b"sig", city="Lisbon"), _call("get_time", "fc-2")
        )
        msgs = [
            {"role": "user", "content": "Weather and time?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "fc-1", "name": "get_weather", "input": {"city": "Lisbon"}},
                    {"id": "fc-2", "name": "get_time", "input": {}},
                ],
                "_raw": raw,
            },
            {"role": "tool", "tool_use_id": "fc-1", "name": "get_weather", "content": "Sunny"},
            {"role": "tool", "tool_use_id": "fc-2", "name": "get_time", "content": "14:05"},
        ]
        _, contents = _messages_to_sdk(msgs)
        assert [c.role for c in contents] == ["user", "model", "user"]
        assert (
            contents[1].parts[0].thought_signature == b"sig"
        )  # the model turn, as Gemini sent it
        responses = [part.function_response for part in contents[2].parts]
        assert [(r.id, r.name) for r in responses] == [
            ("fc-1", "get_weather"),
            ("fc-2", "get_time"),
        ]

    def test_an_id_the_toolkit_made_up_is_not_sent(self):
        # Gemini 2.5 gives calls no id: the toolkit invents one, which Gemini never saw.
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "made-up", "name": "fn"}]},
            {"role": "tool", "tool_use_id": "made-up", "name": "fn", "content": "ok"},
        ]
        _, contents = _messages_to_sdk(msgs)
        assert contents[1].parts[0].function_response.id is None

    def test_an_unknown_role_is_refused(self):
        with pytest.raises(RequestError, match="developer"):
            _messages_to_sdk([{"role": "developer", "content": "x"}])


class TestToolToSdk:
    def test_basic(self):
        fd = _tool_to_sdk(WEATHER)
        assert fd.name == "get_weather"
        assert fd.description == "Get weather"
        # SDK auto-converts dict to types.Schema
        assert fd.parameters is not None
        assert "city" in fd.parameters.properties

    def test_parameters_key_fallback(self):
        tool = {"name": "fn", "parameters": {"type": "object"}}
        fd = _tool_to_sdk(tool)
        assert fd.parameters is not None

    def test_schema_in_the_openapi_subset_keeps_parameters(self):
        fd = _tool_to_sdk(prepare_tools([_lookup])[0])
        assert fd.parameters is not None
        assert fd.parameters_json_schema is None

    def test_tuple_parameter_goes_through_parameters_json_schema(self):
        # The SDK refuses prefixItems in ``parameters`` before any request is sent.
        fd = _tool_to_sdk(prepare_tools([_locate])[0])
        assert fd.parameters is None
        point = fd.parameters_json_schema["properties"]["point"]
        assert point["prefixItems"] == [{"type": "number"}, {"type": "number"}]

    def test_refs_go_through_parameters_json_schema(self):
        tool = {
            "name": "save",
            "input_schema": {
                "type": "object",
                "properties": {"item": {"$ref": "#/$defs/Item"}},
                "$defs": {"Item": {"type": "object", "properties": {"name": {"type": "string"}}}},
            },
        }
        fd = _tool_to_sdk(tool)
        assert fd.parameters is None
        assert fd.parameters_json_schema["$defs"]["Item"]["type"] == "object"

    def test_a_tuple_tool_reaches_the_config(self):
        config = prepare(
            GeminiProvider("gemini-3.8-flash", "test-key"), HI, tools=prepare_tools([_locate])
        ).params["config"]
        assert config.tools[0].function_declarations[0].parameters_json_schema is not None


# ---------------------------------------------------------------------------
# The request: config and profiles
# ---------------------------------------------------------------------------


class TestRequest:
    def test_system_and_forwarded_parameters(self):
        messages = [
            {"role": "system", "content": "From message."},
            {"role": "user", "content": "x"},
        ]
        provider = GeminiProvider("gemini-3.8-flash", "test-key")
        params = prepare(
            provider, messages, system="Explicit.", temperature=0.5, max_tokens=64
        ).params
        assert params["model"] == "gemini-3.8-flash"
        config = params["config"]
        assert config.system_instruction == "Explicit.\n\nFrom message."
        assert (config.temperature, config.max_output_tokens) == (0.5, 64)

    def test_tools_and_tool_choice(self):
        config = _config("gemini-3.8-flash", tools=[WEATHER], tool_choice="get_weather")
        assert config.tools[0].function_declarations[0].name == "get_weather"
        calling = config.tool_config.function_calling_config
        assert calling.mode == types.FunctionCallingConfigMode.ANY
        assert calling.allowed_function_names == ["get_weather"]

    def test_server_tools(self):
        tools = [
            {"_server_tool": True, "type": "web_search"},
            {"_server_tool": True, "type": "code_execution"},
        ]
        config = _config("gemini-3.8-flash", tools=tools)
        assert config.tools[0].google_search is not None
        assert config.tools[1].code_execution is not None

    @pytest.mark.parametrize(
        "server_tool",
        [
            ServerTool(type="web_search", config={"max_uses": 3}),
            ServerTool(type="file_search"),
        ],
        ids=["config", "unknown type"],
    )
    def test_a_server_tool_the_adapter_cannot_send_is_refused(self, server_tool: ServerTool):
        # Both used to be dropped in silence (the config belongs to C05).
        wire = prepare_tools([server_tool])
        _refusal("gemini-3.8-flash", tools=wire)

    def test_output_schema_and_json_mode(self):
        config = _config(
            "gemini-3.8-flash", output_schema=OutputSchema(name="P", schema={"type": "object"})
        )
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == {"type": "object"}
        assert _config("gemini-3.8-flash", json_mode=True).response_mime_type == "application/json"

    def test_a_config_the_sdk_refuses_is_a_request_error(self):
        _refusal("gemini-3.8-flash", temperature="hot")

    def test_unknown_kwargs_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _config("gemini-3.8-flash", typo_param=True)
        assert [str(x.message) for x in w if "typo_param" in str(x.message)]

    def test_known_kwargs_no_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _config("gemini-3.8-flash", temperature=0.5, top_p=0.9, max_output_tokens=1000)
        assert [str(x.message) for x in w] == []

    def test_the_client_takes_one_attempt_the_httpx_stack_and_the_timeout_in_ms(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        captured: dict[str, Any] = {}

        class FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr("ai_arch_toolkit.core._providers._gemini.genai.Client", FakeClient)
        GeminiProvider("gemini-3.8-flash", "test-key", timeout=45)

        options: types.HttpOptions = captured["http_options"]
        assert captured["api_key"] == "test-key"
        assert options.retry_options is not None and options.retry_options.attempts == 1
        assert options.timeout == 45_000
        args = options.async_client_args or {}
        # A transport of its own keeps the SDK off aiohttp, which re-sends on connection errors.
        assert isinstance(args["transport"], httpx.AsyncHTTPTransport)
        assert args["event_hooks"] == {"request": [on_request]}


# Documented thinking controls (https://ai.google.dev/gemini-api/docs/generate-content/thinking).
LEVELS = {
    "gemini-3.8-flash": {"low", "medium", "high"},
    "gemini-3.1-pro-preview": {"low", "medium", "high"},
    "gemini-3.9-flash": {"low", "medium", "high"},  # a model newer than the table
    "gemini-3.5-flash-lite": {"minimal", "low", "medium", "high"},
    "gemini-3-flash-preview": {"minimal", "low", "medium", "high"},
    "gemini-3-pro-preview": {"low", "high"},
}


class TestThinking:
    @pytest.mark.parametrize("model", LEVELS)
    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max"])
    def test_gemini_3_takes_the_levels_its_model_documents(self, model: str, effort: str):
        if effort not in LEVELS[model]:
            assert effort in _refusal(model, thinking_effort=effort)
            return
        config = _config(model, thinking_effort=effort)
        assert config.thinking_config.thinking_level == types.ThinkingLevel(effort.upper())
        assert config.thinking_config.thinking_budget is None

    def test_thinking_asks_for_thoughts_and_defaults_to_high(self):
        thinking = _config("gemini-3.8-flash", thinking=True).thinking_config
        assert thinking.include_thoughts is True
        assert thinking.thinking_level == types.ThinkingLevel.HIGH

    def test_no_thinking_option_sends_no_thinking_config(self):
        assert _config("gemini-3.8-flash").thinking_config is None

    def test_a_thinking_budget_on_gemini_3_only_warns(self):
        with pytest.warns(UserWarning, match="thinking_budget"):
            config = _config("gemini-3.8-flash", thinking=True, thinking_budget=5000)
        assert config.thinking_config.thinking_budget is None

    @pytest.mark.parametrize(
        ("effort", "budget"), [("low", 2048), ("medium", 5000), ("high", 10000)]
    )
    def test_gemini_2_5_turns_the_effort_into_a_budget(self, effort: str, budget: int):
        assert (
            _config("gemini-2.5-flash", thinking_effort=effort).thinking_config.thinking_budget
            == budget
        )

    def test_gemini_2_5_defaults_and_explicit_budget(self):
        default = _config("gemini-2.5-flash", thinking=True).thinking_config
        assert (default.include_thoughts, default.thinking_budget) == (True, 10000)
        assert (
            _config("gemini-2.5-pro", thinking_budget=8000).thinking_config.thinking_budget == 8000
        )

    @pytest.mark.parametrize(
        ("model", "budget"),
        [
            ("gemini-2.5-pro", 0),  # Pro cannot turn thinking off
            ("gemini-2.5-pro", 64),
            ("gemini-2.5-flash", 30000),
            ("gemini-2.5-flash-lite", 256),
        ],
    )
    def test_a_budget_outside_the_models_range_is_refused(self, model: str, budget: int):
        assert str(budget) in _refusal(model, thinking_budget=budget)

    @pytest.mark.parametrize(
        ("model", "budget"), [("gemini-2.5-flash-lite", 0), ("gemini-2.5-pro", -1)]
    )
    def test_off_and_dynamic_budgets_where_documented(self, model: str, budget: int):
        assert _config(model, thinking_budget=budget).thinking_config.thinking_budget == budget

    def test_gemini_2_5_refuses_an_effort_it_cannot_map(self):
        _refusal("gemini-2.5-flash", thinking_effort="xhigh")


# ---------------------------------------------------------------------------
# Usage and assembly
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_basic(self):
        usage = _extract_usage(_usage(prompt=100, candidates=50, cached=10))
        assert usage.input_tokens == 90
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 10
        assert usage.input_tokens + usage.cache_read_tokens == 100

    def test_none_values(self):
        usage = _extract_usage(types.GenerateContentResponseUsageMetadata())
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_thoughts_are_included_in_billable_output(self):
        usage = _extract_usage(
            _usage(prompt=100, candidates=50, cached=10, thoughts=30, tool_use=20)
        )
        assert usage.input_tokens == 110  # 90 uncached prompt + 20 tool-use input
        assert usage.cache_read_tokens == 10
        assert usage.output_tokens == 80  # 50 candidate + 30 thoughts


class TestParseSdkResponse:
    def test_candidate_with_no_parts_is_an_empty_response(self):
        # gemini-2.5-flash cut off by max_tokens while thinking: content exists, parts is None.
        resp = types.GenerateContentResponse(
            candidates=[
                types.Candidate(content=types.Content(role="model"), finish_reason="MAX_TOKENS")
            ]
        )
        r = _parse_sdk_response(resp, "gemini-2.5-flash")
        assert r.text == "" and r.tool_calls == ()
        assert r.stop_reason == "MAX_TOKENS"

    def test_text_response(self):
        r = _parse_sdk_response(_response(types.Part(text="Hello world")), "gemini-3.8-flash")
        assert isinstance(r, Response)
        assert r.text == "Hello world"
        assert r.model == "gemini-3.8-flash"
        assert r.response_id == "resp-1"

    def test_tool_calls(self):
        r = _parse_sdk_response(
            _response(_call("get_weather", "fc-1", city="NYC")), "gemini-3.8-flash"
        )
        assert [(c.id, c.name, c.input) for c in r.tool_calls] == [
            ("fc-1", "get_weather", {"city": "NYC"})
        ]

    def test_a_call_without_an_id_gets_one(self):
        (call,) = _parse_sdk_response(_response(_call("fn")), "gemini-2.5-flash").tool_calls
        assert call.id

    def test_thinking_blocks(self):
        resp = _response(
            types.Part(text="Let me think...", thought=True), types.Part(text="Answer.")
        )
        r = _parse_sdk_response(resp, "gemini-3.8-flash")
        assert [block.text for block in r.thinking] == ["Let me think..."]
        assert r.text == "Answer."

    def test_empty_candidates(self):
        r = _parse_sdk_response(types.GenerateContentResponse(candidates=[]), "gemini-3.8-flash")
        assert r.text == ""

    def test_structured_output_parsed(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        r = _parse_sdk_response(
            _response(types.Part(text='{"name": "Alice"}')),
            "gemini-3.8-flash",
            output_schema=schema,
        )
        assert r.parsed == {"name": "Alice"}

    def test_cost_is_computed(self):
        resp = _response(usage=_usage(prompt=1000, candidates=500))
        r = assembled(GeminiProvider("gemini-3.7-flash", "test-key"), resp)
        assert r.cost == pytest.approx((0.75 * 1000 + 3.75 * 500) / 1_000_000)

    def test_raw_is_preserved(self):
        resp = _response()
        assert _parse_sdk_response(resp, "gemini-3.8-flash").raw is resp

    def test_finish_reason_mapped(self):
        assert _parse_sdk_response(_response(), "gemini-3.8-flash").stop_reason == "STOP"

    def test_grounding_becomes_citations(self):
        resp = _response()
        resp.candidates[0].grounding_metadata = types.GroundingMetadata(
            grounding_chunks=[
                types.GroundingChunk(web=types.GroundingChunkWeb(uri="https://x", title="X"))
            ]
        )
        (citation,) = _parse_sdk_response(resp, "gemini-3.8-flash").citations
        assert (citation.url, citation.title) == ("https://x", "X")


# ---------------------------------------------------------------------------
# Calls through a mocked SDK client (the SDK's own types)
# ---------------------------------------------------------------------------


class TestCalls:
    async def test_complete(self):
        provider = _mocked(generate_content=AsyncMock(return_value=_response()))
        result = await complete(provider, HI)
        assert result.text == "Hello"
        kwargs = provider._client.aio.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-3.8-flash"
        assert isinstance(kwargs["config"], types.GenerateContentConfig)

    async def test_count_tokens_merges_explicit_and_message_system(self):
        provider = _mocked(
            count_tokens=AsyncMock(return_value=types.CountTokensResponse(total_tokens=9))
        )
        messages = [{"role": "system", "content": "A"}, {"role": "user", "content": "x"}]
        assert await provider.count_tokens(messages, system="B") == 9
        kwargs = provider._client.aio.models.count_tokens.call_args.kwargs
        assert kwargs["config"].system_instruction == "B\n\nA"

    async def test_stream_text(self):
        provider = _streaming(
            _chunk(types.Part(text="Hello ")),
            _chunk(types.Part(text="world"), finish="STOP", usage=_usage(candidates=5)),
        )
        events, response = await stream(provider, HI)
        assert [e.text for e in events if e.kind == "text"] == ["Hello ", "world"]
        assert response.text == "Hello world"
        assert response.stop_reason == "STOP"

    async def test_stream_thinking_and_tool_call(self):
        provider = _streaming(
            _chunk(types.Part(text="Let me think...", thought=True)),
            _chunk(_call("get_weather", "fc-1", city="NYC"), finish="STOP"),
        )
        events, response = await stream(provider, HI)
        assert [e.kind for e in events] == ["thinking", "tool_call"]
        assert [(c.name, c.input) for c in response.tool_calls] == [
            ("get_weather", {"city": "NYC"})
        ]

    async def test_stream_raw_keeps_every_chunk_part(self):
        # The final response is the joined stream: the history replays all of it.
        provider = _streaming(
            _chunk(types.Part(text="Checking.")),
            _chunk(_call("get_weather", "fc-1", b"sig", city="NYC"), finish="STOP"),
        )
        _, response = await stream(provider, HI)
        parts = response.raw.candidates[0].content.parts
        assert [part.text for part in parts] == ["Checking.", None]
        assert parts[1].function_call.name == "get_weather"
        assert parts[1].thought_signature == b"sig"

    async def test_a_stream_without_usage_has_an_unknown_cost(self):
        provider = _streaming(_chunk(types.Part(text="ok"), finish="STOP"))
        items = [item async for item in provider.stream(prepare(provider, HI))]
        assert not items[-1].usage_reported
        assert items[-1].response.cost is None


# ---------------------------------------------------------------------------
# Errors: the one mapper
# ---------------------------------------------------------------------------


def _http_error(code: int) -> genai_errors.APIError:
    """What the SDK raises for an HTTP error status (the reply is an ``httpx.Response``)."""
    body = {"error": {"code": code, "message": "scripted", "status": "X"}}
    reply = httpx.Response(code, json=body, request=httpx.Request("POST", "http://gemini"))
    error = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return error(code, body, reply)


# HTTP status → (error, delivery): 400 and 500 are not billed
# (https://ai.google.dev/gemini-api/docs/billing), 429 never is (D20).
STATUSES: dict[int, tuple[type[ProviderError], Delivery]] = {
    429: (RateLimitError, "unbilled"),
    400: (APIError, "unbilled"),
    500: (APIError, "unbilled"),
    401: (APIError, "indeterminate"),
    404: (APIError, "indeterminate"),
    503: (APIError, "indeterminate"),
}


class TestErrors:
    @pytest.mark.parametrize("code", STATUSES)
    def test_http_statuses_are_typed_with_their_delivery(self, code: int):
        error_type, delivery = STATUSES[code]
        error = GeminiProvider("gemini-3.8-flash", "test-key").map_error(
            _http_error(code), sent=True
        )
        assert type(error) is error_type
        assert (error.status_code, error.delivery) == (code, delivery)

    def test_an_error_inside_a_stream_is_indeterminate(self):
        # The SDK raises it from an error object in a 200 stream: tokens may already be billed.
        inside = genai_errors.ServerError(500, {"error": {"code": 500}}, None)
        error = GeminiProvider("gemini-3.8-flash", "test-key").map_error(inside, sent=True)
        assert type(error) is APIError
        assert (error.status_code, error.delivery) == (500, "indeterminate")

    @pytest.mark.parametrize(
        ("exc", "expected", "delivery"),
        [
            (httpx.ConnectError("refused"), TransportError, "not_sent"),
            (httpx.ConnectTimeout("slow"), ProviderTimeout, "not_sent"),
            (httpx.ReadTimeout("slow"), ProviderTimeout, "indeterminate"),
            (httpx.RemoteProtocolError("cut"), TransportError, "indeterminate"),
            (genai_errors.UnknownApiResponseError("not JSON"), ResponseError, "indeterminate"),
        ],
    )
    def test_transport_failures_by_their_cause(self, exc, expected, delivery):
        error = GeminiProvider("gemini-3.8-flash", "test-key").map_error(exc, sent=True)
        assert type(error) is expected
        assert error.delivery == delivery

    async def test_a_rate_limit_inside_complete_stream_and_count_tokens(self):
        async def broken_stream(**kwargs: Any):
            raise _http_error(429)

        provider = _mocked(
            generate_content=AsyncMock(side_effect=_http_error(429)),
            count_tokens=AsyncMock(side_effect=_http_error(429)),
            generate_content_stream=broken_stream,
        )
        with pytest.raises(RateLimitError):
            await complete(provider, HI)
        with pytest.raises(RateLimitError):
            await stream(provider, HI)
        with pytest.raises(RateLimitError):
            await provider.count_tokens(HI)


class TestGeminiRoundtrip:
    def test_to_message_through_gemini_wire(self):
        """Response → to_message → Gemini _messages_to_sdk → correct format."""
        from ai_arch_toolkit.core._response import ToolCall

        r = Response(
            text="Let me check.",
            tool_calls=(ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}),),
        )
        conversation = [{"role": "user", "content": "What's the weather?"}, r.to_message()]
        sys, contents = _messages_to_sdk(conversation)

        assert sys is None
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "What's the weather?"
        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "Let me check."
        assert contents[1].parts[1].function_call.name == "get_weather"
