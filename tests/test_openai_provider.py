"""Tests for _providers/_openai.py — SDK adapter."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._providers._base import StreamState, parse_tool_args
from ai_arch_toolkit.core._providers._openai import (
    OpenAIProvider,
    _build_output_schema_format,
    _extract_usage,
    _messages_to_sdk,
    _parse_sdk_response,
    _tool_to_sdk,
)
from ai_arch_toolkit.core._response import OutputSchema, Response, ToolCall

# ---------------------------------------------------------------------------
# Helpers — build fake SDK objects
# ---------------------------------------------------------------------------


def _sdk_completion(
    *,
    text: str = "Hello!",
    tool_calls: list[dict] | None = None,
    model: str = "gpt-4o",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> SimpleNamespace:
    """Build a fake openai.types.chat.ChatCompletion-like object."""
    tc_objs = None
    if tool_calls:
        tc_objs = [
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=tc.get("arguments", "{}"),
                ),
            )
            for tc in tool_calls
        ]

    message = SimpleNamespace(
        content=text,
        tool_calls=tc_objs,
        role="assistant",
        refusal=None,
    )
    choice = SimpleNamespace(
        finish_reason=finish_reason,
        index=0,
        message=message,
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(
        choices=[choice],
        model=model,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestMessagesToSdk:
    def test_system_as_regular_message(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        wire = _messages_to_sdk(msgs)
        assert wire[0] == {"role": "system", "content": "Be helpful."}
        assert wire[1] == {"role": "user", "content": "Hi"}

    def test_explicit_system_prepended(self):
        msgs = [{"role": "user", "content": "Hi"}]
        wire = _messages_to_sdk(msgs, system="Be helpful.")
        assert wire[0] == {"role": "system", "content": "Be helpful."}
        assert wire[1] == {"role": "user", "content": "Hi"}

    def test_explicit_system_overrides_list_system(self):
        msgs = [
            {"role": "system", "content": "From list."},
            {"role": "user", "content": "Hi"},
        ]
        wire = _messages_to_sdk(msgs, system="Explicit.")
        system_msgs = [m for m in wire if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Explicit."

    def test_multiple_system_messages_without_explicit(self):
        msgs = [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "Hi"},
        ]
        wire = _messages_to_sdk(msgs)
        system_msgs = [m for m in wire if m["role"] == "system"]
        assert len(system_msgs) == 2

    def test_tool_result_uses_role_tool(self):
        msgs = [{"role": "tool", "content": "42", "tool_use_id": "call_1"}]
        wire = _messages_to_sdk(msgs)
        assert wire[0]["role"] == "tool"
        assert wire[0]["tool_call_id"] == "call_1"
        assert wire[0]["content"] == "42"

    def test_no_system(self):
        msgs = [{"role": "user", "content": "Hi"}]
        wire = _messages_to_sdk(msgs)
        assert len(wire) == 1
        assert wire[0]["role"] == "user"

    def test_assistant_with_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"id": "tc_1", "name": "get_weather", "input": {"city": "NYC"}},
                ],
            },
        ]
        wire = _messages_to_sdk(msgs)
        assert wire[0]["role"] == "assistant"
        assert wire[0]["content"] == "Let me check."
        tc = wire[0]["tool_calls"][0]
        assert tc["id"] == "tc_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "NYC"}'


class TestToolToSdk:
    def test_wraps_in_function(self):
        tool = {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
        result = _tool_to_sdk(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["parameters"] == tool["parameters"]

    def test_accepts_input_schema_key(self):
        tool = {
            "name": "search",
            "description": "Search",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
        result = _tool_to_sdk(tool)
        assert result["function"]["parameters"] == tool["input_schema"]

    def test_prefers_input_schema_over_parameters(self):
        tool = {
            "name": "fn",
            "description": "desc",
            "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
            "parameters": {"type": "object", "properties": {"b": {"type": "integer"}}},
        }
        result = _tool_to_sdk(tool)
        assert "a" in result["function"]["parameters"]["properties"]
        assert "b" not in result["function"]["parameters"]["properties"]


class TestParseToolArgs:
    def test_json_string(self):
        result = parse_tool_args('{"city": "NYC"}')
        assert result == {"city": "NYC"}

    def test_dict_passthrough(self):
        d = {"city": "NYC"}
        assert parse_tool_args(d) is d

    def test_invalid_json(self):
        result = parse_tool_args("not json")
        assert result == {"_raw": "not json"}


class TestBuildOutputSchemaFormat:
    def test_creates_json_schema_format(self):
        schema = OutputSchema(
            name="Person",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        fmt = _build_output_schema_format(schema)
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "Person"
        assert fmt["json_schema"]["strict"] is True

    def test_strict_false(self):
        schema = OutputSchema(name="X", schema={"type": "object"}, strict=False)
        fmt = _build_output_schema_format(schema)
        assert fmt["json_schema"]["strict"] is False


class TestExtractUsage:
    def test_basic(self):
        sdk_usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage = _extract_usage(sdk_usage)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_cache_read_tokens(self):
        sdk_usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
        )
        usage = _extract_usage(sdk_usage)
        assert usage.cache_read_tokens == 20

    def test_cache_read_tokens_none_details(self):
        sdk_usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_tokens_details=None,
        )
        usage = _extract_usage(sdk_usage)
        assert usage.cache_read_tokens == 0


class TestParseSdkResponse:
    def test_text_response(self):
        comp = _sdk_completion(text="Hello!")
        r = _parse_sdk_response(comp, "gpt-4o")
        assert r.text == "Hello!"
        assert r.usage.input_tokens == 10
        assert r.usage.output_tokens == 5
        assert r.stop_reason == "stop"
        assert isinstance(r, Response)

    def test_tool_calls(self):
        comp = _sdk_completion(
            text="",
            tool_calls=[
                {"id": "tc_1", "name": "get_weather", "arguments": '{"city": "NYC"}'},
            ],
            finish_reason="tool_calls",
        )
        r = _parse_sdk_response(comp, "gpt-4o")
        assert len(r.tool_calls) == 1
        assert isinstance(r.tool_calls[0], ToolCall)
        assert r.tool_calls[0].name == "get_weather"
        assert r.tool_calls[0].input == {"city": "NYC"}

    def test_empty_choices(self):
        comp = SimpleNamespace(choices=[], model="gpt-4o", usage=None)
        r = _parse_sdk_response(comp, "gpt-4o")
        assert r.text == ""

    def test_cost_is_computed(self):
        comp = _sdk_completion(prompt_tokens=1000, completion_tokens=500)
        r = _parse_sdk_response(comp, "gpt-4o")
        assert r.cost is not None
        assert r.cost > 0

    def test_structured_output_parsed(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        comp = _sdk_completion(text='{"name": "Alice", "age": 30}')
        r = _parse_sdk_response(comp, "gpt-4o", output_schema=schema)
        assert r.parsed == {"name": "Alice", "age": 30}

    def test_raw_is_preserved(self):
        comp = _sdk_completion()
        r = _parse_sdk_response(comp, "gpt-4o")
        assert r.raw is comp


class TestStreamState:
    def test_initial_state(self):
        state = StreamState()
        assert state.usage is None
        assert state.model == ""
        assert state.stop_reason == ""
        assert state.tool_calls == []


# ---------------------------------------------------------------------------
# Provider integration tests (mocked SDK client)
# ---------------------------------------------------------------------------


class TestOpenAIProviderComplete:
    async def test_complete(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion(text="Hello!")

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        result = await provider.complete([{"role": "user", "content": "Hi"}])
        assert result.text == "Hello!"
        assert isinstance(result, Response)
        mock_client.chat.completions.create.assert_called_once()

    async def test_complete_with_tools(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion(
            text="",
            tool_calls=[
                {"id": "tc_1", "name": "search", "arguments": '{"q": "test"}'},
            ],
            finish_reason="tool_calls",
        )

        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        result = await provider.complete([{"role": "user", "content": "Hi"}], tools=tools)
        assert result.has_tool_calls

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["type"] == "function"

    async def test_system_passed_as_message(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion(text="Ok")

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        await provider.complete([{"role": "user", "content": "Hi"}], system="Be brief.")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        msgs = call_kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "Be brief."}
        assert msgs[1] == {"role": "user", "content": "Hi"}

    async def test_explicit_system_overrides_list_system(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion(text="Ok")

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        msgs = [
            {"role": "system", "content": "From list."},
            {"role": "user", "content": "Hi"},
        ]
        await provider.complete(msgs, system="Explicit.")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        wire = call_kwargs["messages"]
        system_msgs = [m for m in wire if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Explicit."

    async def test_thinking_effort_forwarded_as_reasoning_effort(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            thinking=True,
            thinking_effort="medium",
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["reasoning_effort"] == "medium"

    async def test_thinking_defaults_reasoning_effort_to_high(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            thinking=True,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["reasoning_effort"] == "high"

    async def test_gpt5_reasoning_drops_non_default_temperature(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-5.4-mini", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            thinking=True,
            thinking_effort="medium",
            temperature=0.0,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["reasoning_effort"] == "medium"
        assert "temperature" not in call_kwargs

    async def test_gpt5_reasoning_keeps_temperature_one(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-5.4-mini", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            thinking=True,
            thinking_effort="medium",
            temperature=1.0,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["reasoning_effort"] == "medium"
        assert call_kwargs["temperature"] == 1.0

    async def test_gpt5_reasoning_none_keeps_temperature(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-5.4-mini", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            thinking=True,
            thinking_effort="none",
            temperature=0.0,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["reasoning_effort"] == "none"
        assert call_kwargs["temperature"] == 0.0

    async def test_thinking_budget_warns(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await provider.complete(
                [{"role": "user", "content": "Hi"}],
                thinking=True,
                thinking_budget=5000,
            )
            budget_warns = [x for x in w if "thinking_budget" in str(x.message)]
            assert len(budget_warns) == 1

    async def test_output_schema_forwarded(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion(
            text='{"name": "Alice"}'
        )

        schema = OutputSchema(
            name="Person",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        result = await provider.complete(
            [{"role": "user", "content": "Hi"}],
            output_schema=schema,
        )
        assert result.parsed == {"name": "Alice"}
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"

    async def test_output_schema_with_tools_coexist(self):
        """OpenAI supports both tools and output_schema simultaneously."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion(text='{"answer": "42"}')

        schema = OutputSchema(name="X", schema={"type": "object"})
        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            tools=tools,
            output_schema=schema,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert "response_format" in call_kwargs

    async def test_unknown_kwargs_warn(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await provider.complete(
                [{"role": "user", "content": "Hi"}],
                typo_param=True,
            )
            assert len(w) == 1
            assert "typo_param" in str(w[0].message)

    async def test_known_kwargs_no_warn(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await provider.complete(
                [{"role": "user", "content": "Hi"}],
                temperature=0.5,
                top_p=0.9,
            )
            assert len(w) == 0

    async def test_max_completion_tokens_forwarded(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            max_completion_tokens=8192,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_completion_tokens"] == 8192

    async def test_gpt5_translates_max_tokens_to_max_completion_tokens(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-5.4-mini", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            max_tokens=64,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_completion_tokens"] == 64
        assert "max_tokens" not in call_kwargs

    async def test_exact_o3_translates_max_tokens_to_max_completion_tokens(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("o3", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            max_tokens=64,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_completion_tokens"] == 64
        assert "max_tokens" not in call_kwargs

    async def test_gpt5_prefers_explicit_max_completion_tokens(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-5.4-mini", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            max_tokens=64,
            max_completion_tokens=128,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_completion_tokens"] == 128
        assert "max_tokens" not in call_kwargs


class TestOpenAIProviderErrors:
    async def test_rate_limit_error(self):
        import httpx
        import openai as openai_sdk

        mock_client = AsyncMock()
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(429, headers={"retry-after": "3.0"}, request=request)
        mock_client.chat.completions.create.side_effect = openai_sdk.RateLimitError(
            "rate limited", response=resp, body={"error": "too many requests"}
        )

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        with pytest.raises(RateLimitError) as exc_info:
            await provider.complete([{"role": "user", "content": "Hi"}])
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 3.0

    async def test_api_status_error(self):
        import httpx
        import openai as openai_sdk

        mock_client = AsyncMock()
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(500, request=request)
        mock_client.chat.completions.create.side_effect = openai_sdk.APIStatusError(
            "server error", response=resp, body="internal"
        )

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        with pytest.raises(APIError) as exc_info:
            await provider.complete([{"role": "user", "content": "Hi"}])
        assert exc_info.value.status_code == 500


class TestOpenAIProviderLifecycle:
    async def test_close(self):
        provider = OpenAIProvider("gpt-4o", "test-key")
        await provider.close()

    async def test_context_manager(self):
        async with OpenAIProvider("gpt-4o", "test-key") as provider:
            assert provider._client is not None
