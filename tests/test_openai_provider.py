"""Tests for _providers/_openai.py — SDK adapter."""

from __future__ import annotations

import copy
import warnings
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError, RequestError
from ai_arch_toolkit.core._providers._base import on_request, parse_tool_args
from ai_arch_toolkit.core._providers._openai import (
    OpenAIProvider,
    _build_output_schema_format,
    _extract_usage,
    _messages_to_sdk,
    _parse_sdk_response,
    _tool_to_sdk,
)
from ai_arch_toolkit.core._response import (
    OutputSchema,
    Response,
    ThinkingBlock,
    ToolCall,
    Usage,
    _resolve_output_schema,
)
from ai_arch_toolkit.core._server_tools import code_execution, web_search
from ai_arch_toolkit.core._tools import prepare_tools
from tests.provider_calls import complete, prepare, stream

HI = [{"role": "user", "content": "Hi"}]

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
    reasoning: str | None = None,
    reasoning_field: str = "reasoning_content",
    reasoning_tokens: int = 0,
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
    if reasoning is not None:
        setattr(message, reasoning_field, reasoning)
    choice = SimpleNamespace(
        finish_reason=finish_reason,
        index=0,
        message=message,
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
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

    def test_explicit_system_prepended_and_list_system_kept(self):
        msgs = [
            {"role": "system", "content": "From list."},
            {"role": "user", "content": "Hi"},
        ]
        wire = _messages_to_sdk(msgs, system="Explicit.")
        assert wire == [
            {"role": "system", "content": "Explicit."},
            {"role": "system", "content": "From list."},
            {"role": "user", "content": "Hi"},
        ]

    @pytest.mark.parametrize(
        ("explicit", "head"),
        [(None, []), ("B", [{"role": "system", "content": "B"}])],
    )
    def test_mid_conversation_system_keeps_its_position(self, explicit, head):
        msgs = [
            {"role": "user", "content": "x"},
            {"role": "system", "content": "A"},
            {"role": "user", "content": "y"},
        ]
        wire = _messages_to_sdk(msgs, system=explicit)
        assert wire == [
            *head,
            {"role": "user", "content": "x"},
            {"role": "system", "content": "A"},
            {"role": "user", "content": "y"},
        ]

    def test_empty_explicit_system_not_sent(self):
        msgs = [
            {"role": "system", "content": "From list."},
            {"role": "user", "content": "Hi"},
        ]
        wire = _messages_to_sdk(msgs, system="")
        assert wire == [
            {"role": "system", "content": "From list."},
            {"role": "user", "content": "Hi"},
        ]

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

    def test_a_pydantic_schema_is_normalized_to_the_strict_subset(self):
        # OpenAI answers 400 "'additionalProperties' is required to be supplied and to be false"
        # for model_json_schema() as is (verified live, gpt-4.1-nano).
        class Address(BaseModel):
            city: str
            country: str = "Portugal"

        class Person(BaseModel):
            name: str
            nickname: str | None = None
            address: Address = Field(description="Where they live.")

        schema = _resolve_output_schema(Person)
        original = copy.deepcopy(schema.schema)

        sent = _build_output_schema_format(schema)["json_schema"]["schema"]

        assert sent["additionalProperties"] is False
        assert sent["required"] == ["name", "nickname", "address"]
        assert "default" not in sent["properties"]["nickname"]
        # A $ref with a sibling description is inlined, and strict too.
        address = sent["properties"]["address"]
        assert "$ref" not in address
        assert address["description"] == "Where they live."
        assert address["additionalProperties"] is False
        assert address["required"] == ["city", "country"]
        assert sent["$defs"]["Address"]["additionalProperties"] is False
        assert schema.schema == original  # the caller's schema is left alone

    def test_a_self_referencing_model_does_not_recurse_forever(self):
        class Node(BaseModel):
            value: int
            parent: Node | None = Field(default=None, description="The parent node.")
            first: Node = Field(description="Refers to itself with a sibling key.")

        Node.model_rebuild()
        sent = _build_output_schema_format(_resolve_output_schema(Node))["json_schema"]["schema"]

        assert sent["additionalProperties"] is False
        assert sent["required"] == ["value", "parent", "first"]

    def test_a_non_strict_schema_is_sent_as_given(self):
        raw = {"type": "object", "properties": {"a": {"type": "string"}}}
        fmt = _build_output_schema_format(OutputSchema(name="X", schema=raw, strict=False))
        assert fmt["json_schema"]["schema"] == raw


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
        assert usage.input_tokens == 80
        assert usage.cache_read_tokens == 20
        assert usage.input_tokens + usage.cache_read_tokens == 100

    def test_cache_read_tokens_cannot_make_input_negative(self):
        sdk_usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
        )
        usage = _extract_usage(sdk_usage)
        assert usage.input_tokens == 0
        assert usage.cache_read_tokens == 20

    def test_cache_read_tokens_none_details(self):
        sdk_usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_tokens_details=None,
        )
        usage = _extract_usage(sdk_usage)
        assert usage.cache_read_tokens == 0

    def test_completion_tokens_remain_inclusive_of_reasoning(self):
        sdk_usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
        )
        usage = _extract_usage(sdk_usage)
        assert usage.output_tokens == 50

    def test_compatible_api_separate_reasoning_is_recovered_from_total(self):
        sdk_usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=180,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
        )
        usage = _extract_usage(sdk_usage)
        assert usage.output_tokens == 80


class TestParseSdkResponse:
    def test_text_response(self):
        comp = _sdk_completion(text="Hello!")
        r = _parse_sdk_response(comp, "gpt-4o")
        assert r.text == "Hello!"
        assert r.stop_reason == "stop"
        assert isinstance(r, Response)
        # Usage is read by the adapter's usage(); the base puts it on the response.
        usage = OpenAIProvider("gpt-4o", "test-key").usage(comp)
        assert usage == Usage(input_tokens=10, output_tokens=5)

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

    async def test_cost_is_computed(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion(
            prompt_tokens=1000, completion_tokens=500
        )
        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        r = await complete(provider, [{"role": "user", "content": "Hi"}])
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

    def test_reasoning_content_populates_thinking(self):
        comp = _sdk_completion(text="42", reasoning="Let me think.")
        r = _parse_sdk_response(comp, "gpt-4o")
        assert r.thinking == (ThinkingBlock(text="Let me think."),)
        assert r.text == "42"

    def test_reasoning_alt_spelling(self):
        comp = _sdk_completion(text="42", reasoning="Hmm.", reasoning_field="reasoning")
        r = _parse_sdk_response(comp, "gpt-4o")
        assert r.thinking == (ThinkingBlock(text="Hmm."),)

    def test_no_reasoning_empty_thinking(self):
        comp = _sdk_completion(text="42")
        r = _parse_sdk_response(comp, "gpt-4o")
        assert r.thinking == ()


class TestUsageReport:
    def test_a_completion_without_usage_reports_none(self):
        # OpenAI-compatible servers may omit usage; its cost is then unknown, never zero.
        comp = SimpleNamespace(choices=[], model="gpt-4o", usage=None)
        assert OpenAIProvider("gpt-4o", "test-key").usage(comp) is None


# ---------------------------------------------------------------------------
# Provider integration tests (mocked SDK client)
# ---------------------------------------------------------------------------


class TestOpenAIProviderComplete:
    @patch("ai_arch_toolkit.core._providers._openai.openai.AsyncOpenAI")
    def test_disables_hidden_sdk_retries(self, client_cls):
        OpenAIProvider("gpt-4o", "test-key")

        kwargs = client_cls.call_args.kwargs
        assert (kwargs["api_key"], kwargs["max_retries"]) == ("test-key", 0)
        # The HTTP client marks the moment a request is handed to the transport.
        assert kwargs["http_client"].event_hooks["request"] == [on_request]

    async def test_complete(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion(text="Hello!")

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        result = await complete(provider, [{"role": "user", "content": "Hi"}])
        assert result.text == "Hello!"
        assert isinstance(result, Response)
        mock_client.chat.completions.create.assert_called_once()

    async def test_complete_surfaces_reasoning(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion(
            text="42", reasoning="Step by step."
        )

        provider = OpenAIProvider("gemma4:e4b", "not-needed")
        provider._client = mock_client
        result = await complete(provider, [{"role": "user", "content": "Hi"}])
        assert result.thinking == (ThinkingBlock(text="Step by step."),)
        assert result.text == "42"

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
        result = await complete(provider, [{"role": "user", "content": "Hi"}], tools=tools)
        assert result.has_tool_calls

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["type"] == "function"

    async def test_system_passed_as_message(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion(text="Ok")

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        await complete(provider, [{"role": "user", "content": "Hi"}], system="Be brief.")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        msgs = call_kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "Be brief."}
        assert msgs[1] == {"role": "user", "content": "Hi"}

    async def test_explicit_system_sent_before_list_system(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion(text="Ok")

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        msgs = [
            {"role": "system", "content": "From list."},
            {"role": "user", "content": "Hi"},
        ]
        await complete(provider, msgs, system="Explicit.")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "Explicit."},
            {"role": "system", "content": "From list."},
            {"role": "user", "content": "Hi"},
        ]

    async def test_thinking_effort_forwarded_as_reasoning_effort(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-5.4-mini", "test-key")
        provider._client = mock_client
        await complete(
            provider,
            [{"role": "user", "content": "Hi"}],
            thinking=True,
            thinking_effort="medium",
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["reasoning_effort"] == "medium"

    async def test_thinking_defaults_reasoning_effort_to_high(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _sdk_completion()

        provider = OpenAIProvider("gpt-5.4-mini", "test-key")
        provider._client = mock_client
        await complete(
            provider,
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
        await complete(
            provider,
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
        await complete(
            provider,
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
        await complete(
            provider,
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

        provider = OpenAIProvider("gpt-5.4-mini", "test-key")
        provider._client = mock_client
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await complete(
                provider,
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
        result = await complete(
            provider,
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
        await complete(
            provider,
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
            await complete(
                provider,
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
            await complete(
                provider,
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
        await complete(
            provider,
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
        await complete(
            provider,
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
        await complete(
            provider,
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
        await complete(
            provider,
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
            await complete(provider, [{"role": "user", "content": "Hi"}])
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
            await complete(provider, [{"role": "user", "content": "Hi"}])
        assert exc_info.value.status_code == 500


class TestOpenAIProviderNetworkErrors:
    @pytest.mark.parametrize(
        ("sdk_error", "expected"),
        [("APIConnectionError", ConnectionError), ("APITimeoutError", TimeoutError)],
    )
    async def test_network_failures_become_builtin_errors(self, sdk_error, expected):
        import httpx
        import openai as openai_sdk

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = getattr(openai_sdk, sdk_error)(
            request=request
        )
        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client

        with pytest.raises(expected):
            await complete(provider, [{"role": "user", "content": "Hi"}])
        with pytest.raises(expected):
            await stream(provider, [{"role": "user", "content": "Hi"}])

    async def test_a_dropped_connection_is_retried_by_llm(self):
        import httpx
        import openai as openai_sdk

        from ai_arch_toolkit import LLM, RetryConfig

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [
            openai_sdk.APIConnectionError(request=request),
            _sdk_completion(text="recovered"),
        ]
        async with LLM("gpt-4o", api_key="test-key", retry=RetryConfig(base_delay=0.01)) as llm:
            llm._provider._client = mock_client  # type: ignore[attr-defined]
            response = await llm.complete("Hi")

        assert response.text == "recovered"
        assert mock_client.chat.completions.create.await_count == 2


class TestOpenAIProviderLifecycle:
    async def test_close(self):
        provider = OpenAIProvider("gpt-4o", "test-key")
        await provider.close()

    async def test_context_manager(self):
        async with OpenAIProvider("gpt-4o", "test-key") as provider:
            assert provider._client is not None


class TestAstra:
    @pytest.mark.parametrize("thinking", [False, True])
    async def test_request_parameters(self, thinking):
        client = AsyncMock()
        client.chat.completions.create.return_value = _sdk_completion()
        provider = OpenAIProvider("gpt-6-astra", "test-key")
        provider._client = client
        await complete(
            provider,
            [{"role": "user", "content": "Hi"}],
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            logprobs=True,
            top_logprobs=5,
            thinking=thinking,
            thinking_effort="max",
        )
        params = client.chat.completions.create.call_args.kwargs
        assert params["model"] == "gpt-6-astra"
        assert params["max_completion_tokens"] == 4096
        assert (
            not {"max_tokens", "temperature", "top_p", "logprobs", "top_logprobs"} & params.keys()
        )
        assert params.get("reasoning_effort") == ("max" if thinking else None)

    @pytest.mark.parametrize("effort", ["none", "minimal", "ultra"])
    def test_invalid_reasoning_effort(self, effort):
        provider = OpenAIProvider("gpt-6-astra", "test-key")
        with pytest.raises(RequestError, match="thinking_effort"):
            prepare(provider, HI, thinking=True, thinking_effort=effort)

    def test_tools_require_responses(self):
        provider = OpenAIProvider("gpt-6-astra", "test-key")
        with pytest.raises(RequestError, match="requires the Responses API"):
            prepare(provider, HI, tools=[{"name": "lookup"}])


LOOKUP = {"name": "lookup", "description": "d", "input_schema": {"type": "object"}}
SAMPLING = {"temperature": 0.2, "top_p": 0.9, "logprobs": True, "top_logprobs": 3}


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
class TestSolAndLuna:
    """GPT-6 Sol and Luna reason at "medium" unless sent "none", and take tool calls and sampling
    through Chat Completions only at "none" (https://developers.openai.com/api/docs/models/gpt-6-sol,
    https://developers.openai.com/api/docs/guides/latest-model)."""

    @staticmethod
    def _params(model: str, **kwargs: Any) -> dict[str, Any]:
        return prepare(OpenAIProvider(model, "test-key"), HI, max_tokens=64, **kwargs).params

    def test_by_default_they_reason_so_sampling_is_dropped(self, model):
        params = self._params(model, **SAMPLING)
        assert "reasoning_effort" not in params
        assert not {"temperature", "top_p", "logprobs", "top_logprobs"} & params.keys()
        assert params["max_completion_tokens"] == 64

    @pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
    def test_every_documented_effort_is_sent(self, model, effort):
        assert self._params(model, thinking=True, thinking_effort=effort)["reasoning_effort"] == (
            effort
        )

    def test_minimal_is_not_one_of_their_efforts(self, model):
        with pytest.raises(RequestError, match="thinking_effort"):
            self._params(model, thinking=True, thinking_effort="minimal")

    def test_at_none_they_sample(self, model):
        params = self._params(model, thinking=True, thinking_effort="none", **SAMPLING)
        assert params["reasoning_effort"] == "none"
        assert params["temperature"] == 0.2
        assert params["top_p"] == 0.9
        assert params["logprobs"] is True

    def test_tool_calls_without_thinking_are_sent_at_none(self, model):
        params = self._params(model, tools=[LOOKUP], tool_choice="auto", temperature=0.0)
        assert params["reasoning_effort"] == "none"
        assert params["tools"][0]["function"]["name"] == "lookup"
        assert params["tool_choice"] == "auto"
        assert params["temperature"] == 0.0  # at "none" the sampling stays

    @pytest.mark.parametrize("effort", [None, "low", "max"])
    def test_tool_calls_while_reasoning_need_the_responses_api(self, model, effort):
        with pytest.raises(RequestError, match="Responses API"):
            self._params(model, tools=[LOOKUP], thinking=True, thinking_effort=effort)

    def test_tool_calls_at_none_are_accepted_with_thinking_on(self, model):
        params = self._params(model, tools=[LOOKUP], thinking=True, thinking_effort="none")
        assert params["reasoning_effort"] == "none"
        assert "tools" in params


class TestProfiles:
    """Per-model and per-host request rules (R02 step 3), resolved by the id grammar."""

    def test_the_official_host_sends_max_completion_tokens_for_every_model(self):
        # max_tokens is deprecated and not accepted by o-series models:
        # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        for model in ("gpt-4o", "gpt-5.4-mini", "o3", "gpt-7"):
            params = prepare(OpenAIProvider(model, "test-key"), HI, max_tokens=64).params
            assert params["max_completion_tokens"] == 64, model
            assert "max_tokens" not in params, model

    def test_a_compatible_server_gets_max_tokens(self):
        provider = OpenAIProvider("llama3.2", "not-needed", base_url="http://localhost:11434/v1")
        params = prepare(provider, HI, max_tokens=64, thinking=True, temperature=0.0).params
        assert params["max_tokens"] == 64
        assert "max_completion_tokens" not in params
        # No OpenAI model rules on another server: sampling stays as the caller set it.
        assert params["temperature"] == 0.0
        assert params["reasoning_effort"] == "high"

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-2024-08-06", "gpt-4.1-mini", "gpt-4.1"])
    def test_a_model_that_does_not_reason_refuses_thinking(self, model):
        with pytest.raises(RequestError, match="does not reason"):
            prepare(OpenAIProvider(model, "test-key"), HI, thinking=True)

    def test_a_new_model_gets_the_current_generation_rules(self):
        params = prepare(
            OpenAIProvider("gpt-7", "test-key"), HI, thinking=True, temperature=0.0
        ).params
        assert params["reasoning_effort"] == "high"
        assert "temperature" not in params  # reasoning models take only the default

    @pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.5", "gpt-5.6-terra", "gpt-7"])
    def test_from_gpt_5_4_tool_calls_while_reasoning_are_refused(self, model):
        # "Starting with GPT-5.4, Chat Completions does not support tool calling with
        # reasoning_effort values other than none" (docs/guides/migrate-to-responses).
        provider = OpenAIProvider(model, "test-key")
        with pytest.raises(RequestError, match="Responses API"):
            prepare(provider, HI, tools=[LOOKUP], thinking=True)
        at_none = prepare(provider, HI, tools=[LOOKUP], thinking=True, thinking_effort="none")
        assert at_none.params["reasoning_effort"] == "none"
        # Without thinking nothing changes: no effort is sent, as the live probes ran them.
        plain = prepare(provider, HI, tools=[LOOKUP], temperature=0.0).params
        assert "reasoning_effort" not in plain
        assert plain["temperature"] == 0.0

    @pytest.mark.parametrize("model", ["gpt-5", "gpt-5.2", "gpt-5-2025-08-07", "o3"])
    def test_earlier_reasoning_models_call_tools_while_reasoning(self, model):
        params = prepare(OpenAIProvider(model, "test-key"), HI, tools=[LOOKUP], thinking=True)
        assert params.params["reasoning_effort"] == "high"
        assert "tools" in params.params

    def test_a_compatible_server_calls_tools_while_reasoning(self):
        provider = OpenAIProvider("qwen3", "not-needed", base_url="http://localhost:11434/v1")
        params = prepare(provider, HI, tools=[LOOKUP], thinking=True).params
        assert params["reasoning_effort"] == "high"
        assert "tools" in params

    def test_astra_refusals_are_request_errors(self):
        provider = OpenAIProvider("gpt-6-astra", "test-key")
        with pytest.raises(RequestError, match="Responses API"):
            prepare(provider, HI, tools=[{"name": "lookup", "input_schema": {}}])
        with pytest.raises(RequestError, match="thinking_effort"):
            prepare(provider, HI, thinking=True, thinking_effort="none")

    @pytest.mark.parametrize(
        "server_tool", [web_search(), web_search(max_uses=2), code_execution()]
    )
    def test_server_tools_are_refused_before_sending(self, server_tool):
        # Chat Completions takes only function and custom tools (openai 3.14 SDK types).
        tools = prepare_tools([server_tool])
        with pytest.raises(RequestError, match="server tool"):
            prepare(OpenAIProvider("gpt-5.4", "test-key"), HI, tools=tools)
