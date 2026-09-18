"""Tests for _providers/_anthropic.py — SDK adapter."""

from __future__ import annotations

import inspect
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import types as sdk_types
from anthropic.resources.messages import AsyncMessages
from anthropic.types.raw_message_delta_event import Delta

from ai_arch_toolkit.core import LLM, MeterScope
from ai_arch_toolkit.core._content import CachePart, DocumentPart, ImagePart
from ai_arch_toolkit.core._exceptions import (
    APIError,
    RateLimitError,
    RequestError,
    ResponseError,
    TransportError,
)
from ai_arch_toolkit.core._providers._anthropic import (
    AnthropicProvider,
    _build_output_config,
    _content_to_sdk,
    _extract_usage,
    _messages_to_sdk,
    _parse_sdk_response,
    _tool_to_sdk,
)
from ai_arch_toolkit.core._providers._base import on_request
from ai_arch_toolkit.core._response import OutputSchema, Response, ToolCall, Usage
from tests.provider_calls import assembled, prepare
from tests.provider_calls import complete as _complete
from tests.provider_calls import stream as _stream
from tests.sdk_streams import AnthropicStream

# ---------------------------------------------------------------------------
# Helpers — build fake SDK objects
# ---------------------------------------------------------------------------


async def complete(provider, messages, **kwargs):
    """A call as the LLM makes it: the Messages API requires max_tokens, and the LLM sends one."""
    return await _complete(provider, messages, **{"max_tokens": 1024, **kwargs})


async def stream(provider, messages, **kwargs):
    return await _stream(provider, messages, **{"max_tokens": 1024, **kwargs})


def _sdk_accepts(method: str, call_kwargs: dict) -> None:
    """Every kwarg the adapter sends exists in the installed SDK (a mock client accepts any)."""
    inspect.signature(getattr(AsyncMessages, method)).bind_partial(None, **call_kwargs)


def _sdk_message(
    *,
    text: str = "Hello!",
    tool_calls: list[dict] | None = None,
    thinking: list[str] | None = None,
    model: str = "claude-sonnet-4-6",
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    thinking_tokens: int = 0,
) -> SimpleNamespace:
    """Build a fake anthropic.types.Message-like object."""
    content = []
    if thinking:
        for t in thinking:
            content.append(SimpleNamespace(type="thinking", thinking=t, signature="sig"))
    if text:
        content.append(SimpleNamespace(type="text", text=text, citations=None))
    if tool_calls:
        for tc in tool_calls:
            content.append(
                SimpleNamespace(
                    type="tool_use",
                    id=tc["id"],
                    name=tc["name"],
                    input=tc.get("input", {}),
                )
            )
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        output_tokens_details=SimpleNamespace(thinking_tokens=thinking_tokens),
    )
    return SimpleNamespace(
        id="msg_test",
        content=content,
        model=model,
        stop_reason=stop_reason,
        usage=usage,
    )


def _real_message(
    *,
    text: str = "Hello!",
    stop_reason: str | None = "end_turn",
    usage: sdk_types.Usage | None = None,
) -> sdk_types.Message:
    """Build a real ``anthropic.types.Message``; the SDK model validates the API shape."""
    return sdk_types.Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=[sdk_types.TextBlock(type="text", text=text)] if text else [],
        stop_reason=stop_reason,
        usage=usage or sdk_types.Usage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestMessagesToSdk:
    def test_extracts_system(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        sys, wire = _messages_to_sdk(msgs)
        assert sys == "Be helpful."
        assert len(wire) == 1
        assert wire[0] == {"role": "user", "content": "Hi"}

    def test_no_system(self):
        msgs = [{"role": "user", "content": "Hi"}]
        sys, wire = _messages_to_sdk(msgs)
        assert sys is None
        assert len(wire) == 1

    def test_multiple_system_messages_joined(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        sys, wire = _messages_to_sdk(msgs)
        assert sys == "You are helpful.\n\nBe concise."
        assert len(wire) == 1

    def test_tool_result_message(self):
        msgs = [{"role": "user", "content": "42", "tool_use_id": "call_1"}]
        _, wire = _messages_to_sdk(msgs)
        assert wire[0]["role"] == "user"
        assert wire[0]["content"][0]["type"] == "tool_result"
        assert wire[0]["content"][0]["tool_use_id"] == "call_1"

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
        _, wire = _messages_to_sdk(msgs)
        assert wire[0]["role"] == "assistant"
        content = wire[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "Let me check."}
        assert content[1] == {
            "type": "tool_use",
            "id": "tc_1",
            "name": "get_weather",
            "input": {"city": "NYC"},
        }

    def test_assistant_with_tool_calls_no_text(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "name": "search", "input": {"q": "test"}},
                ],
            },
        ]
        _, wire = _messages_to_sdk(msgs)
        content = wire[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"

    def test_assistant_with_multiple_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "Checking both.",
                "tool_calls": [
                    {"id": "tc_1", "name": "get_weather", "input": {"city": "NYC"}},
                    {"id": "tc_2", "name": "get_time", "input": {"tz": "UTC"}},
                ],
            },
        ]
        _, wire = _messages_to_sdk(msgs)
        content = wire[0]["content"]
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "tool_use"
        assert content[2]["type"] == "tool_use"


class TestToolToSdk:
    def test_maps_parameters_to_input_schema(self):
        tool = {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
        result = _tool_to_sdk(tool)
        assert result["input_schema"] == tool["parameters"]
        assert "parameters" not in result

    def test_falls_back_to_input_schema_key(self):
        tool = {"name": "fn", "description": "desc", "input_schema": {"type": "object"}}
        result = _tool_to_sdk(tool)
        assert result["input_schema"] == {"type": "object"}

    def test_prefers_input_schema_over_parameters(self):
        tool = {
            "name": "fn",
            "description": "desc",
            "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
            "parameters": {"type": "object", "properties": {"b": {"type": "integer"}}},
        }
        result = _tool_to_sdk(tool)
        assert "a" in result["input_schema"]["properties"]
        assert "b" not in result["input_schema"]["properties"]


class TestBuildOutputConfig:
    def test_creates_output_config(self):
        schema = OutputSchema(
            name="Person",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        cfg = _build_output_config(schema)
        assert cfg["format"]["type"] == "json_schema"
        assert cfg["format"]["schema"] == schema.schema


class TestExtractUsage:
    def test_basic(self):
        sdk_usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=10,
        )
        usage = _extract_usage(sdk_usage)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_write_tokens == 20
        assert usage.cache_read_tokens == 10

    def test_null_cache_fields_become_zero(self):
        # The SDK types both cache counters as Optional and defaults them to None.
        usage = _extract_usage(sdk_types.Usage(input_tokens=10, output_tokens=5))
        assert usage == Usage(
            input_tokens=10, output_tokens=5, cache_write_tokens=0, cache_read_tokens=0
        )

    def test_output_tokens_remain_inclusive_of_thinking(self):
        sdk_usage = sdk_types.Usage(input_tokens=100, output_tokens=50)
        usage = _extract_usage(sdk_usage)
        assert usage.output_tokens == 50


class TestParseSdkResponse:
    def test_text_response(self):
        msg = _sdk_message(text="Hello!")
        r = assembled(AnthropicProvider("claude-sonnet-4-6", "test-key"), msg)
        assert r.text == "Hello!"
        assert r.usage.input_tokens == 10
        assert r.usage.output_tokens == 5
        assert r.stop_reason == "end_turn"
        assert isinstance(r, Response)

    def test_tool_calls(self):
        msg = _sdk_message(
            text="Let me check.",
            tool_calls=[{"id": "tc_1", "name": "get_weather", "input": {"city": "NYC"}}],
        )
        r = _parse_sdk_response(msg, "claude-sonnet-4-6")
        assert r.text == "Let me check."
        assert len(r.tool_calls) == 1
        assert isinstance(r.tool_calls[0], ToolCall)
        assert r.tool_calls[0].name == "get_weather"
        assert r.tool_calls[0].input == {"city": "NYC"}

    def test_cost_is_computed(self):
        msg = _sdk_message(input_tokens=1000, output_tokens=500)
        r = assembled(AnthropicProvider("claude-sonnet-4-6", "test-key"), msg)
        assert r.cost is not None
        assert r.cost > 0

    def test_cost_unknown_model(self):
        msg = _sdk_message(input_tokens=1000, output_tokens=500)
        r = assembled(AnthropicProvider("unknown-model-v9", "test-key"), msg)
        assert r.cost is None

    def test_cache_tokens(self):
        msg = _sdk_message(cache_creation_input_tokens=20, cache_read_input_tokens=10)
        r = assembled(AnthropicProvider("claude-sonnet-4-6", "test-key"), msg)
        assert r.usage.cache_write_tokens == 20
        assert r.usage.cache_read_tokens == 10

    def test_null_cache_tokens_become_zero_and_are_priced(self):
        msg = _real_message(
            usage=sdk_types.Usage(
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
            )
        )
        r = assembled(AnthropicProvider("claude-sonnet-4-6", "test-key"), msg)
        assert r.usage.cache_write_tokens == 0
        assert r.usage.cache_read_tokens == 0
        assert r.cost is not None

    def test_thinking_blocks(self):
        msg = _sdk_message(text="Answer", thinking=["Let me reason..."])
        r = _parse_sdk_response(msg, "claude-sonnet-4-6")
        assert len(r.thinking) == 1
        assert r.thinking[0].text == "Let me reason..."

    def test_structured_output_parsed(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        msg = _sdk_message(
            text='{"name": "Alice"}',
            stop_reason="end_turn",
        )
        r = _parse_sdk_response(msg, "claude-sonnet-4-6", output_schema=schema)
        assert r.parsed == {"name": "Alice"}

    def test_structured_output_strips_markdown_fence(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        msg = _sdk_message(text='```json\n{"name": "Alice"}\n```')
        r = _parse_sdk_response(msg, "claude-sonnet-4-6", output_schema=schema)
        assert r.parsed == {"name": "Alice"}

    def test_structured_output_invalid_json_returns_none(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        msg = _sdk_message(text="Sorry, I cannot do that.")
        r = _parse_sdk_response(msg, "claude-sonnet-4-6", output_schema=schema)
        assert r.parsed is None

    def test_structured_output_coerces_pydantic_model(self):
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class Person(BaseModel):
            name: str

        schema = OutputSchema(name="Person", schema=Person.model_json_schema(), model_class=Person)
        msg = _sdk_message(text='{"name": "Alice"}')
        r = _parse_sdk_response(msg, "claude-sonnet-4-6", output_schema=schema)
        assert isinstance(r.parsed, Person)
        assert r.parsed.name == "Alice"

    def test_structured_output_validation_failure_falls_back_to_dict(self):
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class Person(BaseModel):
            name: str
            age: int

        schema = OutputSchema(name="Person", schema=Person.model_json_schema(), model_class=Person)
        msg = _sdk_message(text='{"name": "Alice"}')  # missing required 'age'
        r = _parse_sdk_response(msg, "claude-sonnet-4-6", output_schema=schema)
        assert r.parsed == {"name": "Alice"}

    def test_raw_is_preserved(self):
        msg = _sdk_message()
        r = _parse_sdk_response(msg, "claude-sonnet-4-6")
        assert r.raw is msg


# ---------------------------------------------------------------------------
# Provider integration tests (mocked SDK client)
# ---------------------------------------------------------------------------


_SYSTEM_MERGE_MESSAGES = [
    {"role": "system", "content": "A"},
    {"role": "user", "content": "x"},
]


class TestAnthropicSystemPrompts:
    """``system=`` is never dropped for ``system()`` messages, or the other way round."""

    async def test_streams_send_explicit_then_message_system(self):
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _cumulative_text_stream()
        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client

        await stream(provider, _SYSTEM_MERGE_MESSAGES, system="B", max_tokens=64)

        call_kwargs = mock_client.messages.stream.call_args.kwargs
        assert call_kwargs["system"] == "B\n\nA"
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    async def test_count_tokens_sends_explicit_then_message_system(self):
        mock_client = MagicMock()
        mock_client.messages.count_tokens = AsyncMock(return_value=SimpleNamespace(input_tokens=7))
        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client

        assert await provider.count_tokens(_SYSTEM_MERGE_MESSAGES, system="B") == 7
        assert mock_client.messages.count_tokens.call_args.kwargs["system"] == "B\n\nA"

    async def test_empty_explicit_system_keeps_message_system(self):
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _sdk_message(text="Ok")
        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client

        await complete(provider, _SYSTEM_MERGE_MESSAGES, system="")

        assert mock_client.messages.create.call_args.kwargs["system"] == "A"

    @pytest.mark.parametrize(
        ("messages", "extra"),
        [
            ([{"role": "user", "content": "x"}], []),
            (_SYSTEM_MERGE_MESSAGES, [{"type": "text", "text": "A"}]),
        ],
        ids=["alone", "with-message-system"],
    )
    async def test_native_system_blocks_are_kept_and_message_system_follows(self, messages, extra):
        # Native blocks (with their cache markers) pass through; system() messages are appended
        # as text blocks instead of being dropped.
        blocks = [{"type": "text", "text": "B", "cache_control": {"type": "ephemeral"}}]
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _sdk_message(text="Ok")
        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client

        await complete(provider, messages, system=blocks)  # type: ignore[arg-type]

        assert mock_client.messages.create.call_args.kwargs["system"] == [*blocks, *extra]

    async def test_middleware_system_does_not_erase_message_system(self):
        # MemoryMiddleware-style hook: abefore writes request.system, which used to replace
        # the system() message instead of joining it.
        import dataclasses

        from ai_arch_toolkit.core._llm import LLM

        class _WritesSystem:
            def before(self, request):
                return request

            def after(self, request, response):
                return response

            async def abefore(self, request):
                return dataclasses.replace(request, system="MEM")

        llm = LLM("claude-sonnet-4-6", api_key="x", middleware=[_WritesSystem()])
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _sdk_message(text="Ok")
        llm._provider._client = mock_client

        await llm.complete(_SYSTEM_MERGE_MESSAGES)

        assert mock_client.messages.create.call_args.kwargs["system"] == "MEM\n\nA"


class TestAnthropicProviderComplete:
    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_complete(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(text="Hello!")

        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        mock_sdk.AsyncAnthropic.assert_called_once_with(
            api_key="test-key",
            max_retries=0,
            http_client=mock_sdk.DefaultAsyncHttpxClient.return_value,
        )
        provider._client = mock_client
        result = await complete(provider, [{"role": "user", "content": "Hi"}])
        assert result.text == "Hello!"
        assert isinstance(result, Response)
        mock_client.messages.create.assert_called_once()

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_complete_with_tools(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(
            text="",
            tool_calls=[{"id": "tc_1", "name": "search", "input": {"q": "test"}}],
            stop_reason="tool_use",
        )

        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        result = await complete(provider, [{"role": "user", "content": "Hi"}], tools=tools)
        assert result.has_tool_calls

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_system_from_messages(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(text="Ok")

        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        msgs = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]
        await complete(provider, msgs)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be brief."
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_explicit_system_merged_before_message_system(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(text="Ok")

        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        msgs = [
            {"role": "system", "content": "From message."},
            {"role": "user", "content": "Hi"},
        ]
        await complete(provider, msgs, system="Explicit system.")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Explicit system.\n\nFrom message."
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_thinking_forwarded(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(
            text="Answer", thinking=["reasoning"]
        )

        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        result = await complete(
            provider,
            [{"role": "user", "content": "Hi"}],
            temperature=0.0,
            thinking=True,
            thinking_effort="high",
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        # The 4.6 models think adaptively, with the effort in output_config.
        assert call_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert call_kwargs["output_config"] == {"effort": "high"}
        assert "temperature" not in call_kwargs.get("extra_body", {})  # thinking needs the default
        _sdk_accepts("create", call_kwargs)
        assert len(result.thinking) == 1

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_output_schema_forwarded(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(
            text='{"name": "Alice"}',
            stop_reason="end_turn",
        )

        schema = OutputSchema(
            name="Person",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        result = await complete(
            provider,
            [{"role": "user", "content": "Hi"}],
            output_schema=schema,
        )
        assert result.parsed == {"name": "Alice"}

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "output_config" in call_kwargs
        assert call_kwargs["output_config"]["format"]["type"] == "json_schema"

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_output_schema_with_tools_coexist(self, mock_sdk):
        """Anthropic now supports both tools and output_schema via native JSON mode."""
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(
            text='{"answer": "42"}',
        )

        schema = OutputSchema(name="X", schema={"type": "object"})
        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        result = await complete(
            provider,
            [{"role": "user", "content": "Hi"}],
            tools=tools,
            output_schema=schema,
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert "output_config" in call_kwargs
        assert result.parsed == {"answer": "42"}

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_output_schema_prompt_mode_injects_schema(self, mock_sdk):
        """``structured_output_mode='prompt'`` drops output_config and injects the
        schema into the system prompt — for schemas Anthropic's native JSON mode
        rejects as too complex."""
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(text='{"name": "Alice"}')

        schema = OutputSchema(
            name="Person",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        result = await complete(
            provider,
            [{"role": "user", "content": "Hi"}],
            system="You are helpful.",
            output_schema=schema,
            structured_output_mode="prompt",
        )
        assert result.parsed == {"name": "Alice"}

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "output_config" not in call_kwargs
        assert "You are helpful." in call_kwargs["system"]
        assert "must match this schema" in call_kwargs["system"]
        assert '"properties"' in call_kwargs["system"]

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_invalid_structured_output_mode_raises(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message()

        schema = OutputSchema(name="X", schema={"type": "object"})
        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        with pytest.raises(ValueError, match="structured_output_mode"):
            await complete(
                provider,
                [{"role": "user", "content": "Hi"}],
                output_schema=schema,
                structured_output_mode="bogus",
            )

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_unknown_kwargs_warn(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message()

        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await complete(
                provider,
                [{"role": "user", "content": "Hi"}],
                topp=0.9,
                typo_param=True,
            )
            assert len(w) == 1
            assert "topp" in str(w[0].message)

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_known_kwargs_no_warn(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message()

        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
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

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
        ],
    )
    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_deprecated_temperature_model_drops_temperature(self, mock_sdk, model):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message()

        provider = AnthropicProvider(model, "test-key")
        provider._client = mock_client
        await complete(
            provider,
            [{"role": "user", "content": "Hi"}],
            temperature=0.0,
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "temperature" not in call_kwargs.get("extra_body", {})
        _sdk_accepts("create", call_kwargs)

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_older_models_keep_temperature(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message()

        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        await complete(
            provider,
            [{"role": "user", "content": "Hi"}],
            temperature=0.2,
            top_p=0.9,
            top_k=40,
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        # anthropic 1.x removed sampling from its signatures; the API still takes it in the body.
        assert call_kwargs["extra_body"] == {"temperature": 0.2, "top_p": 0.9, "top_k": 40}
        _sdk_accepts("create", call_kwargs)


class TestAnthropicProviderErrors:
    async def test_rate_limit_error(self):
        import anthropic as anthropic_sdk
        import httpx

        mock_client = AsyncMock()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx.Response(429, headers={"retry-after": "5.0"}, request=request)
        mock_client.messages.create.side_effect = anthropic_sdk.RateLimitError(
            "rate limited", response=resp, body={"error": "too many requests"}
        )

        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        with pytest.raises(RateLimitError) as exc_info:
            await complete(provider, [{"role": "user", "content": "Hi"}])
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 5.0

    async def test_api_status_error(self):
        import anthropic as anthropic_sdk
        import httpx

        mock_client = AsyncMock()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx.Response(500, request=request)
        mock_client.messages.create.side_effect = anthropic_sdk.APIStatusError(
            "server error", response=resp, body="internal"
        )

        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        with pytest.raises(APIError) as exc_info:
            await complete(provider, [{"role": "user", "content": "Hi"}])
        assert exc_info.value.status_code == 500


class TestAnthropicProviderNetworkErrors:
    @pytest.mark.parametrize(
        ("sdk_error", "expected"),
        [("APIConnectionError", ConnectionError), ("APITimeoutError", TimeoutError)],
    )
    async def test_network_failures_become_builtin_errors(self, sdk_error, expected):
        import anthropic as anthropic_sdk
        import httpx

        error = getattr(anthropic_sdk, sdk_error)(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=error)
        mock_client.messages.stream.side_effect = error
        mock_client.messages.count_tokens = AsyncMock(side_effect=error)
        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client
        messages = [{"role": "user", "content": "Hi"}]

        with pytest.raises(expected):
            await complete(provider, messages)
        with pytest.raises(expected):
            await provider.count_tokens(messages)
        with pytest.raises(expected):
            await stream(provider, messages)


class TestAnthropicProviderLifecycle:
    async def test_close(self):
        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        await provider.close()

    async def test_context_manager(self):
        async with AnthropicProvider("claude-sonnet-4-6", "test-key") as provider:
            assert provider._client is not None


def _message_start(usage: sdk_types.Usage | None = None) -> sdk_types.RawMessageStartEvent:
    return sdk_types.RawMessageStartEvent(
        type="message_start", message=_real_message(text="", stop_reason=None, usage=usage)
    )


def _message_end(usage: sdk_types.MessageDeltaUsage) -> list:
    return [
        sdk_types.RawMessageDeltaEvent(
            type="message_delta", delta=Delta(stop_reason="end_turn"), usage=usage
        ),
        sdk_types.RawMessageStopEvent(type="message_stop"),
    ]


class TestAnthropicProviderStreamEvents:
    async def test_a_thinking_block_streams_whole_when_it_finishes(self):
        events = [
            _message_start(),
            sdk_types.RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=sdk_types.ThinkingBlock(type="thinking", thinking="", signature=""),
            ),
            sdk_types.RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=sdk_types.ThinkingDelta(type="thinking_delta", thinking="step1 "),
            ),
            sdk_types.RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=sdk_types.ThinkingDelta(type="thinking_delta", thinking="step2"),
            ),
            sdk_types.RawContentBlockStopEvent(type="content_block_stop", index=0),
            *_message_end(sdk_types.MessageDeltaUsage(output_tokens=7)),
        ]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = AnthropicStream(events)
        provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = mock_client

        collected, response = await stream(
            provider, [{"role": "user", "content": "Hi"}], max_tokens=64
        )

        thinking_events = [event for event in collected if event.kind == "thinking"]
        # The SDK buffers the deltas; the block streams whole when it stops.
        assert [e.thinking.text for e in thinking_events if e.thinking] == ["step1 step2"]
        assert [b.text for b in response.thinking] == ["step1 step2"]


def _real_text_stream(
    start_usage: sdk_types.Usage, delta_usage: sdk_types.MessageDeltaUsage
) -> AnthropicStream:
    """Stream a one-block text reply as the real SDK event types, in wire order."""
    return AnthropicStream(
        [
            _message_start(start_usage),
            sdk_types.RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=sdk_types.TextBlock(type="text", text=""),
            ),
            sdk_types.RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=sdk_types.TextDelta(type="text_delta", text="Hello!"),
            ),
            sdk_types.RawContentBlockStopEvent(type="content_block_stop", index=0),
            *_message_end(delta_usage),
        ]
    )


def _cumulative_text_stream() -> AnthropicStream:
    """Current API shape: ``message_delta`` repeats the input counts as running totals."""
    return _real_text_stream(
        sdk_types.Usage(
            input_tokens=25,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        sdk_types.MessageDeltaUsage(
            input_tokens=25,
            output_tokens=15,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


async def _stream_usage(sdk_stream: AnthropicStream) -> Usage:
    provider = AnthropicProvider("claude-sonnet-4-6", "test-key")
    provider._client = MagicMock()
    provider._client.messages.stream.return_value = sdk_stream
    events, response = await stream(provider, [{"role": "user", "content": "Hi"}])
    assert [event.text for event in events] == ["Hello!"]
    return response.usage


class TestAnthropicStreamUsage:
    """``message_delta.usage`` counts are cumulative: the SDK's snapshot replaces, not adds."""

    async def test_cumulative_delta_replaces_message_start_usage(self):
        usage = await _stream_usage(_cumulative_text_stream())
        assert usage == Usage(
            input_tokens=25, output_tokens=15, cache_write_tokens=0, cache_read_tokens=0
        )

    async def test_cumulative_cache_counts_are_not_added_twice(self):
        sdk_stream = _real_text_stream(
            sdk_types.Usage(
                input_tokens=5,
                output_tokens=1,
                cache_creation_input_tokens=100,
                cache_read_input_tokens=200,
            ),
            sdk_types.MessageDeltaUsage(
                input_tokens=5,
                output_tokens=15,
                cache_creation_input_tokens=100,
                cache_read_input_tokens=200,
            ),
        )
        assert await _stream_usage(sdk_stream) == Usage(
            input_tokens=5, output_tokens=15, cache_write_tokens=100, cache_read_tokens=200
        )

    async def test_delta_with_only_output_tokens_keeps_message_start_counts(self):
        # Every MessageDeltaUsage count except output_tokens is Optional in the SDK.
        sdk_stream = _real_text_stream(
            sdk_types.Usage(
                input_tokens=25,
                output_tokens=1,
                cache_creation_input_tokens=7,
                cache_read_input_tokens=3,
            ),
            sdk_types.MessageDeltaUsage(output_tokens=15),
        )
        assert await _stream_usage(sdk_stream) == Usage(
            input_tokens=25, output_tokens=15, cache_write_tokens=7, cache_read_tokens=3
        )


class TestAnthropicUsageMetering:
    """The LLM charge site meters the usage the real Anthropic adapter reports."""

    async def test_complete_with_null_cache_counts_is_metered(self):
        llm = LLM("claude-sonnet-4-6", api_key="x")
        llm._provider._client = AsyncMock()
        llm._provider._client.messages.create.return_value = _real_message(
            usage=sdk_types.Usage(
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
            )
        )

        with MeterScope() as scope:
            response = await llm.complete("Hi")

        assert response.usage == Usage(input_tokens=10, output_tokens=5)
        assert scope.snapshot().input_tokens == 10

    async def test_stream_events_meters_cumulative_input_tokens_once(self):
        llm = LLM("claude-sonnet-4-6", api_key="x")
        llm._provider._client = MagicMock()
        llm._provider._client.messages.stream.return_value = _cumulative_text_stream()

        with MeterScope() as scope:
            stream = llm.stream_events("Hi")
            async for _ in stream:
                pass

        snapshot = scope.snapshot()
        assert snapshot.input_tokens == 25
        assert snapshot.output_tokens == 15
        assert stream.response is not None
        assert stream.response.usage == Usage(input_tokens=25, output_tokens=15)


# ---------------------------------------------------------------------------
# Multimodal content conversion
# ---------------------------------------------------------------------------


class TestContentToSdk:
    def test_string_passthrough(self):
        assert _content_to_sdk("hello") == "hello"

    def test_image_url(self):
        parts = [ImagePart(source="https://example.com/img.png", media_type="image/png")]
        result = _content_to_sdk(parts)
        assert result == [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}}
        ]

    def test_image_b64(self):
        parts = [ImagePart(source="abc123", media_type="image/jpeg")]
        result = _content_to_sdk(parts)
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/jpeg"
        assert result[0]["source"]["data"] == "abc123"

    def test_image_bytes(self):
        parts = [ImagePart(source=b"\x89PNG", media_type="image/png")]
        result = _content_to_sdk(parts)
        assert result[0]["source"]["type"] == "base64"

    def test_document(self):
        parts = [DocumentPart(source="b64data", media_type="application/pdf", name="doc.pdf")]
        result = _content_to_sdk(parts)
        assert result[0]["type"] == "document"
        assert result[0]["title"] == "doc.pdf"
        assert "name" not in result[0]

    def test_cache_part(self):
        parts = [CachePart(content="cached text")]
        result = _content_to_sdk(parts)
        assert result[0] == {
            "type": "text",
            "text": "cached text",
            "cache_control": {"type": "ephemeral"},
        }

    def test_mixed_content(self):
        parts = [
            "Describe this:",
            ImagePart(source="https://example.com/img.png"),
        ]
        result = _content_to_sdk(parts)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Describe this:"}
        assert result[1]["type"] == "image"

    def test_multimodal_messages_to_sdk(self):
        """Multimodal content flows through _messages_to_sdk."""
        msgs = [
            {
                "role": "user",
                "content": ["hello", ImagePart(source="https://img.com/a.png")],
            }
        ]
        _, wire = _messages_to_sdk(msgs)
        assert len(wire) == 1
        assert isinstance(wire[0]["content"], list)
        assert wire[0]["content"][0] == {"type": "text", "text": "hello"}


# ---------------------------------------------------------------------------
# R02: the request by model family, from the documented thinking and effort tables
# (https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting,
# https://platform.claude.com/docs/en/build-with-claude/effort)
# ---------------------------------------------------------------------------

HI = [{"role": "user", "content": "Hi"}]
ADAPTIVE = (
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-5-2",  # a model newer than the table gets the current generation's rules
)
EXTENDED = (
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-sonnet-4-20250514",
)
NO_SAMPLING = ("claude-fable-5-1", "claude-opus-5", "claude-sonnet-5", "claude-opus-4-7")


def _params(model: str, messages: list[dict] | None = None, **kwargs) -> dict:
    kwargs.setdefault("max_tokens", 1024)
    return prepare(AnthropicProvider(model, "test-key"), messages or HI, **kwargs).params


def _refused(model: str, **kwargs) -> str:
    with pytest.raises(RequestError) as refused:
        _params(model, **kwargs)
    return str(refused.value)


class TestThinkingByModel:
    @pytest.mark.parametrize("model", ADAPTIVE)
    def test_adaptive_models_think_adaptively_and_show_a_summary(self, model):
        params = _params(model, thinking=True)
        assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert params["max_tokens"] == 1024
        _sdk_accepts("create", params)

    @pytest.mark.parametrize("model", ADAPTIVE)
    def test_thinking_false_sends_nothing_where_thinking_may_be_on(self, model):
        assert "thinking" not in _params(model)

    @pytest.mark.parametrize("model", ["claude-opus-5", "claude-fable-5-1", "claude-sonnet-5"])
    def test_the_effort_applies_on_its_own_and_merges_with_the_format(self, model):
        assert _params(model, thinking_effort="xhigh")["output_config"] == {"effort": "xhigh"}
        schema = OutputSchema(name="P", schema={"type": "object"})
        config = _params(model, thinking_effort="low", output_schema=schema)["output_config"]
        assert config == {
            "format": {"type": "json_schema", "schema": {"type": "object"}},
            "effort": "low",
        }

    @pytest.mark.parametrize(
        ("model", "effort"),
        [
            ("claude-opus-4-6", "xhigh"),  # the 4.6 models take max but not xhigh
            ("claude-sonnet-4-6", "xhigh"),
            ("claude-opus-5", "minimal"),
            ("claude-opus-5", "none"),
        ],
    )
    def test_an_effort_the_model_does_not_take_is_refused(self, model, effort):
        assert effort in _refused(model, thinking_effort=effort)

    def test_the_4_6_models_take_max(self):
        assert _params("claude-sonnet-4-6", thinking_effort="max")["output_config"] == {
            "effort": "max"
        }

    def test_a_thinking_budget_on_an_adaptive_model_only_warns(self):
        with pytest.warns(UserWarning, match="thinking_budget"):
            params = _params("claude-opus-5", thinking=True, thinking_budget=5000)
        assert params["thinking"] == {"type": "adaptive", "display": "summarized"}

    @pytest.mark.parametrize("model", EXTENDED)
    def test_older_models_take_a_budget_added_to_max_tokens(self, model):
        assert "thinking" not in _params(model)
        # D20: max_tokens = budget + max_tokens, so the budget never eats the answer.
        params = _params(model, thinking=True, max_tokens=1000)
        assert params["thinking"] == {"type": "enabled", "budget_tokens": 10000}
        assert params["max_tokens"] == 11000
        assert "output_config" not in params
        _sdk_accepts("create", params)

    @pytest.mark.parametrize(
        ("options", "budget"),
        [
            ({"thinking_effort": "low"}, 2048),
            ({"thinking_effort": "medium"}, 5000),
            ({"thinking_effort": "high"}, 10000),
            ({"thinking_budget": 4000}, 4000),
            ({"thinking": True, "thinking_budget": 4000, "thinking_effort": "low"}, 4000),
        ],
    )
    def test_on_older_models_an_effort_or_a_budget_turns_thinking_on(self, options, budget):
        params = _params("claude-haiku-4-5", max_tokens=500, **options)
        assert params["thinking"] == {"type": "enabled", "budget_tokens": budget}
        assert params["max_tokens"] == budget + 500

    @pytest.mark.parametrize(
        "options",
        [{"thinking_budget": 512}, {"thinking_effort": "xhigh"}, {"thinking_effort": "max"}],
    )
    def test_older_models_refuse_a_budget_below_1024_or_an_effort_without_a_budget(self, options):
        _refused("claude-haiku-4-5", **options)


class TestSamplingByModel:
    @pytest.mark.parametrize("model", NO_SAMPLING)
    @pytest.mark.parametrize("name", ["top_p", "top_k"])
    def test_models_without_sampling_refuse_it(self, model, name):
        assert name in _refused(model, **{name: 0.5 if name == "top_p" else 40})

    @pytest.mark.parametrize("model", NO_SAMPLING)
    def test_models_without_sampling_drop_the_llm_default_temperature(self, model):
        assert "extra_body" not in _params(model, temperature=0.0)

    def test_thinking_drops_the_temperature_on_a_model_that_samples(self):
        params = _params("claude-haiku-4-5", thinking=True, temperature=0.2, top_p=0.9)
        assert params["extra_body"] == {"top_p": 0.9}

    def test_max_tokens_is_required(self):
        with pytest.raises(RequestError, match="max_tokens"):
            prepare(AnthropicProvider("claude-opus-5", "test-key"), HI)


WEATHER_TOOL = {"name": "get_weather", "description": "d", "input_schema": {"type": "object"}}


class TestToolChoiceAndServerTools:
    TOOLS = (WEATHER_TOOL,)

    @pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-mythos-5-1"])
    @pytest.mark.parametrize("choice", ["required", "get_weather"])
    def test_forced_tool_use_is_refused_where_the_api_refuses_it(self, model, choice):
        # https://platform.claude.com/docs/en/api/errors#forced-tool-use-not-supported
        assert "tool_choice" in _refused(model, tools=list(self.TOOLS), tool_choice=choice)

    @pytest.mark.parametrize("choice", ["auto", "none"])
    def test_auto_and_none_stay_open_on_fable_5_1(self, choice):
        params = _params("claude-fable-5-1", tools=list(self.TOOLS), tool_choice=choice)
        assert params["tool_choice"] == {"type": choice}

    def test_other_models_take_a_forced_tool(self):
        params = _params("claude-opus-5", tools=list(self.TOOLS), tool_choice="get_weather")
        assert params["tool_choice"] == {"type": "tool", "name": "get_weather"}

    def test_server_tools_carry_their_name(self):
        tools = [
            {"_server_tool": True, "type": "web_search"},
            {"_server_tool": True, "type": "code_execution"},
        ]
        assert _params("claude-opus-5", tools=tools)["tools"] == [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "code_execution_20250825", "name": "code_execution"},
        ]

    @pytest.mark.parametrize(
        "tool",
        [
            {"_server_tool": True, "type": "web_search", "max_uses": 3},
            {"_server_tool": True, "type": "file_search"},
        ],
        ids=["config", "unknown type"],
    )
    def test_a_server_tool_the_adapter_cannot_send_is_refused(self, tool):
        _refused("claude-opus-5", tools=[tool])


class TestErrorsR02:
    @staticmethod
    def _status(code: int, body: object = None):
        import anthropic as anthropic_sdk
        import httpx

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        return anthropic_sdk.APIStatusError(
            "error", response=httpx.Response(code, request=request), body=body
        )

    @pytest.mark.parametrize("code", [400, 404, 500, 529])
    def test_an_error_response_is_unbilled(self, code):
        # "Failed requests aren't charged"
        # (https://support.claude.com/en/articles/8977456-how-do-i-pay-for-my-claude-api-usage).
        error = AnthropicProvider("claude-opus-5", "k").map_error(self._status(code), sent=True)
        assert (type(error), error.status_code, error.delivery) == (APIError, code, "unbilled")

    @pytest.mark.parametrize(
        ("kind", "status"),
        [("overloaded_error", 529), ("api_error", 500), ("rate_limit_error", 429)],
    )
    def test_an_error_event_inside_a_stream_takes_its_types_status(self, kind, status):
        # The SDK raises it with the stream's 200 response
        # (https://platform.claude.com/docs/en/api/errors).
        body = {"type": "error", "error": {"type": kind, "message": "x"}}
        error = AnthropicProvider("claude-opus-5", "k").map_error(
            self._status(200, body), sent=True
        )
        assert error.status_code == status
        assert error.delivery == ("unbilled" if status == 429 else "indeterminate")

    def test_an_unknown_error_event_is_an_unreadable_response(self):
        body = {"type": "error", "error": {"type": "brand_new_error", "message": "x"}}
        error = AnthropicProvider("claude-opus-5", "k").map_error(
            self._status(200, body), sent=True
        )
        assert isinstance(error, ResponseError)

    def test_a_connection_that_never_opened_was_not_sent(self):
        import anthropic as anthropic_sdk
        import httpx
        import httpx2

        refused = anthropic_sdk.APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )
        refused.__cause__ = httpx2.ConnectError("refused")
        error = AnthropicProvider("claude-opus-5", "k").map_error(refused, sent=True)
        assert (type(error), error.delivery) == (TransportError, "not_sent")

    async def test_count_tokens_and_batch_map_a_rate_limit(self):
        import anthropic as anthropic_sdk
        import httpx

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        limited = anthropic_sdk.RateLimitError(
            "slow down", response=httpx.Response(429, request=request), body=None
        )
        client = MagicMock()
        client.messages.count_tokens = AsyncMock(side_effect=limited)
        client.messages.batches.create = AsyncMock(side_effect=limited)
        provider = AnthropicProvider("claude-opus-5", "k")
        provider._client = client
        with pytest.raises(RateLimitError):
            await provider.count_tokens(HI)
        with pytest.raises(RateLimitError):
            await provider.batch_submit([{"custom_id": "a", "messages": HI}])

    def test_the_client_marks_the_dispatch(self):
        with patch("ai_arch_toolkit.core._providers._anthropic.anthropic") as sdk:
            AnthropicProvider("claude-opus-5", "test-key")
        kwargs = sdk.AsyncAnthropic.call_args.kwargs
        assert sdk.DefaultAsyncHttpxClient.call_args.kwargs == {
            "event_hooks": {"request": [on_request]}
        }
        assert kwargs["http_client"] is sdk.DefaultAsyncHttpxClient.return_value
