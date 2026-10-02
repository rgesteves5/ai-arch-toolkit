"""Tests for _providers/_openai_compatible.py — OpenAI-compatible servers over Chat Completions.

A ``base_url`` on another host than OpenAI's gets this adapter (``create_provider`` chooses by
host, D43). It applies no OpenAI model rule: the request carries what the caller set. Most of these
tests asserted the Chat Completions wire of the old ``_openai.py`` and moved here with it.
"""

from __future__ import annotations

import copy
import json
import warnings
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from ai_arch_toolkit.core import Response, ThinkingBlock, ToolCall, tool_result, user
from ai_arch_toolkit.core._content import document, image, system
from ai_arch_toolkit.core._exceptions import APIError, RateLimitError, RequestError
from ai_arch_toolkit.core._providers._base import on_request, parse_tool_args
from ai_arch_toolkit.core._providers._openai_compatible import (
    OpenAICompatibleProvider,
    _build_output_schema_format,
    _extract_usage,
    _messages_to_sdk,
    _parse_sdk_response,
    _tool_to_sdk,
)
from ai_arch_toolkit.core._response import OutputSchema, Usage, _resolve_output_schema
from ai_arch_toolkit.core._server_tools import code_execution, web_search
from ai_arch_toolkit.core._tools import prepare_tools
from tests.provider_calls import complete, prepare, stream

LOCAL = "http://localhost:11434/v1"
HI = [{"role": "user", "content": "Hi"}]
LOOKUP = {"name": "lookup", "description": "d", "input_schema": {"type": "object"}}


def _provider(model: str = "llama3.2", client: Any = None) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(model, "not-needed", base_url=LOCAL)
    if client is not None:
        provider._client = client
    return provider


def _sdk_completion(
    *,
    text: str = "Hello!",
    tool_calls: list[dict] | None = None,
    model: str = "llama3.2",
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
                function=SimpleNamespace(name=tc["name"], arguments=tc.get("arguments", "{}")),
            )
            for tc in tool_calls
        ]
    message = SimpleNamespace(content=text, tool_calls=tc_objs, role="assistant", refusal=None)
    if reasoning is not None:
        setattr(message, reasoning_field, reasoning)
    choice = SimpleNamespace(finish_reason=finish_reason, index=0, message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _client(completion: Any = None) -> AsyncMock:
    client = AsyncMock()
    client.chat.completions.create.return_value = completion or _sdk_completion()
    return client


def _sent(client: AsyncMock) -> dict[str, Any]:
    return client.chat.completions.create.call_args.kwargs


# ---------------------------------------------------------------------------
# The request a compatible server got before O03, byte for byte
# ---------------------------------------------------------------------------

CALLS = (
    ToolCall(id="call_1", name="get_weather", input={"city": "Lisbon"}),
    ToolCall(id="call_2", name="get_weather", input={"city": "Porto"}),
)
HISTORY = [
    system("Be brief."),
    user(["Compare these:", image(b"\x89PNG"), document(b"%PDF-1.4", name="a.pdf")]),
    Response(
        text="Checking both.", tool_calls=CALLS, thinking=(ThinkingBlock(text="Two lookups."),)
    ).to_message(),
    *(tool_result("Sunny", tool_use_id=call.id, name=call.name) for call in CALLS),
]
WEATHER = {
    "name": "get_weather",
    "description": "Get the weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}
PERSON = OutputSchema(
    name="Person",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "nickname": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "required": ["name"],
    },
)
ONE = [{"role": "user", "content": "hi"}]
# (messages, prepare's arguments, the JSON of what the adapter built before O03): captured from
# the old _openai.py with base_url="http://localhost:11434/v1" on 2026-10-02, before any change.
BEFORE_O03: dict[str, tuple[list[dict[str, Any]], dict[str, Any], str]] = {
    "plain": (
        ONE,
        {"max_tokens": 64, "temperature": 0.0},
        r'{"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}], '
        r'"temperature": 0.0, "max_tokens": 64}',
    ),
    "history and tools": (
        HISTORY,
        {"system": "Rules.", "tools": [WEATHER], "max_tokens": 1024, "tool_choice": "get_weather"},
        r'{"model": "llama3.2", "messages": [{"role": "system", "content": "Rules."}, '
        r'{"role": "system", "content": "Be brief."}, {"role": "user", "content": '
        r'[{"type": "text", "text": "Compare these:"}, {"type": "image_url", "image_url": '
        r'{"url": "data:image/png;base64,iVBORw=="}}, {"type": "file", "file": {"filename": '
        r'"a.pdf", "file_data": "data:application/pdf;base64,JVBERi0xLjQ="}}]}, {"role": '
        r'"assistant", "content": "Checking both.", "tool_calls": [{"id": "call_1", "type": '
        r'"function", "function": {"name": "get_weather", "arguments": "{\"city\": '
        r'\"Lisbon\"}"}}, {"id": "call_2", "type": "function", "function": {"name": '
        r'"get_weather", "arguments": "{\"city\": \"Porto\"}"}}]}, {"role": "tool", '
        r'"tool_call_id": "call_1", "content": "Sunny"}, {"role": "tool", "tool_call_id": '
        r'"call_2", "content": "Sunny"}], "max_tokens": 1024, "tools": [{"type": "function", '
        r'"function": {"name": "get_weather", "description": "Get the weather", "parameters": '
        r'{"type": "object", "properties": {"city": {"type": "string"}}}}}], "tool_choice": '
        r'{"type": "function", "function": {"name": "get_weather"}}}',
    ),
    "thinking and sampling": (
        ONE,
        {
            "thinking": True,
            "thinking_effort": "low",
            "temperature": 0.2,
            "top_p": 0.9,
            "logprobs": True,
            "top_logprobs": 3,
        },
        r'{"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}], '
        r'"temperature": 0.2, "top_p": 0.9, "top_logprobs": 3, "logprobs": true, '
        r'"reasoning_effort": "low"}',
    ),
    "thinking defaults to high": (
        ONE,
        {"tools": [WEATHER], "thinking": True, "tool_choice": "required"},
        r'{"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}], '
        r'"reasoning_effort": "high", "tools": [{"type": "function", "function": {"name": '
        r'"get_weather", "description": "Get the weather", "parameters": {"type": "object", '
        r'"properties": {"city": {"type": "string"}}}}}], "tool_choice": "required"}',
    ),
    "strict output schema": (
        ONE,
        {"output_schema": PERSON},
        r'{"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}], '
        r'"response_format": {"type": "json_schema", "json_schema": {"name": "Person", '
        r'"schema": {"type": "object", "properties": {"name": {"type": "string"}, "nickname": '
        r'{"anyOf": [{"type": "string"}, {"type": "null"}]}}, "required": ["name", '
        r'"nickname"], "additionalProperties": false}, "strict": true}}}',
    ),
    "loose output schema": (
        ONE,
        {"output_schema": OutputSchema(name="X", schema={"type": "object"}, strict=False)},
        r'{"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}], '
        r'"response_format": {"type": "json_schema", "json_schema": {"name": "X", "schema": '
        r'{"type": "object"}, "strict": false}}}',
    ),
    "json mode and chat parameters": (
        ONE,
        {
            "json_mode": True,
            "stop": ["\n"],
            "seed": 7,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.1,
            "parallel_tool_calls": False,
        },
        r'{"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}], "stop": '
        r'["\n"], "seed": 7, "frequency_penalty": 0.5, "presence_penalty": 0.1, '
        r'"parallel_tool_calls": false, "response_format": {"type": "json_object"}}',
    ),
    "raw response format": (
        ONE,
        {"response_format": {"type": "json_object"}},
        r'{"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}], '
        r'"response_format": {"type": "json_object"}}',
    ),
    "both output limits": (
        ONE,
        {"max_tokens": 64, "max_completion_tokens": 128},
        r'{"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}], '
        r'"max_tokens": 128}',
    ),
}


@pytest.mark.parametrize("case", BEFORE_O03)
def test_a_compatible_server_gets_the_request_it_got_before(case: str) -> None:
    messages, kwargs, before = BEFORE_O03[case]

    params = prepare(_provider(), messages, **kwargs).params

    assert json.dumps(params) == before


# ---------------------------------------------------------------------------
# Pure functions
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
        wire = _messages_to_sdk([{"role": "user", "content": "Hi"}], system="Be helpful.")
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
        assert _messages_to_sdk(msgs, system="") == msgs

    def test_multiple_system_messages_without_explicit(self):
        msgs = [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "Hi"},
        ]
        wire = _messages_to_sdk(msgs)
        assert len([m for m in wire if m["role"] == "system"]) == 2

    def test_tool_result_uses_role_tool(self):
        wire = _messages_to_sdk([{"role": "tool", "content": "42", "tool_use_id": "call_1"}])
        assert wire[0] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}

    def test_no_system(self):
        wire = _messages_to_sdk([{"role": "user", "content": "Hi"}])
        assert len(wire) == 1
        assert wire[0]["role"] == "user"

    def test_assistant_with_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{"id": "tc_1", "name": "get_weather", "input": {"city": "NYC"}}],
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
        assert _tool_to_sdk(tool)["function"]["parameters"] == tool["input_schema"]

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
        assert parse_tool_args('{"city": "NYC"}') == {"city": "NYC"}

    def test_dict_passthrough(self):
        d = {"city": "NYC"}
        assert parse_tool_args(d) is d

    def test_invalid_json(self):
        assert parse_tool_args("not json") == {"_raw": "not json"}


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
        assert _build_output_schema_format(schema)["json_schema"]["strict"] is False

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
        usage = _extract_usage(
            SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        )
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
            prompt_tokens=100, completion_tokens=50, prompt_tokens_details=None
        )
        assert _extract_usage(sdk_usage).cache_read_tokens == 0

    def test_completion_tokens_remain_inclusive_of_reasoning(self):
        sdk_usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
        )
        assert _extract_usage(sdk_usage).output_tokens == 50

    def test_compatible_api_separate_reasoning_is_recovered_from_total(self):
        sdk_usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=180,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
        )
        assert _extract_usage(sdk_usage).output_tokens == 80


class TestParseSdkResponse:
    def test_text_response(self):
        comp = _sdk_completion(text="Hello!")
        r = _parse_sdk_response(comp, "llama3.2")
        assert r.text == "Hello!"
        assert r.stop_reason == "stop"
        assert isinstance(r, Response)
        # Usage is read by the adapter's usage(); the base puts it on the response.
        assert _provider().usage(comp) == Usage(input_tokens=10, output_tokens=5)

    def test_tool_calls(self):
        comp = _sdk_completion(
            text="",
            tool_calls=[{"id": "tc_1", "name": "get_weather", "arguments": '{"city": "NYC"}'}],
            finish_reason="tool_calls",
        )
        r = _parse_sdk_response(comp, "llama3.2")
        assert r.tool_calls == (ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}),)

    def test_empty_choices(self):
        comp = SimpleNamespace(choices=[], model="llama3.2", usage=None)
        assert _parse_sdk_response(comp, "llama3.2").text == ""

    async def test_cost_is_computed(self):
        client = _client(
            _sdk_completion(model="gpt-4o", prompt_tokens=1000, completion_tokens=500)
        )
        r = await complete(_provider("gpt-4o", client), HI)
        assert r.cost is not None
        assert r.cost > 0

    def test_structured_output_parsed(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        comp = _sdk_completion(text='{"name": "Alice", "age": 30}')
        r = _parse_sdk_response(comp, "llama3.2", output_schema=schema)
        assert r.parsed == {"name": "Alice", "age": 30}

    def test_raw_is_preserved(self):
        comp = _sdk_completion()
        assert _parse_sdk_response(comp, "llama3.2").raw is comp

    def test_reasoning_content_populates_thinking(self):
        comp = _sdk_completion(text="42", reasoning="Let me think.")
        r = _parse_sdk_response(comp, "llama3.2")
        assert r.thinking == (ThinkingBlock(text="Let me think."),)
        assert r.text == "42"

    def test_reasoning_alt_spelling(self):
        comp = _sdk_completion(text="42", reasoning="Hmm.", reasoning_field="reasoning")
        assert _parse_sdk_response(comp, "llama3.2").thinking == (ThinkingBlock(text="Hmm."),)

    def test_no_reasoning_empty_thinking(self):
        assert _parse_sdk_response(_sdk_completion(text="42"), "llama3.2").thinking == ()

    def test_a_completion_without_usage_reports_none(self):
        # OpenAI-compatible servers may omit usage; its cost is then unknown, never zero.
        comp = SimpleNamespace(choices=[], model="llama3.2", usage=None)
        assert _provider().usage(comp) is None


# ---------------------------------------------------------------------------
# The provider (mocked SDK client)
# ---------------------------------------------------------------------------


class TestComplete:
    @patch("ai_arch_toolkit.core._providers._openai_compatible.openai.AsyncOpenAI")
    def test_disables_hidden_sdk_retries(self, client_cls):
        OpenAICompatibleProvider("llama3.2", "not-needed", base_url=LOCAL)

        kwargs = client_cls.call_args.kwargs
        assert (kwargs["api_key"], kwargs["base_url"], kwargs["max_retries"]) == (
            "not-needed",
            LOCAL,
            0,
        )
        # The HTTP client marks the moment a request is handed to the transport.
        assert kwargs["http_client"].event_hooks["request"] == [on_request]

    async def test_complete_goes_through_chat_completions(self):
        client = _client(_sdk_completion(text="Hello!"))
        result = await complete(_provider(client=client), HI)
        assert result.text == "Hello!"
        client.chat.completions.create.assert_called_once()

    async def test_complete_surfaces_reasoning(self):
        client = _client(_sdk_completion(text="42", reasoning="Step by step."))
        result = await complete(_provider("gemma4:e4b", client), HI)
        assert result.thinking == (ThinkingBlock(text="Step by step."),)
        assert result.text == "42"

    async def test_complete_with_tools(self):
        calls = [{"id": "tc_1", "name": "search", "arguments": '{"q": "test"}'}]
        client = _client(_sdk_completion(text="", tool_calls=calls, finish_reason="tool_calls"))
        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        result = await complete(_provider(client=client), HI, tools=tools)
        assert result.has_tool_calls
        assert _sent(client)["tools"][0]["type"] == "function"

    async def test_system_passed_as_message(self):
        client = _client()
        await complete(_provider(client=client), HI, system="Be brief.")
        assert _sent(client)["messages"][:2] == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]

    def test_no_openai_model_rule_applies(self):
        # A model with OpenAI's name on another server: sampling stays, any SDK effort goes, and
        # thinking is not refused for a model OpenAI says does not reason.
        params = prepare(
            _provider("gpt-4o"), HI, max_tokens=64, thinking=True, temperature=0.0
        ).params
        assert params["max_tokens"] == 64
        assert "max_completion_tokens" not in params
        assert params["temperature"] == 0.0
        assert params["reasoning_effort"] == "high"

    def test_tools_while_reasoning(self):
        params = prepare(_provider("qwen3"), HI, tools=[LOOKUP], thinking=True).params
        assert params["reasoning_effort"] == "high"
        assert "tools" in params

    def test_an_effort_the_sdk_does_not_name_is_refused(self):
        with pytest.raises(RequestError, match="thinking_effort"):
            prepare(_provider(), HI, thinking=True, thinking_effort="ultra")

    @pytest.mark.parametrize(
        "server_tool", [web_search(), web_search(max_uses=2), code_execution()]
    )
    def test_server_tools_are_refused_before_sending(self, server_tool):
        # Chat Completions takes only function and custom tools (openai SDK types).
        with pytest.raises(RequestError, match="server tool"):
            prepare(_provider(), HI, tools=prepare_tools([server_tool]))

    async def test_thinking_budget_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prepare(_provider(), HI, thinking=True, thinking_budget=5000)
        assert len([w for w in caught if "thinking_budget" in str(w.message)]) == 1

    async def test_unknown_kwargs_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prepare(_provider(), HI, typo_param=True)
        assert len(caught) == 1
        assert "typo_param" in str(caught[0].message)


class TestErrors:
    @staticmethod
    def _status_error(cls: type[Exception], status: int, **headers: str) -> Exception:
        import httpx

        request = httpx.Request("POST", f"{LOCAL}/chat/completions")
        response = httpx.Response(status, headers=headers, request=request)
        return cls("error", response=response, body={"error": "boom"})

    async def test_rate_limit_error(self):
        import openai as openai_sdk

        client = AsyncMock()
        client.chat.completions.create.side_effect = self._status_error(
            openai_sdk.RateLimitError, 429, **{"retry-after": "3.0"}
        )
        with pytest.raises(RateLimitError) as caught:
            await complete(_provider(client=client), HI)
        assert (caught.value.status_code, caught.value.retry_after) == (429, 3.0)

    async def test_api_status_error(self):
        import openai as openai_sdk

        client = AsyncMock()
        client.chat.completions.create.side_effect = self._status_error(
            openai_sdk.APIStatusError, 500
        )
        with pytest.raises(APIError) as caught:
            await complete(_provider(client=client), HI)
        assert caught.value.status_code == 500

    @pytest.mark.parametrize(
        ("sdk_error", "expected"),
        [("APIConnectionError", ConnectionError), ("APITimeoutError", TimeoutError)],
    )
    async def test_network_failures_become_builtin_errors(self, sdk_error, expected):
        import httpx
        import openai as openai_sdk

        request = httpx.Request("POST", f"{LOCAL}/chat/completions")
        client = AsyncMock()
        client.chat.completions.create.side_effect = getattr(openai_sdk, sdk_error)(
            request=request
        )
        provider = _provider(client=client)
        with pytest.raises(expected):
            await complete(provider, HI)
        with pytest.raises(expected):
            await stream(provider, HI)


# ---------------------------------------------------------------------------
# Batch: OpenAI's Batch API on /v1/chat/completions
# ---------------------------------------------------------------------------


def _batch_client(*, endpoint: str = "/v1/chat/completions", output: str = "") -> MagicMock:
    client = MagicMock()
    client.files.create = AsyncMock(return_value=MagicMock(id="file-abc"))
    client.files.content = AsyncMock(return_value=MagicMock(text=output))
    client.batches.create = AsyncMock(return_value=MagicMock(id="batch-1"))
    client.batches.retrieve = AsyncMock(
        return_value=MagicMock(endpoint=endpoint, output_file_id="file-out", status="completed")
    )
    return client


class TestBatch:
    async def test_submits_to_chat_completions_with_the_compatible_request(self):
        client = _batch_client()
        provider = _provider(client=client)

        batch_id = await provider.batch_submit(
            [{"custom_id": "req-1", "messages": HI, "kwargs": {"max_tokens": 100}}]
        )

        assert batch_id == "batch-1"
        assert client.batches.create.call_args.kwargs["endpoint"] == "/v1/chat/completions"
        line = json.loads(client.files.create.call_args.kwargs["file"].read().decode())
        assert line["url"] == "/v1/chat/completions"
        assert line["body"] == {"model": "llama3.2", "messages": HI, "max_tokens": 100}

    async def test_results_are_read_as_chat_completions(self):
        body = {
            "id": "chatcmpl-1",
            "model": "llama3.2",
            "choices": [{"message": {"content": "Hi there"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        output = json.dumps({"custom_id": "req-1", "response": {"body": body}})
        provider = _provider(client=_batch_client(output=output))

        [result] = await provider.batch_results("batch-1")

        assert result.custom_id == "req-1"
        assert result.response is not None
        assert result.response.text == "Hi there"
        assert result.response.usage == Usage(input_tokens=10, output_tokens=5)
