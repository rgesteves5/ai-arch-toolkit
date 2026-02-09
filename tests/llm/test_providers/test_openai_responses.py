"""Tests for the OpenAI Responses API provider."""

from __future__ import annotations

from unittest.mock import MagicMock

from ai_arch_toolkit.llm._providers._openai_responses import OpenAIResponsesProvider
from ai_arch_toolkit.llm._types import (
    JsonSchema,
    Message,
    ServerTool,
    ThinkingConfig,
    Tool,
    ToolCall,
    ToolResult,
)
from tests.conftest import MockResponse

_TEXT_RESPONSE = {
    "id": "resp_123",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello!"}],
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
}

_TOOL_CALL_RESPONSE = {
    "id": "resp_456",
    "status": "completed",
    "output": [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
        }
    ],
    "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
}


def _make_provider() -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider("gpt-4o", "sk-test")


# --- Basic text completion ---


def test_complete_text(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    resp = provider.complete([Message("user", "Hello")])

    assert resp.text == "Hello!"
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 5
    assert resp.usage.total_tokens == 15
    assert resp.stop_reason == "completed"
    assert resp.tool_calls == ()


# --- System instruction ---


def test_system_uses_instructions_key(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")], system="Be helpful")

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["instructions"] == "Be helpful"
    # Should NOT use messages-style system
    assert "messages" not in payload
    assert "system" not in payload


# --- Tool call response ---


def test_complete_with_tools(mock_post: MagicMock, weather_tool: Tool) -> None:
    mock_post.return_value = MockResponse(json_data=_TOOL_CALL_RESPONSE)
    provider = _make_provider()
    resp = provider.complete([Message("user", "Weather?")], tools=[weather_tool])

    assert len(resp.tool_calls) == 1
    assert isinstance(resp.tool_calls, tuple)
    tc = resp.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris"}


# --- Tool result wire format ---


def test_tool_result_wire_format(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
    provider.complete(
        [
            Message("user", "Weather?"),
            Message("assistant", tool_calls=(tc,)),
            ToolResult(tool_call_id="call_1", name="get_weather", content='{"temp": 20}'),
        ]
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    input_items = payload["input"]
    assert isinstance(input_items, list)

    # Find the function_call_output item
    tool_result_items = [i for i in input_items if i.get("type") == "function_call_output"]
    assert len(tool_result_items) == 1
    assert tool_result_items[0]["call_id"] == "call_1"
    assert tool_result_items[0]["output"] == '{"temp": 20}'


# --- JSON schema ---


def test_json_schema_in_payload(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    schema = JsonSchema(
        name="person",
        schema={"type": "object", "properties": {"name": {"type": "string"}}},
    )
    provider.complete([Message("user", "Extract")], json_schema=schema)

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    text_format = payload["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["name"] == "person"
    assert text_format["strict"] is True
    assert text_format["schema"] == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }


# --- Thinking / reasoning ---


def test_thinking_config_reasoning_effort(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Think")], thinking=ThinkingConfig(effort="high"))

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["reasoning"]["effort"] == "high"


def test_thinking_config_with_budget_tokens(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Think")],
        thinking=ThinkingConfig(effort="high", budget_tokens=16384),
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["reasoning"]["effort"] == "high"
    assert payload["reasoning"]["budget_tokens"] == 16384


# --- Server tools ---


def test_server_tools_merged_into_tools(mock_post: MagicMock, weather_tool: Tool) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Search the web")],
        tools=[weather_tool],
        server_tools=[ServerTool(type="web_search")],
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    tools = payload["tools"]
    assert len(tools) == 2

    # First tool is the function tool
    assert tools[0]["type"] == "function"
    assert tools[0]["name"] == "get_weather"

    # Second tool is the server tool
    assert tools[1]["type"] == "web_search"


def test_server_tools_only(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Search the web")],
        server_tools=[ServerTool(type="web_search")],
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    tools = payload["tools"]
    assert len(tools) == 1
    assert tools[0]["type"] == "web_search"


# --- previous_response_id ---


def test_previous_response_id(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Continue")],
        previous_response_id="resp_abc",
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["previous_response_id"] == "resp_abc"


# --- URL ---


def test_url() -> None:
    provider = _make_provider()
    assert provider._url == "https://api.openai.com/v1/responses"


def test_custom_base_url() -> None:
    provider = OpenAIResponsesProvider("gpt-4o", "sk-test", base_url="https://custom.api.com")
    assert provider._url == "https://custom.api.com/v1/responses"


# --- Stream events ---


def test_stream_events_text_deltas(mock_post: MagicMock) -> None:
    completed = (
        "data: "
        '{"type":"response.completed","response":'
        '{"usage":{"input_tokens":10,'
        '"output_tokens":5,"total_tokens":15}}}'
    )
    mock_post.return_value = MockResponse(
        lines=[
            'data: {"type":"response.output_text.delta","delta":"Hello"}',
            'data: {"type":"response.output_text.delta","delta":" world"}',
            completed,
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Hi")]))

    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) == 2
    assert text_events[0].text == "Hello"
    assert text_events[1].text == " world"


def test_stream_events_function_call(mock_post: MagicMock) -> None:
    delta1 = (
        "data: "
        '{"type":"response.function_call_arguments.delta",'
        '"call_id":"call_1","name":"get_weather",'
        '"delta":"{\\"city\\""}'
    )
    delta2 = (
        "data: "
        '{"type":"response.function_call_arguments.delta",'
        '"call_id":"call_1",'
        '"delta":": \\"Paris\\"}"}'
    )
    done = (
        "data: "
        '{"type":"response.function_call_arguments.done",'
        '"call_id":"call_1","name":"get_weather",'
        '"arguments":"{\\"city\\": \\"Paris\\"}"}'
    )
    completed = (
        "data: "
        '{"type":"response.completed","response":'
        '{"usage":{"input_tokens":20,'
        '"output_tokens":10,"total_tokens":30}}}'
    )
    mock_post.return_value = MockResponse(lines=[delta1, delta2, done, completed])
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Weather?")]))

    tc_events = [e for e in events if e.type == "tool_call"]
    assert len(tc_events) == 1
    assert tc_events[0].tool_call.id == "call_1"
    assert tc_events[0].tool_call.name == "get_weather"
    assert tc_events[0].tool_call.arguments == {"city": "Paris"}


def test_stream_events_response_completed(mock_post: MagicMock) -> None:
    completed = (
        "data: "
        '{"type":"response.completed","response":'
        '{"usage":{"input_tokens":10,'
        '"output_tokens":5,"total_tokens":15}}}'
    )
    mock_post.return_value = MockResponse(
        lines=[
            'data: {"type":"response.output_text.delta","delta":"Hi"}',
            completed,
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Hi")]))

    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 1
    assert usage_events[0].usage.input_tokens == 10
    assert usage_events[0].usage.output_tokens == 5
    assert usage_events[0].usage.total_tokens == 15

    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1


# --- Single-message string shortcut ---


def test_single_user_message_string_input(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hello")])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    # Single plain-text user message should be sent as a string shortcut
    assert payload["input"] == "Hello"


# --- Timeout ---


def test_complete_passes_timeout(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")], timeout=30)

    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs.get("timeout") == 30
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "timeout" not in payload
