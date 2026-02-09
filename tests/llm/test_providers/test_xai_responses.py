"""Tests for the xAI Responses API provider."""

from __future__ import annotations

from unittest.mock import MagicMock

from ai_arch_toolkit.llm._providers._xai_responses import XAIResponsesProvider
from ai_arch_toolkit.llm._types import Message, ServerTool
from tests.conftest import MockResponse

_TEXT_RESPONSE = {
    "id": "resp_123",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello from xAI!"}],
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
}


def _make_provider() -> XAIResponsesProvider:
    return XAIResponsesProvider("grok-3", "xai-test-key")


def test_url() -> None:
    provider = _make_provider()
    assert provider._url == "https://api.x.ai/v1/responses"


def test_complete_text(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    resp = provider.complete([Message("user", "Hi")])

    assert resp.text == "Hello from xAI!"
    assert resp.usage.input_tokens == 10
    assert resp.stop_reason == "completed"


def test_web_search_server_tool(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Search for news")],
        server_tools=[ServerTool(type="web_search", config={"allowed_domains": ["example.com"]})],
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    tools = payload["tools"]
    assert any(t["type"] == "web_search" for t in tools)
    ws_tool = next(t for t in tools if t["type"] == "web_search")
    assert ws_tool["allowed_domains"] == ["example.com"]


def test_x_search_server_tool(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Search X")],
        server_tools=[
            ServerTool(
                type="x_search",
                config={"date_range": {"start": "2025-01-01", "end": "2025-12-31"}},
            )
        ],
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    tools = payload["tools"]
    assert any(t["type"] == "x_search" for t in tools)


def test_code_execution_server_tool(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Run code")],
        server_tools=[ServerTool(type="code_execution", config={"pip_packages": ["numpy"]})],
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    tools = payload["tools"]
    ce_tool = next(t for t in tools if t["type"] == "code_execution")
    assert ce_tool["pip_packages"] == ["numpy"]


def test_headers() -> None:
    provider = _make_provider()
    assert provider._headers["Authorization"] == "Bearer xai-test-key"
