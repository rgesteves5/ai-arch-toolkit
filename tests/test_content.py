"""Tests for _content.py message constructors."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._content import assistant, system, tool_result, user


class TestSystem:
    def test_returns_dict_with_role_and_content(self):
        msg = system("You are helpful.")
        assert msg == {"role": "system", "content": "You are helpful."}


class TestUser:
    def test_string_content(self):
        msg = user("Hello")
        assert msg == {"role": "user", "content": "Hello"}

    def test_list_content_for_multimodal(self):
        parts = [{"type": "text", "text": "Hi"}, {"type": "image_url", "url": "http://x"}]
        msg = user(parts)
        assert msg == {"role": "user", "content": parts}


class TestAssistant:
    def test_returns_dict_with_role_and_content(self):
        msg = assistant("Sure!")
        assert msg == {"role": "assistant", "content": "Sure!"}


class TestToolResult:
    def test_includes_tool_use_id(self):
        msg = tool_result("result data", tool_use_id="call_123")
        assert msg["role"] == "tool"
        assert msg["content"] == "result data"
        assert msg["tool_use_id"] == "call_123"

    def test_includes_optional_name(self):
        msg = tool_result("result data", tool_use_id="call_123", name="get_weather")
        assert msg["name"] == "get_weather"


class TestToolResultValidation:
    def test_empty_tool_use_id_raises(self):
        with pytest.raises(ValueError, match="tool_use_id must be a non-empty string"):
            tool_result("result", tool_use_id="")
