"""Tests for Response.to_message()."""

from __future__ import annotations

from ai_arch_toolkit.core._response import Response, ToolCall, Usage


class TestToMessage:
    def test_text_only(self):
        r = Response(text="Hello!", usage=Usage(input_tokens=10, output_tokens=5))
        msg = r.to_message()
        assert msg == {"role": "assistant", "content": "Hello!"}
        assert "tool_calls" not in msg

    def test_with_tool_calls(self):
        r = Response(
            text="Let me check.",
            tool_calls=(
                ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}),
                ToolCall(id="tc_2", name="get_time", input={"tz": "UTC"}),
            ),
            usage=Usage(input_tokens=20, output_tokens=15),
        )
        msg = r.to_message()
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check."
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0] == {
            "id": "tc_1",
            "name": "get_weather",
            "input": {"city": "NYC"},
        }
        assert msg["tool_calls"][1] == {
            "id": "tc_2",
            "name": "get_time",
            "input": {"tz": "UTC"},
        }

    def test_empty_text_with_tool_calls(self):
        r = Response(
            text="",
            tool_calls=(ToolCall(id="tc_1", name="search", input={"q": "test"}),),
        )
        msg = r.to_message()
        assert msg["content"] == ""
        assert len(msg["tool_calls"]) == 1

    def test_empty_response(self):
        r = Response()
        msg = r.to_message()
        assert msg == {"role": "assistant", "content": ""}
        assert "tool_calls" not in msg

    def test_input_dict_is_copied(self):
        """Mutating the message dict must not affect the frozen ToolCall."""
        tc = ToolCall(id="tc_1", name="search", input={"q": "test"})
        r = Response(text="", tool_calls=(tc,))
        msg = r.to_message()
        msg["tool_calls"][0]["input"]["q"] = "mutated"
        assert tc.input["q"] == "test"  # original unchanged

    def test_roundtrip_format_matches_content_helper(self):
        """to_message() format is compatible with _content.assistant()."""
        r = Response(text="Hello")
        msg = r.to_message()
        from ai_arch_toolkit.core._content import assistant

        assert msg == assistant("Hello")
