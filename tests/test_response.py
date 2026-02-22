"""Tests for _response.py — Response, ToolCall, Usage."""

from __future__ import annotations

import pytest

from ai_arch_toolkit._response import Response, ToolCall, Usage


class TestToolCall:
    def test_fields(self):
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        assert tc.id == "tc_1"
        assert tc.name == "get_weather"
        assert tc.input == {"city": "NYC"}

    def test_frozen(self):
        tc = ToolCall(id="tc_1", name="get_weather", input={})
        with pytest.raises(AttributeError):
            tc.name = "other"  # type: ignore[misc]


class TestUsage:
    def test_defaults(self):
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cache_write_tokens == 0
        assert u.cache_read_tokens == 0

    def test_custom_values(self):
        u = Usage(input_tokens=100, output_tokens=50, cache_write_tokens=10, cache_read_tokens=5)
        assert u.input_tokens == 100
        assert u.output_tokens == 50


class TestResponse:
    def test_defaults(self):
        r = Response()
        assert r.text == ""
        assert r.tool_calls == ()
        assert r.cost == 0.0
        assert r.cost_estimated is False
        assert r.stop_reason == ""
        assert r.model == ""
        assert r.raw is None

    def test_cost_estimated_flag(self):
        r1 = Response(cost=0.003, cost_estimated=True)
        assert r1.cost_estimated is True
        r2 = Response(cost=0.0, cost_estimated=False)
        assert r2.cost_estimated is False

    def test_shortcut_tokens(self):
        r = Response(usage=Usage(input_tokens=100, output_tokens=50))
        assert r.tokens == 150
        assert r.input_tokens == 100
        assert r.output_tokens == 50

    def test_has_tool_calls(self):
        r1 = Response()
        assert not r1.has_tool_calls
        r2 = Response(tool_calls=(ToolCall(id="1", name="fn", input={}),))
        assert r2.has_tool_calls

    def test_autocomplete_path(self):
        tc = ToolCall(id="1", name="get_weather", input={"city": "NYC"})
        r = Response(tool_calls=(tc,))
        assert r.tool_calls[0].name == "get_weather"

    def test_str(self):
        r = Response(text="Hello world")
        assert str(r) == "Hello world"

    def test_repr_text_only(self):
        r = Response(text="Hi")
        assert repr(r) == "Response(text='Hi')"

    def test_repr_with_tools(self):
        tc = ToolCall(id="1", name="search", input={})
        r = Response(text="", tool_calls=(tc,))
        assert "search" in repr(r)

    def test_bool_empty(self):
        assert not Response()

    def test_bool_text(self):
        assert Response(text="Hi")

    def test_bool_tools(self):
        assert Response(tool_calls=(ToolCall(id="1", name="fn", input={}),))

    def test_contains(self):
        r = Response(text="hello world")
        assert "world" in r
        assert "xyz" not in r

    def test_add(self):
        r = Response(text="Hello")
        assert r + " world" == "Hello world"

    def test_radd(self):
        r = Response(text="world")
        assert "Hello " + r == "Hello world"

    def test_frozen(self):
        r = Response(text="Hi")
        with pytest.raises(AttributeError):
            r.text = "Bye"  # type: ignore[misc]
