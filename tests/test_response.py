"""Tests for _response.py — Response, ToolCall, Usage, OutputSchema, ThinkingBlock."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._providers._base import _parse_retry_after
from ai_arch_toolkit.core._response import (
    OutputSchema,
    Response,
    SyncStreamResponse,
    ThinkingBlock,
    ToolCall,
    Usage,
    _resolve_output_schema,
)


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
        u = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_write_tokens=10,
            cache_read_tokens=5,
        )
        assert u.input_tokens == 100
        assert u.output_tokens == 50


class TestResponse:
    def test_defaults(self):
        r = Response()
        assert r.text == ""
        assert r.tool_calls == ()
        assert r.cost is None
        assert r.provider_cost is None
        assert r.stop_reason == ""
        assert r.model == ""
        assert r.raw is None

    def test_cost_none_when_unknown(self):
        r = Response()
        assert r.cost is None

    def test_cost_float_when_known(self):
        r = Response(cost=0.003, provider_cost=0.003)
        assert r.cost == 0.003
        assert r.provider_cost == 0.003

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

    def test_frozen(self):
        r = Response(text="Hi")
        with pytest.raises(AttributeError):
            r.text = "Bye"  # type: ignore[misc]

    def test_thinking_field_defaults(self):
        r = Response()
        assert r.thinking == ()
        assert r.parsed is None

    def test_thinking_field_populated(self):
        blocks = (ThinkingBlock(text="Let me think..."),)
        r = Response(text="Answer", thinking=blocks)
        assert len(r.thinking) == 1
        assert r.thinking[0].text == "Let me think..."

    def test_parsed_field(self):
        r = Response(text='{"name": "Alice"}', parsed={"name": "Alice"})
        assert r.parsed == {"name": "Alice"}


class TestThinkingBlock:
    def test_fields(self):
        tb = ThinkingBlock(text="reasoning here")
        assert tb.text == "reasoning here"

    def test_frozen(self):
        tb = ThinkingBlock(text="reasoning")
        with pytest.raises(AttributeError):
            tb.text = "other"  # type: ignore[misc]


class TestOutputSchema:
    def test_fields(self):
        s = OutputSchema(
            name="Person",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        assert s.name == "Person"
        assert s.strict is True

    def test_strict_false(self):
        s = OutputSchema(name="X", schema={"type": "object"}, strict=False)
        assert s.strict is False

    def test_frozen(self):
        s = OutputSchema(name="X", schema={"type": "object"})
        with pytest.raises(AttributeError):
            s.name = "Y"  # type: ignore[misc]


class TestOutputSchemaValidation:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            OutputSchema(name="", schema={"type": "object"})

    def test_empty_schema_raises(self):
        with pytest.raises(ValueError, match="schema must be a non-empty dict"):
            OutputSchema(name="Valid", schema={})


class TestResolveOutputSchema:
    def test_passthrough_output_schema(self):
        s = OutputSchema(name="X", schema={"type": "object"})
        assert _resolve_output_schema(s) is s

    def test_pydantic_model(self):
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class Person(BaseModel):
            name: str
            age: int

        result = _resolve_output_schema(Person)
        assert isinstance(result, OutputSchema)
        assert result.name == "Person"
        assert "properties" in result.schema

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected OutputSchema or Pydantic"):
            _resolve_output_schema("not a schema")

    def test_regular_class_raises(self):
        class NotPydantic:
            name: str

        with pytest.raises(TypeError, match="Expected OutputSchema or Pydantic"):
            _resolve_output_schema(NotPydantic)

    def test_without_pydantic(self, monkeypatch):
        """When pydantic is not installed, non-OutputSchema input raises TypeError."""
        import builtins

        real_import = builtins.__import__

        def _block_pydantic(name, *args, **kwargs):
            if name == "pydantic":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_pydantic)

        with pytest.raises(TypeError, match="Expected OutputSchema or Pydantic"):
            _resolve_output_schema({"not": "a schema"})


class TestSyncStreamResponseContextManager:
    def test_enter_exit(self):
        def _make_iter():
            yield "hello"
            yield " world"

        def _finalizer(text):
            return Response(text=text)

        stream = SyncStreamResponse(_make_iter(), _finalizer)
        with stream:
            for _chunk in stream:
                break  # early exit
        assert stream.response is not None
        assert stream.response.text == "hello"


class TestParseRetryAfter:
    def test_float(self):
        assert _parse_retry_after("5.0") == 5.0

    def test_int(self):
        assert _parse_retry_after("3") == 3.0

    def test_none(self):
        assert _parse_retry_after(None) is None

    def test_non_numeric(self):
        assert _parse_retry_after("Thu, 01 Jan 2026 00:00:00 GMT") is None

    def test_empty_string(self):
        assert _parse_retry_after("") is None
