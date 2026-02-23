"""Tests for _tools/_schema.py — schema inference from type hints."""

from __future__ import annotations

import dataclasses
import enum
from typing import Literal

from ai_arch_toolkit._tools._schema import (
    _get_summary,
    _hint_to_json_schema,
    _parse_param_descriptions,
    infer_schema,
)

# ---------------------------------------------------------------------------
# _hint_to_json_schema
# ---------------------------------------------------------------------------


class TestHintToJsonSchema:
    def test_str(self):
        schema, opt = _hint_to_json_schema(str)
        assert schema == {"type": "string"}
        assert opt is False

    def test_int(self):
        schema, _ = _hint_to_json_schema(int)
        assert schema == {"type": "integer"}

    def test_float(self):
        schema, _ = _hint_to_json_schema(float)
        assert schema == {"type": "number"}

    def test_bool(self):
        schema, _ = _hint_to_json_schema(bool)
        assert schema == {"type": "boolean"}

    def test_optional_str(self):
        schema, opt = _hint_to_json_schema(str | None)
        assert schema == {"type": "string"}
        assert opt is True

    def test_list_bare(self):
        schema, _ = _hint_to_json_schema(list)
        assert schema == {"type": "array"}

    def test_list_typed(self):
        schema, _ = _hint_to_json_schema(list[int])
        assert schema == {"type": "array", "items": {"type": "integer"}}

    def test_dict(self):
        schema, _ = _hint_to_json_schema(dict)
        assert schema == {"type": "object"}

    def test_literal_strings(self):
        schema, _ = _hint_to_json_schema(Literal["a", "b"])
        assert schema == {"type": "string", "enum": ["a", "b"]}

    def test_literal_ints(self):
        schema, _ = _hint_to_json_schema(Literal[1, 2, 3])
        assert schema == {"type": "integer", "enum": [1, 2, 3]}

    def test_enum(self):
        class Color(enum.Enum):
            RED = "red"
            BLUE = "blue"

        schema, _ = _hint_to_json_schema(Color)
        assert schema == {"type": "string", "enum": ["red", "blue"]}

    def test_int_enum(self):
        class Status(enum.IntEnum):
            OK = 200
            NOT_FOUND = 404

        schema, _ = _hint_to_json_schema(Status)
        assert schema == {"type": "integer", "enum": [200, 404]}

    def test_tuple_fixed(self):
        schema, _ = _hint_to_json_schema(tuple[str, int])
        assert schema == {
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "integer"}],
        }

    def test_tuple_variable(self):
        schema, _ = _hint_to_json_schema(tuple[str, ...])
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_dataclass(self):
        @dataclasses.dataclass
        class Point:
            x: float
            y: float

        schema, _ = _hint_to_json_schema(Point)
        assert schema["type"] == "object"
        assert "x" in schema["properties"]
        assert "y" in schema["properties"]

    def test_unknown_fallback(self):
        class Custom:
            pass

        schema, _ = _hint_to_json_schema(Custom)
        assert schema == {"type": "string"}


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------


class TestParseParamDescriptions:
    def test_basic_args(self):
        def fn(city: str, units: str = "metric"):
            """Get weather.

            Args:
                city: The city name.
                units: Temperature units.
            """

        result = _parse_param_descriptions(fn)
        assert result["city"] == "The city name."
        assert result["units"] == "Temperature units."

    def test_multiline_description(self):
        def fn(query: str):
            """Search.

            Args:
                query: The search query to use.
                    Can span multiple lines.
            """

        result = _parse_param_descriptions(fn)
        assert result["query"] == "The search query to use. Can span multiple lines."

    def test_no_docstring(self):
        def fn(x: int):
            pass

        assert _parse_param_descriptions(fn) == {}

    def test_returns_section_ends_args(self):
        def fn(x: int):
            """Do something.

            Args:
                x: A number.

            Returns:
                The result.
            """

        result = _parse_param_descriptions(fn)
        assert result["x"] == "A number."

    def test_type_annotation_in_doc(self):
        def fn(x):
            """Do something.

            Args:
                x (int): A number.
            """

        result = _parse_param_descriptions(fn)
        assert result["x"] == "A number."


class TestGetSummary:
    def test_basic(self):
        def fn():
            """Get the weather for a city."""

        assert _get_summary(fn) == "Get the weather for a city."

    def test_multiline_before_args(self):
        def fn():
            """Get weather.

            More details here.

            Args:
                city: The city.
            """

        result = _get_summary(fn)
        assert "Get weather." in result
        assert "More details here." in result
        assert "Args" not in result

    def test_no_docstring(self):
        def fn():
            pass

        assert _get_summary(fn) == ""


# ---------------------------------------------------------------------------
# infer_schema
# ---------------------------------------------------------------------------


class TestInferSchema:
    def test_basic_function(self):
        def get_weather(city: str, units: str = "metric") -> str:
            """Get the weather for a city.

            Args:
                city: The city name.
                units: Temperature units.
            """
            return f"Sunny in {city}"

        schema = infer_schema(get_weather)
        assert schema["name"] == "get_weather"
        assert schema["description"] == "Get the weather for a city."
        assert "input_schema" in schema
        props = schema["input_schema"]["properties"]
        assert "city" in props
        assert "units" in props
        assert props["city"]["description"] == "The city name."
        assert schema["input_schema"]["required"] == ["city"]

    def test_custom_name(self):
        def fn(x: int):
            """Do stuff."""

        schema = infer_schema(fn, name="custom_name")
        assert schema["name"] == "custom_name"

    def test_default_values_in_schema(self):
        def fn(x: int, y: int = 10):
            """A function."""

        schema = infer_schema(fn)
        props = schema["input_schema"]["properties"]
        assert props["y"]["default"] == 10
        assert "x" in schema["input_schema"]["required"]
        assert "y" not in schema["input_schema"]["required"]

    def test_optional_not_required(self):
        def fn(x: str, y: str | None = None):
            """A function."""

        schema = infer_schema(fn)
        assert schema["input_schema"]["required"] == ["x"]

    def test_overrides(self):
        def fn(x: str):
            """A function."""

        schema = infer_schema(fn, overrides={"x": {"description": "Overridden"}})
        assert schema["input_schema"]["properties"]["x"]["description"] == "Overridden"

    def test_no_type_hints(self):
        def fn(x, y):
            """A function."""

        schema = infer_schema(fn)
        props = schema["input_schema"]["properties"]
        assert props["x"] == {"type": "string"}
        assert props["y"] == {"type": "string"}

    def test_skips_self_cls(self):
        class Foo:
            def method(self, x: int):
                """A method."""

        schema = infer_schema(Foo.method)
        assert "self" not in schema["input_schema"]["properties"]
        assert "x" in schema["input_schema"]["properties"]

    def test_bool_default(self):
        def fn(verbose: bool = False):
            """A function."""

        schema = infer_schema(fn)
        props = schema["input_schema"]["properties"]
        assert props["verbose"]["default"] is False

    def test_none_default_serializable(self):
        def fn(tag: str | None = None):
            """A function."""

        schema = infer_schema(fn)
        props = schema["input_schema"]["properties"]
        assert props["tag"]["default"] is None

    def test_non_serializable_default_omitted(self):
        class Custom:
            pass

        sentinel = Custom()

        def fn(x: str = sentinel):  # type: ignore[assignment]
            """A function."""

        schema = infer_schema(fn)
        props = schema["input_schema"]["properties"]
        assert "default" not in props["x"]
