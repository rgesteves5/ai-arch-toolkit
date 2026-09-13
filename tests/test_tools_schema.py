"""Tests for _tools/_schema.py — schema inference from type hints."""

from __future__ import annotations

import dataclasses
import enum
import json
import warnings
from types import SimpleNamespace
from typing import Any, Literal, Optional, TypedDict, Union
from unittest.mock import AsyncMock, MagicMock

from google.genai import types as genai_types
from pydantic import BaseModel, Field

from ai_arch_toolkit.core._providers import _anthropic, _gemini, _openai, _xai
from ai_arch_toolkit.core._tools import prepare_tools
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._schema import (
    _get_summary,
    _hint_to_json_schema,
    _parse_param_descriptions,
    infer_schema,
)

INT_OR_STR = [{"type": "integer"}, {"type": "string"}]


class _Address(BaseModel):
    city: str


class _Person(BaseModel):
    name: str
    address: _Address = Field(description="Where the person lives.")


class _Node(BaseModel):
    label: str
    children: list[_Node] = []


class _Filter(TypedDict):
    field: str
    value: int | str


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

    def test_any_and_object_accept_every_json_value(self):
        assert _hint_to_json_schema(Any) == ({}, False)
        assert _hint_to_json_schema(object) == ({}, False)

    def test_containers_and_optionals_of_any(self):
        assert _hint_to_json_schema(list[Any]) == ({"type": "array", "items": {}}, False)
        assert _hint_to_json_schema(Any | None) == ({}, True)
        assert _hint_to_json_schema(dict[str, Any]) == ({"type": "object"}, False)

    def test_multi_type_union_is_required_any_of(self):
        assert _hint_to_json_schema(int | str) == ({"anyOf": INT_OR_STR}, False)
        # typing.Union spelling (a distinct origin before Python 3.14)
        typing_union = Union[int, str]  # noqa: UP007
        assert _hint_to_json_schema(typing_union) == ({"anyOf": INT_OR_STR}, False)

    def test_multi_type_union_with_none_is_optional_any_of(self):
        assert _hint_to_json_schema(int | str | None) == ({"anyOf": INT_OR_STR}, True)
        # typing caches unions by equality, and Union[int, str] == Union[str, int], so the member
        # order of a typing.Union depends on which spelling was built first in the process.
        typing_optional = Optional[Union[int, str]]  # noqa: UP007, UP045
        schema, optional = _hint_to_json_schema(typing_optional)
        assert optional is True
        assert sorted(schema["anyOf"], key=repr) == sorted(INT_OR_STR, key=repr)

    def test_multi_type_union_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _hint_to_json_schema(int | str)

    def test_union_members_keep_their_own_schema(self):
        schema, _ = _hint_to_json_schema(int | list[str] | Literal["auto"])
        assert schema == {
            "anyOf": [
                {"type": "integer"},
                {"type": "array", "items": {"type": "string"}},
                {"type": "string", "enum": ["auto"]},
            ]
        }

    def test_union_members_with_identical_schemas_collapse(self):
        schema, opt = _hint_to_json_schema(list[str] | tuple[str, ...])
        assert schema == {"type": "array", "items": {"type": "string"}}
        assert opt is False

    def test_union_inside_list(self):
        schema, opt = _hint_to_json_schema(list[int | str])
        assert schema == {"type": "array", "items": {"anyOf": INT_OR_STR}}
        assert opt is False

    def test_dataclass_union_fields(self):
        @dataclasses.dataclass
        class Query:
            term: int | str
            limit: int | str | None = None

        schema, _ = _hint_to_json_schema(Query)
        assert schema["properties"] == {
            "term": {"anyOf": INT_OR_STR},
            "limit": {"anyOf": INT_OR_STR},
        }
        assert schema["required"] == ["term"]

    def test_typeddict_union_field(self):
        schema, _ = _hint_to_json_schema(_Filter)
        assert schema["properties"] == {
            "field": {"type": "string"},
            "value": {"anyOf": INT_OR_STR},
        }
        assert sorted(schema["required"]) == ["field", "value"]


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

    def test_any_parameter_keeps_only_its_description(self):
        def store(value: Any, items: list[Any]) -> str:
            """Store a value.

            Args:
                value: Anything JSON can hold.
                items: Values of any type.
            """
            return ""

        props = infer_schema(store)["input_schema"]["properties"]
        assert props["value"] == {"description": "Anything JSON can hold."}
        assert props["items"] == {
            "type": "array",
            "items": {},
            "description": "Values of any type.",
        }

    def test_every_adapter_accepts_an_untyped_property(self):
        definition = {
            "name": "store",
            "description": "Store a value.",
            "input_schema": {
                "type": "object",
                "properties": {"value": {}, "items": {"type": "array", "items": {}}},
                "required": ["value"],
            },
        }

        assert _anthropic._tool_to_sdk(definition)["input_schema"]["properties"]["value"] == {}
        assert (
            _openai._tool_to_sdk(definition)["function"]["parameters"]["properties"]["value"] == {}
        )
        declaration = _gemini._tool_to_sdk(definition)
        assert declaration.parameters is not None
        assert declaration.parameters.properties is not None
        assert "value" in declaration.parameters.properties
        assert _xai._tool_to_sdk(definition) is not None

    def test_pydantic_model_references_are_inlined(self):
        def save(person: _Person) -> str:
            """Save a person."""
            return ""

        input_schema = infer_schema(save)["input_schema"]
        person = input_schema["properties"]["person"]

        assert "$defs" not in input_schema and "$defs" not in person
        assert "$ref" not in json.dumps(input_schema)
        assert person["properties"]["address"]["properties"]["city"] == {
            "title": "City",
            "type": "string",
        }
        assert person["properties"]["address"]["description"] == "Where the person lives."

    def test_recursive_model_definitions_are_hoisted_to_the_tool_root(self):
        def store(root: _Node) -> str:
            """Store a tree."""
            return ""

        input_schema = infer_schema(store)["input_schema"]
        root = input_schema["properties"]["root"]

        assert "$defs" not in root
        assert root["properties"]["children"]["items"] == {"$ref": "#/$defs/_Node"}
        assert input_schema["$defs"]["_Node"]["properties"]["label"] == {
            "title": "Label",
            "type": "string",
        }

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

    def test_multi_type_union_parameter_is_required(self):
        def procura(consulta: int | str):
            """Search by id or free text."""

        input_schema = infer_schema(procura)["input_schema"]
        assert input_schema["properties"] == {"consulta": {"anyOf": INT_OR_STR}}
        assert input_schema["required"] == ["consulta"]

    def test_optional_multi_type_union_parameter(self):
        def fn(q: int | str | None = None):
            """A function."""

        input_schema = infer_schema(fn)["input_schema"]
        assert input_schema["properties"] == {"q": {"anyOf": INT_OR_STR, "default": None}}
        assert input_schema["required"] == []

    def test_varargs_never_enter_schema(self):
        def fn(a: int, *args: int, **kwargs: str):
            """A function."""

        input_schema = infer_schema(fn)["input_schema"]
        assert input_schema["properties"] == {"a": {"type": "integer"}}
        assert input_schema["required"] == ["a"]

    def test_untyped_varargs_never_enter_schema(self):
        def flexivel(a, *args, **kwargs):
            """A function."""

        input_schema = infer_schema(flexivel)["input_schema"]
        assert input_schema["properties"] == {"a": {"type": "string"}}
        assert input_schema["required"] == ["a"]

    def test_keyword_only_parameters_after_varargs_are_kept(self):
        def fn(*paths: str, mode: str, limit: int = 5):
            """A function."""

        input_schema = infer_schema(fn)["input_schema"]
        assert input_schema["properties"] == {
            "mode": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        }
        assert input_schema["required"] == ["mode"]


# ---------------------------------------------------------------------------
# anyOf reaches every provider adapter intact
# ---------------------------------------------------------------------------


@tool
def lookup(query: int | str, tags: list[int | str] | None = None) -> str:
    """Look up a record.

    Args:
        query: Record id or free-text query.
        tags: Tags to filter by.
    """
    return f"{query}:{tags}"


QUERY_PROPERTY = {"anyOf": INT_OR_STR, "description": "Record id or free-text query."}
TAGS_PROPERTY = {
    "type": "array",
    "items": {"anyOf": INT_OR_STR},
    "description": "Tags to filter by.",
    "default": None,
}
GEMINI_INT_OR_STR = [genai_types.Type.INTEGER, genai_types.Type.STRING]


def _lookup_definition() -> dict:
    definitions = prepare_tools([lookup])
    assert definitions is not None
    return definitions[0]


def _variant_types(schema: genai_types.Schema | None) -> list[genai_types.Type | None]:
    assert schema is not None
    assert schema.any_of is not None
    return [variant.type for variant in schema.any_of]


def _gemini_text_response() -> SimpleNamespace:
    part = SimpleNamespace(text="ok", thought=False, function_call=None)
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason="STOP")
    usage = SimpleNamespace(
        prompt_token_count=1,
        candidates_token_count=1,
        cached_content_token_count=0,
        thoughts_token_count=0,
        tool_use_prompt_token_count=0,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


class TestAnyOfReachesProviderAdapters:
    def test_provider_definition_uses_any_of(self):
        input_schema = _lookup_definition()["input_schema"]
        assert input_schema["properties"] == {"query": QUERY_PROPERTY, "tags": TAGS_PROPERTY}
        assert input_schema["required"] == ["query"]

    def test_anthropic_sends_any_of(self):
        sdk_tool = _anthropic._tool_to_sdk(_lookup_definition())
        assert sdk_tool["input_schema"]["properties"]["query"] == QUERY_PROPERTY
        assert sdk_tool["input_schema"]["properties"]["tags"] == TAGS_PROPERTY

    def test_openai_sends_any_of(self):
        sdk_tool = _openai._tool_to_sdk(_lookup_definition())
        assert sdk_tool["function"]["parameters"]["properties"]["query"] == QUERY_PROPERTY
        assert sdk_tool["function"]["parameters"]["properties"]["tags"] == TAGS_PROPERTY

    def test_xai_sends_any_of(self):
        sdk_tool = _xai._tool_to_sdk(_lookup_definition())
        parameters = json.loads(sdk_tool.function.parameters)
        assert parameters["properties"]["query"] == QUERY_PROPERTY
        assert parameters["properties"]["tags"] == TAGS_PROPERTY

    def test_gemini_declaration_preserves_any_of_variants(self):
        parameters = _gemini._tool_to_sdk(_lookup_definition()).parameters
        assert parameters is not None
        assert parameters.properties is not None
        assert parameters.required == ["query"]
        query = parameters.properties["query"]
        assert _variant_types(query) == GEMINI_INT_OR_STR
        assert query.description == "Record id or free-text query."
        tags = parameters.properties["tags"]
        assert tags.type == genai_types.Type.ARRAY
        assert _variant_types(tags.items) == GEMINI_INT_OR_STR

    async def test_gemini_request_carries_any_of(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_gemini_text_response())
        provider = _gemini.GeminiProvider("gemini-2.5-flash", "test-key")
        provider._client = mock_client

        await provider.complete([{"role": "user", "content": "Hi"}], tools=[_lookup_definition()])

        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        parameters = config.tools[0].function_declarations[0].parameters
        assert _variant_types(parameters.properties["query"]) == GEMINI_INT_OR_STR
        assert _variant_types(parameters.properties["tags"].items) == GEMINI_INT_OR_STR
