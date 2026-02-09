"""Tests for ToolRegistry and @tool decorator."""

from __future__ import annotations

import asyncio
import dataclasses
import enum
from typing import Literal, TypedDict

import pytest

from ai_arch_toolkit.llm._types import Tool, ToolCall
from ai_arch_toolkit.tools._decorator import tool
from ai_arch_toolkit.tools._registry import ToolRegistry, ValidationError

# Module-level types needed for get_type_hints() resolution with PEP 563


@dataclasses.dataclass
class _Address:
    street: str
    city: str
    zip_code: str = ""


@dataclasses.dataclass
class _Item:
    name: str
    qty: int


class _Config(TypedDict, total=False):
    name: str
    value: int


class _FakeModel:
    @classmethod
    def model_json_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }


# --- ToolRegistry tests ---


def test_register_and_execute():
    reg = ToolRegistry()
    tool_def = Tool(
        name="add",
        description="Add two numbers",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    reg.register("add", lambda a, b: a + b, tool_def)
    result = reg.execute(ToolCall(id="1", name="add", arguments={"a": 2, "b": 3}))
    assert result == "5"


def test_execute_string_passthrough():
    reg = ToolRegistry()
    tool_def = Tool(name="greet", description="", parameters={})
    reg.register("greet", lambda name: f"Hello, {name}!", tool_def)
    result = reg.execute(ToolCall(id="1", name="greet", arguments={"name": "Alice"}))
    assert result == "Hello, Alice!"


def test_execute_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="Unknown tool"):
        reg.execute(ToolCall(id="1", name="nope", arguments={}))


def test_definitions():
    reg = ToolRegistry()
    t1 = Tool(name="a", description="A", parameters={})
    t2 = Tool(name="b", description="B", parameters={})
    reg.register("a", lambda: None, t1)
    reg.register("b", lambda: None, t2)
    assert reg.definitions() == [t1, t2]


def test_contains_and_len():
    reg = ToolRegistry()
    assert len(reg) == 0
    assert "foo" not in reg
    reg.register("foo", lambda: None, Tool(name="foo", description="", parameters={}))
    assert len(reg) == 1
    assert "foo" in reg


def test_async_execute_sync_fn():
    reg = ToolRegistry()
    tool_def = Tool(name="double", description="", parameters={})
    reg.register("double", lambda x: x * 2, tool_def)
    result = asyncio.run(reg.async_execute(ToolCall(id="1", name="double", arguments={"x": 5})))
    assert result == "10"


def test_async_execute_async_fn():
    reg = ToolRegistry()
    tool_def = Tool(name="double", description="", parameters={})

    async def async_double(x: int) -> int:
        return x * 2

    reg.register("double", async_double, tool_def)
    result = asyncio.run(reg.async_execute(ToolCall(id="1", name="double", arguments={"x": 7})))
    assert result == "14"


def test_async_execute_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="Unknown tool"):
        asyncio.run(reg.async_execute(ToolCall(id="1", name="nope", arguments={})))


# --- @tool decorator tests ---


def test_decorator_schema_generation():
    @tool
    def get_weather(city: str, units: str = "celsius") -> str:
        """Get the weather for a city."""
        return f"{city}: 22{units[0]}"

    t = get_weather.__tool__
    assert t.name == "get_weather"
    assert t.description == "Get the weather for a city."
    assert t.parameters["properties"]["city"] == {"type": "string"}
    assert t.parameters["properties"]["units"] == {"type": "string"}
    assert t.parameters["required"] == ["city"]


def test_decorator_with_registry():
    reg = ToolRegistry()

    @tool(registry=reg)
    def ping() -> str:
        """Ping."""
        return "pong"

    assert "ping" in reg
    assert len(reg) == 1
    assert reg.definitions()[0].name == "ping"


def test_decorator_custom_name():
    @tool(name="custom_name")
    def my_fn() -> str:
        """Doc."""
        return "ok"

    assert my_fn.__tool__.name == "custom_name"


def test_decorator_preserves_function():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert add(1, 2) == 3
    assert add.__name__ == "add"


def test_decorator_type_mapping():
    @tool
    def typed(s: str, i: int, f: float, b: bool) -> str:
        """Typed fn."""
        return ""

    props = typed.__tool__.parameters["properties"]
    assert props["s"] == {"type": "string"}
    assert props["i"] == {"type": "integer"}
    assert props["f"] == {"type": "number"}
    assert props["b"] == {"type": "boolean"}


# --- Phase 1: @tool type coverage tests ---


def test_decorator_list_type():
    @tool
    def fn(items: list) -> str:
        """Doc."""
        return ""

    assert fn.__tool__.parameters["properties"]["items"] == {"type": "array"}


def test_decorator_list_str_type():
    @tool
    def fn(names: list[str]) -> str:
        """Doc."""
        return ""

    assert fn.__tool__.parameters["properties"]["names"] == {
        "type": "array",
        "items": {"type": "string"},
    }


def test_decorator_list_int_type():
    @tool
    def fn(numbers: list[int]) -> str:
        """Doc."""
        return ""

    assert fn.__tool__.parameters["properties"]["numbers"] == {
        "type": "array",
        "items": {"type": "integer"},
    }


def test_decorator_dict_type():
    @tool
    def fn(data: dict) -> str:
        """Doc."""
        return ""

    assert fn.__tool__.parameters["properties"]["data"] == {"type": "object"}


def test_decorator_optional_not_required():
    @tool
    def fn(name: str, age: int | None = None) -> str:
        """Doc."""
        return ""

    t = fn.__tool__
    assert t.parameters["required"] == ["name"]
    assert t.parameters["properties"]["age"] == {"type": "integer"}


def test_decorator_optional_no_default_not_required():
    """Optional params without a default are still not in required."""

    @tool
    def fn(name: str, tag: str | None) -> str:
        """Doc."""
        return ""

    t = fn.__tool__
    assert t.parameters["required"] == ["name"]
    assert t.parameters["properties"]["tag"] == {"type": "string"}


def test_decorator_unknown_type_fallback():
    class Custom:
        pass

    @tool
    def fn(x: Custom) -> str:
        """Doc."""
        return ""

    assert fn.__tool__.parameters["properties"]["x"] == {"type": "string"}


# --- Phase 2: Docstring parameter descriptions tests ---


def test_decorator_google_docstring_params():
    @tool
    def get_weather(city: str, units: str = "celsius") -> str:
        """Get current weather.

        Args:
            city: The city name to look up.
            units: Temperature units.
        """
        return ""

    t = get_weather.__tool__
    assert t.description == "Get current weather."
    props = t.parameters["properties"]
    assert props["city"] == {
        "type": "string",
        "description": "The city name to look up.",
    }
    assert props["units"] == {
        "type": "string",
        "description": "Temperature units.",
    }


def test_decorator_docstring_summary_only():
    @tool
    def fn(x: str) -> str:
        """Just a summary."""
        return ""

    assert fn.__tool__.description == "Just a summary."
    assert fn.__tool__.parameters["properties"]["x"] == {"type": "string"}


def test_decorator_no_docstring():
    @tool
    def fn(x: str) -> str:
        return ""

    assert fn.__tool__.description == ""
    assert fn.__tool__.parameters["properties"]["x"] == {"type": "string"}


def test_decorator_multiline_param_description():
    @tool
    def fn(city: str) -> str:
        """Search.

        Args:
            city: The city name
                to look up in the database.
        """
        return ""

    desc = fn.__tool__.parameters["properties"]["city"]["description"]
    assert desc == "The city name to look up in the database."


def test_decorator_docstring_with_returns_section():
    @tool
    def fn(x: str) -> str:
        """Do something.

        Args:
            x: The input value.

        Returns:
            The output value.
        """
        return ""

    t = fn.__tool__
    assert t.description == "Do something."
    assert t.parameters["properties"]["x"] == {
        "type": "string",
        "description": "The input value.",
    }


# --- Phase 3: Extended type support tests ---


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Priority(enum.IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


def test_decorator_enum_type():
    @tool
    def fn(color: Color) -> str:
        """Doc."""
        return ""

    assert fn.__tool__.parameters["properties"]["color"] == {
        "type": "string",
        "enum": ["red", "green", "blue"],
    }


def test_decorator_enum_int_type():
    @tool
    def fn(priority: Priority) -> str:
        """Doc."""
        return ""

    assert fn.__tool__.parameters["properties"]["priority"] == {
        "type": "integer",
        "enum": [1, 2, 3],
    }


def test_decorator_literal_str():
    @tool
    def fn(mode: Literal["fast", "slow"]) -> str:
        """Doc."""
        return ""

    assert fn.__tool__.parameters["properties"]["mode"] == {
        "type": "string",
        "enum": ["fast", "slow"],
    }


def test_decorator_literal_int():
    @tool
    def fn(level: Literal[1, 2, 3]) -> str:
        """Doc."""
        return ""

    assert fn.__tool__.parameters["properties"]["level"] == {
        "type": "integer",
        "enum": [1, 2, 3],
    }


def test_decorator_tuple_fixed():
    @tool
    def fn(point: tuple[str, int]) -> str:
        """Doc."""
        return ""

    assert fn.__tool__.parameters["properties"]["point"] == {
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "integer"}],
    }


def test_decorator_tuple_variable():
    @tool
    def fn(tags: tuple[str, ...]) -> str:
        """Doc."""
        return ""

    assert fn.__tool__.parameters["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
    }


def test_decorator_dataclass_type():
    @tool
    def fn(addr: _Address) -> str:
        """Doc."""
        return ""

    schema = fn.__tool__.parameters["properties"]["addr"]
    assert schema["type"] == "object"
    assert "street" in schema["properties"]
    assert "city" in schema["properties"]
    assert "zip_code" in schema["properties"]
    assert "street" in schema["required"]
    assert "city" in schema["required"]
    assert "zip_code" not in schema["required"]


def test_decorator_typeddict_type():
    @tool
    def fn(cfg: _Config) -> str:
        """Doc."""
        return ""

    schema = fn.__tool__.parameters["properties"]["cfg"]
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "value" in schema["properties"]


def test_decorator_pydantic_model():
    """Pydantic-like model support via duck-typing."""

    @tool
    def fn(user: _FakeModel) -> str:
        """Doc."""
        return ""

    schema = fn.__tool__.parameters["properties"]["user"]
    assert schema["type"] == "object"
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["required"] == ["name"]


def test_decorator_custom_schema_override():
    @tool(
        schema={
            "location": {
                "type": "string",
                "description": "City,Country format",
            }
        }
    )
    def weather(location: str) -> str:
        """Get weather."""
        return ""

    props = weather.__tool__.parameters["properties"]
    assert props["location"]["description"] == "City,Country format"
    assert props["location"]["type"] == "string"


def test_decorator_custom_schema_partial():
    """Schema override merges with auto-generated schema."""

    @tool(schema={"city": {"description": "The city name"}})
    def fn(city: str, units: str = "c") -> str:
        """Doc."""
        return ""

    props = fn.__tool__.parameters["properties"]
    assert props["city"] == {
        "type": "string",
        "description": "The city name",
    }
    assert props["units"] == {"type": "string"}


def test_decorator_nested_list_of_dataclass():
    @tool
    def fn(items: list[_Item]) -> str:
        """Doc."""
        return ""

    schema = fn.__tool__.parameters["properties"]["items"]
    assert schema["type"] == "array"
    inner = schema["items"]
    assert inner["type"] == "object"
    assert "name" in inner["properties"]


# --- Phase 4: Validation tests ---


def test_validation_missing_required():
    reg = ToolRegistry()
    tool_def = Tool(
        name="add",
        description="",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    )
    reg.register("add", lambda a, b: a + b, tool_def)

    with pytest.raises(ValidationError, match="Missing required argument 'a'"):
        reg.execute(ToolCall(id="1", name="add", arguments={"b": 2}))


def test_validation_wrong_type():
    reg = ToolRegistry()
    tool_def = Tool(
        name="greet",
        description="",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )
    reg.register("greet", lambda name: f"Hi {name}", tool_def)

    with pytest.raises(ValidationError, match="expected string"):
        reg.execute(ToolCall(id="1", name="greet", arguments={"name": 123}))


def test_validation_passes_valid():
    reg = ToolRegistry()
    tool_def = Tool(
        name="add",
        description="",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    )
    reg.register("add", lambda a, b: a + b, tool_def)
    result = reg.execute(ToolCall(id="1", name="add", arguments={"a": 1, "b": 2}))
    assert result == "3"


def test_validation_async():
    reg = ToolRegistry()
    tool_def = Tool(
        name="add",
        description="",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    )
    reg.register("add", lambda a, b: a + b, tool_def)

    with pytest.raises(ValidationError, match="Missing required"):
        asyncio.run(reg.async_execute(ToolCall(id="1", name="add", arguments={"a": 1})))


# --- Phase 5: Group tests ---


def test_group_definitions():
    reg = ToolRegistry()
    t1 = Tool(name="math.add", description="", parameters={})
    t2 = Tool(name="math.sub", description="", parameters={})
    t3 = Tool(name="io.read", description="", parameters={})
    reg.register("math.add", lambda: None, t1)
    reg.register("math.sub", lambda: None, t2)
    reg.register("io.read", lambda: None, t3)

    defs = reg.definitions(group="math")
    assert len(defs) == 2
    names = {d.name for d in defs}
    assert names == {"math.add", "math.sub"}


def test_group_view_execute():
    reg = ToolRegistry()
    tool_def = Tool(name="math.add", description="", parameters={})
    reg.register("math.add", lambda a, b: a + b, tool_def)

    view = reg.group("math")
    result = view.execute(ToolCall(id="1", name="math.add", arguments={"a": 2, "b": 3}))
    assert result == "5"


def test_group_view_contains():
    reg = ToolRegistry()
    reg.register("math.add", lambda: None, Tool(name="math.add", description="", parameters={}))
    reg.register("io.read", lambda: None, Tool(name="io.read", description="", parameters={}))

    view = reg.group("math")
    assert "math.add" in view
    assert "io.read" not in view
    assert len(view) == 1


# --- Phase 6: Enable/disable tests ---


def test_disable_removes_from_definitions():
    reg = ToolRegistry()
    reg.register("foo", lambda: None, Tool(name="foo", description="", parameters={}))
    assert len(reg.definitions()) == 1

    reg.disable("foo")
    assert len(reg.definitions()) == 0


def test_disable_blocks_execution():
    reg = ToolRegistry()
    reg.register("foo", lambda: "ok", Tool(name="foo", description="", parameters={}))
    reg.disable("foo")

    with pytest.raises(KeyError, match="disabled"):
        reg.execute(ToolCall(id="1", name="foo", arguments={}))


def test_enable_restores_tool():
    reg = ToolRegistry()
    reg.register("foo", lambda: "ok", Tool(name="foo", description="", parameters={}))
    reg.disable("foo")
    reg.enable("foo")

    assert len(reg.definitions()) == 1
    result = reg.execute(ToolCall(id="1", name="foo", arguments={}))
    assert result == "ok"


# --- Phase 7: Alias tests ---


def test_alias_execute():
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b, Tool(name="add", description="", parameters={}))
    reg.alias("add", "plus")

    result = reg.execute(ToolCall(id="1", name="plus", arguments={"a": 3, "b": 4}))
    assert result == "7"
    assert "plus" in reg
    assert len(reg) == 2


def test_alias_independent_disable():
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b, Tool(name="add", description="", parameters={}))
    reg.alias("add", "plus")
    reg.disable("add")

    # Original is disabled but alias is not
    with pytest.raises(KeyError, match="disabled"):
        reg.execute(ToolCall(id="1", name="add", arguments={"a": 1, "b": 2}))

    result = reg.execute(ToolCall(id="1", name="plus", arguments={"a": 1, "b": 2}))
    assert result == "3"
