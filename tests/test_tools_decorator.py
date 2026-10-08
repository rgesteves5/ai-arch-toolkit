"""Tests for _tools/_decorator.py — @tool decorator."""

from __future__ import annotations

import re

import pytest

from ai_arch_toolkit.core._tools import ToolGroup, prepare_tools
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._definition import ToolDefinition, ToolSchema


class TestToolDecorator:
    def test_bare_decorator(self):
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city.

            Args:
                city: The city name.
            """
            return f"Sunny in {city}"

        assert hasattr(get_weather, "__tool_definition__")
        td = get_weather.__tool_definition__
        assert isinstance(td, ToolDefinition)
        assert td.schema.name == "get_weather"
        assert td.schema.description == "Get weather for a city."
        assert "properties" in td.schema.input_schema

    def test_definition_fn_points_to_wrapper(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert add.__tool_definition__.fn is add

    def test_decorator_with_name(self):
        @tool(name="weather")
        def get_weather(city: str) -> str:
            """Get weather."""
            return f"Sunny in {city}"

        assert get_weather.__tool_definition__.schema.name == "weather"

    def test_decorator_with_schema_override(self):
        @tool(schema={"city": {"description": "Override desc"}})
        def get_weather(city: str) -> str:
            """Get weather."""
            return f"Sunny in {city}"

        schema = get_weather.__tool_definition__.schema
        assert schema.input_schema["properties"]["city"]["description"] == "Override desc"

    def test_decorator_with_runtime_policy(self):
        @tool(
            capability="shell",
            risk_level="critical",
            requires_approval=True,
            approval_reason="Needs human review.",
        )
        def run(command: str) -> str:
            """Run a command."""
            return command

        policy = run.__tool_definition__.policy
        assert policy.capability == "shell"
        assert policy.risk_level == "critical"
        assert policy.requires_approval is True
        assert policy.approval_reason == "Needs human review."

    def test_default_policy_is_low_risk_no_approval(self):
        @tool
        def safe(x: int) -> int:
            """Safe tool."""
            return x

        policy = safe.__tool_definition__.policy
        assert policy.risk_level == "low"
        assert policy.requires_approval is False
        assert policy.capability is None

    def test_decorated_function_still_callable(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert add(2, 3) == 5

    def test_preserves_function_metadata(self):
        @tool
        def my_func(x: str) -> str:
            """My docstring."""
            return x

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."

    def test_provider_dict_uses_input_schema_key(self):
        """Provider-facing dicts use input_schema (not parameters)."""

        @tool
        def fn(x: int) -> int:
            """Do stuff."""
            return x

        provider = fn.__tool_definition__.schema.to_provider_dict()
        assert "input_schema" in provider
        assert "parameters" not in provider

    def test_default_values_included(self):
        @tool
        def fn(x: int, y: int = 5) -> int:
            """Add."""
            return x + y

        props = fn.__tool_definition__.schema.input_schema["properties"]
        assert props["y"]["default"] == 5
        assert "x" in fn.__tool_definition__.schema.input_schema["required"]
        assert "y" not in fn.__tool_definition__.schema.input_schema["required"]


class TestSchemaOverrides:
    """``schema=`` maps a parameter to keywords; a complete schema belongs to tool_from_schema."""

    def test_a_complete_schema_is_a_type_error(self):
        complete = {"type": "object", "properties": {"city": {"type": "string"}}}

        with pytest.raises(TypeError, match="tool_from_schema") as raised:
            tool(schema=complete)  # type: ignore[arg-type]

        assert "'type'" in str(raised.value)

    @pytest.mark.parametrize("schema", [["city"], "city", {"city": "string"}])
    def test_anything_but_a_mapping_of_mappings_is_a_type_error(self, schema):
        with pytest.raises(TypeError, match="schema="):
            tool(schema=schema)

    def test_the_error_comes_before_the_function_is_decorated(self):
        with pytest.raises(TypeError):

            @tool(schema={"properties": {"city": {}}, "required": ["city"]})  # type: ignore[dict-item]
            def get_weather(city: str) -> str:
                """Get weather."""
                return city


class TestPreviewHook:
    """``preview=`` takes a plain function of the arguments that returns text (C07c, D64)."""

    @pytest.mark.parametrize("preview", ["a picture", 42])
    def test_a_preview_that_is_not_a_function_is_a_type_error(self, preview):
        with pytest.raises(TypeError, match="plain function"):
            tool(preview=preview)  # type: ignore[arg-type]

    def test_an_async_preview_is_a_type_error_before_the_function_is_decorated(self):
        async def describe(arguments: dict) -> str:
            return "a picture"

        with pytest.raises(TypeError, match="plain function"):
            tool(preview=describe)  # type: ignore[arg-type]

    def test_a_definition_built_by_hand_checks_its_preview_too(self):
        def fn() -> str:
            return "ok"

        with pytest.raises(TypeError, match="plain function"):
            ToolDefinition(fn=fn, schema=ToolSchema(name="fn"), preview="text")  # type: ignore[arg-type]


class TestPortableNames:
    """One name rule, ``^[A-Za-z_][A-Za-z0-9_-]{0,63}$``, the one every provider and MCP take."""

    @pytest.mark.parametrize("name", ["a.b", "1a", "a" * 65, "get weather", "ação"])
    def test_a_decorator_name_outside_the_rule_is_a_value_error(self, name):
        with pytest.raises(ValueError, match="portable"):

            @tool(name=name)
            def fn() -> str:
                """Do stuff."""
                return "ok"

    @pytest.mark.parametrize("name", ["a.b", "1a", "a" * 65])
    def test_a_tool_schema_checks_its_name(self, name):
        with pytest.raises(ValueError, match=re.escape(repr(name))):
            ToolSchema(name=name)

    def test_the_longest_portable_name_is_accepted(self):
        assert ToolSchema(name="_" + "a" * 63).name == "_" + "a" * 63

    def test_a_lambda_is_a_value_error_wherever_it_becomes_a_tool(self):
        with pytest.raises(ValueError, match="'<lambda>'"):
            ToolGroup(lambda x: x)
        with pytest.raises(ValueError, match="'<lambda>'"):
            prepare_tools([lambda x: x])

    def test_prepare_tools_checks_the_name_of_a_dict(self):
        with pytest.raises(ValueError, match=re.escape("'a.b'")):
            prepare_tools([{"name": "a.b", "input_schema": {"type": "object"}}])

    def test_every_tool_the_package_ships_has_a_portable_name(self):
        from tests.toolkit.tool_catalog import TOOLS

        names = [fn.__tool_definition__.schema.name for fn in TOOLS.values()]

        assert len(names) > 100
        assert [n for n in names if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", n)] == []
