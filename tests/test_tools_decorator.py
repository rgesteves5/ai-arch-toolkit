"""Tests for _tools/_decorator.py — @tool decorator."""

from __future__ import annotations

from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._definition import ToolDefinition


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
