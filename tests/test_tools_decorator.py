"""Tests for _tools/_decorator.py — @tool decorator."""

from __future__ import annotations

from ai_arch_toolkit._tools._decorator import tool


class TestToolDecorator:
    def test_bare_decorator(self):
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city.

            Args:
                city: The city name.
            """
            return f"Sunny in {city}"

        assert hasattr(get_weather, "__tool__")
        td = get_weather.__tool__
        assert td["name"] == "get_weather"
        assert td["description"] == "Get weather for a city."
        assert "input_schema" in td

    def test_decorator_with_name(self):
        @tool(name="weather")
        def get_weather(city: str) -> str:
            """Get weather."""
            return f"Sunny in {city}"

        assert get_weather.__tool__["name"] == "weather"

    def test_decorator_with_schema_override(self):
        @tool(schema={"city": {"description": "Override desc"}})
        def get_weather(city: str) -> str:
            """Get weather."""
            return f"Sunny in {city}"

        td = get_weather.__tool__
        assert td["input_schema"]["properties"]["city"]["description"] == "Override desc"

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

    def test_input_schema_key(self):
        """Tool defs use input_schema (not parameters)."""

        @tool
        def fn(x: int) -> int:
            """Do stuff."""
            return x

        td = fn.__tool__
        assert "input_schema" in td
        assert "parameters" not in td

    def test_default_values_included(self):
        @tool
        def fn(x: int, y: int = 5) -> int:
            """Add."""
            return x + y

        td = fn.__tool__
        props = td["input_schema"]["properties"]
        assert props["y"]["default"] == 5
        assert "x" in td["input_schema"]["required"]
        assert "y" not in td["input_schema"]["required"]
