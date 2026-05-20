"""Tool registry and resolution for configurable agents."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._tools._schema import infer_schema
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._config import ToolsConfig
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._profiles import (
    ALL_TOOL_NAMES,
)
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._web_search import (
    web_search_query,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedTools:
    """Resolved tools for a run."""

    group: ToolGroup
    names: tuple[str, ...]


DANGEROUS_TOOLS = frozenset({"run_command", "python_repl"})


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolGovernance:
    """Runtime governance settings for tool execution."""

    allow_dangerous: bool = False
    dry_run: bool = False
    max_calls: int | None = None


class GovernedToolGroup(ToolGroup):
    """ToolGroup with execution-time governance checks."""

    __slots__ = ("_calls", "_governance")

    def __init__(self, base: ToolGroup, governance: ToolGovernance) -> None:
        super().__init__(*base.tools)
        self._governance = governance
        self._calls = 0

    def execute(self, tool_call: ToolCall) -> str:
        decision = self._check(tool_call)
        if decision is not None:
            return decision
        return super().execute(tool_call)

    async def async_execute(self, tool_call: ToolCall) -> str:
        decision = self._check(tool_call)
        if decision is not None:
            return decision
        return await super().async_execute(tool_call)

    def _check(self, tool_call: ToolCall) -> str | None:
        if self._governance.max_calls is not None and self._calls >= self._governance.max_calls:
            return (
                "Tool blocked by governance: max tool calls exceeded "
                f"({self._governance.max_calls})."
            )
        self._calls += 1

        if tool_call.name in DANGEROUS_TOOLS and not self._governance.allow_dangerous:
            return (
                f"Tool blocked by governance: {tool_call.name!r} requires --allow-dangerous-tools."
            )
        if self._governance.dry_run:
            return f"Dry run: would execute {tool_call.name} with input {tool_call.input!r}."
        return None


class ToolRegistry:
    """Small registry mapping stable tool names to callables."""

    def __init__(self, tools: Iterable[Callable[..., Any]] | None = None) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        if tools:
            for fn in tools:
                self.register(fn)

    @classmethod
    def from_mapping(cls, tools: Mapping[str, Callable[..., Any]]) -> ToolRegistry:
        """Create a registry from explicit names."""
        registry = cls()
        for name, fn in tools.items():
            registry.register(fn, name=name)
        return registry

    def register(self, fn: Callable[..., Any], *, name: str | None = None) -> None:
        """Register a tool callable."""
        tool_name = name or _tool_name(fn)
        self._tools[tool_name] = fn

    def copy(self) -> ToolRegistry:
        """Return a shallow copy of this registry."""
        return ToolRegistry.from_mapping(dict(self._tools))

    def extend(self, group: ToolGroup) -> None:
        """Register all tools from a ToolGroup."""
        for fn in group.tools:
            self.register(fn)

    def get(self, name: str) -> Callable[..., Any]:
        """Get a registered tool by name."""
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Unknown enabled tool: {name!r}") from None

    @property
    def names(self) -> tuple[str, ...]:
        """Registered tool names."""
        return tuple(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def resolve_tools(config: ToolsConfig, registry: ToolRegistry | None) -> ResolvedTools:
    """Resolve enabled/disabled tool names into a ToolGroup."""
    return resolve_tools_with_limits(config, registry)


def resolve_tools_with_limits(
    config: ToolsConfig,
    registry: ToolRegistry | None,
    *,
    max_tool_calls: int | None = None,
) -> ResolvedTools:
    """Resolve enabled/disabled tool names into a ToolGroup with optional run limits."""
    registry = registry or ToolRegistry()
    disabled = set(config.disabled)
    enabled = registry.names if "all" in config.enabled else config.enabled
    names = tuple(name for name in enabled if name not in disabled)
    fns = [registry.get(name) for name in names]
    group: ToolGroup = ToolGroup(*fns)
    governance = _tool_governance(config, max_tool_calls=max_tool_calls)
    if governance != ToolGovernance() or any(name in DANGEROUS_TOOLS for name in names):
        group = GovernedToolGroup(group, governance)
    return ResolvedTools(group=group, names=names)


def _tool_governance(
    config: ToolsConfig,
    *,
    max_tool_calls: int | None = None,
) -> ToolGovernance:
    permissions = dict(config.permissions)
    max_calls = permissions.get("max_calls", max_tool_calls)
    return ToolGovernance(
        allow_dangerous=bool(permissions.get("allow_dangerous", False)),
        dry_run=bool(permissions.get("dry_run", False)),
        max_calls=int(max_calls) if max_calls is not None else None,
    )


def built_in_tool_registry() -> ToolRegistry:
    """Return a registry containing all built-in toolkit tools exposed to the chat."""
    from ai_arch_toolkit.toolkit.tools import (
        base64_decode,
        base64_encode,
        country_info,
        csv_read,
        date_add,
        date_diff,
        date_format,
        datetime_now,
        define_word,
        distance_between,
        geocode,
        get_forecast,
        get_forecast_by_coords,
        get_weather,
        get_weather_by_coords,
        hacker_news,
        http_get,
        ip_lookup,
        json_extract,
        list_directory,
        math_eval,
        python_repl,
        read_file,
        regex_search,
        reverse_geocode,
        run_command,
        scrape_text,
        search_files,
        text_stats,
        timezone_convert,
        timezone_lookup,
        unit_convert,
        weather_units,
        wikipedia_article,
        wikipedia_related,
        wikipedia_search,
    )

    registry = ToolRegistry.from_mapping(
        {
            "base64_decode": base64_decode,
            "base64_encode": base64_encode,
            "country_info": country_info,
            "csv_read": csv_read,
            "date_add": date_add,
            "date_diff": date_diff,
            "date_format": date_format,
            "datetime_now": datetime_now,
            "define_word": define_word,
            "distance_between": distance_between,
            "geocode": geocode,
            "get_forecast": get_forecast,
            "get_forecast_by_coords": get_forecast_by_coords,
            "get_weather": get_weather,
            "get_weather_by_coords": get_weather_by_coords,
            "hacker_news": hacker_news,
            "http_get": http_get,
            "ip_lookup": ip_lookup,
            "json_extract": json_extract,
            "list_directory": list_directory,
            "math_eval": math_eval,
            "python_repl": python_repl,
            "read_file": read_file,
            "regex_search": regex_search,
            "reverse_geocode": reverse_geocode,
            "run_command": run_command,
            "scrape_text": scrape_text,
            "search_files": search_files,
            "text_stats": text_stats,
            "timezone_convert": timezone_convert,
            "timezone_lookup": timezone_lookup,
            "unit_convert": unit_convert,
            "weather_units": weather_units,
            "web_search_query": web_search_query,
            "wikipedia_article": wikipedia_article,
            "wikipedia_related": wikipedia_related,
            "wikipedia_search": wikipedia_search,
        }
    )
    for name in ALL_TOOL_NAMES:
        if name not in registry:
            raise RuntimeError(f"Built-in tool profile references unregistered tool: {name}")
    return registry


def _tool_name(fn: Callable[..., Any]) -> str:
    tool_def = getattr(fn, "__tool__", None)
    if tool_def is None:
        tool_def = infer_schema(fn)
    return str(tool_def["name"])
