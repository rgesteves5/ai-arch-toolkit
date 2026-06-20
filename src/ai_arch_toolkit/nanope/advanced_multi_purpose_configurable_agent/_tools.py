"""Tool registry and resolution for configurable agents."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core._tools._governance import (
    DangerousToolGate,
    DryRunGate,
    ToolGate,
)
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


DANGEROUS_TOOLS = frozenset(
    {
        "http_get",
        "list_directory",
        "python_repl",
        "read_file",
        "run_command",
        "scrape_text",
        "search_files",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolGovernance:
    """Runtime governance settings for tool execution."""

    allow_dangerous: bool = False
    dry_run: bool = False
    max_calls: int | None = None


def _governance_gates(governance: ToolGovernance) -> tuple[ToolGate, ...]:
    """Translate configurable-agent governance into core execution gates."""
    gates: list[ToolGate] = [
        DangerousToolGate(blocked=DANGEROUS_TOOLS, allow=governance.allow_dangerous),
    ]
    if governance.dry_run:
        gates.append(DryRunGate(dry_run=True))
    return tuple(gates)


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
    governance = _tool_governance(config, max_tool_calls=max_tool_calls)
    group = ToolGroup(
        *fns,
        gates=_governance_gates(governance),
        max_calls=governance.max_calls,
    )
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
        ip_lookup,
        json_extract,
        math_eval,
        regex_search,
        reverse_geocode,
        text_stats,
        timezone_convert,
        timezone_lookup,
        unit_convert,
        weather_units,
        wikipedia_article,
        wikipedia_related,
        wikipedia_search,
    )
    from ai_arch_toolkit.toolkit.tools.dangerous import (
        http_get,
        list_directory,
        python_repl,
        read_file,
        run_command,
        scrape_text,
        search_files,
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
    definition = getattr(fn, "__tool_definition__", None)
    if definition is not None:
        return definition.schema.name
    return str(infer_schema(fn)["name"])
