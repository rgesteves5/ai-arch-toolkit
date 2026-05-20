"""Built-in capability profiles for the configurable agent nano project."""

from __future__ import annotations

from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._config import (
    CapabilityProfile,
)

SAFE_CHAT_TOOLS = (
    "datetime_now",
    "date_add",
    "date_diff",
    "date_format",
    "timezone_convert",
    "define_word",
    "wikipedia_search",
    "wikipedia_article",
    "wikipedia_related",
    "math_eval",
    "unit_convert",
    "text_stats",
    "regex_search",
    "base64_encode",
    "base64_decode",
)

WEB_TOOLS = (
    "web_search_query",
    "wikipedia_search",
    "wikipedia_article",
    "wikipedia_related",
    "define_word",
    "http_get",
    "scrape_text",
    "hacker_news",
)

LOCAL_READ_TOOLS = (
    "list_directory",
    "read_file",
    "search_files",
)

LOCAL_EXEC_TOOLS = (
    "run_command",
    "python_repl",
)

DATA_TOOLS = (
    "json_extract",
    "csv_read",
    "math_eval",
    "unit_convert",
)

GEO_WEATHER_TOOLS = (
    "get_weather",
    "get_weather_by_coords",
    "get_forecast",
    "get_forecast_by_coords",
    "weather_units",
    "geocode",
    "reverse_geocode",
    "distance_between",
    "country_info",
    "ip_lookup",
    "timezone_lookup",
)

MEMORY_TOOLS = (
    "remember",
    "recall",
    "explore_memory",
    "forget_memory",
    "list_memories",
    "find_duplicate_memories",
    "consolidate_memories",
)

ALL_TOOL_NAMES = (
    *SAFE_CHAT_TOOLS,
    *WEB_TOOLS,
    *LOCAL_READ_TOOLS,
    *LOCAL_EXEC_TOOLS,
    *DATA_TOOLS,
    *GEO_WEATHER_TOOLS,
)

PROFILE_DESCRIPTIONS = {
    "basic_chat": (
        "General assistant profile with safe utility, text, math, date, and reference tools."
    ),
    "web_researcher": (
        "Research profile with web search, page fetching, Wikipedia, and Hacker News tools."
    ),
    "math_helper": "Focused calculation and unit conversion profile.",
    "data_helper": "Small structured-data profile for JSON, CSV, math, and unit conversion.",
    "geo_weather": "Location, weather, forecast, country, distance, IP, and timezone profile.",
    "local_reader": "Read-only local file discovery profile.",
    "local_operator": "Local file reading plus command and Python execution profile.",
    "private_memory_user": (
        "Private graph memory profile with recall, write, inspect, and consolidation tools."
    ),
    "deep_reasoner": "Self-discovery reasoning profile for harder tasks.",
    "reviewer": "Generate-review reasoning profile for iterative answer improvement.",
    "all_tools": (
        "All built-in tools. Dangerous execution tools remain blocked unless explicitly allowed."
    ),
}


def built_in_profiles() -> dict[str, CapabilityProfile]:
    """Return built-in capability profiles."""
    return {
        "basic_chat": CapabilityProfile(
            "basic_chat",
            {
                "tools": {"enabled": list(SAFE_CHAT_TOOLS)},
                "reasoning": {"strategy": "react", "max_iterations": 6},
            },
        ),
        "web_researcher": CapabilityProfile(
            "web_researcher",
            {
                "tools": {"enabled": list(WEB_TOOLS)},
                "reasoning": {"strategy": "react", "max_iterations": 8},
            },
        ),
        "math_helper": CapabilityProfile(
            "math_helper",
            {"tools": {"enabled": ["math_eval", "unit_convert"]}},
        ),
        "data_helper": CapabilityProfile(
            "data_helper",
            {"tools": {"enabled": list(DATA_TOOLS)}},
        ),
        "geo_weather": CapabilityProfile(
            "geo_weather",
            {"tools": {"enabled": list(GEO_WEATHER_TOOLS)}},
        ),
        "local_reader": CapabilityProfile(
            "local_reader",
            {"tools": {"enabled": list(LOCAL_READ_TOOLS)}},
        ),
        "local_operator": CapabilityProfile(
            "local_operator",
            {"tools": {"enabled": [*LOCAL_READ_TOOLS, *LOCAL_EXEC_TOOLS]}},
        ),
        "private_memory_user": CapabilityProfile(
            "private_memory_user",
            {
                "tools": {"enabled": list(MEMORY_TOOLS)},
                "memory": {"private_enabled": True, "read": True, "write": True},
            },
        ),
        "deep_reasoner": CapabilityProfile(
            "deep_reasoner",
            {"reasoning": {"strategy": "self_discovery", "max_iterations": 8}},
        ),
        "reviewer": CapabilityProfile(
            "reviewer",
            {"reasoning": {"strategy": "generate_review", "max_iterations": 8}},
        ),
        "all_tools": CapabilityProfile(
            "all_tools",
            {"tools": {"enabled": list(dict.fromkeys(ALL_TOOL_NAMES))}},
        ),
    }


def profile_details(name: str | None = None) -> dict[str, dict[str, object]]:
    """Return built-in profile descriptions and config fragments."""
    profiles = built_in_profiles()
    names = (name,) if name else tuple(sorted(profiles))
    details: dict[str, dict[str, object]] = {}
    for profile_name in names:
        profile = profiles[profile_name]
        details[profile_name] = {
            "description": PROFILE_DESCRIPTIONS.get(profile_name, ""),
            "config": profile.to_dict(),
        }
    return details


__all__ = [
    "ALL_TOOL_NAMES",
    "DATA_TOOLS",
    "GEO_WEATHER_TOOLS",
    "LOCAL_EXEC_TOOLS",
    "LOCAL_READ_TOOLS",
    "MEMORY_TOOLS",
    "PROFILE_DESCRIPTIONS",
    "SAFE_CHAT_TOOLS",
    "WEB_TOOLS",
    "built_in_profiles",
    "profile_details",
]
