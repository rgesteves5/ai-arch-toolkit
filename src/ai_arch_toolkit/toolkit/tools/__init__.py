"""Pre-built tools — real, working tools for agents and examples."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.tools._datetime import (
    date_add,
    date_diff,
    date_format,
    datetime_now,
    timezone_convert,
)
from ai_arch_toolkit.toolkit.tools._dictionary import define_word
from ai_arch_toolkit.toolkit.tools._filesystem import list_directory, read_file, search_files
from ai_arch_toolkit.toolkit.tools._geo import (
    country_info,
    distance_between,
    geocode,
    ip_lookup,
    reverse_geocode,
    timezone_lookup,
)
from ai_arch_toolkit.toolkit.tools._json import csv_read, json_extract
from ai_arch_toolkit.toolkit.tools._math import math_eval, unit_convert
from ai_arch_toolkit.toolkit.tools._news import hacker_news
from ai_arch_toolkit.toolkit.tools._python import python_repl
from ai_arch_toolkit.toolkit.tools._shell import run_command
from ai_arch_toolkit.toolkit.tools._text import (
    base64_decode,
    base64_encode,
    regex_search,
    text_stats,
)
from ai_arch_toolkit.toolkit.tools._weather import (
    get_forecast,
    get_forecast_by_coords,
    get_weather,
    get_weather_by_coords,
    weather_units,
)
from ai_arch_toolkit.toolkit.tools._web import http_get, scrape_text
from ai_arch_toolkit.toolkit.tools._wikipedia import (
    wikipedia_article,
    wikipedia_related,
    wikipedia_search,
)

__all__ = [
    "base64_decode",
    "base64_encode",
    "country_info",
    "csv_read",
    "date_add",
    "date_diff",
    "date_format",
    "datetime_now",
    "define_word",
    "distance_between",
    "geocode",
    "get_forecast",
    "get_forecast_by_coords",
    "get_weather",
    "get_weather_by_coords",
    "hacker_news",
    "http_get",
    "ip_lookup",
    "json_extract",
    "list_directory",
    "math_eval",
    "python_repl",
    "read_file",
    "regex_search",
    "reverse_geocode",
    "run_command",
    "scrape_text",
    "search_files",
    "text_stats",
    "timezone_convert",
    "timezone_lookup",
    "unit_convert",
    "weather_units",
    "wikipedia_article",
    "wikipedia_related",
    "wikipedia_search",
]
