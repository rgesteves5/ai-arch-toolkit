"""Pre-built tools — real, working tools for agents and examples."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.tools._datetime import datetime_now, timezone_convert
from ai_arch_toolkit.toolkit.tools._filesystem import list_directory, read_file, search_files
from ai_arch_toolkit.toolkit.tools._geo import country_info, geocode, ip_lookup
from ai_arch_toolkit.toolkit.tools._json import csv_read, json_extract
from ai_arch_toolkit.toolkit.tools._knowledge import (
    define_word,
    wikipedia_article,
    wikipedia_search,
)
from ai_arch_toolkit.toolkit.tools._math import math_eval, unit_convert
from ai_arch_toolkit.toolkit.tools._news import hacker_news
from ai_arch_toolkit.toolkit.tools._shell import run_command
from ai_arch_toolkit.toolkit.tools._text import (
    base64_decode,
    base64_encode,
    regex_search,
    text_stats,
)
from ai_arch_toolkit.toolkit.tools._weather import get_forecast, get_weather
from ai_arch_toolkit.toolkit.tools._web import http_get, scrape_text

__all__ = [
    "base64_decode",
    "base64_encode",
    "country_info",
    "csv_read",
    "datetime_now",
    "define_word",
    "geocode",
    "get_forecast",
    "get_weather",
    "hacker_news",
    "http_get",
    "ip_lookup",
    "json_extract",
    "list_directory",
    "math_eval",
    "read_file",
    "regex_search",
    "run_command",
    "scrape_text",
    "search_files",
    "text_stats",
    "timezone_convert",
    "unit_convert",
    "wikipedia_article",
    "wikipedia_search",
]
