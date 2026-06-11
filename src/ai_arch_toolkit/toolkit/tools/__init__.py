"""Pre-built tools — real, working tools for agents and examples."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.tools._arxiv import arxiv_paper, arxiv_search
from ai_arch_toolkit.toolkit.tools._clinical_trials import (
    clinical_trial_study,
    clinical_trials_search,
)
from ai_arch_toolkit.toolkit.tools._crossref import crossref_search, crossref_work
from ai_arch_toolkit.toolkit.tools._datacite import datacite_doi, datacite_search
from ai_arch_toolkit.toolkit.tools._datetime import (
    date_add,
    date_diff,
    date_format,
    datetime_now,
    timezone_convert,
)
from ai_arch_toolkit.toolkit.tools._dictionary import define_word
from ai_arch_toolkit.toolkit.tools._filesystem import list_directory, read_file, search_files
from ai_arch_toolkit.toolkit.tools._gdelt import gdelt_news_search, gdelt_timeline
from ai_arch_toolkit.toolkit.tools._geo import (
    country_info,
    distance_between,
    geocode,
    ip_lookup,
    reverse_geocode,
    timezone_lookup,
)
from ai_arch_toolkit.toolkit.tools._internet_archive import (
    internet_archive_item,
    internet_archive_search,
)
from ai_arch_toolkit.toolkit.tools._json import csv_read, json_extract
from ai_arch_toolkit.toolkit.tools._math import math_eval, unit_convert
from ai_arch_toolkit.toolkit.tools._news import hacker_news
from ai_arch_toolkit.toolkit.tools._nvd import nvd_cve, nvd_cve_search
from ai_arch_toolkit.toolkit.tools._open_library import (
    open_library_isbn,
    open_library_search,
    open_library_work,
)
from ai_arch_toolkit.toolkit.tools._pubmed import pubmed_article, pubmed_search
from ai_arch_toolkit.toolkit.tools._python import python_repl
from ai_arch_toolkit.toolkit.tools._semantic_scholar import (
    semantic_scholar_citations,
    semantic_scholar_paper,
    semantic_scholar_search,
)
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
from ai_arch_toolkit.toolkit.tools._wikidata import (
    wikidata_entity,
    wikidata_search,
    wikidata_sparql,
)
from ai_arch_toolkit.toolkit.tools._wikipedia import (
    wikipedia_article,
    wikipedia_related,
    wikipedia_search,
)

__all__ = [
    "arxiv_paper",
    "arxiv_search",
    "base64_decode",
    "base64_encode",
    "clinical_trial_study",
    "clinical_trials_search",
    "country_info",
    "crossref_search",
    "crossref_work",
    "csv_read",
    "datacite_doi",
    "datacite_search",
    "date_add",
    "date_diff",
    "date_format",
    "datetime_now",
    "define_word",
    "distance_between",
    "gdelt_news_search",
    "gdelt_timeline",
    "geocode",
    "get_forecast",
    "get_forecast_by_coords",
    "get_weather",
    "get_weather_by_coords",
    "hacker_news",
    "http_get",
    "internet_archive_item",
    "internet_archive_search",
    "ip_lookup",
    "json_extract",
    "list_directory",
    "math_eval",
    "nvd_cve",
    "nvd_cve_search",
    "open_library_isbn",
    "open_library_search",
    "open_library_work",
    "pubmed_article",
    "pubmed_search",
    "python_repl",
    "read_file",
    "regex_search",
    "reverse_geocode",
    "run_command",
    "scrape_text",
    "search_files",
    "semantic_scholar_citations",
    "semantic_scholar_paper",
    "semantic_scholar_search",
    "text_stats",
    "timezone_convert",
    "timezone_lookup",
    "unit_convert",
    "weather_units",
    "wikidata_entity",
    "wikidata_search",
    "wikidata_sparql",
    "wikipedia_article",
    "wikipedia_related",
    "wikipedia_search",
]
