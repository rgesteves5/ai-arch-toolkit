"""World Bank tools — public development indicators catalog and series lookup."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://api.worldbank.org/v2"
_TIMEOUT = 15
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_RESULTS_LIMIT = 100
_INDICATOR_SEARCH_PAGE_SIZE = 1000
_INDICATOR_SCAN_PAGES_LIMIT = 30
_COUNTRIES_PAGE_SIZE = 500
_COMPARE_COUNTRIES_LIMIT = 10
_COMPARE_POINTS_LIMIT = 300
_NOTE_MAX_CHARS = 500
_YEAR_RE = re.compile(r"^\d{4}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_COUNTRY_RE = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True, slots=True, kw_only=True)
class _WorldBankTopic:
    """Normalized World Bank topic metadata."""

    id: str
    name: str
    note: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _WorldBankSource:
    """Normalized World Bank source/database metadata."""

    id: str
    name: str
    code: str
    last_updated: str
    data_available: str
    metadata_available: str
    description: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _WorldBankCountry:
    """Normalized World Bank country or aggregate metadata."""

    id: str
    iso2: str
    name: str
    region_id: str
    region: str
    income_level_id: str
    income_level: str
    lending_type_id: str
    lending_type: str
    capital_city: str
    latitude: str
    longitude: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _WorldBankIndicator:
    """Normalized World Bank indicator metadata."""

    id: str
    name: str
    unit: str
    source_id: str
    source: str
    source_note: str
    source_organization: str
    topics: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _WorldBankSeriesPoint:
    """Normalized World Bank indicator observation."""

    country_id: str
    country_iso3: str
    country: str
    indicator_id: str
    indicator: str
    date: str
    value: int | float | str | None
    unit: str
    obs_status: str
    decimal: int | None


@tool
def world_bank_topics(max_results: int = 50, page: int = 1) -> str:
    """List World Bank indicator topics.

    Args:
        max_results: Number of topics to return (1-100). Defaults to 50.
        page: One-based result page. Defaults to 1.
    """
    if page < 1:
        return "World Bank topics failed: page must be greater than or equal to 1."

    try:
        metadata, items = _fetch_world_bank(
            "/topic",
            {"page": str(page), "per_page": str(_bounded(max_results))},
        )
        topics = [_parse_topic(item) for item in items if isinstance(item, dict)]
    except urllib.error.HTTPError as e:
        return _http_error("World Bank topics failed", e)
    except urllib.error.URLError as e:
        return f"World Bank topics failed: URL error: {e.reason}"
    except TimeoutError:
        return "World Bank topics failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"World Bank topics failed: could not parse API response: {e}"

    topics = [topic for topic in topics if topic is not None]
    if not topics:
        return "No World Bank topics found."
    return _pagination_header("World Bank topics", metadata) + "\n" + _format_topics(topics)


@tool
def world_bank_sources(max_results: int = 50, page: int = 1) -> str:
    """List World Bank data sources/databases.

    Args:
        max_results: Number of sources to return (1-100). Defaults to 50.
        page: One-based result page. Defaults to 1.
    """
    if page < 1:
        return "World Bank sources failed: page must be greater than or equal to 1."

    try:
        metadata, items = _fetch_world_bank(
            "/source",
            {"page": str(page), "per_page": str(_bounded(max_results))},
        )
        sources = [_parse_source(item) for item in items if isinstance(item, dict)]
    except urllib.error.HTTPError as e:
        return _http_error("World Bank sources failed", e)
    except urllib.error.URLError as e:
        return f"World Bank sources failed: URL error: {e.reason}"
    except TimeoutError:
        return "World Bank sources failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"World Bank sources failed: could not parse API response: {e}"

    sources = [source for source in sources if source is not None]
    if not sources:
        return "No World Bank sources found."
    return _pagination_header("World Bank sources", metadata) + "\n" + _format_sources(sources)


@tool
def world_bank_countries(
    query: str = "",
    region: str = "",
    income_level: str = "",
    lending_type: str = "",
    max_results: int = 50,
    page: int = 1,
) -> str:
    """List or search World Bank countries and aggregates.

    Args:
        query: Optional text or code filter, e.g. "Portugal", "PT", or "WLD".
        region: Optional official region ID filter, e.g. "ECS" or "NA".
        income_level: Optional official income level ID filter, e.g. "HIC".
        lending_type: Optional official lending type ID filter, e.g. "IBD".
        max_results: Number of countries/aggregates to return (1-100). Defaults to 50.
        page: One-based result page. Defaults to 1.
    """
    if page < 1:
        return "World Bank countries failed: page must be greater than or equal to 1."

    max_results = _bounded(max_results)
    query = query.strip()
    region = region.strip().upper()
    income_level = income_level.strip().upper()
    lending_type = lending_type.strip().upper()
    filtered = any((query, region, income_level, lending_type))

    try:
        metadata, items = _fetch_world_bank(
            "/country",
            {
                "page": "1" if filtered else str(page),
                "per_page": str(_COUNTRIES_PAGE_SIZE if filtered else max_results),
            },
        )
        countries = [_parse_country(item) for item in items if isinstance(item, dict)]
    except urllib.error.HTTPError as e:
        return _http_error("World Bank countries failed", e)
    except urllib.error.URLError as e:
        return f"World Bank countries failed: URL error: {e.reason}"
    except TimeoutError:
        return "World Bank countries failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"World Bank countries failed: could not parse API response: {e}"

    countries = [country for country in countries if country is not None]
    if filtered:
        countries = [
            country
            for country in countries
            if _country_matches(country, query, region, income_level, lending_type)
        ]
        metadata = _local_metadata(page=page, per_page=max_results, total=len(countries))
        countries = _slice_page(countries, page, max_results)

    if not countries:
        return "No World Bank countries found."
    return (
        _pagination_header("World Bank countries", metadata) + "\n" + _format_countries(countries)
    )


@tool
def world_bank_indicators(
    query: str = "",
    topic: str = "",
    source: str = "",
    max_results: int = 20,
    page: int = 1,
    scan_pages: int = 10,
) -> str:
    """List or search World Bank indicators.

    Args:
        query: Optional text filter across indicator ID, name, source, topics, and definition.
        topic: Optional official topic ID filter, e.g. "3" for Economy & Growth.
        source: Optional official source ID filter, e.g. "2" for World Development Indicators.
        max_results: Number of indicators to return (1-100). Defaults to 20.
        page: One-based page to browse, or first page to scan when query is provided.
        scan_pages: Number of catalog pages to scan for query matches (1-30). Defaults to 10.
    """
    if page < 1:
        return "World Bank indicators failed: page must be greater than or equal to 1."
    if scan_pages < 1:
        return "World Bank indicators failed: scan_pages must be greater than or equal to 1."

    max_results = _bounded(max_results)
    query = query.strip()
    topic = topic.strip()
    source = source.strip()
    scan_pages = min(scan_pages, _INDICATOR_SCAN_PAGES_LIMIT)

    try:
        if query:
            metadata, indicators = _scan_indicators(query, topic, source, page, scan_pages)
            metadata = _local_metadata(page=1, per_page=max_results, total=len(indicators))
            indicators = _rank_indicators(indicators, query)[:max_results]
        else:
            metadata, items = _fetch_world_bank(
                _indicator_path(topic, source),
                {"page": str(page), "per_page": str(max_results)},
            )
            indicators = [_parse_indicator(item) for item in items if isinstance(item, dict)]
            if topic and source:
                indicators = [
                    indicator
                    for indicator in indicators
                    if indicator is not None and indicator.source_id == source
                ]
    except urllib.error.HTTPError as e:
        return _http_error("World Bank indicators failed", e)
    except urllib.error.URLError as e:
        return f"World Bank indicators failed: URL error: {e.reason}"
    except TimeoutError:
        return "World Bank indicators failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"World Bank indicators failed: could not parse API response: {e}"

    indicators = [indicator for indicator in indicators if indicator is not None]
    if not indicators:
        hint = " Try a topic/source filter or increase scan_pages." if query else ""
        return f"No World Bank indicators found.{hint}"

    header = _pagination_header("World Bank indicators", metadata)
    if query:
        header += f" | scanned_pages: {scan_pages}"
    return header + "\n" + _format_indicators(indicators, include_note=True)


@tool
def world_bank_indicator(indicator: str) -> str:
    """Fetch metadata for a specific World Bank indicator.

    Args:
        indicator: World Bank indicator ID, e.g. "SP.POP.TOTL".
    """
    indicator = indicator.strip()
    if not _valid_indicator_id(indicator):
        return f"World Bank indicator lookup failed: invalid indicator ID: {indicator!r}"

    try:
        _metadata, items = _fetch_world_bank(f"/indicator/{_quote_path(indicator)}", {})
        indicators = [_parse_indicator(item) for item in items if isinstance(item, dict)]
        indicators = [item for item in indicators if item is not None]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"World Bank indicator not found: {indicator}"
        return _http_error("World Bank indicator lookup failed", e)
    except urllib.error.URLError as e:
        return f"World Bank indicator lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "World Bank indicator lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"World Bank indicator lookup failed: could not parse API response: {e}"

    if not indicators:
        return f"World Bank indicator not found: {indicator}"
    return f"World Bank indicator {indicators[0].id}:\n" + _format_indicators(
        [indicators[0]],
        include_index=False,
        include_note=True,
        include_organization=True,
    )


@tool
def world_bank_series(
    country: str,
    indicator: str,
    start_year: str = "",
    end_year: str = "",
    max_results: int = 100,
    page: int = 1,
) -> str:
    """Fetch a World Bank indicator time series for a country or aggregate.

    Args:
        country: Country, economy, or aggregate code, e.g. "PRT", "US", "WLD", or "all".
        indicator: World Bank indicator ID, e.g. "SP.POP.TOTL".
        start_year: Optional first year as YYYY.
        end_year: Optional last year as YYYY.
        max_results: Number of observations to return (1-100). Defaults to 100.
        page: One-based result page. Defaults to 1.
    """
    validation = _validate_series_inputs(country, indicator, start_year, end_year, page)
    if validation:
        return f"World Bank series failed: {validation}"

    try:
        metadata, items = _fetch_world_bank(
            f"/country/{_quote_country(country)}/indicator/{_quote_path(indicator.strip())}",
            _series_params(start_year, end_year, page, _bounded(max_results)),
        )
        points = [_parse_series_point(item) for item in items if isinstance(item, dict)]
        points = [point for point in points if point is not None]
    except urllib.error.HTTPError as e:
        return _http_error("World Bank series failed", e)
    except urllib.error.URLError as e:
        return f"World Bank series failed: URL error: {e.reason}"
    except TimeoutError:
        return "World Bank series failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"World Bank series failed: could not parse API response: {e}"

    if not points:
        return "No World Bank series observations found."
    return _pagination_header("World Bank series", metadata) + "\n" + _format_series(points)


@tool
def world_bank_compare(
    indicator: str,
    countries: str,
    year: str = "",
    start_year: str = "",
    end_year: str = "",
    max_points: int = 100,
) -> str:
    """Compare a World Bank indicator across multiple countries or aggregates.

    Args:
        indicator: World Bank indicator ID, e.g. "NY.GDP.MKTP.CD".
        countries: Comma-separated country/economy codes, e.g. "PRT,ESP,DEU".
        year: Optional single year as YYYY. Overrides start_year/end_year when provided.
        start_year: Optional first year as YYYY.
        end_year: Optional last year as YYYY.
        max_points: Maximum observations to return (1-300). Defaults to 100.
    """
    indicator = indicator.strip()
    if not _valid_indicator_id(indicator):
        return f"World Bank compare failed: invalid indicator ID: {indicator!r}"

    country_codes = _parse_country_list(countries)
    if not country_codes:
        return "World Bank compare failed: provide at least one country code."
    if len(country_codes) > _COMPARE_COUNTRIES_LIMIT:
        return (
            f"World Bank compare failed: at most {_COMPARE_COUNTRIES_LIMIT} countries are allowed."
        )

    if year.strip():
        if not _valid_year(year):
            return f"World Bank compare failed: invalid year: {year!r}. Use YYYY."
        start_year = year.strip()
        end_year = year.strip()
    validation = _validate_year_range(start_year, end_year)
    if validation:
        return f"World Bank compare failed: {validation}"

    max_points = max(1, min(max_points, _COMPARE_POINTS_LIMIT))
    try:
        metadata, items = _fetch_world_bank(
            f"/country/{_quote_country(';'.join(country_codes))}/indicator/{_quote_path(indicator)}",
            _series_params(start_year, end_year, 1, max_points),
        )
        points = [_parse_series_point(item) for item in items if isinstance(item, dict)]
        points = [point for point in points if point is not None]
    except urllib.error.HTTPError as e:
        return _http_error("World Bank compare failed", e)
    except urllib.error.URLError as e:
        return f"World Bank compare failed: URL error: {e.reason}"
    except TimeoutError:
        return "World Bank compare failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"World Bank compare failed: could not parse API response: {e}"

    if not points:
        return "No World Bank comparison observations found."
    return _pagination_header("World Bank comparison", metadata) + "\n" + _format_compare(points)


def _fetch_world_bank(path: str, params: dict[str, str]) -> tuple[dict[str, Any], list[Any]]:
    params = {"format": "json", **params}
    url = f"{_BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    if isinstance(payload, list) and len(payload) >= 2:
        metadata = payload[0] if isinstance(payload[0], dict) else {}
        items = payload[1] if isinstance(payload[1], list) else []
        return metadata, items
    if (
        isinstance(payload, list)
        and payload
        and isinstance(payload[0], dict)
        and "message" in payload[0]
    ):
        return {}, []
    raise TypeError("unexpected World Bank response shape")


def _scan_indicators(
    query: str,
    topic: str,
    source: str,
    start_page: int,
    scan_pages: int,
) -> tuple[dict[str, Any], list[_WorldBankIndicator]]:
    matches: list[_WorldBankIndicator] = []
    last_metadata: dict[str, Any] = {}
    tokens = _query_tokens(query)
    path = _indicator_path(topic, source)

    for page in range(start_page, start_page + scan_pages):
        metadata, items = _fetch_world_bank(
            path,
            {"page": str(page), "per_page": str(_INDICATOR_SEARCH_PAGE_SIZE)},
        )
        last_metadata = metadata
        indicators = [_parse_indicator(item) for item in items if isinstance(item, dict)]
        for indicator in indicators:
            if indicator is None:
                continue
            if topic and not _indicator_has_topic(indicator, topic):
                continue
            if source and indicator.source_id != source:
                continue
            if _indicator_matches(indicator, tokens):
                matches.append(indicator)
        if page >= _int_or_none(metadata.get("pages"), page):
            break

    return last_metadata, list(dict.fromkeys(matches))


def _indicator_path(topic: str, source: str) -> str:
    if topic:
        return f"/topic/{_quote_path(topic)}/indicator"
    if source:
        return f"/source/{_quote_path(source)}/indicator"
    return "/indicator"


def _series_params(
    start_year: str,
    end_year: str,
    page: int,
    per_page: int,
) -> dict[str, str]:
    params = {"page": str(page), "per_page": str(per_page)}
    if start_year.strip() or end_year.strip():
        start = start_year.strip() or end_year.strip()
        end = end_year.strip() or start_year.strip()
        params["date"] = f"{start}:{end}"
    return params


def _parse_topic(data: dict[str, Any]) -> _WorldBankTopic | None:
    topic_id = _string(data.get("id"))
    name = _string(data.get("value"))
    if not topic_id and not name:
        return None
    return _WorldBankTopic(
        id=topic_id, name=name or "(unnamed)", note=_string(data.get("sourceNote"))
    )


def _parse_source(data: dict[str, Any]) -> _WorldBankSource | None:
    source_id = _string(data.get("id"))
    name = _string(data.get("name"))
    if not source_id and not name:
        return None
    return _WorldBankSource(
        id=source_id,
        name=name or "(unnamed)",
        code=_string(data.get("code")),
        last_updated=_string(data.get("lastupdated")),
        data_available=_string(data.get("dataavailability")),
        metadata_available=_string(data.get("metadataavailability")),
        description=_string(data.get("description")),
    )


def _parse_country(data: dict[str, Any]) -> _WorldBankCountry | None:
    country_id = _string(data.get("id"))
    name = _string(data.get("name"))
    if not country_id and not name:
        return None
    region = data.get("region") if isinstance(data.get("region"), dict) else {}
    income_level = data.get("incomeLevel") if isinstance(data.get("incomeLevel"), dict) else {}
    lending_type = data.get("lendingType") if isinstance(data.get("lendingType"), dict) else {}
    return _WorldBankCountry(
        id=country_id,
        iso2=_string(data.get("iso2Code")),
        name=name or "(unnamed)",
        region_id=_string(region.get("id")),
        region=_string(region.get("value")),
        income_level_id=_string(income_level.get("id")),
        income_level=_string(income_level.get("value")),
        lending_type_id=_string(lending_type.get("id")),
        lending_type=_string(lending_type.get("value")),
        capital_city=_string(data.get("capitalCity")),
        latitude=_string(data.get("latitude")),
        longitude=_string(data.get("longitude")),
    )


def _parse_indicator(data: dict[str, Any]) -> _WorldBankIndicator | None:
    indicator_id = _string(data.get("id"))
    name = _string(data.get("name"))
    if not indicator_id and not name:
        return None
    source = data.get("source") if isinstance(data.get("source"), dict) else {}
    return _WorldBankIndicator(
        id=indicator_id,
        name=name or "(unnamed)",
        unit=_string(data.get("unit")),
        source_id=_string(source.get("id")),
        source=_string(source.get("value")),
        source_note=_string(data.get("sourceNote")),
        source_organization=_string(data.get("sourceOrganization")),
        topics=_topics(data.get("topics")),
    )


def _parse_series_point(data: dict[str, Any]) -> _WorldBankSeriesPoint | None:
    indicator = data.get("indicator") if isinstance(data.get("indicator"), dict) else {}
    country = data.get("country") if isinstance(data.get("country"), dict) else {}
    date = _string(data.get("date"))
    if not date:
        return None
    return _WorldBankSeriesPoint(
        country_id=_string(country.get("id")),
        country_iso3=_string(data.get("countryiso3code")),
        country=_string(country.get("value")),
        indicator_id=_string(indicator.get("id")),
        indicator=_string(indicator.get("value")),
        date=date,
        value=data.get("value"),
        unit=_string(data.get("unit")),
        obs_status=_string(data.get("obs_status")),
        decimal=_int_or_none(data.get("decimal")),
    )


def _topics(value: Any) -> tuple[tuple[str, str], ...]:
    topics: list[tuple[str, str]] = []
    for item in value or []:
        if isinstance(item, dict):
            topic_id = _string(item.get("id"))
            name = _string(item.get("value"))
            if topic_id or name:
                topics.append((topic_id, name))
    return tuple(topics)


def _format_topics(topics: list[_WorldBankTopic]) -> str:
    blocks: list[str] = []
    for index, topic in enumerate(topics, start=1):
        lines = [f"{index}. {topic.name} ({topic.id})"]
        if topic.note:
            lines.append(f"   Note: {_truncate(topic.note, _NOTE_MAX_CHARS)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_sources(sources: list[_WorldBankSource]) -> str:
    blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        title = f"{index}. {source.name} ({source.id})"
        if source.code:
            title += f" [{source.code}]"
        lines = [title]
        meta = []
        if source.last_updated:
            meta.append(f"last updated: {source.last_updated}")
        if source.data_available:
            meta.append(f"data: {source.data_available}")
        if source.metadata_available:
            meta.append(f"metadata: {source.metadata_available}")
        if meta:
            lines.append("   " + " | ".join(meta))
        if source.description:
            lines.append(f"   Description: {_truncate(source.description, _NOTE_MAX_CHARS)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_countries(countries: list[_WorldBankCountry]) -> str:
    blocks: list[str] = []
    for index, country in enumerate(countries, start=1):
        lines = [f"{index}. {country.name} ({country.id})"]
        meta = []
        if country.iso2:
            meta.append(f"ISO2: {country.iso2}")
        if country.region:
            meta.append(f"region: {country.region} ({country.region_id})")
        if country.income_level:
            meta.append(f"income: {country.income_level} ({country.income_level_id})")
        if country.lending_type:
            meta.append(f"lending: {country.lending_type} ({country.lending_type_id})")
        if meta:
            lines.append("   " + " | ".join(meta))
        if country.capital_city:
            lines.append(f"   Capital: {country.capital_city}")
        if country.latitude and country.longitude:
            lines.append(f"   Coordinates: {country.latitude}, {country.longitude}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_indicators(
    indicators: list[_WorldBankIndicator],
    *,
    include_index: bool = True,
    include_note: bool = False,
    include_organization: bool = False,
) -> str:
    blocks: list[str] = []
    for index, indicator in enumerate(indicators, start=1):
        title = f"{index}. {indicator.id} — {indicator.name}" if include_index else indicator.name
        lines = [title]
        meta = []
        if not include_index:
            meta.append(f"ID: {indicator.id}")
        if indicator.unit:
            meta.append(f"unit: {indicator.unit}")
        if indicator.source:
            meta.append(f"source: {indicator.source} ({indicator.source_id})")
        if indicator.topics:
            topics = ", ".join(f"{name} ({topic_id})" for topic_id, name in indicator.topics[:6])
            meta.append(f"topics: {topics}")
        if meta:
            lines.append("   " + " | ".join(meta))
        if include_note and indicator.source_note:
            lines.append(f"   Definition: {_truncate(indicator.source_note, _NOTE_MAX_CHARS)}")
        if include_organization and indicator.source_organization:
            organization = _truncate(indicator.source_organization, _NOTE_MAX_CHARS)
            lines.append(f"   Source organization: {organization}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_series(points: list[_WorldBankSeriesPoint]) -> str:
    first = points[0]
    lines = [
        f"{first.indicator_id} — {first.indicator} for {first.country} ({first.country_iso3})"
    ]
    for index, point in enumerate(points, start=1):
        value = "missing" if point.value is None else _format_value(point.value)
        suffix = f" {point.unit}" if point.unit else ""
        status = f" | status: {point.obs_status}" if point.obs_status else ""
        lines.append(f"{index}. {point.date}: {value}{suffix}{status}")
    return "\n".join(lines)


def _format_compare(points: list[_WorldBankSeriesPoint]) -> str:
    indicator = points[0].indicator_id
    indicator_name = points[0].indicator
    lines = [f"{indicator} — {indicator_name} comparison:"]
    for index, point in enumerate(
        sorted(
            points, key=lambda item: (item.date, item.country_iso3 or item.country), reverse=True
        ),
        start=1,
    ):
        value = "missing" if point.value is None else _format_value(point.value)
        country = point.country_iso3 or point.country_id or point.country
        label = f"{point.country} ({country})" if point.country else country
        lines.append(f"{index}. {point.date} | {label}: {value}")
    return "\n".join(lines)


def _pagination_header(label: str, metadata: dict[str, Any]) -> str:
    page = _string(metadata.get("page")) or "?"
    pages = _string(metadata.get("pages")) or "?"
    total = _string(metadata.get("total")) or "?"
    per_page = _string(metadata.get("per_page")) or "?"
    return f"{label} (page {page}/{pages}, per_page {per_page}, total {total}):"


def _local_metadata(page: int, per_page: int, total: int) -> dict[str, Any]:
    pages = (total + per_page - 1) // per_page if total else 0
    return {"page": page, "pages": pages, "per_page": per_page, "total": total}


def _slice_page(items: list[Any], page: int, per_page: int) -> list[Any]:
    start = (page - 1) * per_page
    return items[start : start + per_page]


def _country_matches(
    country: _WorldBankCountry,
    query: str,
    region: str,
    income_level: str,
    lending_type: str,
) -> bool:
    if query:
        haystack = " ".join((country.id, country.iso2, country.name)).lower()
        if query.lower() not in haystack:
            return False
    if region and country.region_id.upper() != region:
        return False
    if income_level and country.income_level_id.upper() != income_level:
        return False
    return not (lending_type and country.lending_type_id.upper() != lending_type)


def _indicator_matches(indicator: _WorldBankIndicator, tokens: tuple[str, ...]) -> bool:
    haystack = _indicator_haystack(indicator)
    return all(token in haystack for token in tokens)


def _rank_indicators(
    indicators: list[_WorldBankIndicator],
    query: str,
) -> list[_WorldBankIndicator]:
    phrase = query.lower()

    def score(indicator: _WorldBankIndicator) -> int:
        name = indicator.name.lower()
        indicator_id = indicator.id.lower()
        haystack = _indicator_haystack(indicator)
        value = 0
        if indicator_id == phrase:
            value += 100
        if name == phrase:
            value += 80
        if name.startswith(phrase):
            value += 40
        if phrase in name:
            value += 25
        if phrase in indicator_id:
            value += 20
        if phrase in haystack:
            value += 10
        return value

    return sorted(indicators, key=score, reverse=True)


def _indicator_haystack(indicator: _WorldBankIndicator) -> str:
    topic_text = " ".join(f"{topic_id} {name}" for topic_id, name in indicator.topics)
    return " ".join(
        (
            indicator.id,
            indicator.name,
            indicator.unit,
            indicator.source_id,
            indicator.source,
            indicator.source_note,
            indicator.source_organization,
            topic_text,
        )
    ).lower()


def _indicator_has_topic(indicator: _WorldBankIndicator, topic: str) -> bool:
    return any(topic_id == topic for topic_id, _name in indicator.topics)


def _query_tokens(query: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"\W+", query.lower()) if token)


def _parse_country_list(countries: str) -> tuple[str, ...]:
    codes: list[str] = []
    for raw in countries.replace(";", ",").split(","):
        code = raw.strip().upper()
        if code and _COUNTRY_RE.fullmatch(code):
            codes.append(code)
    return tuple(dict.fromkeys(codes))


def _validate_series_inputs(
    country: str,
    indicator: str,
    start_year: str,
    end_year: str,
    page: int,
) -> str:
    if page < 1:
        return "page must be greater than or equal to 1."
    if not _valid_country(country):
        return f"invalid country code: {country!r}"
    if not _valid_indicator_id(indicator.strip()):
        return f"invalid indicator ID: {indicator!r}"
    return _validate_year_range(start_year, end_year)


def _validate_year_range(start_year: str, end_year: str) -> str:
    start = start_year.strip()
    end = end_year.strip()
    if start and not _valid_year(start):
        return f"invalid start_year: {start_year!r}. Use YYYY."
    if end and not _valid_year(end):
        return f"invalid end_year: {end_year!r}. Use YYYY."
    if start and end and int(start) > int(end):
        return "start_year must be before or equal to end_year."
    return ""


def _valid_year(value: str) -> bool:
    return bool(_YEAR_RE.fullmatch(value.strip()))


def _valid_indicator_id(value: str) -> bool:
    return bool(value and _ID_RE.fullmatch(value))


def _valid_country(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped and (stripped.lower() == "all" or _COUNTRY_RE.fullmatch(stripped)))


def _quote_path(value: str) -> str:
    return urllib.parse.quote(value.strip(), safe="")


def _quote_country(value: str) -> str:
    return urllib.parse.quote(value.strip(), safe=";")


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 429:
        return f"{prefix}: rate limited by World Bank (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_RESULTS_LIMIT))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _int_or_none(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, int):
        return value
    try:
        if value is not None and str(value).strip():
            return int(value)
    except ValueError:
        return default
    return default


def _format_value(value: int | float | str) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"
