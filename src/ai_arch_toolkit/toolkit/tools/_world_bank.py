"""World Bank tools: the development indicators catalogue and its series.

The API (https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures)
pages with ``page`` and ``per_page`` and says the total; every list here reads on by ``page``,
through the window (D39). Several countries go in one request, joined by ``;``. Values are
written with every digit the API sent, never in scientific notation. The API explains its
errors as ``[{"message": [{"id", "key", "value"}]}]``, with HTTP 200 (seen 2026-09-29), with the
ids of its error table
(https://datahelpdesk.worldbank.org/knowledgebase/articles/898620-api-error-codes).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._numbers import plain_number
from ai_arch_toolkit.toolkit.tools._window import Window, list_window

# The errors of a request at fault: a missing parameter (115), an invalid value (120), an
# unsupported language (150), a value filter without dates (160).
_REQUEST_ERRORS = frozenset({"115", "120", "150", "160"})
# The service is temporarily unavailable (105).
_UNAVAILABLE = frozenset({"105"})
_INVALID_VALUE = "120"
_LISTS = (
    "check the codes and IDs given (world_bank_countries, world_bank_indicators, "
    "world_bank_sources and world_bank_topics list them)"
)


def _messages(payload: object) -> list[dict[str, Any]] | None:
    """The messages a World Bank error answer carries; ``None`` for a result."""
    first = payload[0] if isinstance(payload, list) and payload else None
    messages = first.get("message") if isinstance(first, dict) else None
    if not isinstance(messages, list):
        return None
    return [item for item in messages if isinstance(item, dict)]


def _api_message(reply: Reply) -> ToolFailure | str | None:
    """The error a World Bank answer reports, in the source's words (``key: value``); ``None``
    for a result. Errors of the request are a ``validation_error``; error 105, the service
    unavailable, is worth a retry; any other is the words."""
    messages = _messages(reply.body)
    if messages is None:
        return None
    said = _said(messages)
    ids = {_string(item.get("id")) for item in messages}
    if ids and ids <= _REQUEST_ERRORS:
        return ToolFailure("validation_error", f"{said.rstrip('.')}; {_LISTS}")
    if ids and ids <= _UNAVAILABLE:
        return ToolFailure("upstream", f"{said.rstrip('.')}; try again later", retryable=True)
    return said


def _said(messages: list[dict[str, Any]]) -> str:
    """The messages in the source's words, ``key: value`` each."""
    reported = [
        ": ".join(text for text in (_string(item.get("key")), _string(item.get("value"))) if text)
        for item in messages
    ]
    return "; ".join(text for text in reported if text) or "unknown error"


def _lookup_error(what: str, next_step: str) -> Callable[[Reply], ToolFailure | str | None]:
    """The reader of a request for named codes: error 120 ("Invalid value") is a code the World
    Bank does not have (``not_found``: ``what``, the source's words, ``next_step``); any other
    error as :func:`_api_message` reads it."""

    def read(reply: Reply) -> ToolFailure | str | None:
        messages = _messages(reply.body)
        if messages and all(_string(item.get("id")) == _INVALID_VALUE for item in messages):
            said = _said(messages).rstrip(".")
            return ToolFailure("not_found", f"{what} (error 120, {said}); {next_step}")
        return _api_message(reply)

    return read


# Country lists travel as one path segment, the codes joined by ";".
_API = Api(
    base="https://api.worldbank.org/v2",
    name="World Bank",
    timeout_s=15,
    params={"format": "json"},
    segment_safe=";",
    error_reader=_api_message,
)
_CATALOGUE_PAGE = 1000
_SCAN_PAGES = 30
# Every country and aggregate in one page (296 in 2026): the filters apply to all of them.
_ALL_COUNTRIES = 1000
_YEAR_RE = re.compile(r"^\d{4}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_COUNTRY_RE = re.compile(r"^[A-Za-z0-9_]+$")


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
class _Scan:
    """What a search read of the catalogue: its matches, how many indicators it read, and the
    catalogue's size."""

    matches: list[_WorldBankIndicator]
    read: int
    total: int
    pages: int


@tool(capability="network")
def world_bank_topics(
    max_results: Annotated[int, Range(1, 100)] = 50, page: Annotated[int, Range(1)] = 1
) -> ToolResult:
    """List the World Bank's indicator topics, each with its note.

    Args:
        max_results: How many topics to show.
        page: The page to show; the footer gives the next one.
    """
    params = {"page": str(page), "per_page": str(max_results)}
    return _API.get_json_list(
        "topic",
        params=params,
        parse=lambda payload: _listing(
            payload,
            page=page,
            per_page=max_results,
            line=_topic_line,
            heading="World Bank topics (world_bank_indicators(topic=ID) lists one's indicators):",
            empty="The World Bank lists no topics.",
        ),
    )


@tool(capability="network")
def world_bank_sources(
    max_results: Annotated[int, Range(1, 100)] = 50, page: Annotated[int, Range(1)] = 1
) -> ToolResult:
    """List the World Bank's data sources (its databases).

    Args:
        max_results: How many sources to show.
        page: The page to show; the footer gives the next one.
    """
    params = {"page": str(page), "per_page": str(max_results)}
    return _API.get_json_list(
        "source",
        params=params,
        parse=lambda payload: _listing(
            payload,
            page=page,
            per_page=max_results,
            line=_source_line,
            heading="World Bank sources (world_bank_indicators(source=ID) lists one's "
            "indicators):",
            empty="The World Bank lists no sources.",
        ),
    )


@tool(capability="network")
def world_bank_countries(
    query: str = "",
    region: str = "",
    income_level: str = "",
    lending_type: str = "",
    max_results: Annotated[int, Range(1, 100)] = 50,
    page: Annotated[int, Range(1)] = 1,
) -> ToolResult:
    """List or search World Bank countries and aggregates, with the codes world_bank_series
    takes.

    Args:
        query: Text or code to look for, e.g. "Portugal", "PT", or "WLD".
        region: Region ID to keep, e.g. "ECS" or "NAC".
        income_level: Income level ID to keep, e.g. "HIC".
        lending_type: Lending type ID to keep, e.g. "IBD".
        max_results: How many countries or aggregates to show.
        page: The page to show; the footer gives the next one.
    """
    filters = (
        query.strip(),
        region.strip().upper(),
        income_level.strip().upper(),
        lending_type.strip().upper(),
    )
    heading = "World Bank countries and aggregates (world_bank_series takes their codes):"
    if not any(filters):
        params = {"page": str(page), "per_page": str(max_results)}
        return _API.get_json_list(
            "country",
            params=params,
            parse=lambda payload: _listing(
                payload,
                page=page,
                per_page=max_results,
                line=_country_line,
                heading=heading,
                empty="The World Bank lists no countries or aggregates.",
            ),
        )
    params = {"page": "1", "per_page": str(_ALL_COUNTRIES)}
    return _API.get_json_list(
        "country",
        params=params,
        parse=lambda payload: _countries_answer(payload, filters, page, max_results, heading),
    )


@tool(capability="network")
def world_bank_indicators(
    query: str = "",
    topic: str = "",
    source: str = "",
    max_results: Annotated[int, Range(1, 100)] = 20,
    page: Annotated[int, Range(1)] = 1,
    scan_pages: Annotated[int, Range(1, _SCAN_PAGES)] = 10,
) -> ToolResult:
    """List or search World Bank indicators, with the IDs world_bank_series takes.

    Without a query, it lists the catalogue (or a topic's or a source's indicators) page by
    page. With one, it searches the IDs, names, sources, topics and definitions of the first
    ``scan_pages`` catalogue pages of 1000 indicators, best matches first.

    Args:
        query: Words to look for, e.g. "inflation consumer prices".
        topic: Topic ID to keep, e.g. "3" for Economy & Growth (world_bank_topics).
        source: Source ID to keep, e.g. "2" for World Development Indicators
            (world_bank_sources).
        max_results: How many indicators to show.
        page: The page to show, of the catalogue or of the matches; the footer gives the next.
        scan_pages: With a query, how many catalogue pages of 1000 indicators to search.
    """
    query, topic, source = query.strip(), topic.strip(), source.strip()
    if query or (topic and source):
        scan = _scan_indicators(query, topic, source, scan_pages)
        return _search_answer(scan, query, topic, source, page, max_results)
    params = {"page": str(page), "per_page": str(max_results)}
    under = f" {_under(topic, source)}" if topic or source else ""
    return _API.get_json_list(
        *_indicator_segments(topic, source),
        params=params,
        parse=lambda payload: _listing(
            payload,
            page=page,
            per_page=max_results,
            line=_indicator_line,
            heading=f"World Bank indicators{under} (world_bank_indicator gives a definition):",
            empty=f"No World Bank indicators are listed{under}.",
        ),
    )


@tool(capability="network")
def world_bank_indicator(indicator: str) -> ToolResult:
    """Read a World Bank indicator: its name, source, topics, definition and source
    organization.

    Args:
        indicator: World Bank indicator ID, e.g. "SP.POP.TOTL".

    Raises:
        ToolFailure: validation_error when the indicator ID is malformed; not_found when the World
            Bank has no indicator with that ID.
    """
    indicator = _indicator_id(indicator)
    what = f"the World Bank has no indicator {indicator}"
    api = replace(
        _API, error_reader=_lookup_error(what, "search for one with world_bank_indicators")
    )
    return api.get_json_list(
        "indicator",
        indicator,
        parse=lambda payload: _indicator_answer(payload, indicator),
        missing=_no_indicator(indicator),
    )


@tool(capability="network")
def world_bank_series(
    country: str,
    indicator: str,
    start_year: str = "",
    end_year: str = "",
    max_results: Annotated[int, Range(1, 100)] = 100,
    page: Annotated[int, Range(1)] = 1,
) -> ToolResult:
    """Read a World Bank indicator's values for one or several countries or aggregates, in the
    API's order; several codes compare them.

    Args:
        country: Country, economy or aggregate codes, comma-separated, e.g. "PRT",
            "PRT,ESP,DEU", "WLD", or "all" (world_bank_countries lists the codes).
        indicator: World Bank indicator ID, e.g. "NY.GDP.MKTP.CD".
        start_year: First year, as YYYY; alone, the only year.
        end_year: Last year, as YYYY.
        max_results: How many values to show.
        page: The page to show; the footer gives the next one.

    Raises:
        ToolFailure: validation_error when a country code, the indicator ID or a year is
            invalid; not_found when the World Bank has no such indicator or country.
    """
    codes = _country_codes(country)
    indicator = _indicator_id(indicator)
    _check_year_range(start_year, end_year)
    what = f"the World Bank has no indicator {indicator} or no country or aggregate among {codes}"
    reader = _lookup_error(
        what,
        "find the indicator with world_bank_indicators and the codes with world_bank_countries",
    )
    asked = f"{indicator} for {codes}{_years(start_year, end_year)}"
    api = replace(_API, error_reader=reader)
    return api.get_json_list(
        "country",
        codes,
        "indicator",
        indicator,
        params=_series_params(start_year, end_year, page, max_results),
        parse=lambda payload: _series_answer(payload, asked, page, max_results),
    )


# --- Answers -----------------------------------------------------------------------------------


def _page(payload: list[Any]) -> tuple[dict[str, Any], list[Any]]:
    """A World Bank answer's ``[metadata, items]``; any other shape raises ``ToolFailure``.

    An error message never reaches here: ``_API``'s reader raises it first (``_api_message``).
    """
    if len(payload) >= 2:
        return _dict(payload[0]), payload[1] if isinstance(payload[1], list) else []
    raise ToolFailure(
        "upstream", "could not parse API response: unexpected World Bank response shape"
    )


def _paged(lines: list[str], *, page: int, per_page: int, total: int | None, pages: int) -> Window:
    """A page of a list the API (or the tool) paged: numbered from the page's start, with the
    total and the next page."""
    next_call = {"page": page + 1} if lines and page < pages else None
    return list_window(lines, first=(page - 1) * per_page + 1, total=total, next_call=next_call)


def _listing(
    payload: list[Any],
    *,
    page: int,
    per_page: int,
    line: Callable[[int, dict[str, Any]], str],
    heading: str,
    empty: str,
) -> ToolResult:
    """One page of a list the API pages, each item a line from ``line``; ``empty`` says the
    list has nothing."""
    metadata, items = _page(payload)
    records = [item for item in items if isinstance(item, dict)]
    if not records and page == 1:
        return ToolResult.success(empty)
    first = (page - 1) * per_page + 1
    lines = [line(number, item) for number, item in enumerate(records, start=first)]
    pages = _int_or_none(metadata.get("pages")) or 0
    window = _paged(
        lines,
        page=page,
        per_page=per_page,
        total=_int_or_none(metadata.get("total")),
        pages=pages,
    )
    return window.result(heading=heading)


def _topic_line(number: int, topic: dict[str, Any]) -> str:
    line = (
        f"{number}. {_string(topic.get('value')) or '(unnamed)'} (ID {_string(topic.get('id'))})"
    )
    note = _string(topic.get("sourceNote"))
    return line + (f"\n   {note}" if note else "")


def _source_line(number: int, source: dict[str, Any]) -> str:
    code = _string(source.get("code"))
    ids = f"ID {_string(source.get('id'))}" + (f", code {code}" if code else "")
    lines = [f"{number}. {_string(source.get('name')) or '(unnamed)'} ({ids})"]
    meta = [
        f"{label}: {value}"
        for label, key in (
            ("last updated", "lastupdated"),
            ("data", "dataavailability"),
            ("metadata", "metadataavailability"),
        )
        if (value := _string(source.get(key)))
    ]
    if meta:
        lines.append("   " + " | ".join(meta))
    if description := _string(source.get("description")):
        lines.append(f"   {description}")
    return "\n".join(lines)


def _country_line(number: int, country: dict[str, Any]) -> str:
    iso2 = _string(country.get("iso2Code"))
    codes = _string(country.get("id")) + (f", ISO2 {iso2}" if iso2 else "")
    lines = [f"{number}. {_string(country.get('name')) or '(unnamed)'} ({codes})"]
    meta = [
        f"{label}: {_string(group.get('value'))} ({_string(group.get('id'))})"
        for label, key in (
            ("region", "region"),
            ("income", "incomeLevel"),
            ("lending", "lendingType"),
        )
        if _string((group := _dict(country.get(key))).get("value"))
    ]
    if meta:
        lines.append("   " + " | ".join(meta))
    place = []
    if capital := _string(country.get("capitalCity")):
        place.append(f"capital: {capital}")
    latitude, longitude = _string(country.get("latitude")), _string(country.get("longitude"))
    if latitude and longitude:
        place.append(f"latitude {latitude}, longitude {longitude}")
    if place:
        lines.append("   " + " | ".join(place))
    return "\n".join(lines)


def _countries_answer(
    payload: list[Any],
    filters: tuple[str, str, str, str],
    page: int,
    per_page: int,
    heading: str,
) -> ToolResult:
    """The page of the countries and aggregates that match ``filters``, among all of them."""
    metadata, items = _page(payload)
    matches = [
        item for item in items if isinstance(item, dict) and _country_matches(item, *filters)
    ]
    if not matches:
        named = zip(("query", "region", "income_level", "lending_type"), filters, strict=True)
        asked = ", ".join(
            f"{key}={value!r}" if key == "query" else f"{key}={value}"
            for key, value in named
            if value
        )
        return ToolResult.success(f"No World Bank country or aggregate matches {asked}.")
    start = (page - 1) * per_page
    lines = [
        _country_line(number, item)
        for number, item in enumerate(matches[start : start + per_page], start=start + 1)
    ]
    total = _int_or_none(metadata.get("total"))
    if total is not None and total > len(items):
        heading += f"\n(searched the first {len(items)} of {total} the World Bank lists)"
    pages = -(-len(matches) // per_page)
    window = _paged(lines, page=page, per_page=per_page, total=len(matches), pages=pages)
    return window.result(heading=heading)


def _country_matches(
    country: dict[str, Any], query: str, region: str, income_level: str, lending_type: str
) -> bool:
    if query:
        fields = ("id", "iso2Code", "name")
        haystack = " ".join(_string(country.get(key)) for key in fields).lower()
        if query.lower() not in haystack:
            return False
    wanted = (("region", region), ("incomeLevel", income_level), ("lendingType", lending_type))
    return all(
        not value or _string(_dict(country.get(key)).get("id")).upper() == value
        for key, value in wanted
    )


def _scan_indicators(query: str, topic: str, source: str, scan_pages: int) -> _Scan:
    """The indicators that match ``query`` (and ``topic`` and ``source``) in the first
    ``scan_pages`` catalogue pages."""
    tokens = _query_tokens(query)
    segments = _indicator_segments(topic, source)
    matches: list[_WorldBankIndicator] = []
    read = total = pages = 0
    for page in range(1, scan_pages + 1):
        params = {"page": str(page), "per_page": str(_CATALOGUE_PAGE)}
        metadata, indicators = _API.get_json_list(*segments, params=params, parse=_indicator_page)
        total = _int_or_none(metadata.get("total")) or total
        pages = _int_or_none(metadata.get("pages")) or pages
        read += len(indicators)
        matches.extend(
            indicator
            for indicator in indicators
            if (not topic or any(topic_id == topic for topic_id, _ in indicator.topics))
            and (not source or indicator.source_id == source)
            and all(token in _indicator_haystack(indicator) for token in tokens)
        )
        if page >= pages:
            break
    return _Scan(
        matches=list(dict.fromkeys(matches)), read=read, total=max(total, read), pages=pages
    )


def _search_answer(
    scan: _Scan, query: str, topic: str, source: str, page: int, per_page: int
) -> ToolResult:
    looked = f"searched {scan.read} of {scan.total} indicators"
    if scan.read < scan.total:
        looked += (
            f"; scan_pages={scan.pages} searches them all"
            if scan.pages <= _SCAN_PAGES
            else f"; a search reads at most {_SCAN_PAGES} pages: narrow it with topic or source"
        )
    under = _under(topic, source)
    asked = " ".join(part for part in (f"that match {query!r}" if query else "", under) if part)
    if not scan.matches:
        return ToolResult.success(f"No World Bank indicators {asked} ({looked}).")
    ranked = _rank_indicators(scan.matches, query)
    start = (page - 1) * per_page
    lines = [
        _indicator_text(number, indicator)
        for number, indicator in enumerate(ranked[start : start + per_page], start=start + 1)
    ]
    pages = -(-len(ranked) // per_page)
    window = _paged(lines, page=page, per_page=per_page, total=len(ranked), pages=pages)
    return window.result(
        heading=f"World Bank indicators {asked} ({looked}; world_bank_indicator gives a "
        "definition):"
    )


def _under(topic: str, source: str) -> str:
    """``under topic 3 and source 2``; empty without either."""
    named = [f"{kind} {value}" for kind, value in (("topic", topic), ("source", source)) if value]
    return f"under {' and '.join(named)}" if named else ""


def _indicator_page(payload: list[Any]) -> tuple[dict[str, Any], list[_WorldBankIndicator]]:
    metadata, items = _page(payload)
    parsed = [_parse_indicator(item) for item in items if isinstance(item, dict)]
    return metadata, [indicator for indicator in parsed if indicator is not None]


def _indicator_line(number: int, item: dict[str, Any]) -> str:
    indicator = _parse_indicator(item)
    return _indicator_text(number, indicator) if indicator else f"{number}. (unnamed)"


def _indicator_text(number: int, indicator: _WorldBankIndicator) -> str:
    parts = [f"{number}. {indicator.id}: {indicator.name}"]
    if indicator.unit:
        parts.append(f"unit: {indicator.unit}")
    if indicator.source:
        parts.append(f"source: {indicator.source} ({indicator.source_id})")
    if indicator.topics:
        parts.append("topics: " + ", ".join(f"{name} ({key})" for key, name in indicator.topics))
    return " | ".join(parts)


def _indicator_answer(payload: list[Any], indicator: str) -> ToolResult:
    _metadata, indicators = _indicator_page(payload)
    if not indicators:
        raise ToolFailure("not_found", _no_indicator(indicator))
    found = indicators[0]
    lines = [f"World Bank indicator {found.id}: {found.name}"]
    meta = [f"unit: {found.unit}"] if found.unit else []
    if found.source:
        meta.append(f"source: {found.source} ({found.source_id})")
    if found.topics:
        meta.append("topics: " + ", ".join(f"{name} ({key})" for key, name in found.topics))
    if meta:
        lines.append("   " + " | ".join(meta))
    if found.source_note:
        lines.append(f"   Definition: {found.source_note}")
    if found.source_organization:
        lines.append(f"   Source organization: {found.source_organization}")
    return ToolResult.success("\n".join(lines))


def _series_answer(payload: list[Any], asked: str, page: int, per_page: int) -> ToolResult:
    metadata, items = _page(payload)
    points = [item for item in items if isinstance(item, dict) and _string(item.get("date"))]
    if not points and page == 1:
        return ToolResult.success(f"No World Bank observations of {asked}.")
    lines = [
        _point_text(number, point)
        for number, point in enumerate(points, start=(page - 1) * per_page + 1)
    ]
    indicator = _dict(points[0].get("indicator")) if points else {}
    name = f"{_string(indicator.get('id'))}: {_string(indicator.get('value'))}".strip(": ")
    units = {unit for point in points if (unit := _string(point.get("unit")))}
    heading = (
        f"World Bank, {name or asked}"
        + (f" (unit: {', '.join(sorted(units))})" if units else "")
        + ":"
    )
    total = _int_or_none(metadata.get("total"))
    pages = _int_or_none(metadata.get("pages")) or 0
    window = _paged(lines, page=page, per_page=per_page, total=total, pages=pages)
    return window.result(heading=heading)


def _point_text(number: int, point: dict[str, Any]) -> str:
    """``3. Portugal (PRT), 2023: 29184912345600``: the value with every digit it has."""
    country = _dict(point.get("country"))
    code = _string(point.get("countryiso3code")) or _string(country.get("id"))
    name = _string(country.get("value"))
    place = f"{name} ({code})" if name and code else name or code or "(no country)"
    value = point.get("value")
    shown = plain_number(value) if isinstance(value, int | float | str) else "no value"
    status = _string(point.get("obs_status"))
    return f"{number}. {place}, {_date(_string(point.get('date')))}: {shown}" + (
        f" (status: {status})" if status else ""
    )


def _date(date: str) -> str:
    """A World Bank period in ISO 8601: ``2021M07`` as ``2021-07``, ``2021Q1`` as ``2021-Q1``."""
    found = re.fullmatch(r"(\d{4})([MQ])(\d{1,2})", date)
    if found is None:
        return date
    year, kind, step = found.groups()
    return f"{year}-{step.zfill(2)}" if kind == "M" else f"{year}-Q{step}"


# --- Requests ----------------------------------------------------------------------------------


def _indicator_segments(topic: str, source: str) -> tuple[str, ...]:
    if topic:
        return ("topic", topic, "indicator")
    if source:
        return ("source", source, "indicator")
    return ("indicator",)


def _series_params(start_year: str, end_year: str, page: int, per_page: int) -> dict[str, str]:
    params = {"page": str(page), "per_page": str(per_page)}
    if start_year.strip() or end_year.strip():
        start = start_year.strip() or end_year.strip()
        end = end_year.strip() or start_year.strip()
        params["date"] = f"{start}:{end}"
    return params


def _years(start_year: str, end_year: str) -> str:
    start = start_year.strip() or end_year.strip()
    end = end_year.strip() or start_year.strip()
    if not start:
        return ""
    return f" in {start}" if start == end else f" from {start} to {end}"


def _parse_indicator(data: dict[str, Any]) -> _WorldBankIndicator | None:
    indicator_id = _string(data.get("id"))
    name = _string(data.get("name"))
    if not indicator_id and not name:
        return None
    source = _dict(data.get("source"))
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


def _topics(value: Any) -> tuple[tuple[str, str], ...]:
    items = value if isinstance(value, list) else []
    pairs = [
        (_string(item.get("id")), _string(item.get("value")))
        for item in items
        if isinstance(item, dict)
    ]
    return tuple(pair for pair in pairs if any(pair))


def _rank_indicators(
    indicators: list[_WorldBankIndicator], query: str
) -> list[_WorldBankIndicator]:
    phrase = query.lower()

    def score(indicator: _WorldBankIndicator) -> int:
        name, indicator_id = indicator.name.lower(), indicator.id.lower()
        checks = (
            (indicator_id == phrase, 100),
            (name == phrase, 80),
            (name.startswith(phrase), 40),
            (phrase in name, 25),
            (phrase in indicator_id, 20),
            (phrase in _indicator_haystack(indicator), 10),
        )
        return sum(points for hit, points in checks if hit)

    return sorted(indicators, key=score, reverse=True) if phrase else indicators


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


def _query_tokens(query: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"\W+", query.lower()) if token)


def _country_codes(country: str) -> str:
    """The codes of ``country``, joined by ``;`` as the API takes them.

    Raises:
        ToolFailure: validation_error for an empty or malformed code, or "all" with others.
    """
    codes = [code.strip() for code in country.replace(";", ",").split(",") if code.strip()]
    if not codes or not all(_COUNTRY_RE.fullmatch(code) for code in codes):
        raise ToolFailure(
            "validation_error",
            f"invalid country code in {country!r}; use codes such as 'PRT', 'PRT,ESP', 'WLD' "
            "or 'all' (find them with world_bank_countries)",
        )
    if any(code.lower() == "all" for code in codes):
        if len(codes) > 1:
            raise ToolFailure(
                "validation_error", "give 'all' alone, or the countries' codes without it"
            )
        return "all"
    return ";".join(dict.fromkeys(code.upper() for code in codes))


def _indicator_id(indicator: str) -> str:
    """The stripped indicator ID; a malformed one raises ``ToolFailure`` (validation_error)."""
    stripped = indicator.strip()
    if not (stripped and _ID_RE.fullmatch(stripped)):
        raise ToolFailure(
            "validation_error",
            f"invalid indicator ID {indicator!r}; an ID looks like SP.POP.TOTL "
            "(find one with world_bank_indicators)",
        )
    return stripped


def _no_indicator(indicator: str) -> str:
    """The not_found message for an indicator ID the World Bank does not have."""
    return (
        f"the World Bank has no indicator {indicator}; search for one with world_bank_indicators"
    )


def _check_year_range(start_year: str, end_year: str) -> None:
    """Raises ``ToolFailure`` (validation_error) for a malformed year or a reversed range."""
    start, end = start_year.strip(), end_year.strip()
    problem = ""
    if start and not _YEAR_RE.fullmatch(start):
        problem = f"invalid start_year {start_year!r}; use YYYY"
    elif end and not _YEAR_RE.fullmatch(end):
        problem = f"invalid end_year {end_year!r}; use YYYY"
    elif start and end and int(start) > int(end):
        problem = f"start_year {start} is after end_year {end}; swap them"
    if problem:
        raise ToolFailure("validation_error", problem)


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip()) if value is not None else None
    except ValueError:
        return None
