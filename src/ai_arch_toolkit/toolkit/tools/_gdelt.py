"""GDELT tools — public global news search and timeline lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply


def _query_error(reply: Reply) -> ToolFailure | None:
    """The error GDELT sends in place of the JSON; ``None`` for a result.

    GDELT answers a request it cannot run with a line of text instead of JSON, still with HTTP
    200 (seen 2026-09-29: "Your query was too short or too long.", "Invalid/Unsupported
    Country."): the request is at fault, a ``validation_error``. It explains an error status (a
    429) the same way, and the door reads that text itself. Text that starts like JSON is a
    broken answer, and markup is a page, not a message.
    """
    body = reply.body
    message = " ".join(body.split()) if isinstance(body, str) else ""
    if reply.status >= 400 or not message or message.startswith(("{", "[", "<")):
        return None
    return ToolFailure(
        "validation_error", f"{message.rstrip('.')}; change the query or its options"
    )


# At most one request every 5 seconds, with a margin: GDELT answers faster callers with a 429.
# After a 429 its gate stays shut a minute or more, with no Retry-After
# (https://github.com/cyanheads/gdelt-mcp-server/issues/44; every request of 2026-10-04 got one,
# whatever the User-Agent): the host rests 60 s instead of prolonging it (D53).
_API = Api(
    base="https://api.gdeltproject.org/api/v2/doc/doc",
    name="GDELT",
    timeout_s=15,
    min_interval_s=5.1,
    cooldown_s=60.0,
    error_reader=_query_error,
)
_MAX_RESULTS_LIMIT = 20
_TIMESPAN_RE = re.compile(r"^\d+[mhdw]$", re.IGNORECASE)
_SORT_VALUES = {
    "hybrid": "HybridRel",
    "date": "DateDesc",
    "tone": "ToneDesc",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class _GdeltArticle:
    """Normalized GDELT article."""

    title: str
    url: str
    source_country: str
    domain: str
    language: str
    seendate: str
    social_image: str
    tone: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class _GdeltTimelinePoint:
    """Normalized GDELT timeline point."""

    date: str
    value: float | None


@tool(capability="network")
def gdelt_news_search(
    query: str,
    max_results: int = 10,
    timespan: str = "7d",
    sort: str = "hybrid",
) -> str:
    """Search global news articles using the public GDELT DOC 2.0 API.

    Args:
        query: GDELT full-text query.
        max_results: Number of articles to return (1-20). Defaults to 10.
        timespan: Recent time window, e.g. "24h", "7d", or "4w".
        sort: Sort mode: hybrid, date, or tone.

    Raises:
        ToolFailure: validation_error when ``query`` is empty, ``timespan`` or ``sort`` is
            invalid, or GDELT cannot run the query.
    """
    query = _query(query)
    timespan = _timespan(timespan, "7d")
    sort = sort.strip() or "hybrid"
    if sort not in _SORT_VALUES:
        msg = f"sort must be one of hybrid, date, tone, got {sort!r}."
        raise ToolFailure("validation_error", msg)

    max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max_results),
        "timespan": timespan,
        "sort": _SORT_VALUES[sort],
    }
    return _API.get_json(params=params, parse=lambda data: _articles_text(data, query))


@tool(capability="network")
def gdelt_timeline(query: str, timespan: str = "30d") -> str:
    """Fetch a GDELT volume timeline for a query.

    Args:
        query: GDELT full-text query.
        timespan: Recent time window, e.g. "24h", "30d", or "12w".

    Raises:
        ToolFailure: validation_error when ``query`` is empty, ``timespan`` is invalid, or
            GDELT cannot run the query.
    """
    query = _query(query)
    timespan = _timespan(timespan, "30d")
    params = {"query": query, "mode": "timelinevol", "format": "json", "timespan": timespan}
    return _API.get_json(params=params, parse=lambda data: _timeline_text(data, query))


def _query(query: str) -> str:
    query = query.strip()
    if not query:
        msg = "query cannot be empty; pass GDELT full-text query terms, e.g. 'climate'."
        raise ToolFailure("validation_error", msg)
    return query


def _timespan(timespan: str, default: str) -> str:
    timespan = timespan.strip() or default
    if not _TIMESPAN_RE.fullmatch(timespan):
        msg = (
            f"invalid timespan {timespan!r}; use a number and a unit m, h, d or w, "
            "e.g. '24h' or '7d'."
        )
        raise ToolFailure("validation_error", msg)
    return timespan


def _articles_text(data: dict[str, Any], query: str) -> str:
    articles = [
        _parse_article(item) for item in data.get("articles", []) if isinstance(item, dict)
    ]
    articles = [article for article in articles if article is not None]
    if not articles:
        return f"No GDELT articles found for: {query!r}"
    return f"GDELT articles for {query!r}:\n" + _format_articles(articles)


def _timeline_text(data: dict[str, Any], query: str) -> str:
    """The points of the timeline's one series: ``{"timeline": [{"series", "data": [...]}]}``."""
    series = next((item for item in data.get("timeline", []) if isinstance(item, dict)), {})
    points = [_parse_timeline_point(item) for item in series.get("data", [])]
    points = [point for point in points if point is not None]
    if not points:
        return f"No GDELT timeline points found for: {query!r}"
    name = str(series.get("series", "") or "").strip()
    heading = f"GDELT timeline for {query!r}" + (f" ({name})" if name else "")
    if len(points) > _MAX_RESULTS_LIMIT:
        heading += f", first {_MAX_RESULTS_LIMIT} of {len(points)} points"
    return f"{heading}:\n" + _format_timeline(points)


def _parse_article(data: dict[str, Any]) -> _GdeltArticle | None:
    title = str(data.get("title", "") or "").strip()
    url = str(data.get("url", "") or "").strip()
    if not title and not url:
        return None
    return _GdeltArticle(
        title=title or "(untitled)",
        url=url,
        source_country=str(data.get("sourcecountry", "") or "").strip(),
        domain=str(data.get("domain", "") or "").strip(),
        language=str(data.get("language", "") or "").strip(),
        seendate=str(data.get("seendate", "") or "").strip(),
        social_image=str(data.get("socialimage", "") or "").strip(),
        tone=_float_or_none(data.get("tone")),
    )


def _parse_timeline_point(data: Any) -> _GdeltTimelinePoint | None:
    if not isinstance(data, dict):
        return None
    date = str(data.get("date", "") or data.get("datetime", "") or "").strip()
    value = _float_or_none(data.get("value") if "value" in data else data.get("norm"))
    if not date and value is None:
        return None
    return _GdeltTimelinePoint(date=date, value=value)


def _format_articles(articles: list[_GdeltArticle]) -> str:
    blocks: list[str] = []
    for index, article in enumerate(articles, start=1):
        lines = [f"{index}. {article.title}"]
        meta = []
        if article.seendate:
            meta.append(f"seen: {article.seendate}")
        if article.domain:
            meta.append(f"domain: {article.domain}")
        if article.source_country:
            meta.append(f"country: {article.source_country}")
        if article.language:
            meta.append(f"language: {article.language}")
        if article.tone is not None:
            meta.append(f"tone: {article.tone:.2f}")
        if meta:
            lines.append("   " + " | ".join(meta))
        if article.social_image:
            lines.append(f"   Image: {article.social_image}")
        if article.url:
            lines.append(f"   URL: {article.url}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_timeline(points: list[_GdeltTimelinePoint]) -> str:
    lines: list[str] = []
    for index, point in enumerate(points[:_MAX_RESULTS_LIMIT], start=1):
        value = "" if point.value is None else f" | value: {point.value:.6g}"
        lines.append(f"{index}. {point.date}{value}")
    return "\n".join(lines)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if value is not None and str(value).strip():
            return float(value)
    except ValueError:
        return None
    return None
