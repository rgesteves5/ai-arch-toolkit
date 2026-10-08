"""GDELT tools: global news search and the volume timeline of a query (DOC 2.0 API).

The API (https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) lists at most 250 articles a
query (``maxrecords``), with no offset and no total: the search asks for the articles up to the
end of the page shown and reads on by ``offset``, through the window (D39). A timeline has one
point per 15 minutes under 72 hours, per hour up to a week, and per day beyond; its points read
on by ``offset`` too. Times are ISO 8601 UTC.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._numbers import plain_number
from ai_arch_toolkit.toolkit.tools._window import list_window, page_window


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
# The most articles GDELT lists for a query (MAXRECORDS, the API's announcement).
_MAX_RECORDS = 250
_TIMELINE_PAGE = 100
# A number and a unit: minutes, hours, days, weeks or months (TIMESPAN).
_TIMESPAN_RE = re.compile(r"^\d+(?:min|h|d|w|m)$", re.IGNORECASE)
_SORT_VALUES = {
    "hybrid": "HybridRel",
    "date": "DateDesc",
    "tone": "ToneDesc",
}
_SORT_WORDS = {"hybrid": "by relevance", "date": "newest first", "tone": "most positive first"}


@dataclass(frozen=True, slots=True, kw_only=True)
class _GdeltArticle:
    """Normalized GDELT article."""

    title: str
    url: str
    source_country: str
    domain: str
    language: str
    seendate: str
    tone: float | None


@tool(capability="network")
def gdelt_news_search(
    query: str,
    max_results: Annotated[int, Range(1, 50)] = 10,
    offset: Annotated[int, Range(0, _MAX_RECORDS - 1)] = 0,
    timespan: str = "7d",
    sort: str = "hybrid",
) -> ToolResult:
    """Search global news articles with the GDELT DOC 2.0 API.

    Args:
        query: GDELT full-text query, e.g. 'climate sourcecountry:france'.
        max_results: How many articles to show.
        offset: How many articles to skip; the footer gives the next offset.
        timespan: Recent time window: a number and min, h, d, w or m (months), e.g. "24h",
            "7d" or "3m"; GDELT searches the last 3 months.
        sort: Sort mode: hybrid (relevance), date (newest first) or tone (most positive first).

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

    # GDELT has no offset: the tool asks for the articles up to the page's end, and one more.
    end = offset + max_results
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(min(end + 1, _MAX_RECORDS)),
        "timespan": timespan,
        "sort": _SORT_VALUES[sort],
    }
    where = f"{query!r} in the last {timespan}"
    return _API.get_json(
        params=params,
        parse=lambda data: _articles_answer(
            data, where=where, order=_SORT_WORDS[sort], offset=offset, end=end
        ),
    )


@tool(capability="network")
def gdelt_timeline(
    query: str, timespan: str = "30d", offset: Annotated[int, Range(0)] = 0
) -> ToolResult:
    """Get the GDELT volume timeline of a query: the share of all the news coverage GDELT
    monitored that matched it, step by step.

    Args:
        query: GDELT full-text query.
        timespan: Recent time window: a number and min, h, d, w or m (months), e.g. "24h",
            "30d" or "12w"; steps are 15 minutes under 72 hours, hours up to a week, days beyond.
        offset: How many points to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when ``query`` is empty, ``timespan`` is invalid, or
            GDELT cannot run the query.
    """
    query = _query(query)
    timespan = _timespan(timespan, "30d")
    params = {"query": query, "mode": "timelinevol", "format": "json", "timespan": timespan}
    where = f"{query!r} in the last {timespan}"
    return _API.get_json(
        params=params, parse=lambda data: _timeline_answer(data, where=where, offset=offset)
    )


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
            f"invalid timespan {timespan!r}; use a number and a unit min, h, d, w or m "
            "(months), e.g. '24h' or '7d'."
        )
        raise ToolFailure("validation_error", msg)
    return timespan


def _articles_answer(
    data: dict[str, Any], *, where: str, order: str, offset: int, end: int
) -> ToolResult:
    """The page of articles from ``offset`` to ``end``: GDELT's list holds every article up to
    the page's end, and one more.

    The one more says there is a next page; a list that stops before it is all there is (its
    length is the total); a list cut at GDELT's cap may go on past it, where no call reads.
    """
    parsed = (_parse_article(item) for item in data.get("articles", []) if isinstance(item, dict))
    articles = [article for article in parsed if article is not None]
    if not articles:
        return ToolResult.success(f"No GDELT articles match {where}.")
    page = articles[offset:end]
    lines = [_article_text(number, article) for number, article in enumerate(page, offset + 1)]
    complete = len(articles) < min(end + 1, _MAX_RECORDS)
    more = len(articles) > end
    window = list_window(
        lines,
        first=offset + 1,
        total=len(articles) if complete else None,
        next_call={"offset": end} if more and page else None,
    )
    heading = f"GDELT articles that match {where}, {order}:"
    if not complete and not more:  # cut at GDELT's cap, at the page's end
        heading += (
            f"\nGDELT lists at most {_MAX_RECORDS} articles for a query: narrow the timespan or "
            "the query, or change the sort, to see others."
        )
    return window.result(heading=heading)


def _timeline_answer(data: dict[str, Any], *, where: str, offset: int) -> ToolResult:
    """The points of the timeline's one series: ``{"timeline": [{"series", "data": [...]}]}``."""
    series = next((item for item in data.get("timeline", []) if isinstance(item, dict)), {})
    points = [line for item in series.get("data", []) if (line := _point_text(item))]
    if not points:
        return ToolResult.success(f"No GDELT timeline points for {where}.")
    name = _string(series.get("series")) or "volume"
    details = data.get("query_details")
    step = _string(details.get("date_resolution")) if isinstance(details, Mapping) else ""
    heading = (
        f"GDELT timeline for {where} ({name}: the share of all the coverage GDELT monitored "
        f"that matched, in %{f', by {step}' if step else ''}):"
    )
    return page_window(points, offset=offset, limit=_TIMELINE_PAGE).result(heading=heading)


def _parse_article(data: dict[str, Any]) -> _GdeltArticle | None:
    title = _string(data.get("title"))
    url = _string(data.get("url"))
    if not title and not url:
        return None
    return _GdeltArticle(
        title=title or "(untitled)",
        url=url,
        source_country=_string(data.get("sourcecountry")),
        domain=_string(data.get("domain")),
        language=_string(data.get("language")),
        seendate=_iso(_string(data.get("seendate"))),
        tone=_float_or_none(data.get("tone")),
    )


def _article_text(number: int, article: _GdeltArticle) -> str:
    meta = [f"seen {article.seendate}"] if article.seendate else []
    for label, value in (
        ("domain", article.domain),
        ("country", article.source_country),
        ("language", article.language),
    ):
        if value:
            meta.append(f"{label}: {value}")
    if article.tone is not None:
        meta.append(f"tone: {plain_number(round(article.tone, 2))}")
    lines = [f"{number}. {article.title}"]
    if meta:
        lines.append("   " + " | ".join(meta))
    if article.url:
        lines.append(f"   {article.url}")
    return "\n".join(lines)


def _point_text(data: object) -> str:
    """A timeline point as ``time: value%``; empty for one without either."""
    if not isinstance(data, dict):
        return ""
    date = _iso(_string(data.get("date") or data.get("datetime")))
    value = _float_or_none(data.get("value"))
    if not date and value is None:
        return ""
    return f"{date or '(no time)'}: {'no value' if value is None else plain_number(value) + '%'}"


def _iso(stamp: str) -> str:
    """GDELT's ``20260611T120000Z`` (or ``20260611120000``) as ISO 8601 UTC; other text as it
    is."""
    digits = stamp.replace("T", "").removesuffix("Z")
    if len(digits) != 14 or not digits.isdigit():
        return stamp
    return (
        f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}T{digits[8:10]}:{digits[10:12]}:{digits[12:]}Z"
    )


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if value is not None and str(value).strip():
            return float(value)
    except ValueError:
        return None
    return None


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
