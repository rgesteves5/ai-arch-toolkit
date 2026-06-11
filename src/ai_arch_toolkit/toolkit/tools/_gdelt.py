"""GDELT tools — public global news search and timeline lookup."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool

_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_TIMEOUT = 15
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_RESULTS_LIMIT = 20
_GDELT_INTERVAL_SECONDS = 5.1
_LAST_REQUEST_AT = 0.0
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


@tool
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
    """
    query = query.strip()
    if not query:
        return "GDELT news search failed: query cannot be empty."
    timespan = timespan.strip() or "7d"
    if not _TIMESPAN_RE.fullmatch(timespan):
        return f"GDELT news search failed: invalid timespan: {timespan!r}"
    sort = sort.strip() or "hybrid"
    if sort not in _SORT_VALUES:
        return "GDELT news search failed: sort must be one of hybrid, date, tone."

    max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))
    try:
        data = _fetch_json(
            {
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": str(max_results),
                "timespan": timespan,
                "sort": _SORT_VALUES[sort],
            }
        )
        articles = [
            _parse_article(item) for item in data.get("articles", []) if isinstance(item, dict)
        ]
        articles = [article for article in articles if article is not None]
    except urllib.error.HTTPError as e:
        return _http_error("GDELT news search failed", e)
    except urllib.error.URLError as e:
        return f"GDELT news search failed: URL error: {e.reason}"
    except TimeoutError:
        return "GDELT news search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"GDELT news search failed: could not parse API response: {e}"

    if not articles:
        return f"No GDELT articles found for: {query!r}"

    return f"GDELT articles for {query!r}:\n" + _format_articles(articles)


@tool
def gdelt_timeline(query: str, timespan: str = "30d") -> str:
    """Fetch a GDELT volume timeline for a query.

    Args:
        query: GDELT full-text query.
        timespan: Recent time window, e.g. "24h", "30d", or "12w".
    """
    query = query.strip()
    if not query:
        return "GDELT timeline failed: query cannot be empty."
    timespan = timespan.strip() or "30d"
    if not _TIMESPAN_RE.fullmatch(timespan):
        return f"GDELT timeline failed: invalid timespan: {timespan!r}"

    try:
        data = _fetch_json(
            {
                "query": query,
                "mode": "timelinevol",
                "format": "json",
                "timespan": timespan,
            }
        )
        points = [_parse_timeline_point(item) for item in data.get("timeline", [])]
        points = [point for point in points if point is not None]
    except urllib.error.HTTPError as e:
        return _http_error("GDELT timeline failed", e)
    except urllib.error.URLError as e:
        return f"GDELT timeline failed: URL error: {e.reason}"
    except TimeoutError:
        return "GDELT timeline failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"GDELT timeline failed: could not parse API response: {e}"

    if not points:
        return f"No GDELT timeline points found for: {query!r}"

    return f"GDELT timeline for {query!r}:\n" + _format_timeline(points)


def _fetch_json(params: dict[str, str]) -> dict[str, Any]:
    url = f"{_DOC_API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    _throttle()
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _throttle() -> None:
    global _LAST_REQUEST_AT

    now = time.monotonic()
    elapsed = now - _LAST_REQUEST_AT
    if elapsed < _GDELT_INTERVAL_SECONDS:
        time.sleep(_GDELT_INTERVAL_SECONDS - elapsed)
    _LAST_REQUEST_AT = time.monotonic()


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 429:
        detail = _read_error_body(error)
        if detail:
            return f"{prefix}: rate limited by GDELT (HTTP 429): {detail}"
        return f"{prefix}: rate limited by GDELT (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _read_error_body(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    return " ".join(body.split())


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
