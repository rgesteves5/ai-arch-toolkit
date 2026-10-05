"""Web search with a key of one's own: Brave Search and Tavily (D55), each priced per unit (D56).

Unlike the provider-hosted ``web_search()`` server tool, these run on the toolkit's side, so they
serve any model, local ones included, and the meter counts their cost: each request a service
accepts is billed at the price table's ``[tools]`` entry of the tool.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_MAX_RESULTS = 20
_SNIPPET_CHARS = 500
_MARKUP = re.compile(r"<[^>]+>")
_TOPICS = ("general", "news")
_TIME_RANGES = ("day", "week", "month", "year")
_FRESHNESS = ("pd", "pw", "pm", "py")


def _error_detail(answer: object) -> str | None:
    """The error Brave or Tavily explains in an error body, in its own words."""
    if not isinstance(answer, dict):
        return None
    error = answer.get("error") or answer.get("detail")
    if isinstance(error, dict):
        text = error.get("detail") or error.get("error") or error.get("message")
        return str(text) if text else None
    return str(error) if isinstance(error, str) and error else None


# Brave Search API: GET /res/v1/web/search, the key in X-Subscription-Token, up to 20 results
# (https://api-dashboard.search.brave.com/app/documentation/web-search/get-started).
_BRAVE = Api(
    base="https://api.search.brave.com/res/v1/web/search",
    name="Brave Search",
    key_env="BRAVE_SEARCH_API_KEY",
    key_header="X-Subscription-Token",
    key_url="https://brave.com/search/api/",
    key_required=True,
    billed_as="brave_search",
    status_messages={
        401: "Brave rejected the key in BRAVE_SEARCH_API_KEY (HTTP 401).",
        403: "Brave rejected the key in BRAVE_SEARCH_API_KEY (HTTP 403).",
    },
    body_error=_error_detail,
)


def _tavily_credits(text: str) -> int:
    """The credits Tavily says a search used (``include_usage``)."""
    return int(json.loads(text)["usage"]["credits"])


# Tavily: POST /search, the key as a Bearer token; a basic search is one credit, and the answer
# reports the credits it used (https://docs.tavily.com/documentation/api-reference/endpoint/search).
_TAVILY = Api(
    base="https://api.tavily.com/search",
    name="Tavily",
    timeout_s=30.0,
    key_env="TAVILY_API_KEY",
    key_header="Authorization",
    key_prefix="Bearer ",
    key_url="https://app.tavily.com",
    key_required=True,
    billed_as="tavily_search",
    bill_units=_tavily_credits,
    status_messages={
        401: "Tavily rejected the key in TAVILY_API_KEY (HTTP 401).",
        432: "the search exceeds your Tavily plan's limit (HTTP 432).",
        433: "the search exceeds your Tavily plan's limit for pay-as-you-go (HTTP 433).",
    },
    body_error=_error_detail,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _WebResult:
    title: str
    url: str
    snippet: str
    note: str = ""


def _plain(text: object) -> str:
    """A result's text without its markup (Brave bolds the query terms) or extra spaces."""
    flat = " ".join(html.unescape(_MARKUP.sub("", str(text or ""))).split())
    return flat if len(flat) <= _SNIPPET_CHARS else flat[: _SNIPPET_CHARS - 1] + "…"


def _brave_results(data: dict[str, Any]) -> list[_WebResult]:
    web = data.get("web") or {}
    return [
        _WebResult(
            title=_plain(item.get("title")) or item["url"],
            url=str(item["url"]),
            snippet=_plain(item.get("description")),
            note=str(item.get("age") or ""),
        )
        for item in web.get("results") or []
        if isinstance(item, dict) and item.get("url")
    ]


def _tavily_answer(data: dict[str, Any]) -> tuple[str, list[_WebResult]]:
    results = [
        _WebResult(
            title=_plain(item.get("title")) or item["url"],
            url=str(item["url"]),
            snippet=_plain(item.get("content")),
        )
        for item in data.get("results") or []
        if isinstance(item, dict) and item.get("url")
    ]
    return _plain(data.get("answer")), results


def _format(query: str, service: str, results: list[_WebResult], answer: str = "") -> str:
    if not results and not answer:
        return f"No web results for {query!r} ({service})."
    lines = [f"Web results for {query!r} ({service}):"]
    if answer:
        lines.append(f"Answer: {answer}")
    for index, result in enumerate(results, 1):
        lines.append(f"{index}. {result.title}")
        lines.append(f"   {result.url}")
        if result.snippet:
            note = f" ({result.note})" if result.note else ""
            lines.append(f"   {result.snippet}{note}")
    return "\n".join(lines)


@tool(capability="network")
def brave_search(query: str, max_results: int = 10, country: str = "", freshness: str = "") -> str:
    """Search the web with Brave Search (needs BRAVE_SEARCH_API_KEY; each search is billed).

    Args:
        query: What to search for; Brave takes operators such as quotes, -word and site:.
        max_results: Number of results to return (1-20). Defaults to 10.
        country: Optional two-letter country code to search from, e.g. "PT" or "US".
        freshness: Optional age limit: "pd" (a day), "pw" (a week), "pm" (a month), "py" (a year).
    """
    query = query.strip()
    if not query:
        return "Brave search failed: query cannot be empty."
    if freshness and freshness not in _FRESHNESS:
        return "Brave search failed: freshness must be one of pd, pw, pm, py."
    params = {"q": query, "count": str(max(1, min(max_results, _MAX_RESULTS)))}
    if country.strip():
        params["country"] = country.strip().upper()
    if freshness:
        params["freshness"] = freshness
    try:
        results = _BRAVE.get_json(params=params, parse=_brave_results)
    except HttpError as e:
        return f"Brave search failed: {e}"
    return _format(query, "Brave", results)


@tool(capability="network")
def tavily_search(
    query: str,
    max_results: int = 5,
    topic: str = "general",
    time_range: str = "",
    include_answer: bool = False,
) -> str:
    """Search the web with Tavily, which returns an excerpt of each page (needs TAVILY_API_KEY;
    each search is billed).

    Args:
        query: What to search for.
        max_results: Number of results to return (1-20). Defaults to 5.
        topic: "general" or "news". Defaults to "general".
        time_range: Optional age limit: "day", "week", "month" or "year".
        include_answer: Whether Tavily also writes a short answer from the results.
    """
    query = query.strip()
    if not query:
        return "Tavily search failed: query cannot be empty."
    if topic not in _TOPICS:
        return "Tavily search failed: topic must be 'general' or 'news'."
    if time_range and time_range not in _TIME_RANGES:
        return "Tavily search failed: time_range must be one of day, week, month, year."
    payload: dict[str, Any] = {
        "query": query,
        "search_depth": "basic",  # one credit; "advanced" costs two
        "max_results": max(1, min(max_results, _MAX_RESULTS)),
        "topic": topic,
        "include_answer": include_answer,
        "include_usage": True,  # the credits it used, which the meter charges
    }
    if time_range:
        payload["time_range"] = time_range
    try:
        answer, results = _TAVILY.post_json(payload=payload, parse=_tavily_answer)
    except HttpError as e:
        return f"Tavily search failed: {e}"
    return _format(query, "Tavily", results, answer)
