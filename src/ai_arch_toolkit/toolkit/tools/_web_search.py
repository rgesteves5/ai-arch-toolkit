"""Web search with a key of one's own: Brave Search and Tavily (D55), each priced per unit (D56).

Unlike the provider-hosted ``web_search()`` server tool, these run on the toolkit's side, so they
serve any model, local ones included, and the meter counts their cost: each request a service
accepts is billed at the price table's ``[tools]`` entry of the tool. They are network tools of
low risk that run without approval, as the others do (D65): the budget bounds their cost.

Each result shows whole, and only a result with an http(s) URL reaches the model (D65). Brave
pages its results, so ``brave_search`` ends a page with the window's footer and the call for the
next (D39); Tavily serves one page, and ``tavily_search`` says so when that page is full.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Annotated, Any
from urllib.parse import urlsplit

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._window import Window

# Brave serves at most 20 results a page and 10 pages, ``offset`` 0 to 9 counting pages of
# ``count`` results (https://api-dashboard.search.brave.com/api-reference/web/search/get).
_BRAVE_PAGE = 20
_BRAVE_LAST_OFFSET = 9
# Tavily returns 0 to 20 results, in one page: it takes no offset
# (https://docs.tavily.com/documentation/api-reference/endpoint/search).
_TAVILY_MOST = 20
_MARKUP = re.compile(r"<[^>]+>")
_WEB_SCHEMES = frozenset({"http", "https"})
_TOPICS = ("general", "news")
_TIME_RANGES = ("day", "week", "month", "year")
_FRESHNESS = ("pd", "pw", "pm", "py")


# --- Errors ------------------------------------------------------------------------------------


def _said(reply: Reply) -> str | None:
    """What Brave or Tavily says in an error answer, in its own words: Brave's ``error.detail``
    (its ``ErrorResponse``), Tavily's ``detail.error``, or the fields Tavily's 422 names (a
    ``detail`` list of ``loc`` and ``msg``). A success says nothing."""
    answer = reply.body
    if reply.status < 400 or not isinstance(answer, dict):
        return None
    said = answer.get("error") or answer.get("detail")
    if isinstance(said, dict):
        said = said.get("detail") or said.get("error") or said.get("message")
    if isinstance(said, list):
        said = "; ".join(_field_error(item) for item in said if isinstance(item, dict))
    if not isinstance(said, str):
        return None
    return " ".join(said.split()) or None


def _field_error(item: dict[str, Any]) -> str:
    """One field a 422 rejects, as ``query: Input should be a valid string``."""
    where = [str(part) for part in item.get("loc") or [] if part != "body"]
    message = " ".join(str(item.get("msg") or "").split())
    return f"{'.'.join(where)}: {message}" if where else message


def _refused_key(service: str, env: str, url: str, reply: Reply) -> ToolFailure:
    """A key the service refused: not retryable until someone sets a valid one."""
    said = _said(reply)
    msg = (
        f"{service} rejected the key in {env} (HTTP {reply.status})"
        + (f": {said.rstrip('.')}" if said else "")
        + f"; set a valid key in {env} (get one: {url})."
    )
    return ToolFailure("upstream", msg, retryable=False)


def _refused_search(service: str, reply: Reply, step: str) -> ToolFailure:
    """Arguments the service refused: the caller's to change, so a ``validation_error``."""
    said = _said(reply)
    msg = (
        f"{service} refused the search (HTTP {reply.status})"
        + (f": {said.rstrip('.')}" if said else "")
        + f"; {step}."
    )
    return ToolFailure("validation_error", msg, retryable=False)


def _brave_error(reply: Reply) -> ToolFailure | str | None:
    """Brave's errors: 401 and 403 refuse the key; 422 refuses the arguments; any other (404,
    429, a server's), in Brave's words, typed by its status."""
    if reply.status in (401, 403):
        return _refused_key(
            "Brave", "BRAVE_SEARCH_API_KEY", "https://brave.com/search/api/", reply
        )
    if reply.status == 422:
        return _refused_search("Brave", reply, "a query takes at most 600 characters and 75 words")
    return _said(reply)


def _tavily_error(reply: Reply) -> ToolFailure | str | None:
    """Tavily's errors (https://docs.tavily.com/documentation/api-reference/endpoint/search):
    401 refuses the key; 400 and 422 refuse the arguments; 432 and 433 are the plan's and the
    pay-as-you-go usage limits, which stand until the account changes, so they are
    ``rate_limited`` but not retryable; any other (429, 500), in Tavily's words."""
    if reply.status == 401:
        return _refused_key("Tavily", "TAVILY_API_KEY", "https://app.tavily.com", reply)
    if reply.status in (400, 422):
        return _refused_search("Tavily", reply, "change the arguments")
    limits = {
        432: "the search exceeds your Tavily plan's usage limit (HTTP 432)",
        433: "the search exceeds your Tavily pay-as-you-go limit (HTTP 433)",
    }
    if (limit := limits.get(reply.status)) is not None:
        said = _said(reply)
        msg = (
            limit
            + (f": {said.rstrip('.')}" if said else "")
            + "; raise it at https://app.tavily.com."
        )
        return ToolFailure("rate_limited", msg, retryable=False)
    return _said(reply)


# --- The APIs ----------------------------------------------------------------------------------

# Brave Search API: GET /res/v1/web/search, the key in X-Subscription-Token
# (https://api-dashboard.search.brave.com/app/documentation/web-search/get-started); a failed
# request is not billed (https://api-dashboard.search.brave.com/documentation/guides/rate-limiting).
_BRAVE = Api(
    base="https://api.search.brave.com/res/v1/web/search",
    name="Brave Search",
    key_env="BRAVE_SEARCH_API_KEY",
    key_header="X-Subscription-Token",
    key_url="https://brave.com/search/api/",
    key_required=True,
    billed_as="brave_search",
    error_reader=_brave_error,
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
    error_reader=_tavily_error,
)


# --- Results -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class _WebResult:
    """One result: its place in the source's ranking (1 is the first result of the first page),
    and what the model reads."""

    position: int
    title: str
    url: str
    snippet: str
    note: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Found:
    """A page of results as the source sent it.

    Attributes:
        results: Those with an http(s) URL.
        sent: How many the source sent.
        more: The source has another page (Brave's ``more_results_available``).
        answer: The short answer Tavily writes when asked.
    """

    results: tuple[_WebResult, ...]
    sent: int
    more: bool = False
    answer: str = ""

    @property
    def left_out(self) -> int:
        """Results without an http(s) URL, which never reach the model."""
        return self.sent - len(self.results)


def _plain(text: object) -> str:
    """A result's text without its markup (Brave bolds the query terms) or extra spaces."""
    return " ".join(html.unescape(_MARKUP.sub("", str(text or ""))).split())


def _web_url(value: object) -> str | None:
    """``value`` when it is an http(s) URL with a host; ``None`` for anything else, such as a
    ``javascript:`` or ``data:`` link."""
    try:
        parts = urlsplit(str(value or "").strip())
    except ValueError:  # a malformed IPv6 host
        return None
    return parts.geturl() if parts.scheme in _WEB_SCHEMES and parts.hostname else None


def _listed(items: object, *, first: int, snippet: str, note: str = "") -> _Found:
    """The results of one page, numbered from ``first``: each with an http(s) URL, whole."""
    sent = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    results = []
    for position, item in enumerate(sent, first):
        if (url := _web_url(item.get("url"))) is None:
            continue
        results.append(
            _WebResult(
                position=position,
                title=_plain(item.get("title")) or url,
                url=url,
                snippet=_plain(item.get(snippet)),
                note=_plain(item.get(note)) if note else "",
            )
        )
    return _Found(results=tuple(results), sent=len(sent))


def _lines(found: _Found) -> list[str]:
    lines = [f"Answer: {found.answer}"] if found.answer else []
    for result in found.results:
        lines += [f"{result.position}. {result.title}", f"   {result.url}"]
        if result.snippet:
            note = f" ({result.note})" if result.note else ""
            lines.append(f"   {result.snippet}{note}")
    if found.left_out:
        plural = "s" if found.left_out > 1 else ""
        lines.append(f"({found.left_out} result{plural} without an http(s) URL left out.)")
    return lines


def _nothing(query: str, service: str, found: _Found) -> ToolResult:
    """Zero results: a success that says so, with the query."""
    text = f"No web results for {query!r} ({service})."
    if found.left_out:
        text += f" {found.left_out} without an http(s) URL left out."
    return ToolResult.success(text)


def _brave_page(data: dict[str, Any], offset: int, count: int) -> _Found:
    web = data.get("web") or {}
    found = _listed(
        web.get("results"), first=offset * count + 1, snippet="description", note="age"
    )
    query = data.get("query") or {}
    return replace(found, more=query.get("more_results_available") is True)


def _brave_answer(found: _Found, query: str, offset: int, count: int) -> ToolResult:
    """A page of Brave's results; while Brave has more, the footer names the next page."""
    if not found.results and offset == 0 and not found.more:
        return _nothing(query, "Brave", found)
    lines = _lines(found)
    onward = found.more and offset < _BRAVE_LAST_OFFSET
    if found.more and not onward:
        lines.append(
            f"(Brave serves no page past offset {_BRAVE_LAST_OFFSET}: refine the query for "
            "other results.)"
        )
    first = offset * count + 1
    window = Window(
        body="\n".join(lines),
        unit="results",
        first=first,
        last=first + found.sent - 1,
        total=None,
        next_call={"offset": offset + 1, "max_results": count} if onward else None,
    )
    return window.result(heading=f"Web results for {query!r} (Brave):")


def _tavily_page(data: dict[str, Any]) -> _Found:
    found = _listed(data.get("results"), first=1, snippet="content")
    return replace(found, answer=_plain(data.get("answer")))


def _tavily_answer(found: _Found, query: str, max_results: int) -> ToolResult:
    """Tavily's one page; when it is full, how to get other results."""
    if not found.results and not found.answer:
        return _nothing(query, "Tavily", found)
    lines = [f"Web results for {query!r} (Tavily):", *_lines(found)]
    if max_results and found.sent >= max_results:
        lines.append(
            "Tavily returns one page: for more results, raise max_results (up to "
            f"{_TAVILY_MOST}) or refine the query."
            if max_results < _TAVILY_MOST
            else f"Tavily returns one page, of {_TAVILY_MOST} results at most: refine the query "
            "for others."
        )
    return ToolResult.success("\n".join(lines))


# --- The tools ---------------------------------------------------------------------------------


def _query(query: str) -> str:
    """The query, stripped; an empty one is refused before any request."""
    if not query.strip():
        raise ToolFailure("validation_error", "empty query; say what to search for")
    return query.strip()


def _choice(value: str, allowed: Sequence[str], name: str) -> None:
    """Refuse ``value`` unless it is empty or one of ``allowed``."""
    if value and value not in allowed:
        choices = ", ".join(allowed)
        raise ToolFailure("validation_error", f"invalid {name} {value!r}; use {choices}, or ''")


@tool(capability="network")
def brave_search(
    query: str,
    max_results: Annotated[int, Range(1, _BRAVE_PAGE)] = 10,
    country: str = "",
    freshness: str = "",
    offset: Annotated[int, Range(0, _BRAVE_LAST_OFFSET)] = 0,
) -> ToolResult:
    """Search the web with Brave Search (needs BRAVE_SEARCH_API_KEY; each search is billed).

    Args:
        query: What to search for; Brave takes operators such as quotes, -word and site:.
        max_results: How many results a page holds.
        country: Optional two-letter country code to search from, e.g. "PT" or "US".
        freshness: Optional age limit: "pd" (a day), "pw" (a week), "pm" (a month), "py" (a year).
        offset: How many pages of ``max_results`` results to skip; the footer gives the next
            call.

    Raises:
        ToolFailure: validation_error when the query is empty, the freshness is unknown,
            BRAVE_SEARCH_API_KEY is not set or Brave refuses the arguments; upstream (not
            retryable) when Brave refuses the key.
    """
    query = _query(query)
    _choice(freshness, _FRESHNESS, "freshness")
    params = {"q": query, "count": str(max_results)}
    if country.strip():
        params["country"] = country.strip().upper()
    if freshness:
        params["freshness"] = freshness
    if offset:
        params["offset"] = str(offset)
    return _BRAVE.get_json(
        params=params,
        parse=lambda data: _brave_answer(
            _brave_page(data, offset, max_results), query, offset, max_results
        ),
    )


@tool(capability="network")
def tavily_search(
    query: str,
    max_results: Annotated[int, Range(0, _TAVILY_MOST)] = 5,
    topic: str = "general",
    time_range: str = "",
    include_answer: bool = False,
) -> ToolResult:
    """Search the web with Tavily, which returns an excerpt of each page (needs TAVILY_API_KEY;
    each search is billed). Tavily serves one page of results.

    Args:
        query: What to search for.
        max_results: How many results to return; 0 asks only for the answer
            (``include_answer``).
        topic: "general" or "news". Defaults to "general".
        time_range: Optional age limit: "day", "week", "month" or "year".
        include_answer: Whether Tavily also writes a short answer from the results.

    Raises:
        ToolFailure: validation_error when the query is empty, the topic or the time range is
            unknown, max_results is 0 without include_answer, TAVILY_API_KEY is not set or
            Tavily refuses the arguments; upstream (not retryable) when Tavily refuses the key;
            rate_limited (not retryable) when the search exceeds the plan's limit.
    """
    query = _query(query)
    if topic not in _TOPICS:
        raise ToolFailure("validation_error", f"invalid topic {topic!r}; use 'general' or 'news'")
    _choice(time_range, _TIME_RANGES, "time_range")
    if max_results == 0 and not include_answer:
        raise ToolFailure(
            "validation_error",
            "max_results=0 asks only for Tavily's answer: set include_answer=True, or ask for "
            f"1 to {_TAVILY_MOST} results",
        )
    payload: dict[str, Any] = {
        "query": query,
        "search_depth": "basic",  # one credit; "advanced" costs two
        "max_results": max_results,
        "topic": topic,
        "include_answer": include_answer,
        "include_usage": True,  # the credits it used, which the meter charges
    }
    if time_range:
        payload["time_range"] = time_range
    return _TAVILY.post_json(
        payload=payload,
        parse=lambda data: _tavily_answer(_tavily_page(data), query, max_results),
    )
