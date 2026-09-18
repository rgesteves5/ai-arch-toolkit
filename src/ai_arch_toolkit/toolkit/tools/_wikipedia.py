"""Wikipedia tools — article search, summaries, and related pages."""

from __future__ import annotations

from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_API = Api(base="https://en.wikipedia.org/w/api.php", name="Wikipedia")
_MAX_CHARS_LIMIT = 100_000


@tool(capability="network")
def wikipedia_search(query: str, results: int = 3) -> str:
    """Search Wikipedia and return article titles with summaries.

    Args:
        query: The search query.
        results: Number of results to return (1-10). Defaults to 3.
    """
    results = max(1, min(results, 10))
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": results,
        "format": "json",
        "utf8": "1",
    }
    try:
        return _API.get_json(params=params, parse=lambda data: _search_text(data, query))
    except HttpError as e:
        return f"Wikipedia search failed: {e}"


@tool(capability="network")
def wikipedia_article(title: str, max_chars: int = 4000) -> str:
    """Get the summary extract of a Wikipedia article.

    Args:
        title: Exact article title, e.g. "Python (programming language)".
        max_chars: Maximum characters to return (1-100000). Defaults to 4000.
    """
    max_chars = max(1, min(max_chars, _MAX_CHARS_LIMIT))
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "format": "json",
        "utf8": "1",
    }
    try:
        return _API.get_json(
            params=params, parse=lambda data: _article_text(data, title, max_chars)
        )
    except HttpError as e:
        return f"Wikipedia API failed: {e}"


@tool(capability="network")
def wikipedia_related(title: str, limit: int = 5) -> str:
    """Get related Wikipedia article titles from a page's outgoing links.

    Falls back to a regular Wikipedia search if the page is missing.

    Args:
        title: Exact article title, e.g. "Python (programming language)".
        limit: Number of related pages to return (1-20). Defaults to 5.
    """
    limit = max(1, min(limit, 20))
    params = {
        "action": "query",
        "titles": title,
        "prop": "links",
        "plnamespace": "0",
        "pllimit": limit,
        "redirects": "1",
        "format": "json",
        "utf8": "1",
    }
    try:
        related = _API.get_json(
            params=params, parse=lambda data: _related_text(data, title, limit)
        )
    except HttpError as e:
        return f"Wikipedia related lookup failed: {e}"
    if related is None:
        return wikipedia_search(title, results=limit)
    return related


def _search_text(data: dict[str, Any], query: str) -> str:
    items = data.get("query", {}).get("search", [])
    if not items:
        return f"No Wikipedia results for: {query!r}"

    lines: list[str] = []
    for item in items:
        title = item.get("title", "")
        snippet = _strip_html(item.get("snippet", ""))
        lines.append(f"  - {title}: {snippet}")

    return f"Wikipedia results for {query!r}:\n" + "\n".join(lines)


def _article_text(data: dict[str, Any], title: str, max_chars: int) -> str:
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return f"Article not found: {title!r}"
        extract = page.get("extract", "")
        if not extract:
            return f"No extract available for: {title!r}"
        if len(extract) > max_chars:
            return extract[:max_chars] + "\n\n[Truncated]"
        return f"{page.get('title', title)}:\n{extract}"

    return f"Article not found: {title!r}"


def _related_text(data: dict[str, Any], title: str, limit: int) -> str | None:
    """The page's outgoing links, or ``None`` to fall back to a search."""
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return None

        links = page.get("links", [])
        if not links:
            return None

        lines = [f"Related Wikipedia pages for {page.get('title', title)!r}:"]
        for item in links[:limit]:
            link_title = item.get("title", "")
            if link_title:
                lines.append(f"  - {link_title}")
        if len(lines) == 1:
            return None
        return "\n".join(lines)

    return None


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    import re

    return re.sub(r"<[^>]+>", "", text)
