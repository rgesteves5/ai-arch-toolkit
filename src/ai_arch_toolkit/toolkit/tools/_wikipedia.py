"""Wikipedia tools — article search, summaries, and related pages."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from ai_arch_toolkit.core import tool

_TIMEOUT = 10
_USER_AGENT = "ai-arch-toolkit/1.0"


@tool
def wikipedia_search(query: str, results: int = 3) -> str:
    """Search Wikipedia and return article titles with summaries.

    Args:
        query: The search query.
        results: Number of results to return (1-10). Defaults to 3.
    """
    results = max(1, min(results, 10))
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={urllib.parse.quote(query)}"
        f"&srlimit={results}&format=json&utf8=1"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Wikipedia search failed: {e}"

    items = data.get("query", {}).get("search", [])
    if not items:
        return f"No Wikipedia results for: {query!r}"

    lines: list[str] = []
    for item in items:
        title = item.get("title", "")
        snippet = _strip_html(item.get("snippet", ""))
        lines.append(f"  - {title}: {snippet}")

    return f"Wikipedia results for {query!r}:\n" + "\n".join(lines)


@tool
def wikipedia_article(title: str, max_chars: int = 4000) -> str:
    """Get the summary extract of a Wikipedia article.

    Args:
        title: Exact article title, e.g. "Python (programming language)".
        max_chars: Maximum characters to return. Defaults to 4000.
    """
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={urllib.parse.quote(title)}"
        f"&prop=extracts&exintro=1&explaintext=1&format=json&utf8=1"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Wikipedia API failed: {e}"

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


@tool
def wikipedia_related(title: str, limit: int = 5) -> str:
    """Get related Wikipedia article titles from a page's outgoing links.

    Falls back to a regular Wikipedia search if the page is missing.

    Args:
        title: Exact article title, e.g. "Python (programming language)".
        limit: Number of related pages to return (1-20). Defaults to 5.
    """
    limit = max(1, min(limit, 20))
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={urllib.parse.quote(title)}"
        f"&prop=links&plnamespace=0&pllimit={limit}&redirects=1&format=json&utf8=1"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Wikipedia related lookup failed: {e}"

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return wikipedia_search(title, results=limit)

        links = page.get("links", [])
        if not links:
            return wikipedia_search(title, results=limit)

        lines = [f"Related Wikipedia pages for {page.get('title', title)!r}:"]
        for item in links[:limit]:
            link_title = item.get("title", "")
            if link_title:
                lines.append(f"  - {link_title}")
        if len(lines) == 1:
            return wikipedia_search(title, results=limit)
        return "\n".join(lines)

    return wikipedia_search(title, results=limit)


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    import re

    return re.sub(r"<[^>]+>", "", text)
