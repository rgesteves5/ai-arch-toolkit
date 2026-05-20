"""Simple general web search tool for the configurable agent."""

from __future__ import annotations

import html.parser
import urllib.parse
import urllib.request
from io import StringIO

from ai_arch_toolkit.core._tools._decorator import tool

_USER_AGENT = "ai-arch-toolkit/1.0"


class _LinkExtractor(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href = ""
        self._buf = StringIO()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href") or ""
        if "/l/?" in href or href.startswith("http"):
            self._href = href
            self._buf = StringIO()

    def handle_data(self, data: str) -> None:
        if self._href:
            self._buf.write(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            title = " ".join(self._buf.getvalue().split())
            url = _clean_url(self._href)
            if title and url:
                self.links.append((title, url))
            self._href = ""


@tool
def web_search_query(query: str, max_results: int = 5) -> str:
    """Search the web and return result titles and URLs.

    Args:
        query: Search query.
        max_results: Maximum results to return.
    """
    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Web search failed: {exc}"

    parser = _LinkExtractor()
    parser.feed(html)
    seen: set[str] = set()
    lines: list[str] = []
    for title, link in parser.links:
        if link in seen:
            continue
        seen.add(link)
        lines.append(f"- {title}\n  {link}")
        if len(lines) >= max_results:
            break
    if not lines:
        return "No search results found."
    return "\n".join(lines)


def _clean_url(href: str) -> str:
    if href.startswith("http"):
        return href
    parsed = urllib.parse.urlparse(href)
    query = urllib.parse.parse_qs(parsed.query)
    uddg = query.get("uddg", [""])[0]
    return urllib.parse.unquote(uddg)
