"""Web tools: any page a person approved, as it came or as its visible text, read window by window
(T09; D39).

Both tools stay gated (``requires_approval``, ``risk_level="high"``): a URL comes from the caller,
so it can reach internal services. Every call fetches the page again; the window's footer names the
call that reads on, and following the footers rebuilds the whole text.
"""

from __future__ import annotations

import html.parser
from dataclasses import replace
from io import StringIO
from typing import Annotated

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.toolkit.tools._http import Page, fetch_page
from ai_arch_toolkit.toolkit.tools._window import Window, find_window, text_window

_DEFAULT_MAX_CHARS = 8000
_MAX_CHARS = 100_000
# UTF-8 needs at most four bytes a character: reading four bytes a character reaches as far as a
# window asks, in most pages.
_BYTES_PER_CHAR = 4
# A read that falls short (a BOM, a stateful encoding's escapes, characters of more than four
# bytes) is made again, at least twice as far and at least one socket read (the door's 64 KiB).
_REREAD_MIN_BYTES = 64 * 1024
# The most of a page http_get reads, its offset as deep as it goes (the door's default body bound).
_READ_BYTES = 10_000_000
# A page's text is a fraction of its HTML: scrape_text reads this much HTML, whatever it returns.
_SCRAPE_MAX_BYTES = 2_000_000


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Strip HTML tags and extract visible text."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "head"})

    def __init__(self) -> None:
        super().__init__()
        self._buf = StringIO()
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag in ("br", "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
            self._buf.write("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._buf.write(data)

    def get_text(self) -> str:
        raw = self._buf.getvalue()
        # Collapse whitespace but preserve newlines
        lines = (line.strip() for line in raw.splitlines())
        return "\n".join(line for line in lines if line)


@tool(
    capability="network",
    risk_level="high",
    requires_approval=True,
    approval_reason="Fetching arbitrary URLs can reach internal services or leak data.",
)
def http_get(
    url: str,
    max_chars: Annotated[int, Range(1, _MAX_CHARS)] = _DEFAULT_MAX_CHARS,
    offset: Annotated[int, Range(0)] = 0,
    find: str = "",
) -> ToolResult:
    """Fetch a URL and return the response text as it came.

    It reads only as much of the page as the window needs, four bytes a character; a page whose
    characters take more (a byte-order mark, a stateful encoding such as ISO-2022-JP) is fetched
    again, further, until the window is whole.

    Args:
        url: The URL to fetch (http:// or https://). Redirects stay on its host.
        max_chars: How many characters to return.
        offset: Where to start, in characters of the text (with ``find``, where to search on
            from); the footer gives the next offset.
        find: A term to look for: the answer is the passages around each match, not the text.

    Raises:
        ToolFailure: validation_error when the URL is not http(s) or carries credentials;
            upstream or rate_limited when the request fails.
    """
    term = find.strip()
    # A window needs the text up to its end; a search, all the text there is to read.
    wanted = _READ_BYTES if term else (offset + max_chars) * _BYTES_PER_CHAR
    budget = min(wanted, _READ_BYTES)
    page = fetch_page(url, max_bytes=budget)
    text = _read_text(page)
    while not page.complete and len(text) < offset + max_chars and budget < _READ_BYTES:
        budget = min(max(budget * 2, _REREAD_MIN_BYTES), _READ_BYTES)
        page = fetch_page(url, max_bytes=budget)
        text = _read_text(page)
    if page.complete and not text.strip():
        return ToolResult.success(f"The answer from {url} has no text.")
    at_most = not page.complete and budget == _READ_BYTES  # nothing past this read is reachable
    window = _window(
        text, whole=page.complete, onward=not at_most, term=term, offset=offset, limit=max_chars
    )
    stopped = at_most and (bool(term) or window.next_call is None)
    note = f"({url} goes on past its first {_READ_BYTES} bytes, all http_get reads)"
    return window.result(heading=note if stopped else "")


@tool(
    capability="network",
    risk_level="high",
    requires_approval=True,
    approval_reason="Fetching arbitrary URLs can reach internal services or leak data.",
)
def scrape_text(
    url: str,
    max_chars: Annotated[int, Range(1, _MAX_CHARS)] = _DEFAULT_MAX_CHARS,
    offset: Annotated[int, Range(0)] = 0,
    find: str = "",
) -> ToolResult:
    """Fetch a web page and return its visible text, without the HTML.

    Args:
        url: The URL to fetch (http:// or https://). Redirects stay on its host.
        max_chars: How many characters to return.
        offset: Where to start, in characters of the text (with ``find``, where to search on
            from); the footer gives the next offset.
        find: A term to look for: the answer is the passages around each match, not the text.

    Raises:
        ToolFailure: validation_error when the URL is not http(s) or carries credentials;
            upstream or rate_limited when the request fails.
    """
    page = fetch_page(url, max_bytes=_SCRAPE_MAX_BYTES)
    html_text = _read_text(page)
    extractor = _HTMLTextExtractor()
    extractor.feed(html_text)
    text = extractor.get_text()
    note = (
        f"(only the first {_SCRAPE_MAX_BYTES} bytes of {url}'s HTML were read: its text stops "
        f"there; http_get with offset={len(html_text)} reads the HTML on, or find= searches it)"
    )
    if not text:
        cut = "" if page.complete else f" {note}"
        return ToolResult.success(f"The page at {url} has no visible text.{cut}")
    window = _window(
        text, whole=page.complete, onward=False, term=find.strip(), offset=offset, limit=max_chars
    )
    return window.result(heading="" if page.complete else note)


def _read_text(page: Page) -> str:
    """The page's text; a read cut short may end inside a character, which decodes to a
    replacement character: the text stops before it (the next window reads it whole)."""
    return page.text if page.complete else page.text.removesuffix("�")


def _window(text: str, *, whole: bool, onward: bool, term: str, offset: int, limit: int) -> Window:
    """At most ``limit`` characters of ``text`` from ``offset``, or the passages around ``term``.

    ``whole`` is ``False`` when the page goes on past ``text``: its length is then unknown, and
    when the window reaches the end of ``text`` the next call reads further only if more can be
    fetched (``onward``). An offset past ``text`` then shows nothing there, at that offset.
    """
    if term:
        return find_window(text, term, offset=offset, limit=limit)
    window = text_window(text, offset=offset, limit=limit)
    if not whole:
        if offset > len(text):  # past all that can be read: nothing there, and no way on
            return replace(window, first=offset, last=offset, total=None, next_call=None)
        more = window.last < len(text) or onward
        window = replace(window, total=None, next_call={"offset": window.last} if more else None)
    # Every footer reads on: a short read is made again further (``http_get``) until it does.
    assert window.next_call is None or window.last > offset, "a footer that does not read on"
    return window
