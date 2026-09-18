"""Web tools — HTTP fetching and HTML text extraction."""

from __future__ import annotations

import html.parser
from io import StringIO

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import HttpError, Page, fetch_page

_DEFAULT_MAX_CHARS = 8000
_MAX_CHARS_LIMIT = 100_000
# A page's text is a fraction of its HTML: read this much HTML whatever max_chars asks for.
_SCRAPE_MAX_BYTES = 2_000_000
# UTF-8 needs at most four bytes a character.
_BYTES_PER_CHAR = 4


def _clamp(max_chars: int) -> int:
    return max(1, min(max_chars, _MAX_CHARS_LIMIT))


def _cut(text: str, max_chars: int, page: Page) -> str:
    """``text`` up to ``max_chars``, marked when the page had more than what is returned."""
    if len(text) <= max_chars and page.complete:
        return text
    total = f"{len(text)} total chars" if page.complete else "the page is longer"
    return text[:max_chars] + f"\n\n[Truncated — {total}]"


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
def http_get(url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Fetch a URL and return the raw response text.

    Args:
        url: The URL to fetch (http:// or https://). Redirects stay on its host.
        max_chars: Maximum characters to return (1-100000). Defaults to 8000.
    """
    max_chars = _clamp(max_chars)
    try:
        page = fetch_page(url, max_bytes=max_chars * _BYTES_PER_CHAR)
    except HttpError as e:
        return str(e)
    return _cut(page.text, max_chars, page)


@tool(
    capability="network",
    risk_level="high",
    requires_approval=True,
    approval_reason="Fetching arbitrary URLs can reach internal services or leak data.",
)
def scrape_text(url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Fetch a web page and extract visible text (HTML tags stripped).

    Args:
        url: The URL to fetch (http:// or https://). Redirects stay on its host.
        max_chars: Maximum characters to return (1-100000). Defaults to 8000.
    """
    max_chars = _clamp(max_chars)
    try:
        page = fetch_page(url, max_bytes=_SCRAPE_MAX_BYTES)
    except HttpError as e:
        return str(e)
    extractor = _HTMLTextExtractor()
    extractor.feed(page.text)
    return _cut(extractor.get_text(), max_chars, page)
