"""Web tools — HTTP fetching and HTML text extraction."""

from __future__ import annotations

import html.parser
import urllib.error
import urllib.request
from io import StringIO

from ai_arch_toolkit.core import tool

_DEFAULT_MAX_CHARS = 8000
_DEFAULT_TIMEOUT = 10
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"


def _fetch(url: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Fetch URL content as text."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


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


@tool
def http_get(url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Fetch a URL and return the raw response text.

    Args:
        url: The URL to fetch (must start with http:// or https://).
        max_chars: Maximum characters to return. Defaults to 8000.
    """
    if not url.startswith(("http://", "https://")):
        return f"Invalid URL: {url!r}. Must start with http:// or https://."
    try:
        content = _fetch(url)
    except urllib.error.HTTPError as e:
        return f"HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"URL error: {e.reason}"
    except TimeoutError:
        return f"Request timed out for {url}"
    if len(content) > max_chars:
        return content[:max_chars] + f"\n\n[Truncated — {len(content)} total chars]"
    return content


@tool
def scrape_text(url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Fetch a web page and extract visible text (HTML tags stripped).

    Args:
        url: The URL to fetch (must start with http:// or https://).
        max_chars: Maximum characters to return. Defaults to 8000.
    """
    if not url.startswith(("http://", "https://")):
        return f"Invalid URL: {url!r}. Must start with http:// or https://."
    try:
        raw_html = _fetch(url)
    except urllib.error.HTTPError as e:
        return f"HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"URL error: {e.reason}"
    except TimeoutError:
        return f"Request timed out for {url}"

    extractor = _HTMLTextExtractor()
    extractor.feed(raw_html)
    text = extractor.get_text()

    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[Truncated — {len(text)} total chars]"
    return text
