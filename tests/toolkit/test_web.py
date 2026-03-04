"""Tests for toolkit/tools/_web.py."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.toolkit.tools._web import http_get, scrape_text


def _mock_urlopen(content: str, charset: str = "utf-8"):
    """Create a mock for urllib.request.urlopen."""
    resp = MagicMock()
    resp.read.return_value = content.encode(charset)
    resp.headers.get_content_charset.return_value = charset
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestHttpGet:
    def test_invalid_url(self):
        result = http_get("not-a-url")
        assert "Invalid URL" in result

    @patch("ai_arch_toolkit.toolkit.tools._web.urllib.request.urlopen")
    def test_fetches_content(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen("Hello World")
        result = http_get("https://example.com")
        assert result == "Hello World"

    @patch("ai_arch_toolkit.toolkit.tools._web.urllib.request.urlopen")
    def test_truncation(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen("x" * 500)
        result = http_get("https://example.com", max_chars=100)
        assert "Truncated" in result
        assert len(result) < 500

    @patch("ai_arch_toolkit.toolkit.tools._web.urllib.request.urlopen")
    def test_http_error(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://example.com", 404, "Not Found", {}, BytesIO()
        )
        result = http_get("https://example.com")
        assert "404" in result

    @patch("ai_arch_toolkit.toolkit.tools._web.urllib.request.urlopen")
    def test_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = http_get("https://example.com")
        assert "timed out" in result.lower()


class TestScrapeText:
    def test_invalid_url(self):
        result = scrape_text("not-a-url")
        assert "Invalid URL" in result

    @patch("ai_arch_toolkit.toolkit.tools._web.urllib.request.urlopen")
    def test_strips_html(self, mock_urlopen):
        html = "<html><body><p>Hello</p><script>evil()</script><p>World</p></body></html>"
        mock_urlopen.return_value = _mock_urlopen(html)
        result = scrape_text("https://example.com")
        assert "Hello" in result
        assert "World" in result
        assert "<p>" not in result
        assert "evil()" not in result

    @patch("ai_arch_toolkit.toolkit.tools._web.urllib.request.urlopen")
    def test_truncation(self, mock_urlopen):
        html = "<p>" + "word " * 2000 + "</p>"
        mock_urlopen.return_value = _mock_urlopen(html)
        result = scrape_text("https://example.com", max_chars=100)
        assert "Truncated" in result
