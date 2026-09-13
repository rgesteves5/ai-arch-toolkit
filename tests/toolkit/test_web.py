"""Tests for toolkit/tools/_web.py."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.core import ApprovalDecision, ApprovalRequest, ToolCall, ToolGroup
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


class TestHttpGetGovernance:
    @patch("ai_arch_toolkit.toolkit.tools._web.urllib.request.urlopen")
    def test_denied_without_approval_handler(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen("Hello World")
        call = ToolCall(id="tc_1", name="http_get", input={"url": "https://example.com"})

        result = ToolGroup(http_get).execute(call)

        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "approval_denied"
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._web.urllib.request.urlopen")
    async def test_fetches_when_handler_approves(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen("Hello World")
        requests: list[ApprovalRequest] = []

        async def approve(request: ApprovalRequest) -> ApprovalDecision:
            requests.append(request)
            return ApprovalDecision.approve()

        call = ToolCall(id="tc_1", name="http_get", input={"url": "https://example.com"})
        result = await ToolGroup(http_get, approval_handler=approve).async_execute(call)

        assert result.ok is True
        assert result.value == "Hello World"
        assert [(r.tool_name, r.capability, r.risk_level) for r in requests] == [
            ("http_get", "network", "high")
        ]
