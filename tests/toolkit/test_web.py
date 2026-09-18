"""Tests for toolkit/tools/_web.py."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

from ai_arch_toolkit.core import ApprovalDecision, ApprovalRequest, ToolCall, ToolGroup
from ai_arch_toolkit.toolkit.tools._web import http_get, scrape_text
from tests.toolkit.http_fakes import HTTP_OPEN, respond


class TestHttpGet:
    def test_invalid_url(self):
        result = http_get("not-a-url")
        assert "Invalid URL" in result

    @patch(HTTP_OPEN)
    def test_fetches_content(self, mock_urlopen):
        mock_urlopen.return_value = respond("Hello World")
        result = http_get("https://example.com")
        assert result == "Hello World"

    @patch(HTTP_OPEN)
    def test_truncation(self, mock_urlopen):
        mock_urlopen.return_value = respond("x" * 500)
        result = http_get("https://example.com", max_chars=100)
        assert "Truncated" in result
        assert len(result) < 500

    @patch(HTTP_OPEN)
    def test_a_negative_or_huge_max_chars_is_clamped(self, mock_urlopen):
        mock_urlopen.return_value = respond("x" * 300_000)
        assert http_get("https://example.com", max_chars=-1).startswith("x\n\n[Truncated")

        mock_urlopen.return_value = respond("x" * 300_000)
        result = http_get("https://example.com", max_chars=10**9)
        assert result.startswith("x" * 100_000 + "\n\n[Truncated")

    @patch(HTTP_OPEN)
    def test_reads_no_more_of_the_body_than_it_can_return(self, mock_urlopen):
        body = respond("x" * 3_000_000)
        mock_urlopen.return_value = body

        http_get("https://example.com", max_chars=100)

        assert body.bytes_read <= 4 * 100 + 1

    @patch(HTTP_OPEN)
    def test_plain_http_is_still_fetched(self, mock_urlopen):
        mock_urlopen.return_value = respond("Hello")

        assert http_get("http://example.com/") == "Hello"
        assert mock_urlopen.call_args.args[0].full_url == "http://example.com/"

    @patch(HTTP_OPEN)
    def test_credentials_in_the_url_are_refused(self, mock_urlopen):
        assert "Invalid URL" in http_get("https://user:secret@example.com/")
        assert "Invalid URL" in scrape_text("https://user:secret@example.com/")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_http_error(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://example.com", 404, "Not Found", {}, BytesIO()
        )
        result = http_get("https://example.com")
        assert "404" in result

    @patch(HTTP_OPEN)
    def test_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = http_get("https://example.com")
        assert "timed out" in result.lower()


class TestScrapeText:
    def test_invalid_url(self):
        result = scrape_text("not-a-url")
        assert "Invalid URL" in result

    @patch(HTTP_OPEN)
    def test_strips_html(self, mock_urlopen):
        html = "<html><body><p>Hello</p><script>evil()</script><p>World</p></body></html>"
        mock_urlopen.return_value = respond(html)
        result = scrape_text("https://example.com")
        assert "Hello" in result
        assert "World" in result
        assert "<p>" not in result
        assert "evil()" not in result

    @patch(HTTP_OPEN)
    def test_truncation(self, mock_urlopen):
        html = "<p>" + "word " * 2000 + "</p>"
        mock_urlopen.return_value = respond(html)
        result = scrape_text("https://example.com", max_chars=100)
        assert "Truncated" in result


class TestHttpGetGovernance:
    @patch(HTTP_OPEN)
    def test_denied_without_approval_handler(self, mock_urlopen):
        mock_urlopen.return_value = respond("Hello World")
        call = ToolCall(id="tc_1", name="http_get", input={"url": "https://example.com"})

        result = ToolGroup(http_get).execute(call)

        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "approval_denied"
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    async def test_fetches_when_handler_approves(self, mock_urlopen):
        mock_urlopen.return_value = respond("Hello World")
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
