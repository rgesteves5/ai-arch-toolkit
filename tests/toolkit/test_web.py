"""Tests for toolkit/tools/_web.py: any page, read window by window (T09)."""

from __future__ import annotations

from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_arch_toolkit.core import (
    ApprovalDecision,
    ApprovalRequest,
    ToolCall,
    ToolGroup,
    ToolResult,
)
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools import _web
from ai_arch_toolkit.toolkit.tools._web import http_get, scrape_text
from tests.toolkit.http_fakes import HTTP_OPEN, respond


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


def _window(result: ToolResult | str) -> dict[str, Any]:
    assert isinstance(result, ToolResult)
    return result.metadata["window"]


def _approve_all(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision.approve()


def _rows(count: int) -> str:
    return "".join(f"row {n}\n" for n in range(count))


# Lines of one to four bytes a character, so a window's end in characters is not one in bytes.
_MIXED = "".join(f"linha {n}: café, 20 €, 🙂 {'x' * (n % 7)}\n" for n in range(120))


def _invalid_url(fn, url):
    with pytest.raises(ToolFailure) as caught:
        fn(url)

    assert "Invalid URL" in caught.value.error.message
    return caught.value.error


@pytest.mark.parametrize("fn", [http_get, scrape_text])
def test_an_invalid_url_is_a_validation_error(fn):
    assert _invalid_url(fn, "not-a-url").type == "validation_error"


class TestHttpGet:
    def test_invalid_url(self):
        _invalid_url(http_get, "not-a-url")

    @patch(HTTP_OPEN)
    def test_fetches_content(self, mock_urlopen):
        mock_urlopen.return_value = respond("Hello World")
        result = http_get("https://example.com")
        assert _text(result) == "Hello World"

    @patch(HTTP_OPEN)
    def test_following_the_footers_rebuilds_the_whole_text(self, mock_urlopen):
        mock_urlopen.side_effect = lambda request, timeout: respond(_MIXED)
        parts: list[str] = []
        offset: int | None = 0
        calls = 0

        while offset is not None:
            result = http_get("https://example.com", max_chars=300, offset=offset)
            window = _window(result)
            parts.append(_text(result)[: window["last"] - window["first"]])
            assert window["first"] == offset
            offset = (window["next_call"] or {}).get("offset")
            calls += 1

        assert "".join(parts) == _MIXED
        assert calls > 5
        assert window["total"] == len(_MIXED)  # the last window read the page to its end

    @pytest.mark.parametrize("page", ["🙂" * 40, " " * 500 + "late text"])
    @patch(HTTP_OPEN)
    def test_the_smallest_window_still_reads_on_to_the_end(self, mock_urlopen, page):
        # Four bytes a character, one character a window; or a start that is only spaces.
        mock_urlopen.side_effect = lambda request, timeout: respond(page)
        parts: list[str] = []
        offset: int | None = 0

        while offset is not None and len(parts) <= len(page):
            result = http_get("https://example.com", max_chars=1, offset=offset)
            window = _window(result)
            parts.append(_text(result)[: window["last"] - window["first"]])
            offset = (window["next_call"] or {}).get("offset")

        assert "".join(parts) == page

    @pytest.mark.parametrize(
        ("charset", "page", "max_chars"),
        [
            # A BOM, then a character of four bytes: four bytes a character fall short.
            ("utf-16", "😀abc" * 20, 1),
            ("utf-32", "😀" * 50, 1),
            ("utf-32", "😀" * 50, 3),
            # Three-byte escapes at each switch of script: 4.5 bytes a character.
            ("iso-2022-jp", "a日" * 50, 1),
            ("iso-2022-jp", "a日" * 50, 2),
            ("iso-2022-jp", "a日" * 50, 3),
            ("iso-2022-jp", "日本語のテキスト" * 20, 7),
        ],
        ids=["utf16-bom", "utf32-bom", "utf32-bom-3", "jis-1", "jis-2", "jis-3", "jis-kana-7"],
    )
    @patch(HTTP_OPEN)
    def test_every_footer_reads_on_past_its_offset_whatever_the_encoding(
        self, mock_urlopen, charset, page, max_chars
    ):
        # Reading four bytes a character stopped short of the window: the footer then named its
        # own offset (or one before it), and following it never ended.
        body = page.encode(charset)
        mock_urlopen.side_effect = lambda request, timeout: respond(
            body, content_type=f"text/plain; charset={charset}"
        )
        parts: list[str] = []
        offset: int | None = 0

        while offset is not None and len(parts) <= len(page):
            result = http_get("https://example.com", max_chars=max_chars, offset=offset)
            window = _window(result)
            assert window["first"] == offset
            parts.append(_text(result)[: window["last"] - window["first"]])
            following = (window["next_call"] or {}).get("offset")
            assert following is None or following > offset
            offset = following

        assert "".join(parts) == page
        assert window["total"] == len(page)

    @patch(HTTP_OPEN)
    def test_a_short_read_is_read_again_further_only_as_far_as_it_needs(self, mock_urlopen):
        body = ("😀" * 50).encode("utf-32")  # a BOM, then four bytes a character
        mock_urlopen.side_effect = lambda request, timeout: respond(
            body, content_type="text/plain; charset=utf-32"
        )

        result = http_get("https://example.com", max_chars=1)

        assert _text(result).startswith("😀\n[chars 0-1 ")
        assert mock_urlopen.call_count == 2

    @patch(HTTP_OPEN)
    def test_an_offset_past_what_http_get_reads_is_not_moved_back(self, mock_urlopen, monkeypatch):
        monkeypatch.setattr(_web, "_READ_BYTES", 200)
        mock_urlopen.side_effect = lambda request, timeout: respond(_rows(100))

        result = http_get("https://example.com", max_chars=100, offset=300)

        lines = _text(result).splitlines()
        assert lines[0] == (
            "(https://example.com goes on past its first 200 bytes, all http_get reads)"
        )
        assert lines[-1] == "[chars 300-300 | end]"
        assert (_window(result)["first"], _window(result)["next_call"]) == (300, None)

    @patch(HTTP_OPEN)
    def test_a_cut_page_names_the_next_offset_and_reads_no_more_than_it_needs(self, mock_urlopen):
        body = respond(_rows(100_000))
        mock_urlopen.return_value = body

        result = http_get("https://example.com", max_chars=100)

        window = _window(result)
        assert _text(result).startswith("row 0\nrow 1\n")
        assert window["total"] is None  # the page goes on: its length is not known yet
        assert _text(result).endswith(
            f"[chars 0-{window['last']} | next: offset={window['last']}]"
        )
        assert body.bytes_read <= 4 * 100 + 1

        body = respond(_rows(100_000))
        mock_urlopen.return_value = body

        http_get("https://example.com", max_chars=100, offset=1000)

        assert body.bytes_read <= 4 * 1100 + 1

    @patch(HTTP_OPEN)
    def test_a_page_read_whole_gives_its_length(self, mock_urlopen):
        mock_urlopen.return_value = respond(_rows(10))

        result = http_get("https://example.com", max_chars=30, offset=42)

        assert _window(result)["total"] == 60
        assert _text(result) == "row 7\nrow 8\nrow 9\n[chars 42-60 of 60 | end]"

    @patch(HTTP_OPEN)
    def test_find_returns_the_passages_around_a_term(self, mock_urlopen):
        page = _rows(2000) + "the needle is here\n" + "tail\n" * 50
        mock_urlopen.return_value = respond(page)

        text = _text(http_get("https://example.com", find="NEEDLE"))

        assert "the needle is here" in text
        assert text.startswith("[at char ")
        assert text.endswith('[matches 1-1 of 1 for "NEEDLE" | end]')

    @patch(HTTP_OPEN)
    def test_a_page_longer_than_what_http_get_reads_says_so(self, mock_urlopen, monkeypatch):
        monkeypatch.setattr(_web, "_READ_BYTES", 200)
        mock_urlopen.return_value = respond(_rows(100))

        result = http_get("https://example.com", max_chars=100, offset=150)

        lines = _text(result).splitlines()
        assert lines[0] == (
            "(https://example.com goes on past its first 200 bytes, all http_get reads)"
        )
        assert lines[-1] == "[chars 150-200 | end]"
        assert _window(result)["next_call"] is None

    @patch(HTTP_OPEN)
    def test_an_empty_answer_says_so(self, mock_urlopen):
        mock_urlopen.return_value = respond("")

        assert _text(http_get("https://example.com")) == (
            "The answer from https://example.com has no text."
        )

    @pytest.mark.parametrize(("max_chars", "kept"), [(1, True), (0, False), (100_001, False)])
    @patch(HTTP_OPEN)
    def test_max_chars_is_refused_outside_its_limits_through_the_executor(
        self, mock_urlopen, max_chars, kept
    ):
        mock_urlopen.return_value = respond("Hello")
        call = ToolCall(
            id="c1", name="http_get", input={"url": "https://example.com", "max_chars": max_chars}
        )

        result = ToolGroup(http_get, approval_handler=_approve_all).execute(call)

        assert result.ok is kept
        if not kept:
            assert result.error is not None
            assert result.error.type == "validation_error"

    @patch(HTTP_OPEN)
    def test_plain_http_is_still_fetched(self, mock_urlopen):
        mock_urlopen.return_value = respond("Hello")

        assert _text(http_get("http://example.com/")) == "Hello"
        assert mock_urlopen.call_args.args[0].full_url == "http://example.com/"

    @patch(HTTP_OPEN)
    def test_credentials_in_the_url_are_refused(self, mock_urlopen):
        _invalid_url(http_get, "https://user:secret@example.com/")
        _invalid_url(scrape_text, "https://user:secret@example.com/")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_http_error(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://example.com", 404, "Not Found", {}, BytesIO()
        )

        with pytest.raises(ToolFailure) as caught:
            http_get("https://example.com")

        assert caught.value.error.type == "upstream"
        assert not caught.value.error.retryable
        assert "404" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()

        with pytest.raises(ToolFailure) as caught:
            http_get("https://example.com")

        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable
        assert "timed out" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_rate_limited(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://example.com", 429, "Too Many Requests", {}, BytesIO()
        )

        with pytest.raises(ToolFailure) as caught:
            scrape_text("https://example.com")

        assert caught.value.error.type == "rate_limited"


class TestScrapeText:
    def test_invalid_url(self):
        _invalid_url(scrape_text, "not-a-url")

    @patch(HTTP_OPEN)
    def test_strips_html(self, mock_urlopen):
        html = "<html><body><p>Hello</p><script>evil()</script><p>World</p></body></html>"
        mock_urlopen.return_value = respond(html)
        result = _text(scrape_text("https://example.com"))
        assert result == "Hello\nWorld"
        assert "evil()" not in result

    @patch(HTTP_OPEN)
    def test_following_the_footers_rebuilds_the_visible_text(self, mock_urlopen):
        html = "<html><body>" + "".join(f"<p>Para {n}: café.</p>" for n in range(200)) + "</body>"
        mock_urlopen.side_effect = lambda request, timeout: respond(html)
        whole = "\n".join(f"Para {n}: café." for n in range(200))
        parts: list[str] = []
        offset: int | None = 0

        while offset is not None:
            result = scrape_text("https://example.com", max_chars=250, offset=offset)
            window = _window(result)
            parts.append(_text(result)[: window["last"] - window["first"]])
            offset = (window["next_call"] or {}).get("offset")

        assert "".join(parts) == whole
        assert window["total"] == len(whole)

    @patch(HTTP_OPEN)
    def test_find_returns_the_passages_around_a_term(self, mock_urlopen):
        html = "".join(f"<p>Para {n}.</p>" for n in range(500)) + "<p>The answer is 42.</p>"
        mock_urlopen.return_value = respond(html)

        text = _text(scrape_text("https://example.com", find="answer"))

        assert "The answer is 42." in text
        assert text.endswith('[matches 1-1 of 1 for "answer" | end]')

    @patch(HTTP_OPEN)
    def test_html_longer_than_what_scrape_text_reads_says_so(self, mock_urlopen, monkeypatch):
        monkeypatch.setattr(_web, "_SCRAPE_MAX_BYTES", 200)
        mock_urlopen.return_value = respond("".join(f"<p>Para {n}.</p>" for n in range(100)))

        result = scrape_text("https://example.com")

        lines = _text(result).splitlines()
        assert lines[0] == (
            "(only the first 200 bytes of https://example.com's HTML were read: its text stops "
            "there; http_get with offset=200 reads the HTML on, or find= searches it)"
        )
        assert lines[1] == "Para 0."
        assert _window(result)["next_call"] is None

    @patch(HTTP_OPEN)
    def test_the_html_scrape_text_did_not_read_is_where_http_get_reads_on(
        self, mock_urlopen, monkeypatch
    ):
        monkeypatch.setattr(_web, "_SCRAPE_MAX_BYTES", 200)
        html = "".join(f"<p>Pará {n}.</p>" for n in range(100))  # two bytes for one "á"
        mock_urlopen.side_effect = lambda request, timeout: respond(html)

        note = _text(scrape_text("https://example.com")).splitlines()[0]
        offset = int(note.split("offset=")[1].split()[0])
        rest = ToolGroup(http_get, approval_handler=_approve_all).execute(
            ToolCall(
                id="c1",
                name="http_get",
                input={"url": "https://example.com", "offset": offset, "max_chars": 100_000},
            )
        )

        assert offset < 200  # characters, not bytes
        assert html[offset:].startswith(_text(rest).splitlines()[0][:20])

    @patch(HTTP_OPEN)
    def test_a_page_without_visible_text_says_so(self, mock_urlopen):
        mock_urlopen.return_value = respond("<html><script>x()</script></html>")

        assert _text(scrape_text("https://example.com")) == (
            "The page at https://example.com has no visible text."
        )


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

    def test_the_window_does_not_change_the_web_tools_governance(self) -> None:
        for fn in (http_get, scrape_text):
            policy = fn.__tool_definition__.policy
            assert (policy.capability, policy.risk_level, policy.requires_approval) == (
                "network",
                "high",
                True,
            )

    @patch(HTTP_OPEN)
    def test_reading_on_is_approved_call_by_call(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = lambda request, timeout: respond(_rows(100))
        asked: list[ApprovalRequest] = []

        def approve(request: ApprovalRequest) -> ApprovalDecision:
            asked.append(request)
            return ApprovalDecision.approve()

        group = ToolGroup(http_get, approval_handler=approve)
        url = "https://example.com"
        first = group.execute(
            ToolCall(id="c1", name="http_get", input={"url": url, "max_chars": 50})
        )
        onward = first.metadata["window"]["next_call"]
        second = group.execute(
            ToolCall(id="c2", name="http_get", input={"url": url, "max_chars": 50, **onward})
        )

        assert second.ok
        assert second.metadata["window"]["first"] == first.metadata["window"]["last"]
        assert len(asked) == 2
