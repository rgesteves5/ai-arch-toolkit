"""Tests for toolkit/tools/_gdelt.py."""

from __future__ import annotations

import io
import urllib.error
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolFailure, ToolGroup, ToolResult
from ai_arch_toolkit.toolkit.tools._gdelt import gdelt_news_search, gdelt_timeline
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _failure(call) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult) and result.ok
    assert isinstance(result.value, str)
    return result.value


def _article(number: int) -> dict[str, object]:
    return {
        "title": f"Story {number}",
        "url": f"https://news.example/{number}",
        "sourcecountry": "United States",
        "domain": "news.example",
        "language": "English",
        "seendate": "20260611T120000Z",
        "tone": "-1.25",
    }


def _articles(count: int) -> dict[str, object]:
    return {"articles": [_article(number) for number in range(1, count + 1)]}


class TestGdeltNewsSearch:
    @patch(HTTP_OPEN)
    def test_returns_articles(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "articles": [
                    {
                        "title": "Climate story",
                        "url": "https://news.example/story",
                        "sourcecountry": "US",
                        "domain": "news.example",
                        "language": "English",
                        "seendate": "20260611T120000Z",
                        "socialimage": "https://news.example/image.jpg",
                        "tone": "-1.25",
                    }
                ]
            }
        )

        text = _text(gdelt_news_search("climate", max_results=2, timespan="24h", sort="date"))

        assert text.splitlines()[:4] == [
            "GDELT articles that match 'climate' in the last 24h, newest first:",
            "1. Climate story",
            "   seen 2026-06-11T12:00:00Z | domain: news.example | country: US | language: "
            "English | tone: -1.25",
            "   https://news.example/story",
        ]

        params = _called_params(mock_urlopen)
        assert params["query"] == ["climate"]
        assert params["mode"] == ["artlist"]
        assert params["maxrecords"] == ["3"]
        assert params["timespan"] == ["24h"]
        assert params["sort"] == ["DateDesc"]

    @patch(HTTP_OPEN)
    def test_the_results_read_on_past_the_first_page(self, mock_urlopen):
        # It stopped at 20 results, with no total and no way on; GDELT lists up to 250.
        mock_urlopen.return_value = respond(_articles(31))

        result = gdelt_news_search("climate", max_results=10, offset=20)
        text = _text(result)

        # GDELT has no offset: the tool asks for the articles up to the page's end, and one more.
        assert _called_params(mock_urlopen)["maxrecords"] == ["31"]
        assert text.splitlines()[1] == "21. Story 21"
        assert "Story 20\n" not in text
        assert text.endswith("[results 21-30 | next: offset=30]")
        assert result.metadata["window"]["next_call"] == {"offset": 30}

    @patch(HTTP_OPEN)
    def test_a_page_that_ends_the_list_has_no_next_call(self, mock_urlopen):
        # Without the one more, a list of exactly 20 would send the agent to an empty page.
        mock_urlopen.return_value = respond(_articles(20))

        text = _text(gdelt_news_search("climate", max_results=10, offset=10))

        assert text.endswith("[results 11-20 of 20 | end]")

    @patch(HTTP_OPEN)
    def test_fewer_articles_than_asked_are_all_there_is(self, mock_urlopen):
        mock_urlopen.return_value = respond(_articles(14))

        text = _text(gdelt_news_search("climate", max_results=10, offset=10))

        assert text.endswith("[results 11-14 of 14 | end]")

    @patch(HTTP_OPEN)
    def test_at_gdelts_cap_the_answer_says_how_to_see_others(self, mock_urlopen):
        mock_urlopen.return_value = respond(_articles(250))

        text = _text(gdelt_news_search("climate", max_results=50, offset=240))

        # The page of GDELT's first results is _first_results' (T08b), the cap said at its end.
        assert _called_params(mock_urlopen)["maxrecords"] == ["250"]
        assert text.splitlines()[1] == "241. Story 241"
        assert text.endswith(
            "   https://news.example/250\n"
            "(the source returns no more than 250 results; narrow the timespan or the query, or "
            "change the sort, to see others)\n"
            "[results 241-250 | end]"
        )

    @patch(HTTP_OPEN)
    def test_zero_results_say_so_with_the_query(self, mock_urlopen):
        # GDELT answers a query that matches nothing with an empty object (2026-09-29).
        mock_urlopen.return_value = respond(b"{}", content_type="application/json; charset=utf-8")

        text = _text(gdelt_news_search("zzqqxxyyvvww"))

        assert text == "No GDELT articles match 'zzqqxxyyvvww' in the last 7d."

    def test_the_limits_are_in_the_schema(self):
        group = ToolGroup(gdelt_news_search)
        for args in ({"max_results": 51}, {"offset": 250}, {"offset": -1}):
            call = ToolCall(id="c", name="gdelt_news_search", input={"query": "x", **args})
            result = group.execute(call)
            assert not result.ok and result.error is not None
            assert result.error.type == "validation_error", args

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        for call, words in (
            (lambda: gdelt_news_search(""), "query cannot be empty"),
            (lambda: gdelt_news_search("test", timespan="yesterday"), "invalid timespan"),
            (lambda: gdelt_news_search("test", sort="random"), "sort must be"),
            (lambda: gdelt_timeline(" "), "query cannot be empty"),
            (lambda: gdelt_timeline("test", timespan="1y"), "invalid timespan"),
        ):
            failure = _failure(call)
            assert failure.error.type == "validation_error"
            assert words in str(failure)
        mock_urlopen.assert_not_called()

    @pytest.mark.parametrize("timespan", ["15min", "24h", "7d", "4w", "3m"])
    @patch(HTTP_OPEN)
    def test_every_unit_gdelt_takes_is_accepted(self, mock_urlopen, timespan):
        # Minutes are "min", months "m" (https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/).
        mock_urlopen.return_value = respond({})

        gdelt_news_search("climate", timespan=timespan)

        assert _called_params(mock_urlopen)["timespan"] == [timespan]

    @patch(HTTP_OPEN)
    def test_rate_limited_includes_body_hint(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.gdeltproject.org/api/v2/doc/doc",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b"Please limit requests to one every 5 seconds."),
        )

        failure = _failure(lambda: gdelt_news_search("test"))

        assert failure.error.type == "rate_limited"
        assert failure.error.retryable
        assert str(failure) == (
            "rate limited by GDELT (HTTP 429). Try again later. GDELT said: Please limit "
            "requests to one every 5 seconds."
        )

    @patch(HTTP_OPEN)
    def test_after_a_429_gdelt_rests_and_the_next_call_says_when(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            429, "Too Many Requests", body=b"Please limit requests to one every 5 seconds."
        )
        _failure(lambda: gdelt_news_search("test"))

        failure = _failure(lambda: gdelt_timeline("test"))

        assert failure.error.type == "rate_limited"
        assert str(failure).startswith("GDELT asked to slow down: ")
        assert "try again in 60 s." in str(failure)
        assert mock_urlopen.call_count == 1  # the second call did not go out


def _timeline(*values: float) -> dict[str, object]:
    """A ``timelinevol`` answer: one series, its daily points under ``data``."""
    start = datetime(2026, 9, 1, tzinfo=UTC)
    points = [
        {"date": (start + timedelta(days=day)).strftime("%Y%m%dT%H%M%SZ"), "value": value}
        for day, value in enumerate(values)
    ]
    return {
        "query_details": {"title": "climate", "date_resolution": "day"},
        "timeline": [{"series": "Volume Intensity", "data": points}],
    }


class TestGdeltTimeline:
    @patch(HTTP_OPEN)
    def test_returns_timeline(self, mock_urlopen):
        # It read the series as if it were a point, so no answer ever had points.
        mock_urlopen.return_value = respond(_timeline(0.2, 0.00001))

        text = _text(gdelt_timeline("climate", timespan="7d"))

        assert text == (
            "GDELT timeline for 'climate' in the last 7d (Volume Intensity: the share of all the "
            "coverage GDELT monitored that matched, in %, by day):\n"
            "2026-09-01T00:00:00Z: 0.2%\n"
            "2026-09-02T00:00:00Z: 0.00001%"
        )
        params = _called_params(mock_urlopen)
        assert params["mode"] == ["timelinevol"]
        assert params["timespan"] == ["7d"]

    @patch(HTTP_OPEN)
    def test_a_long_timeline_reads_on_to_its_latest_points(self, mock_urlopen):
        # It kept the first 20 points, the oldest, with no way to the recent ones.
        answer = _timeline(*[0.1] * 130, 0.9)
        mock_urlopen.side_effect = [respond(answer), respond(answer)]

        first = _text(gdelt_timeline("climate", timespan="3m"))
        second = _text(gdelt_timeline("climate", timespan="3m", offset=100))

        assert first.endswith("[results 1-100 of 131 | next: offset=100]")
        assert second.splitlines()[-2] == "2027-01-09T00:00:00Z: 0.9%"
        assert second.endswith("[results 101-131 of 131 | end]")

    @pytest.mark.parametrize("body", [{}, {"timeline": []}, {"timeline": [{"series": "x"}]}])
    @patch(HTTP_OPEN)
    def test_a_timeline_without_points_says_so_with_the_query(self, mock_urlopen, body):
        mock_urlopen.return_value = respond(body)

        assert _text(gdelt_timeline("climate")) == (
            "No GDELT timeline points for 'climate' in the last 30d."
        )

    @patch(HTTP_OPEN)
    def test_a_broken_json_answer_is_a_parse_error(self, mock_urlopen):
        mock_urlopen.return_value = respond(b'{"timeline": [{"date": "20260611000000", "val')

        failure = _failure(lambda: gdelt_timeline("test"))

        assert failure.error.type == "upstream"
        assert str(failure).startswith("could not parse API response: ")


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: gdelt_timeline("a"), "Your query was too short or too long."),
        (lambda: gdelt_news_search("climate sourcecountry:zz"), "Invalid/Unsupported Country."),
    ],
)
@patch(HTTP_OPEN)
def test_a_request_gdelt_cannot_run_is_a_validation_error(mock_urlopen, call, message):
    # As answered live (2026-09-29): HTTP 200, text in place of the JSON, which read as a parse
    # error.
    mock_urlopen.return_value = respond(
        f"{message}\n".encode(), content_type="text/html; charset=utf-8"
    )

    failure = _failure(call)

    assert failure.error.type == "validation_error"
    assert not failure.error.retryable
    assert str(failure) == f"{message.rstrip('.')}; change the query or its options"


@pytest.mark.parametrize("body", [b"", b" \n"])
@patch(HTTP_OPEN)
def test_an_empty_body_is_a_parse_error_not_a_message(mock_urlopen, body):
    mock_urlopen.return_value = respond(body, content_type="text/html; charset=utf-8")

    failure = _failure(lambda: gdelt_news_search("climate"))

    assert failure.error.type == "upstream"
    assert str(failure).startswith("could not parse API response: ")


@patch(HTTP_OPEN)
def test_an_error_page_is_not_a_message(mock_urlopen):
    mock_urlopen.side_effect = http_error(
        500, "Internal Server Error", body=b"<html><body>Server Error</body></html>"
    )

    failure = _failure(lambda: gdelt_timeline("climate"))

    assert failure.error.type == "upstream"
    assert failure.error.retryable
    assert str(failure) == "HTTP error 500: Internal Server Error"


@patch(HTTP_OPEN)
def test_a_404_is_an_endpoint_not_found(mock_urlopen):
    mock_urlopen.side_effect = http_error(404, "Not Found", body=b"Not Found")

    failure = _failure(lambda: gdelt_news_search("climate"))

    assert failure.error.type == "upstream"
    assert not failure.error.retryable
    assert str(failure) == (
        "GDELT: endpoint not found (HTTP 404); the API may have changed: Not Found"
    )
