"""Tests for toolkit/tools/_gdelt.py."""

from __future__ import annotations

import io
import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.toolkit.tools._gdelt import gdelt_news_search, gdelt_timeline
from tests.toolkit.http_fakes import HTTP_OPEN, respond


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


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

        result = gdelt_news_search("climate", max_results=2, timespan="24h", sort="date")

        assert "GDELT articles for 'climate'" in result
        assert "Climate story" in result
        assert "domain: news.example" in result
        assert "tone: -1.25" in result
        assert "https://news.example/story" in result

        params = _called_params(mock_urlopen)
        assert params["query"] == ["climate"]
        assert params["mode"] == ["artlist"]
        assert params["maxrecords"] == ["2"]
        assert params["timespan"] == ["24h"]
        assert params["sort"] == ["DateDesc"]

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "query cannot be empty" in gdelt_news_search("")
        assert "invalid timespan" in gdelt_news_search("test", timespan="yesterday")
        assert "sort must be" in gdelt_news_search("test", sort="random")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_rate_limited_includes_body_hint(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.gdeltproject.org/api/v2/doc/doc",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b"Please limit requests to one every 5 seconds."),
        )

        result = gdelt_news_search("test")

        assert "rate limited by GDELT" in result
        assert "one every 5 seconds" in result


def _timeline(*values: float) -> dict[str, object]:
    """A ``timelinevol`` answer: one series, its points under ``data``."""
    points = [
        {"date": f"202609{day:02d}T000000Z", "value": value}
        for day, value in enumerate(values, start=1)
    ]
    return {
        "query_details": {"title": "climate", "date_resolution": "day"},
        "timeline": [{"series": "Volume Intensity", "data": points}],
    }


class TestGdeltTimeline:
    @patch(HTTP_OPEN)
    def test_returns_timeline(self, mock_urlopen):
        # It read the series as if it were a point, so no answer ever had points.
        mock_urlopen.return_value = respond(_timeline(0.2, 0.25))

        result = gdelt_timeline("climate", timespan="7d")

        assert result == (
            "GDELT timeline for 'climate' (Volume Intensity):\n"
            "1. 20260901T000000Z | value: 0.2\n"
            "2. 20260902T000000Z | value: 0.25"
        )
        params = _called_params(mock_urlopen)
        assert params["mode"] == ["timelinevol"]
        assert params["timespan"] == ["7d"]

    @patch(HTTP_OPEN)
    def test_a_longer_timeline_says_how_many_points_it_left_out(self, mock_urlopen):
        # The default 30-day timespan has 31 daily points.
        mock_urlopen.return_value = respond(_timeline(*[0.1] * 31))

        lines = gdelt_timeline("climate").splitlines()

        assert lines[0] == (
            "GDELT timeline for 'climate' (Volume Intensity), first 20 of 31 points:"
        )
        assert lines[-1] == "20. 20260920T000000Z | value: 0.1"

    @pytest.mark.parametrize("body", [{}, {"timeline": []}, {"timeline": [{"series": "x"}]}])
    @patch(HTTP_OPEN)
    def test_a_timeline_without_points_says_so(self, mock_urlopen, body):
        mock_urlopen.return_value = respond(body)

        assert gdelt_timeline("climate") == "No GDELT timeline points found for: 'climate'"

    @patch(HTTP_OPEN)
    def test_a_broken_json_answer_is_a_parse_error(self, mock_urlopen):
        mock_urlopen.return_value = respond(b'{"timeline": [{"date": "20260611000000", "val')

        result = gdelt_timeline("test")

        assert result.startswith("GDELT timeline failed: could not parse API response: ")


@pytest.mark.parametrize(
    ("call", "failure", "message"),
    [
        (lambda: gdelt_timeline("a"), "GDELT timeline", "Your query was too short or too long."),
        (
            lambda: gdelt_news_search("climate sourcecountry:zz"),
            "GDELT news search",
            "Invalid/Unsupported Country.",
        ),
    ],
)
@patch(HTTP_OPEN)
def test_a_request_gdelt_cannot_run_is_the_tools_error(mock_urlopen, call, failure, message):
    # As answered live (2026-09-29): HTTP 200, text in place of the JSON, which read as a parse
    # error.
    mock_urlopen.return_value = respond(
        f"{message}\n".encode(), content_type="text/html; charset=utf-8"
    )

    assert call() == f"{failure} failed: {message}"


@pytest.mark.parametrize("body", [b"", b" \n"])
@patch(HTTP_OPEN)
def test_an_empty_body_is_a_parse_error_not_a_message(mock_urlopen, body):
    mock_urlopen.return_value = respond(body, content_type="text/html; charset=utf-8")

    result = gdelt_news_search("climate")

    assert result.startswith("GDELT news search failed: could not parse API response: ")


@patch(HTTP_OPEN)
def test_a_query_that_matches_nothing_is_not_an_error(mock_urlopen):
    # GDELT answers it with an empty object (2026-09-29).
    mock_urlopen.return_value = respond(b"{}", content_type="application/json; charset=utf-8")

    assert gdelt_news_search("zzqqxxyyvvww") == "No GDELT articles found for: 'zzqqxxyyvvww'"
