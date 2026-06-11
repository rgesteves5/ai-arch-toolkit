"""Tests for toolkit/tools/_gdelt.py."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._gdelt import gdelt_news_search, gdelt_timeline


def _mock_urlopen(data):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestGdeltNewsSearch:
    @patch("ai_arch_toolkit.toolkit.tools._gdelt._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._gdelt.urllib.request.urlopen")
    def test_returns_articles(self, mock_urlopen, _mock_throttle):
        mock_urlopen.return_value = _mock_urlopen(
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

    @patch("ai_arch_toolkit.toolkit.tools._gdelt.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "query cannot be empty" in gdelt_news_search("")
        assert "invalid timespan" in gdelt_news_search("test", timespan="yesterday")
        assert "sort must be" in gdelt_news_search("test", sort="random")
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._gdelt._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._gdelt.urllib.request.urlopen")
    def test_rate_limited_includes_body_hint(self, mock_urlopen, _mock_throttle):
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


class TestGdeltTimeline:
    @patch("ai_arch_toolkit.toolkit.tools._gdelt._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._gdelt.urllib.request.urlopen")
    def test_returns_timeline(self, mock_urlopen, _mock_throttle):
        mock_urlopen.return_value = _mock_urlopen(
            {"timeline": [{"date": "20260611000000", "value": 0.2}]}
        )

        result = gdelt_timeline("climate", timespan="7d")

        assert "GDELT timeline for 'climate'" in result
        assert "20260611000000 | value: 0.2" in result
        params = _called_params(mock_urlopen)
        assert params["mode"] == ["timelinevol"]
        assert params["timespan"] == ["7d"]

    @patch("ai_arch_toolkit.toolkit.tools._gdelt._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._gdelt.urllib.request.urlopen")
    def test_parse_failure(self, mock_urlopen, _mock_throttle):
        resp = MagicMock()
        resp.read.return_value = b"not json"
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = gdelt_timeline("test")

        assert "could not parse" in result
