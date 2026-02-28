"""Tests for toolkit/tools/_news.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.toolkit.tools._news import hacker_news


def _mock_urlopen(data):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestHackerNews:
    @patch("ai_arch_toolkit.toolkit.tools._news.urllib.request.urlopen")
    def test_returns_stories(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _mock_urlopen([100, 200]),
            _mock_urlopen(
                {
                    "title": "Show HN: Cool Project",
                    "url": "https://example.com",
                    "score": 150,
                    "by": "alice",
                    "descendants": 42,
                }
            ),
            _mock_urlopen(
                {
                    "title": "Ask HN: Best Languages?",
                    "url": "https://example2.com",
                    "score": 80,
                    "by": "bob",
                    "descendants": 20,
                }
            ),
        ]
        result = hacker_news(count=2)
        assert "Cool Project" in result
        assert "Best Languages" in result
        assert "150 points" in result
        assert "alice" in result
        assert "42 comments" in result
        assert "Top 2" in result

    @patch("ai_arch_toolkit.toolkit.tools._news.urllib.request.urlopen")
    def test_clamps_count(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _mock_urlopen([100]),
            _mock_urlopen(
                {
                    "title": "Story",
                    "score": 10,
                    "by": "user",
                    "descendants": 0,
                }
            ),
        ]
        result = hacker_news(count=99)
        assert "Story" in result

    @patch("ai_arch_toolkit.toolkit.tools._news.urllib.request.urlopen")
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = hacker_news()
        assert "Failed" in result

    @patch("ai_arch_toolkit.toolkit.tools._news.urllib.request.urlopen")
    def test_skips_failed_items(self, mock_urlopen):
        # First call returns IDs, second fails, third succeeds
        mock_urlopen.side_effect = [
            _mock_urlopen([100, 200]),
            TimeoutError(),
            _mock_urlopen(
                {
                    "title": "Good Story",
                    "score": 50,
                    "by": "user",
                    "descendants": 5,
                }
            ),
        ]
        result = hacker_news(count=2)
        assert "Good Story" in result
