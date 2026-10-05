"""Tests for toolkit/tools/_news.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._news import hacker_news
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


class TestHackerNews:
    @patch(HTTP_OPEN)
    def test_returns_stories(self, mock_urlopen):
        mock_urlopen.side_effect = [
            respond([100, 200]),
            respond(
                {
                    "title": "Show HN: Cool Project",
                    "url": "https://example.com",
                    "score": 150,
                    "by": "alice",
                    "descendants": 42,
                }
            ),
            respond(
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

    @patch(HTTP_OPEN)
    def test_clamps_count(self, mock_urlopen):
        mock_urlopen.side_effect = [
            respond([100]),
            respond(
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

    @patch(HTTP_OPEN)
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        with pytest.raises(ToolFailure) as caught:
            hacker_news()
        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable
        assert "timed out" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_says_which_stories_it_could_not_load(self, mock_urlopen):
        # First call returns IDs, second fails, third succeeds
        mock_urlopen.side_effect = [
            respond([100, 200]),
            TimeoutError(),
            respond(
                {
                    "title": "Good Story",
                    "score": 50,
                    "by": "user",
                    "descendants": 5,
                }
            ),
        ]
        result = hacker_news(count=2)

        assert result.startswith("Hacker News — Top 2 stories:\n\n  2. Good Story")
        assert result.endswith("Could not load 1 of them: #1 (item 100: request timed out.)")

    @patch(HTTP_OPEN)
    def test_says_why_when_no_story_loads(self, mock_urlopen):
        # The API answers an item it does not have with null (2026-09-30).
        mock_urlopen.side_effect = [respond([100]), respond(b"null")]

        with pytest.raises(ToolFailure) as caught:
            hacker_news(count=1)

        assert caught.value.error.type == "upstream"
        assert not caught.value.error.retryable
        assert caught.value.error.message == (
            "could not load any of the Hacker News top stories: #1 (item 100: could not parse "
            "API response: expected a JSON object, got null)"
        )

    @patch(HTTP_OPEN)
    def test_a_rate_limited_list_is_rate_limited(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(429)

        with pytest.raises(ToolFailure) as caught:
            hacker_news(count=1)

        assert caught.value.error.type == "rate_limited"
        assert caught.value.error.retryable
