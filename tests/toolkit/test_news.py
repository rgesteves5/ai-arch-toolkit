"""Tests for toolkit/tools/_news.py."""

from __future__ import annotations

from unittest.mock import patch

from ai_arch_toolkit.toolkit.tools._news import hacker_news
from tests.toolkit.http_fakes import HTTP_OPEN, respond


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
        result = hacker_news()
        assert "Failed" in result

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

        assert hacker_news(count=1) == (
            "Failed to fetch HN top stories: #1 (item 100: could not parse API response: "
            "expected a JSON object, got null)"
        )
