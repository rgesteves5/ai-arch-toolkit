"""Tests for toolkit/tools/_news.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_arch_toolkit.core import ToolCall, ToolFailure, ToolGroup, ToolResult
from ai_arch_toolkit.toolkit.tools._news import hacker_news
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

# The API lists up to 500 top stories (https://github.com/HackerNews/API).
_IDS = list(range(1000, 1500))


def _story(number: int, **fields: object) -> dict[str, object]:
    return {
        "id": number,
        "type": "story",
        "title": f"Story {number}",
        "url": f"https://example.com/{number}",
        "score": 10,
        "by": "alice",
        "descendants": 3,
        "time": 1_760_000_000,
        **fields,
    }


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult) and result.ok
    assert isinstance(result.value, str)
    return result.value


def _sent(mock_urlopen: MagicMock) -> list[str]:
    return [call.args[0].full_url for call in mock_urlopen.call_args_list]


class TestHackerNews:
    @patch(HTTP_OPEN)
    def test_returns_stories_with_the_posting_time_in_utc(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [
            respond([100, 200]),
            respond(_story(100, title="Show HN: Cool &amp; Project", score=150, descendants=42)),
            respond(_story(200, title="Ask HN: Best Languages?", url=None)),
        ]

        text = _text(hacker_news(count=2))

        assert text.splitlines()[0] == "Hacker News top stories (2 on the list):"
        assert "1. Show HN: Cool & Project" in text  # the title is HTML (the API's README)
        assert "150 points by alice | 42 comments | posted 2025-10-09T08:53:20Z" in text
        # A story without a URL is its discussion page.
        assert "https://news.ycombinator.com/item?id=200" in text

    @patch(HTTP_OPEN)
    def test_the_list_reads_on_past_the_first_stories(self, mock_urlopen: MagicMock) -> None:
        # It showed at most 30 of about 500, with no way to read the rest.
        mock_urlopen.side_effect = [respond(_IDS), respond(_story(1030)), respond(_story(1031))]

        result = hacker_news(count=2, offset=30)
        text = _text(result)

        assert text.splitlines()[1] == "31. Story 1030"
        assert text.endswith("[results 31-32 of 500 | next: offset=32]")
        assert result.metadata["window"]["next_call"] == {"offset": 32}
        assert _sent(mock_urlopen)[1:] == [
            "https://hacker-news.firebaseio.com/v0/item/1030.json",
            "https://hacker-news.firebaseio.com/v0/item/1031.json",
        ]

    @patch(HTTP_OPEN)
    def test_the_last_stories_say_end(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [respond(_IDS), respond(_story(1499))]

        text = _text(hacker_news(count=5, offset=499))

        assert text.endswith("[results 500-500 of 500 | end]")

    @patch(HTTP_OPEN)
    def test_an_offset_past_the_list_says_so(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond([100, 200])

        text = _text(hacker_news(offset=10))

        assert text.endswith("[no results from 11 of 2 | end]")
        assert mock_urlopen.call_count == 1

    def test_count_and_offset_are_bounded_in_the_schema(self) -> None:
        group = ToolGroup(hacker_news)

        for args in ({"count": 31}, {"count": 0}, {"offset": 500}, {"offset": -1}):
            result = group.execute(ToolCall(id="c", name="hacker_news", input=args))
            assert not result.ok and result.error is not None
            assert result.error.type == "validation_error", args

    @patch(HTTP_OPEN)
    def test_a_story_that_fails_is_named_in_its_place(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [respond([100, 200]), TimeoutError(), respond(_story(200))]

        lines = _text(hacker_news(count=2)).splitlines()

        assert lines[1] == "1. (could not load item 100: request timed out.)"
        assert lines[2] == "2. Story 200"

    @patch(HTTP_OPEN)
    def test_api_failure(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = TimeoutError()
        with pytest.raises(ToolFailure) as caught:
            hacker_news()
        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable
        assert "timed out" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_says_why_when_no_story_loads(self, mock_urlopen: MagicMock) -> None:
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
    def test_a_rate_limited_list_is_rate_limited(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = http_error(429)

        with pytest.raises(ToolFailure) as caught:
            hacker_news(count=1)

        assert caught.value.error.type == "rate_limited"
        assert caught.value.error.retryable

    @patch(HTTP_OPEN)
    def test_an_error_firebase_explains_carries_its_words(self, mock_urlopen: MagicMock) -> None:
        # Firebase's REST errors are {"error": "..."} with the status
        # (https://firebase.google.com/docs/reference/rest/database#section-error-conditions).
        mock_urlopen.side_effect = http_error(401, body=b'{"error": "Permission denied"}')

        with pytest.raises(ToolFailure) as caught:
            hacker_news()

        assert caught.value.error.type == "upstream"
        assert caught.value.error.message == "HTTP error 401: Permission denied"
