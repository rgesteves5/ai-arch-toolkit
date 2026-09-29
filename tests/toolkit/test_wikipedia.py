"""Tests for toolkit/tools/_wikipedia.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_arch_toolkit.toolkit.tools._wikipedia import (
    wikipedia_article,
    wikipedia_related,
    wikipedia_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, respond


class TestWikipediaSearch:
    @patch(HTTP_OPEN)
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "query": {
                    "search": [
                        {"title": "Python", "snippet": "A <b>programming</b> language."},
                        {"title": "Monty Python", "snippet": "A comedy group."},
                    ]
                }
            }
        )
        result = wikipedia_search("Python")
        assert "Python" in result
        assert "programming" in result
        assert "<b>" not in result

    @patch(HTTP_OPEN)
    def test_no_results(self, mock_urlopen):
        mock_urlopen.return_value = respond({"query": {"search": []}})
        result = wikipedia_search("xyznonexistent")
        assert "No Wikipedia results" in result

    @patch(HTTP_OPEN)
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = wikipedia_search("test")
        assert "failed" in result.lower()


class TestWikipediaArticle:
    @patch(HTTP_OPEN)
    def test_returns_extract(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"query": {"pages": {"123": {"title": "Python", "extract": "Python is a language."}}}}
        )
        result = wikipedia_article("Python")
        assert "Python" in result
        assert "language" in result

    @patch(HTTP_OPEN)
    def test_missing_article(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"query": {"pages": {"-1": {"title": "Xyz", "missing": ""}}}}
        )
        result = wikipedia_article("Xyz")
        assert "not found" in result.lower()

    @patch(HTTP_OPEN)
    def test_truncation(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"query": {"pages": {"1": {"title": "Big", "extract": "x" * 10000}}}}
        )
        result = wikipedia_article("Big", max_chars=100)
        assert "Truncated" in result


class TestWikipediaRelated:
    @patch(HTTP_OPEN)
    def test_returns_links(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "query": {
                    "pages": {
                        "1": {
                            "title": "Python",
                            "links": [
                                {"title": "Guido van Rossum"},
                                {"title": "Programming language"},
                            ],
                        }
                    }
                }
            }
        )
        result = wikipedia_related("Python")
        assert "Related Wikipedia pages" in result
        assert "Guido van Rossum" in result

    @patch(HTTP_OPEN)
    def test_falls_back_to_search_when_missing(self, mock_urlopen):
        mock_urlopen.side_effect = [
            respond({"query": {"pages": {"-1": {"title": "Missing", "missing": ""}}}}),
            respond({"query": {"search": [{"title": "Python", "snippet": "A language."}]}}),
        ]
        result = wikipedia_related("Missing")
        assert "Wikipedia results" in result


_RATELIMITED = {
    "error": {
        "code": "ratelimited",
        "info": "You've exceeded your rate limit. Please wait some time and try again.",
    }
}


@pytest.mark.parametrize(
    ("fn", "failure"),
    [
        (wikipedia_search, "Wikipedia search failed"),
        (wikipedia_article, "Wikipedia API failed"),
        (wikipedia_related, "Wikipedia related lookup failed"),
    ],
)
@patch(HTTP_OPEN)
def test_an_error_the_api_reports_is_the_tools_error(mock_urlopen, fn, failure):
    # MediaWiki sends it with HTTP 200; wikipedia_related does not fall back to a search.
    mock_urlopen.return_value = respond(_RATELIMITED)

    result = fn("Python")

    assert result == (
        f"{failure}: ratelimited: You've exceeded your rate limit. Please wait some time and try "
        "again."
    )
    assert mock_urlopen.call_count == 1


_INVALID_TITLE = {
    "batchcomplete": "",
    "query": {
        "pages": {
            "-1": {
                "title": "a[b",
                "invalidreason": 'The requested page title contains invalid characters: "[".',
                "invalid": "",
            }
        }
    },
}


@pytest.mark.parametrize(
    ("fn", "failure"),
    [
        (wikipedia_article, "Wikipedia API failed"),
        (wikipedia_related, "Wikipedia related lookup failed"),
    ],
)
@patch(HTTP_OPEN)
def test_an_invalid_title_is_the_tools_error_with_the_apis_reason(mock_urlopen, fn, failure):
    # As answered live (2026-09-29): wikipedia_article said "No extract available", and
    # wikipedia_related searched for the title instead.
    mock_urlopen.return_value = respond(_INVALID_TITLE)

    result = fn("a[b")

    assert result == (
        f'{failure}: invalid title: The requested page title contains invalid characters: "[".'
    )
    assert mock_urlopen.call_count == 1


@patch(HTTP_OPEN)
def test_an_invalid_title_without_a_reason_still_says_so(mock_urlopen):
    mock_urlopen.return_value = respond({"query": {"pages": {"-1": {"invalid": ""}}}})

    assert wikipedia_article("Talk:") == "Wikipedia API failed: invalid title"


@patch(HTTP_OPEN)
def test_an_answer_without_pages_is_not_found(mock_urlopen):
    # An interwiki title, like "fr:Paris", comes back with no pages.
    mock_urlopen.return_value = respond({"query": {"interwiki": [{"title": "fr:Paris"}]}})

    assert wikipedia_article("fr:Paris") == "Article not found: 'fr:Paris'"


@patch(HTTP_OPEN)
def test_an_empty_search_reports_the_missing_parameter(mock_urlopen):
    mock_urlopen.return_value = respond(
        {"error": {"code": "missingparam", "info": 'The "srsearch" parameter must be set.'}}
    )

    assert wikipedia_search("") == (
        'Wikipedia search failed: missingparam: The "srsearch" parameter must be set.'
    )


@patch(HTTP_OPEN)
def test_max_chars_is_clamped(mock_urlopen):
    page = {"query": {"pages": {"1": {"title": "T", "extract": "e" * 2_000_000}}}}

    mock_urlopen.return_value = respond(page)
    assert wikipedia_article("T", max_chars=-1).startswith("e\n\n[Truncated]")
    mock_urlopen.return_value = respond(page)
    assert len(wikipedia_article("T", max_chars=10**9)) < 100_100
