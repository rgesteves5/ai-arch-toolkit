"""Tests for toolkit/tools/_wikipedia.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._wikipedia import (
    wikipedia_article,
    wikipedia_related,
    wikipedia_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, respond


def _failure(call):
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value.error


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

        error = _failure(lambda: wikipedia_search("test"))

        assert error.type == "upstream"
        assert error.retryable


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

        error = _failure(lambda: wikipedia_article("Xyz"))

        assert error.type == "not_found"
        assert error.message == (
            "no Wikipedia article titled 'Xyz'; find the exact title with wikipedia_search"
        )

    @patch(HTTP_OPEN)
    def test_a_page_without_an_extract_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond({"query": {"pages": {"1": {"title": "Xyz"}}}})

        assert wikipedia_article("Xyz") == "No extract available for: 'Xyz'"

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


@pytest.mark.parametrize("fn", [wikipedia_search, wikipedia_article, wikipedia_related])
@patch(HTTP_OPEN)
def test_an_error_the_api_reports_is_the_tools_error(mock_urlopen, fn):
    # MediaWiki sends it with HTTP 200; wikipedia_related does not fall back to a search.
    mock_urlopen.return_value = respond(_RATELIMITED)

    error = _failure(lambda: fn("Python"))

    assert error.type == "upstream"
    assert error.message == (
        "ratelimited: You've exceeded your rate limit. Please wait some time and try again."
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


@pytest.mark.parametrize("fn", [wikipedia_article, wikipedia_related])
@patch(HTTP_OPEN)
def test_an_invalid_title_is_the_tools_error_with_the_apis_reason(mock_urlopen, fn):
    # As answered live (2026-09-29): wikipedia_article said "No extract available", and
    # wikipedia_related searched for the title instead.
    mock_urlopen.return_value = respond(_INVALID_TITLE)

    error = _failure(lambda: fn("a[b"))

    assert error.type == "validation_error"
    assert error.message == (
        'invalid title: The requested page title contains invalid characters: "["; '
        "give an article title, e.g. from wikipedia_search"
    )
    assert mock_urlopen.call_count == 1


@patch(HTTP_OPEN)
def test_an_invalid_title_without_a_reason_still_says_so(mock_urlopen):
    mock_urlopen.return_value = respond({"query": {"pages": {"-1": {"invalid": ""}}}})

    error = _failure(lambda: wikipedia_article("Talk:"))

    assert error.type == "validation_error"
    assert error.message.startswith("invalid title; ")


@patch(HTTP_OPEN)
def test_an_answer_without_pages_is_not_found(mock_urlopen):
    # An interwiki title, like "fr:Paris", comes back with no pages.
    mock_urlopen.return_value = respond({"query": {"interwiki": [{"title": "fr:Paris"}]}})

    error = _failure(lambda: wikipedia_article("fr:Paris"))

    assert error.type == "not_found"
    assert "'fr:Paris'" in error.message


@patch(HTTP_OPEN)
def test_an_empty_search_is_refused_before_the_request(mock_urlopen):
    error = _failure(lambda: wikipedia_search("  "))

    assert error.type == "validation_error"
    assert "query cannot be empty" in error.message
    mock_urlopen.assert_not_called()


@patch(HTTP_OPEN)
def test_max_chars_is_clamped(mock_urlopen):
    page = {"query": {"pages": {"1": {"title": "T", "extract": "e" * 2_000_000}}}}

    mock_urlopen.return_value = respond(page)
    assert wikipedia_article("T", max_chars=-1).startswith("e\n\n[Truncated]")
    mock_urlopen.return_value = respond(page)
    assert len(wikipedia_article("T", max_chars=10**9)) < 100_100


@pytest.mark.parametrize(
    ("page", "why"),
    [
        ({"title": "Xyz", "missing": ""}, "No Wikipedia page 'Xyz'"),
        ({"title": "Xyz", "links": []}, "The Wikipedia page 'Xyz' links to no articles"),
    ],
)
@patch(HTTP_OPEN)
def test_the_search_that_stands_in_for_related_pages_says_why(mock_urlopen, page, why):
    # The switch to a search was silent.
    mock_urlopen.side_effect = [
        respond({"query": {"pages": {"-1": page}}}),
        respond({"query": {"search": [{"title": "Python", "snippet": "A language."}]}}),
    ]

    result = wikipedia_related("Xyz")

    assert result.startswith(f"{why}; searching instead.\nWikipedia results for 'Xyz':")
