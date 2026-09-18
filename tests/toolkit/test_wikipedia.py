"""Tests for toolkit/tools/_wikipedia.py."""

from __future__ import annotations

from unittest.mock import patch

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


@patch(HTTP_OPEN)
def test_max_chars_is_clamped(mock_urlopen):
    page = {"query": {"pages": {"1": {"title": "T", "extract": "e" * 2_000_000}}}}

    mock_urlopen.return_value = respond(page)
    assert wikipedia_article("T", max_chars=-1).startswith("e\n\n[Truncated]")
    mock_urlopen.return_value = respond(page)
    assert len(wikipedia_article("T", max_chars=10**9)) < 100_100
