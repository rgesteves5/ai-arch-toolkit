"""Tests for toolkit/tools/_wikipedia.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.toolkit.tools._wikipedia import (
    wikipedia_article,
    wikipedia_related,
    wikipedia_search,
)


def _mock_urlopen(data):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestWikipediaSearch:
    @patch("ai_arch_toolkit.toolkit.tools._wikipedia.urllib.request.urlopen")
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
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

    @patch("ai_arch_toolkit.toolkit.tools._wikipedia.urllib.request.urlopen")
    def test_no_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({"query": {"search": []}})
        result = wikipedia_search("xyznonexistent")
        assert "No Wikipedia results" in result

    @patch("ai_arch_toolkit.toolkit.tools._wikipedia.urllib.request.urlopen")
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = wikipedia_search("test")
        assert "failed" in result.lower()


class TestWikipediaArticle:
    @patch("ai_arch_toolkit.toolkit.tools._wikipedia.urllib.request.urlopen")
    def test_returns_extract(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            {"query": {"pages": {"123": {"title": "Python", "extract": "Python is a language."}}}}
        )
        result = wikipedia_article("Python")
        assert "Python" in result
        assert "language" in result

    @patch("ai_arch_toolkit.toolkit.tools._wikipedia.urllib.request.urlopen")
    def test_missing_article(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            {"query": {"pages": {"-1": {"title": "Xyz", "missing": ""}}}}
        )
        result = wikipedia_article("Xyz")
        assert "not found" in result.lower()

    @patch("ai_arch_toolkit.toolkit.tools._wikipedia.urllib.request.urlopen")
    def test_truncation(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            {"query": {"pages": {"1": {"title": "Big", "extract": "x" * 10000}}}}
        )
        result = wikipedia_article("Big", max_chars=100)
        assert "Truncated" in result


class TestWikipediaRelated:
    @patch("ai_arch_toolkit.toolkit.tools._wikipedia.urllib.request.urlopen")
    def test_returns_links(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
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

    @patch("ai_arch_toolkit.toolkit.tools._wikipedia.urllib.request.urlopen")
    def test_falls_back_to_search_when_missing(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _mock_urlopen({"query": {"pages": {"-1": {"title": "Missing", "missing": ""}}}}),
            _mock_urlopen({"query": {"search": [{"title": "Python", "snippet": "A language."}]}}),
        ]
        result = wikipedia_related("Missing")
        assert "Wikipedia results" in result
