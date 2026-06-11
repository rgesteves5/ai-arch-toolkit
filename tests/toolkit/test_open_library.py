"""Tests for toolkit/tools/_open_library.py."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._open_library import (
    open_library_isbn,
    open_library_search,
    open_library_work,
)

_SEARCH_DOC = {
    "key": "/works/OL27448W",
    "title": "The Lord of the Rings",
    "author_name": ["J.R.R. Tolkien"],
    "first_publish_year": 1954,
    "edition_count": 251,
    "publisher": ["Allen & Unwin"],
    "language": ["eng", "por"],
    "subject": ["Fantasy fiction", "Middle Earth"],
    "isbn": ["9780618640157", "0618640150"],
    "cover_i": 14625765,
    "ebook_access": "borrowable",
    "has_fulltext": True,
}
_WORK = {
    "key": "/works/OL27448W",
    "title": "The Lord of the Rings",
    "authors": [{"author": {"key": "/authors/OL26320A"}}],
    "first_publish_date": "1954",
    "subjects": ["Fantasy fiction", "Quests"],
    "covers": [14625765],
    "description": {"value": "An epic fantasy novel."},
    "links": [
        {"title": "Wikipedia", "url": "https://en.wikipedia.org/wiki/The_Lord_of_the_Rings"}
    ],
}
_ISBN = {
    "key": "/books/OL7353617M",
    "title": "Fantastic Mr. Fox",
    "authors": [{"key": "/authors/OL34184A"}],
    "publish_date": "October 1, 1988",
    "publishers": ["Puffin"],
    "languages": [{"key": "/languages/eng"}],
    "isbn_10": ["0140328726"],
    "isbn_13": ["9780140328721"],
    "number_of_pages": 96,
    "works": [{"key": "/works/OL45804W"}],
    "covers": [15152634],
    "description": "A story about a clever fox.",
}


def _mock_urlopen(data: dict | str):
    resp = MagicMock()
    if isinstance(data, dict):
        resp.read.return_value = json.dumps(data).encode()
    else:
        resp.read.return_value = data.encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestOpenLibrarySearch:
    @patch("ai_arch_toolkit.toolkit.tools._open_library.urllib.request.urlopen")
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({"docs": [_SEARCH_DOC]})

        result = open_library_search("lord rings", max_results=2)

        assert "Open Library results:" in result
        assert "The Lord of the Rings" in result
        assert "key: /works/OL27448W | first published: 1954 | editions: 251" in result
        assert "Authors: J.R.R. Tolkien" in result
        assert "ISBN: 9780618640157, 0618640150" in result
        assert "Languages: eng, por" in result
        assert "Cover: https://covers.openlibrary.org/b/id/14625765-M.jpg" in result
        assert "Description:" not in result

        params = _called_params(mock_urlopen)
        assert params["q"] == ["lord rings"]
        assert params["limit"] == ["2"]
        assert params["offset"] == ["0"]

    @patch("ai_arch_toolkit.toolkit.tools._open_library.urllib.request.urlopen")
    def test_filters_and_caps_max_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({"docs": []})

        open_library_search(
            "",
            max_results=99,
            start=40,
            title="Dune",
            author="Frank Herbert",
            subject="science fiction",
            isbn="9780441172719",
        )

        params = _called_params(mock_urlopen)
        assert params["limit"] == ["20"]
        assert params["offset"] == ["40"]
        assert params["title"] == ["Dune"]
        assert params["author"] == ["Frank Herbert"]
        assert params["subject"] == ["science fiction"]
        assert params["isbn"] == ["9780441172719"]

    @patch("ai_arch_toolkit.toolkit.tools._open_library.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "provide query" in open_library_search("")
        assert "start must be greater than or equal to 0" in open_library_search("test", start=-1)
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._open_library.urllib.request.urlopen")
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen("not json")

        result = open_library_search("test")

        assert "could not parse" in result


class TestOpenLibraryWork:
    @patch("ai_arch_toolkit.toolkit.tools._open_library.urllib.request.urlopen")
    def test_returns_work_by_url(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_WORK)

        result = open_library_work("https://openlibrary.org/works/OL27448W")

        assert result.startswith("Open Library work OL27448W:")
        assert "The Lord of the Rings" in result
        assert "Authors: /authors/OL26320A" in result
        assert "Description: An epic fantasy novel." in result
        assert "Wikipedia: https://en.wikipedia.org/wiki/The_Lord_of_the_Rings" in result
        assert "https://openlibrary.org/works/OL27448W" in result

        request = _called_request(mock_urlopen)
        assert urlparse(request.full_url).path == "/works/OL27448W.json"

    @patch("ai_arch_toolkit.toolkit.tools._open_library.urllib.request.urlopen")
    def test_invalid_work_id(self, mock_urlopen):
        result = open_library_work("OL123M")

        assert "invalid work_id" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._open_library.urllib.request.urlopen")
    def test_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://openlibrary.org/works/OL000W.json",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

        result = open_library_work("OL000W")

        assert "not found" in result.lower()


class TestOpenLibraryIsbn:
    @patch("ai_arch_toolkit.toolkit.tools._open_library.urllib.request.urlopen")
    def test_returns_isbn(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_ISBN)

        result = open_library_isbn("978-0140328721")

        assert result.startswith("Open Library ISBN 9780140328721:")
        assert "Fantastic Mr. Fox" in result
        assert "key: /books/OL7353617M | published: October 1, 1988 | pages: 96" in result
        assert "Authors: /authors/OL34184A" in result
        assert "Publishers: Puffin" in result
        assert "Languages: eng" in result
        assert "Works: /works/OL45804W" in result
        assert "Description: A story about a clever fox." in result

    @patch("ai_arch_toolkit.toolkit.tools._open_library.urllib.request.urlopen")
    def test_invalid_isbn(self, mock_urlopen):
        result = open_library_isbn("bad")

        assert "invalid ISBN" in result
        mock_urlopen.assert_not_called()
