"""Tests for toolkit/tools/_open_library.py."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._open_library import (
    open_library_isbn,
    open_library_search,
    open_library_work,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _failure(fn, *args, **kwargs) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _invalid(fn, *args, **kwargs) -> str:
    failure = _failure(fn, *args, **kwargs)
    assert failure.error.type == "validation_error"
    return failure.error.message


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


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestOpenLibrarySearch:
    @patch(HTTP_OPEN)
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = respond({"docs": [_SEARCH_DOC]})

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

    @patch(HTTP_OPEN)
    def test_filters_and_caps_max_results(self, mock_urlopen):
        mock_urlopen.return_value = respond({"docs": []})

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

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "provide query" in _invalid(open_library_search, "")
        assert "start must be greater than or equal to 0" in _invalid(
            open_library_search, "test", start=-1
        )
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond("not json")

        failure = _failure(open_library_search, "test")

        assert failure.error.type == "upstream"
        assert "could not parse" in str(failure)

    @patch(HTTP_OPEN)
    def test_no_results_is_an_answer(self, mock_urlopen):
        mock_urlopen.return_value = respond({"numFound": 0, "docs": []})

        assert open_library_search("zzqqxx") == "No Open Library results found."

    @patch(HTTP_OPEN)
    def test_a_404_on_the_search_is_an_endpoint_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(open_library_search, "test")

        assert failure.error.type == "upstream"
        assert "Open Library: endpoint not found (HTTP 404)" in failure.error.message


class TestOpenLibraryWork:
    @patch(HTTP_OPEN)
    def test_returns_work_by_url(self, mock_urlopen):
        mock_urlopen.return_value = respond(_WORK)

        result = open_library_work("https://openlibrary.org/works/OL27448W")

        assert result.startswith("Open Library work OL27448W:")
        assert "The Lord of the Rings" in result
        assert "Authors: /authors/OL26320A" in result
        assert "Description: An epic fantasy novel." in result
        assert "Wikipedia: https://en.wikipedia.org/wiki/The_Lord_of_the_Rings" in result
        assert "https://openlibrary.org/works/OL27448W" in result

        request = _called_request(mock_urlopen)
        assert urlparse(request.full_url).path == "/works/OL27448W.json"

    @patch(HTTP_OPEN)
    def test_invalid_work_id(self, mock_urlopen):
        assert "invalid work_id 'OL123M'" in _invalid(open_library_work, "OL123M")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://openlibrary.org/works/OL000W.json",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

        failure = _failure(open_library_work, "OL000W")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "Open Library has no work OL000W; find works with open_library_search"
        )

    @patch(HTTP_OPEN)
    def test_another_status_stays_the_requests_failure(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://openlibrary.org/works/OL1W.json",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=None,
        )

        failure = _failure(open_library_work, "OL1W")

        assert failure.error.type == "upstream"
        assert failure.error.retryable


class TestOpenLibraryIsbn:
    @patch(HTTP_OPEN)
    def test_returns_isbn(self, mock_urlopen):
        mock_urlopen.return_value = respond(_ISBN)

        result = open_library_isbn("978-0140328721")

        assert result.startswith("Open Library ISBN 9780140328721:")
        assert "Fantastic Mr. Fox" in result
        assert "key: /books/OL7353617M | published: October 1, 1988 | pages: 96" in result
        assert "Authors: /authors/OL34184A" in result
        assert "Publishers: Puffin" in result
        assert "Languages: eng" in result
        assert "Works: /works/OL45804W" in result
        assert "Description: A story about a clever fox." in result

    @patch(HTTP_OPEN)
    def test_invalid_isbn(self, mock_urlopen):
        assert "invalid ISBN 'bad'" in _invalid(open_library_isbn, "bad")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_isbn_without_a_record_is_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond({})

        failure = _failure(open_library_isbn, "9780140328721")

        assert failure.error.type == "not_found"
        assert "no edition with ISBN 9780140328721" in failure.error.message

    @patch(HTTP_OPEN)
    def test_a_404_for_an_isbn_is_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found", body=b'{"error": "notfound"}')

        failure = _failure(open_library_isbn, "9780140328721")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "Open Library has no edition with ISBN 9780140328721; find books with "
            "open_library_search"
        )


# Records as Open Library answered them live, with HTTP 200 (2026-09-30).
_MERGED = {
    "location": "/works/OL27448W",
    "key": "/works/OL100005W",
    "type": {"key": "/type/redirect"},
    "latest_revision": 4,
    "revision": 4,
}
_DELETED = {
    "key": "/works/OL1000619W",
    "type": {"key": "/type/delete"},
    "latest_revision": 2,
    "revision": 2,
}


@patch(HTTP_OPEN)
def test_a_merged_work_is_followed_to_the_work_it_went_into(mock_urlopen):
    # It read as an untitled work.
    mock_urlopen.side_effect = [respond(_MERGED), respond(_WORK)]

    result = open_library_work("OL100005W")

    assert result.startswith("Open Library work OL100005W (merged into OL27448W):\nThe Lord")
    paths = [urlparse(call.args[0].full_url).path for call in mock_urlopen.call_args_list]
    assert paths == ["/works/OL100005W.json", "/works/OL27448W.json"]


@patch(HTTP_OPEN)
def test_a_deleted_work_says_so(mock_urlopen):
    mock_urlopen.return_value = respond(_DELETED)

    failure = _failure(open_library_work, "OL1000619W")

    assert failure.error.type == "not_found"
    assert failure.error.message == (
        "Open Library deleted /works/OL1000619W; find another with open_library_search"
    )


@patch(HTTP_OPEN)
def test_redirects_that_do_not_end_are_an_error(mock_urlopen):
    loop = {**_MERGED, "location": "/works/OL100005W"}
    mock_urlopen.side_effect = [respond(loop) for _ in range(4)]

    failure = _failure(open_library_work, "OL100005W")

    assert failure.error.type == "upstream"
    assert str(failure) == "more than 3 redirects between Open Library records"
    assert mock_urlopen.call_count == 4


@patch(HTTP_OPEN)
def test_an_isbn_whose_edition_was_merged_reads_the_edition_it_went_into(mock_urlopen):
    merged = {"key": "/books/OL1M", "type": {"key": "/type/redirect"}, "location": "/books/OL2M"}
    mock_urlopen.side_effect = [respond(merged), respond(_ISBN)]

    result = open_library_isbn("9780140328721")

    assert result.startswith("Open Library ISBN 9780140328721:\nFantastic Mr. Fox")
    assert urlparse(_called_request(mock_urlopen).full_url).path == "/books/OL2M.json"


@patch(HTTP_OPEN)
def test_a_redirect_to_something_that_is_not_a_record_is_an_error(mock_urlopen):
    mock_urlopen.return_value = respond({**_MERGED, "location": "/../admin"})

    failure = _failure(open_library_work, "OL100005W")

    assert failure.error.type == "upstream"
    assert str(failure) == ("could not parse API response: a redirect without a record to go to")
