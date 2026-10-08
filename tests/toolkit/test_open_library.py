"""Tests for toolkit/tools/_open_library.py: the search with its total, and whole records with
their authors' names, read window by window (T09)."""

from __future__ import annotations

import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools import _open_library
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


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


def _window(result: ToolResult | str) -> dict[str, Any]:
    assert isinstance(result, ToolResult)
    return result.metadata["window"]


_SEARCH_DOC = {
    "key": "/works/OL27448W",
    "title": "The Lord of the Rings",
    "author_name": ["J.R.R. Tolkien"],
    "author_key": ["OL26320A"],
    "first_publish_year": 1954,
    "edition_count": 251,
    "publisher": ["Allen & Unwin"],
    "language": ["eng", "por"],
    "isbn": ["9780618640157", "0618640150"],
    "ebook_access": "borrowable",
}
_WORK = {
    "key": "/works/OL27448W",
    "title": "The Lord of the Rings",
    "authors": [{"author": {"key": "/authors/OL26320A"}, "type": {"key": "/type/author_role"}}],
    "first_publish_date": "1954",
    "subjects": ["Fantasy fiction", "Quests"],
    "covers": [14625765],
    "description": {"type": "/type/text", "value": "An epic fantasy novel."},
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


def _authors(*pairs: tuple[str, str]) -> dict[str, Any]:
    """A search/authors.json answer: the bare key and the name of each author
    (https://openlibrary.org/dev/docs/api/authors)."""
    docs = [{"key": key, "name": name} for key, name in pairs]
    return {"numFound": len(docs), "start": 0, "numFoundExact": True, "docs": docs}


_TOLKIEN = _authors(("OL26320A", "J. R. R. Tolkien"))
_DAHL = _authors(("OL34184A", "Roald Dahl"))


def _search(docs: list[dict[str, Any]], *, total: int, start: int = 0) -> dict[str, Any]:
    return {"numFound": total, "num_found": total, "start": start, "docs": docs}


def _requests(mock_urlopen: MagicMock) -> list[tuple[str, dict[str, list[str]]]]:
    urls = [urlparse(call.args[0].full_url) for call in mock_urlopen.call_args_list]
    return [(url.path, parse_qs(url.query)) for url in urls]


def _execute(name: str, fn: Any, **arguments: Any) -> Any:
    return ToolGroup(fn).execute(ToolCall(id="c1", name=name, input=arguments))


class TestOpenLibrarySearch:
    @patch(HTTP_OPEN)
    def test_returns_results_numbered_with_the_total_and_the_next_start(self, mock_urlopen):
        mock_urlopen.return_value = respond(_search([_SEARCH_DOC, _SEARCH_DOC], total=629))

        result = open_library_search("lord rings", max_results=2)

        lines = _text(result).splitlines()
        assert lines[:4] == [
            "Open Library books that match query 'lord rings':",
            "1. The Lord of the Rings (work OL27448W)",
            "   by J.R.R. Tolkien (OL26320A) | first published 1954 | 251 editions | "
            "languages: eng, por | ebook: borrowable",
            "   ISBN: 9780618640157, 0618640150 | publishers: Allen & Unwin",
        ]
        assert lines[-1] == "[results 1-2 of 629 | next: start=2]"
        assert _window(result)["next_call"] == {"start": 2}

        ((path, params),) = _requests(mock_urlopen)
        assert path == "/search.json"
        assert params["q"] == ["lord rings"]
        assert params["limit"] == ["2"]
        assert params["offset"] == ["0"]
        assert set(params["fields"][0].split(",")) >= {"key", "author_name", "author_key"}

    @patch(HTTP_OPEN)
    def test_the_next_page_is_numbered_on(self, mock_urlopen):
        mock_urlopen.return_value = respond(_search([_SEARCH_DOC], total=3, start=2))

        result = open_library_search("lord rings", max_results=2, start=2)

        assert _text(result).splitlines()[1].startswith("3. The Lord of the Rings")
        assert _text(result).endswith("[results 3-3 of 3 | end]")

    @patch(HTTP_OPEN)
    def test_long_lists_in_a_result_say_how_many_more(self, mock_urlopen):
        doc = {**_SEARCH_DOC, "isbn": [f"97800000000{n:02d}" for n in range(40)]}
        mock_urlopen.return_value = respond(_search([doc], total=1))

        text = _text(open_library_search("lord rings"))

        assert "ISBN: 9780000000000, 9780000000001, 9780000000002 (+37 more)" in text

    @patch(HTTP_OPEN)
    def test_filters_are_sent_and_named(self, mock_urlopen):
        mock_urlopen.return_value = respond(_search([], total=0, start=40))

        result = open_library_search(
            "",
            max_results=20,
            start=40,
            title="Dune",
            author="Frank Herbert",
            subject="science fiction",
            isbn="9780441172719",
        )

        params = _requests(mock_urlopen)[0][1]
        assert params["limit"] == ["20"]
        assert params["offset"] == ["40"]
        assert params["title"] == ["Dune"]
        assert params["author"] == ["Frank Herbert"]
        assert params["subject"] == ["science fiction"]
        assert params["isbn"] == ["9780441172719"]
        assert _text(result).endswith("[no results from 41 of 0 | end]")

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "provide query" in _invalid(open_library_search, "")
        mock_urlopen.assert_not_called()

    @pytest.mark.parametrize("arguments", [{"start": -1}, {"max_results": 0}, {"max_results": 21}])
    @patch(HTTP_OPEN)
    def test_limits_are_refused_through_the_executor(self, mock_urlopen, arguments):
        result = _execute("open_library_search", open_library_search, query="x", **arguments)

        assert result.error is not None
        assert result.error.type == "validation_error"
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond("not json")

        failure = _failure(open_library_search, "test")

        assert failure.error.type == "upstream"
        assert "could not parse" in str(failure)

    @patch(HTTP_OPEN)
    def test_no_results_is_an_answer_that_names_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond(_search([], total=0))

        assert _text(open_library_search("zzqqxx", author="Nobody")) == (
            "No Open Library books match query 'zzqqxx', author 'Nobody'."
        )

    @patch(HTTP_OPEN)
    def test_a_404_on_the_search_is_an_endpoint_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(open_library_search, "test")

        assert failure.error.type == "upstream"
        assert "Open Library: endpoint not found (HTTP 404)" in failure.error.message

    @patch(HTTP_OPEN)
    def test_a_429_is_rate_limited(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        failure = _failure(open_library_search, "test")

        assert (failure.error.type, failure.error.retryable) == ("rate_limited", True)


class TestOpenLibraryWork:
    @patch(HTTP_OPEN)
    def test_returns_the_work_with_its_authors_names(self, mock_urlopen):
        # The authors came out as /authors/OL…A keys.
        mock_urlopen.side_effect = [respond(_WORK), respond(_TOLKIEN)]

        result = open_library_work("https://openlibrary.org/works/OL27448W")

        assert _text(result) == (
            "Open Library work OL27448W:\n"
            "Title: The Lord of the Rings\n"
            "Authors: J. R. R. Tolkien (OL26320A)\n"
            "First published: 1954\n"
            "Page: https://openlibrary.org/works/OL27448W\n"
            "Cover: https://covers.openlibrary.org/b/id/14625765-M.jpg\n"
            "Description: An epic fantasy novel.\n"
            "Subjects (2): Fantasy fiction, Quests\n"
            "Links: Wikipedia: https://en.wikipedia.org/wiki/The_Lord_of_the_Rings"
        )
        (work_path, _), (names_path, names) = _requests(mock_urlopen)
        assert work_path == "/works/OL27448W.json"
        assert names_path == "/search/authors.json"
        assert names["q"] == ["key:(/authors/OL26320A)"]

    @patch(HTTP_OPEN)
    def test_every_author_is_named_in_one_request(self, mock_urlopen):
        keys = [f"OL{n}A" for n in range(1, 4)]
        work = {**_WORK, "authors": [{"author": {"key": f"/authors/{key}"}} for key in keys]}
        names = _authors(("OL3A", "Third"), ("OL1A", "First"))  # in any order; one unknown
        mock_urlopen.side_effect = [respond(work), respond(names)]

        text = _text(open_library_work("OL27448W"))

        assert "Authors: First (OL1A), OL2A (no name on Open Library), Third (OL3A)" in text
        assert _requests(mock_urlopen)[1][1]["q"] == [
            "key:(/authors/OL1A OR /authors/OL2A OR /authors/OL3A)"
        ]

    @patch(HTTP_OPEN)
    def test_a_work_without_authors_asks_for_no_names(self, mock_urlopen):
        mock_urlopen.side_effect = [respond({**_WORK, "authors": []})]

        text = _text(open_library_work("OL27448W"))

        assert "Authors" not in text
        assert mock_urlopen.call_count == 1

    @patch(HTTP_OPEN)
    def test_names_are_asked_for_the_first_fifty_authors_at_most(self, mock_urlopen):
        work = {**_WORK, "authors": [{"author": {"key": f"/authors/OL{n}A"}} for n in range(60)]}
        mock_urlopen.side_effect = [respond(work), respond(_authors(("OL0A", "Zero")))]

        text = _text(open_library_work("OL27448W"))

        assert _requests(mock_urlopen)[1][1]["q"][0].count("/authors/") == 50
        assert "Zero (OL0A)" in text
        assert "OL59A" in text  # listed, with its key, though not named

    @patch(HTTP_OPEN)
    def test_no_description_or_list_is_cut_short_and_a_long_work_reads_on(self, mock_urlopen):
        work = {
            **_WORK,
            "description": "word " * 600,
            "subjects": [f"Subject {n}" for n in range(500)],
        }
        mock_urlopen.side_effect = [respond(body) for body in (work, _TOLKIEN) * 2]

        first = open_library_work("OL27448W", max_chars=20_000)
        second = open_library_work("OL27448W", max_chars=500, offset=_window(first)["last"] - 300)

        text = _text(first)
        assert "Description: " + "word " * 599 + "word\n" in text
        assert "Subject 499" in text
        assert "Subjects (500): Subject 0, " in text
        assert _window(second)["first"] == _window(first)["last"] - 300

    @patch(HTTP_OPEN)
    def test_a_long_work_names_the_next_offset(self, mock_urlopen):
        work = {**_WORK, "subjects": [f"Subject {n}" for n in range(500)]}
        mock_urlopen.side_effect = [respond(work), respond(_TOLKIEN)]

        result = open_library_work("OL27448W", max_chars=500)

        window = _window(result)
        assert _text(result).endswith(
            f"[chars 0-{window['last']} of {window['total']} | next: offset={window['last']}]"
        )

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

    @patch(HTTP_OPEN)
    def test_a_failure_of_the_names_request_is_the_tools_failure(self, mock_urlopen):
        mock_urlopen.side_effect = [respond(_WORK), http_error(503, "Unavailable")]

        failure = _failure(open_library_work, "OL27448W")

        assert (failure.error.type, failure.error.retryable) == ("upstream", True)


class TestOpenLibraryIsbn:
    @patch(HTTP_OPEN)
    def test_returns_the_edition(self, mock_urlopen):
        mock_urlopen.side_effect = [respond(_ISBN), respond(_DAHL)]

        result = open_library_isbn("978-0140328721")

        assert _text(result) == (
            "Open Library ISBN 9780140328721:\n"
            "Title: Fantastic Mr. Fox\n"
            "Authors: Roald Dahl (OL34184A)\n"
            "Published: 1988-10-01 | Pages: 96\n"
            "Publishers: Puffin\n"
            "Languages: eng\n"
            "ISBN-13: 9780140328721 | ISBN-10: 0140328726\n"
            "Work: OL45804W (open_library_work reads it)\n"
            "Page: https://openlibrary.org/books/OL7353617M\n"
            "Cover: https://covers.openlibrary.org/b/id/15152634-M.jpg\n"
            "Description: A story about a clever fox."
        )
        paths = [path for path, _ in _requests(mock_urlopen)]
        assert paths == ["/isbn/9780140328721.json", "/search/authors.json"]

    @pytest.mark.parametrize(
        ("written", "iso"),
        [
            ("October 1, 1988", "1988-10-01"),
            ("Oct 1, 1988", "1988-10-01"),
            ("1 October 1988", "1988-10-01"),
            ("September 1970", "1970-09"),
            ("1954", "1954"),
            ("1988-10-01", "1988-10-01"),
            ("c1988", "c1988"),  # kept as written when it is no date the tool reads
        ],
    )
    @patch(HTTP_OPEN)
    def test_dates_come_in_iso_8601(self, mock_urlopen, written, iso):
        edition = {**_ISBN, "publish_date": written, "authors": []}
        mock_urlopen.return_value = respond(edition)

        text = _text(open_library_isbn("9780140328721"))

        assert f"Published: {iso} | Pages: 96" in text

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


def test_requests_are_spaced_as_open_library_asks() -> None:
    # One request a second for a caller that sends no email (https://openlibrary.org/developers/api).
    assert _open_library._API.min_interval_s == 1.0


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
    mock_urlopen.side_effect = [respond(_MERGED), respond(_WORK), respond(_TOLKIEN)]

    result = open_library_work("OL100005W")

    assert _text(result).startswith(
        "Open Library work OL100005W (merged into OL27448W):\nTitle: The Lord"
    )
    paths = [path for path, _ in _requests(mock_urlopen)]
    assert paths == ["/works/OL100005W.json", "/works/OL27448W.json", "/search/authors.json"]


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
    mock_urlopen.side_effect = [respond(merged), respond(_ISBN), respond(_DAHL)]

    result = open_library_isbn("9780140328721")

    assert _text(result).startswith("Open Library ISBN 9780140328721:\nTitle: Fantastic Mr. Fox")
    assert [path for path, _ in _requests(mock_urlopen)][1] == "/books/OL2M.json"


@patch(HTTP_OPEN)
def test_a_redirect_to_something_that_is_not_a_record_is_an_error(mock_urlopen):
    mock_urlopen.return_value = respond({**_MERGED, "location": "/../admin"})

    failure = _failure(open_library_work, "OL100005W")

    assert failure.error.type == "upstream"
    assert str(failure) == ("could not parse API response: a redirect without a record to go to")
