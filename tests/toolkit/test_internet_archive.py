"""Tests for toolkit/tools/_internet_archive.py: the search with its total, and whole item records
read window by window (T09)."""

from __future__ import annotations

import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._internet_archive import (
    internet_archive_item,
    internet_archive_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _failure(fn, *args, **kwargs) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


def _window(result: ToolResult | str) -> dict[str, Any]:
    assert isinstance(result, ToolResult)
    return result.metadata["window"]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


_DOC = {
    "identifier": "goodytwoshoes00newyiala",
    "title": "Goody Two Shoes",
    "creator": ["Newbery"],
    "date": "1888-01-01T00:00:00Z",
    "mediatype": "texts",
    "collection": ["americana"],
    "subject": ["Children"],
    "downloads": "10",
    "item_size": "123",
}


def _search_answer(docs: list[dict[str, Any]], *, total: int, start: int = 0) -> dict[str, Any]:
    """An advancedsearch.php answer, as Solr shapes it (``response.numFound``, ``start``)."""
    return {
        "responseHeader": {"status": 0, "QTime": 12},
        "response": {"numFound": total, "start": start, "docs": docs},
    }


def _docs(first: int, count: int) -> list[dict[str, Any]]:
    return [
        {**_DOC, "identifier": f"item{n}", "title": f"Item {n}"}
        for n in range(first, first + count)
    ]


def _execute(name: str, fn: Any, **arguments: Any) -> Any:
    return ToolGroup(fn).execute(ToolCall(id="c1", name=name, input=arguments))


class TestInternetArchiveSearch:
    @patch(HTTP_OPEN)
    def test_a_query_the_search_cannot_run_says_why_and_what_to_check(self, mock_urlopen):
        # Sent with HTTP 200 (archive.org, 2026-09-29); it used to read as no items.
        mock_urlopen.return_value = respond(
            {"error": 'a group is empty (near char ")" at position 10)'}
        )

        failure = _failure(internet_archive_search, "title:(")

        assert failure.error.type == "upstream"
        assert str(failure) == (
            "the Internet Archive's search could not run the query: a group is empty "
            '(near char ")" at position 10); check its syntax (field:value, AND, OR, quotes, '
            "parentheses)"
        )

    @patch(HTTP_OPEN)
    def test_an_error_status_says_what_the_search_said(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            400, "Bad Request", body=b'{"error": "rows must be a number"}'
        )

        failure = _failure(internet_archive_search, "apple")

        assert failure.error.type == "upstream"
        assert not failure.error.retryable
        assert str(failure).startswith(
            "HTTP error 400: the Internet Archive's search could not run the query: rows must be "
            "a number"
        )

    @patch(HTTP_OPEN)
    def test_a_server_error_is_retryable_and_says_only_what_the_search_said(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(503, "Unavailable", body=b'{"error": "overloaded"}')

        failure = _failure(internet_archive_search, "apple")

        assert (failure.error.type, failure.error.retryable) == ("upstream", True)
        assert str(failure) == "HTTP error 503: overloaded"

    @patch(HTTP_OPEN)
    def test_a_404_on_the_search_is_an_endpoint_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(internet_archive_search, "apple")

        assert failure.error.type == "upstream"
        assert str(failure) == (
            "Internet Archive: endpoint not found (HTTP 404); the API may have changed"
        )

    @patch(HTTP_OPEN)
    def test_no_items_is_an_answer_that_names_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond(_search_answer([], total=0))

        assert _text(internet_archive_search("zzqqxx", mediatype="texts")) == (
            "No Internet Archive items match 'zzqqxx' (mediatype: texts)."
        )

    @patch(HTTP_OPEN)
    def test_returns_items_numbered_with_the_total_and_the_next_page(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            _search_answer([_DOC, *_docs(2, 1)], total=30, start=4)
        )

        result = internet_archive_search(
            "goody two shoes", max_results=2, page=3, mediatype="texts", collection="americana"
        )

        lines = _text(result).splitlines()
        assert lines[:4] == [
            "Internet Archive items that match 'goody two shoes' (mediatype: texts, collection: "
            "americana):",
            "5. Goody Two Shoes",
            "   identifier: goodytwoshoes00newyiala | mediatype: texts | date: 1888-01-01 | "
            "downloads: 10 | size: 123 bytes",
            "   creator: Newbery | collections: americana | subjects: Children",
        ]
        assert lines[-1] == "[results 5-6 of 30 | next: page=4]"
        assert _window(result)["next_call"] == {"page": 4}

        params = _called_params(mock_urlopen)
        assert urlparse(mock_urlopen.call_args.args[0].full_url).path == "/advancedsearch.php"
        assert params["q"] == ["(goody two shoes) AND mediatype:texts AND collection:americana"]
        assert params["rows"] == ["2"]
        assert params["page"] == ["3"]
        assert set(params["fl[]"]) >= {"identifier", "title", "item_size"}

    @patch(HTTP_OPEN)
    def test_long_lists_in_a_result_say_how_many_more_the_item_has(self, mock_urlopen):
        doc = {**_DOC, "subject": [f"s{n}" for n in range(12)], "creator": ["A", "B", "C", "D"]}
        mock_urlopen.return_value = respond(_search_answer([doc], total=1))

        text = _text(internet_archive_search("goody"))

        assert "creator: A, B, C (+1 more)" in text
        assert "subjects: s0, s1, s2, s3, s4 (+7 more)" in text

    @patch(HTTP_OPEN)
    def test_the_search_stops_at_the_depth_the_archive_pages_and_says_so(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            _search_answer(_docs(9981, 20), total=50_000, start=9980)
        )

        result = internet_archive_search("book", max_results=20, page=500)

        text = _text(result)
        assert text.splitlines()[0] == (
            "Internet Archive items that match 'book' (the search pages only its first 10000 "
            "results: narrow the query for the rest):"
        )
        assert text.endswith("[results 9981-10000 of 50000 | the rest cannot be read here]")

    @patch(HTTP_OPEN)
    def test_a_page_past_that_depth_is_refused_before_asking(self, mock_urlopen):
        failure = _failure(internet_archive_search, "book", max_results=20, page=501)

        assert failure.error.type == "validation_error"
        assert "past the first 10000" in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_empty_query_is_refused_before_asking(self, mock_urlopen):
        failure = _failure(internet_archive_search, "  ")

        assert failure.error.type == "validation_error"
        assert "query cannot be empty" in failure.error.message
        mock_urlopen.assert_not_called()

    @pytest.mark.parametrize(
        "arguments", [{"page": 0}, {"page": 10_001}, {"max_results": 0}, {"max_results": 21}]
    )
    @patch(HTTP_OPEN)
    def test_limits_are_refused_through_the_executor(self, mock_urlopen, arguments):
        result = _execute(
            "internet_archive_search", internet_archive_search, query="test", **arguments
        )

        assert result.error is not None
        assert result.error.type == "validation_error"
        mock_urlopen.assert_not_called()


_ITEM = {
    "created": 1727000000,
    "files_count": 2,
    "item_size": 12468,
    "metadata": {
        "identifier": "goodytwoshoes00newyiala",
        "title": "Goody Two Shoes",
        "creator": "Newbery",
        "date": "1888",
        "mediatype": "texts",
        "collection": ["americana"],
        "subject": ["Children"],
        "description": {"value": "A public domain book."},
    },
    "files": [
        {"name": "goody.pdf", "format": "PDF", "size": "12345"},
        {"name": "goody.txt", "format": "Text"},
    ],
}


def _big_item(files: int, description: str = "A public domain book.") -> dict[str, Any]:
    return {
        **_ITEM,
        "files_count": files,
        "metadata": {**_ITEM["metadata"], "description": description},
        "files": [
            {"name": f"page{n:04d}.jpg", "format": "JPEG", "size": str(1000 + n)}
            for n in range(files)
        ],
    }


class TestInternetArchiveItem:
    @patch(HTTP_OPEN)
    def test_returns_the_whole_item(self, mock_urlopen):
        mock_urlopen.return_value = respond(_ITEM)

        result = internet_archive_item("goodytwoshoes00newyiala")

        assert _text(result) == (
            "Internet Archive item goodytwoshoes00newyiala:\n"
            "Title: Goody Two Shoes\n"
            "Mediatype: texts | Date: 1888 | Size: 12468 bytes | Files: 2\n"
            "Page: https://archive.org/details/goodytwoshoes00newyiala\n"
            "Creator: Newbery\n"
            "Collections: americana\n"
            "Subjects: Children\n"
            "Description: A public domain book.\n"
            "Files (download one at "
            "https://archive.org/download/goodytwoshoes00newyiala/<name>):\n"
            "- goody.pdf (PDF), 12345 bytes\n"
            "- goody.txt (Text)"
        )
        request = mock_urlopen.call_args.args[0]
        assert urlparse(request.full_url).path == "/metadata/goodytwoshoes00newyiala"
        assert _called_params(mock_urlopen)["extended_err"] == ["1"]

    @patch(HTTP_OPEN)
    def test_no_list_or_description_is_cut_short(self, mock_urlopen):
        # Files stopped at eight and descriptions at 1000 characters, without a word.
        mock_urlopen.return_value = respond(_big_item(30, "word " * 600))

        text = _text(internet_archive_item("goodytwoshoes00newyiala", max_chars=20_000))

        assert "- page0029.jpg (JPEG), 1029 bytes" in text
        assert "Description: " + "word " * 599 + "word\n" in text

    @patch(HTTP_OPEN)
    def test_a_long_item_reads_window_by_window(self, mock_urlopen):
        mock_urlopen.side_effect = lambda request, timeout: respond(_big_item(400))

        first = internet_archive_item("goodytwoshoes00newyiala", max_chars=1000)
        second = internet_archive_item(
            "goodytwoshoes00newyiala", max_chars=1000, **_window(first)["next_call"]
        )

        assert _text(first).endswith(
            f"[chars 0-{_window(first)['last']} of {_window(first)['total']} | "
            f"next: offset={_window(first)['last']}]"
        )
        assert _window(second)["first"] == _window(first)["last"]
        assert _text(second).splitlines()[1].startswith("- page")

    @patch(HTTP_OPEN)
    def test_find_returns_the_passages_around_a_term(self, mock_urlopen):
        item = _big_item(400)
        item["files"].insert(200, {"name": "goody.pdf", "format": "PDF", "size": "99"})
        mock_urlopen.return_value = respond(item)

        text = _text(internet_archive_item("goodytwoshoes00newyiala", find="pdf"))

        assert text.splitlines()[0] == (
            "Internet Archive item goodytwoshoes00newyiala, passages that mention 'pdf':"
        )
        assert "- goody.pdf (PDF), 99 bytes" in text
        assert text.endswith('[matches 1-2 of 2 for "pdf" | end]')

    @patch(HTTP_OPEN)
    def test_invalid_identifier(self, mock_urlopen):
        failure = _failure(internet_archive_item, "../bad")

        assert failure.error.type == "validation_error"
        assert "invalid identifier" in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://archive.org/metadata/missing",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

        failure = _failure(internet_archive_item, "missing")

        assert failure.error.type == "not_found"
        assert not failure.error.retryable
        assert failure.error.message == (
            "no Internet Archive item 'missing'; find its identifier with internet_archive_search"
        )

    @patch(HTTP_OPEN)
    def test_a_server_error_is_upstream_and_retryable(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://archive.org/metadata/x", code=503, msg="Unavailable", hdrs=None, fp=None
        )

        failure = _failure(internet_archive_item, "x")

        assert failure.error.type == "upstream"
        assert failure.error.retryable

    @patch(HTTP_OPEN)
    def test_an_error_the_metadata_api_reports_is_not_a_missing_item(self, mock_urlopen):
        mock_urlopen.return_value = respond({"error": "Item is temporarily unavailable."})

        failure = _failure(internet_archive_item, "goodytwoshoes00newyiala")

        assert failure.error.type == "upstream"
        assert str(failure) == "Item is temporarily unavailable."

    @pytest.mark.parametrize(
        ("errcode", "kind", "retryable", "words"),
        [
            (104, "not_found", False, "the Internet Archive deleted the item (Item was deleted)"),
            (102, "upstream", True, "the item cannot be read now (Item is unavailable"),
            (101, "upstream", True, "try again later"),
            (400, "upstream", False, "Inaccurate lookahead"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_an_extended_error_code_says_what_happened(
        self, mock_urlopen: MagicMock, errcode, kind, retryable, words
    ):
        said = {
            104: "Item was deleted",
            102: "Item is unavailable (data node(s) are offline or not responding)",
            101: "Item creation is pending",
            400: "Inaccurate lookahead",
        }[errcode]
        mock_urlopen.return_value = respond({"error": said, "errcode": errcode})

        failure = _failure(internet_archive_item, "goodytwoshoes00newyiala")

        assert (failure.error.type, failure.error.retryable) == (kind, retryable)
        assert words in failure.error.message

    @patch(HTTP_OPEN)
    def test_an_unknown_identifier_is_still_not_found(self, mock_urlopen):
        # The metadata API answers an identifier it does not know with an empty object.
        mock_urlopen.return_value = respond({})

        failure = _failure(internet_archive_item, "zzqqxx")

        assert failure.error.type == "not_found"
        assert "no Internet Archive item 'zzqqxx'" in failure.error.message

    @pytest.mark.parametrize("max_chars", [499, 20_001])
    @patch(HTTP_OPEN)
    def test_max_chars_is_refused_outside_its_limits_through_the_executor(
        self, mock_urlopen, max_chars
    ):
        result = _execute(
            "internet_archive_item", internet_archive_item, identifier="x", max_chars=max_chars
        )

        assert result.error is not None
        assert result.error.type == "validation_error"
        mock_urlopen.assert_not_called()
