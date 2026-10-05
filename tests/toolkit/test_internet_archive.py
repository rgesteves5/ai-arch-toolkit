"""Tests for toolkit/tools/_internet_archive.py."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._internet_archive import (
    internet_archive_item,
    internet_archive_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, respond


def _failure(fn, *args, **kwargs) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestInternetArchiveSearch:
    @patch(HTTP_OPEN)
    def test_a_query_the_search_cannot_run_is_the_tools_error(self, mock_urlopen):
        # Sent with HTTP 200 (archive.org, 2026-09-29); it used to read as no items.
        mock_urlopen.return_value = respond(
            {"error": 'a group is empty (near char ")" at position 10)'}
        )

        failure = _failure(internet_archive_search, "title:(")

        assert failure.error.type == "upstream"
        assert str(failure) == 'a group is empty (near char ")" at position 10)'

    @patch(HTTP_OPEN)
    def test_no_items_is_an_answer(self, mock_urlopen):
        mock_urlopen.return_value = respond({"response": {"docs": []}})

        assert internet_archive_search("zzqqxx") == "No Internet Archive items found for: 'zzqqxx'"

    @patch(HTTP_OPEN)
    def test_returns_items(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "response": {
                    "docs": [
                        {
                            "identifier": "goodytwoshoes00newyiala",
                            "title": "Goody Two Shoes",
                            "creator": ["Newbery"],
                            "date": "1888",
                            "mediatype": "texts",
                            "collection": ["americana"],
                            "subject": ["Children"],
                            "downloads": "10",
                            "item_size": "123",
                        }
                    ]
                }
            }
        )

        result = internet_archive_search(
            "goody two shoes",
            max_results=2,
            page=3,
            mediatype="texts",
            collection="americana",
        )

        assert "Internet Archive items for 'goody two shoes'" in result
        assert "Goody Two Shoes" in result
        assert "identifier: goodytwoshoes00newyiala" in result
        assert "downloads: 10" in result
        assert "size: 123" in result
        assert "Newbery" in result

        params = _called_params(mock_urlopen)
        assert params["q"] == ["(goody two shoes) AND mediatype:texts AND collection:americana"]
        assert params["rows"] == ["2"]
        assert params["page"] == ["3"]
        assert set(params["fl[]"]) >= {"identifier", "title", "item_size"}

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        empty = _failure(internet_archive_search, "")
        bad_page = _failure(internet_archive_search, "test", page=0)

        assert empty.error.type == "validation_error"
        assert "query cannot be empty" in empty.error.message
        assert bad_page.error.type == "validation_error"
        assert "page must" in bad_page.error.message
        mock_urlopen.assert_not_called()


class TestInternetArchiveItem:
    @patch(HTTP_OPEN)
    def test_returns_item(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
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
        )

        result = internet_archive_item("goodytwoshoes00newyiala")

        assert result.startswith("Internet Archive item goodytwoshoes00newyiala:")
        assert "Description: A public domain book." in result
        assert "goody.pdf (PDF), 12345 bytes" in result
        assert "goody.txt (Text)" in result

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

    @patch(HTTP_OPEN)
    def test_an_unknown_identifier_is_still_not_found(self, mock_urlopen):
        # The metadata API answers an identifier it does not know with an empty object.
        mock_urlopen.return_value = respond({})

        failure = _failure(internet_archive_item, "zzqqxx")

        assert failure.error.type == "not_found"
        assert "no Internet Archive item 'zzqqxx'" in failure.error.message
