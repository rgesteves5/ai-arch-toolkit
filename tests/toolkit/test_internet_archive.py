"""Tests for toolkit/tools/_internet_archive.py."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._internet_archive import (
    internet_archive_item,
    internet_archive_search,
)


def _mock_urlopen(data):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestInternetArchiveSearch:
    @patch("ai_arch_toolkit.toolkit.tools._internet_archive.urllib.request.urlopen")
    def test_returns_items(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
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

    @patch("ai_arch_toolkit.toolkit.tools._internet_archive.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "query cannot be empty" in internet_archive_search("")
        assert "page must" in internet_archive_search("test", page=0)
        mock_urlopen.assert_not_called()


class TestInternetArchiveItem:
    @patch("ai_arch_toolkit.toolkit.tools._internet_archive.urllib.request.urlopen")
    def test_returns_item(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
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

    @patch("ai_arch_toolkit.toolkit.tools._internet_archive.urllib.request.urlopen")
    def test_invalid_identifier(self, mock_urlopen):
        result = internet_archive_item("../bad")

        assert "invalid identifier" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._internet_archive.urllib.request.urlopen")
    def test_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://archive.org/metadata/missing",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

        result = internet_archive_item("missing")

        assert "not found" in result
