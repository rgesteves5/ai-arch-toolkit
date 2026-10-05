"""Tests for toolkit/tools/_crossref.py."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._crossref import crossref_search, crossref_work
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_WORK = {
    "DOI": "10.5555/example",
    "title": ["Attention Is All You Need"],
    "subtitle": ["Transformer paper"],
    "author": [
        {"given": "Ashish", "family": "Vaswani"},
        {"name": "Noam Shazeer"},
    ],
    "issued": {"date-parts": [[2017, 6, 12]]},
    "container-title": ["Advances in Neural Information Processing Systems"],
    "publisher": "NeurIPS",
    "type": "proceedings-article",
    "URL": "https://doi.org/10.5555/example",
    "abstract": "<jats:p>The dominant <i>sequence</i> transduction model.</jats:p>",
    "is-referenced-by-count": 1234,
    "license": [{"URL": "https://license.example"}],
    "link": [{"URL": "https://content.example/full.pdf"}],
    "reference": [
        {
            "author": "Smith",
            "article-title": "Related Work",
            "journal-title": "Journal of Tests",
            "year": "2016",
            "DOI": "10.5555/ref",
        }
    ],
}


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestCrossrefSearch:
    @patch(HTTP_OPEN)
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = respond({"message": {"items": [_WORK]}})

        result = crossref_search("transformers", max_results=2)

        assert "Crossref results for 'transformers'" in result
        assert "Attention Is All You Need: Transformer paper" in result
        assert "Ashish Vaswani, Noam Shazeer" in result
        assert "DOI: 10.5555/example | type: proceedings-article" in result
        assert "published: 2017-06-12" in result
        assert "Venue: Advances in Neural Information Processing Systems" in result
        assert "Publisher: NeurIPS" in result
        assert "Referenced by: 1234" in result
        assert "https://doi.org/10.5555/example" in result
        assert "https://license.example" in result
        assert "https://content.example/full.pdf" in result
        assert "Abstract:" not in result

        params = _called_params(mock_urlopen)
        assert params["query"] == ["transformers"]
        assert params["rows"] == ["2"]
        assert params["offset"] == ["0"]

    @patch(HTTP_OPEN)
    def test_filters_dates_type_start_and_caps_max_results(self, mock_urlopen):
        mock_urlopen.return_value = respond({"message": {"items": []}})

        crossref_search(
            "language agents",
            max_results=99,
            start=40,
            from_date="2024-01-01",
            to_date="2024-12-31",
            type_filter="journal-article",
        )

        params = _called_params(mock_urlopen)
        assert params["query"] == ["language agents"]
        assert params["rows"] == ["20"]
        assert params["offset"] == ["40"]
        assert params["filter"] == [
            "from-pub-date:2024-01-01,until-pub-date:2024-12-31,type:journal-article"
        ]

    @patch(HTTP_OPEN)
    def test_no_results(self, mock_urlopen):
        mock_urlopen.return_value = respond({"message": {"items": []}})

        result = crossref_search("no such paper")

        assert "No Crossref results" in result

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"query": ""}, "query cannot be empty"),
            ({"query": "test", "start": -1}, "start must be greater than or equal to 0"),
            ({"query": "test", "from_date": "01-01-2024"}, "invalid from_date"),
            ({"query": "test", "to_date": "2024-13-01"}, "invalid to_date"),
            (
                {"query": "test", "from_date": "2024-02-01", "to_date": "2024-01-01"},
                "must be before or equal to to_date",
            ),
            ({"query": "test", "type_filter": "journal article"}, "invalid type_filter"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen, kwargs, words):
        with pytest.raises(ToolFailure) as caught:
            crossref_search(**kwargs)

        assert caught.value.error.type == "validation_error"
        assert words in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()

        with pytest.raises(ToolFailure) as caught:
            crossref_search("test")

        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable
        assert "timed out" in caught.value.error.message.lower()

    @patch(HTTP_OPEN)
    def test_404_is_endpoint_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found", body=b"Resource not found.")

        with pytest.raises(ToolFailure) as caught:
            crossref_search("test")

        assert caught.value.error.type == "upstream"
        assert caught.value.error.message == (
            "Crossref: endpoint not found (HTTP 404); the API may have changed: "
            "Resource not found."
        )

    @patch(HTTP_OPEN)
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond(b"not json")

        with pytest.raises(ToolFailure) as caught:
            crossref_search("test")

        assert caught.value.error.type == "upstream"
        assert "could not parse" in caught.value.error.message


class TestCrossrefWork:
    @patch(HTTP_OPEN)
    def test_returns_work_by_doi_url(self, mock_urlopen):
        mock_urlopen.return_value = respond({"message": _WORK})

        result = crossref_work("https://doi.org/10.5555/example")

        assert result.startswith("Crossref work 10.5555/example:")
        assert "Attention Is All You Need: Transformer paper" in result
        assert "Abstract: The dominant sequence transduction model." in result
        assert "References (1 deposited):" in result
        assert "Smith; Related Work; Journal of Tests; 2016; 10.5555/ref" in result
        assert "1. Attention" not in result

        request = _called_request(mock_urlopen)
        assert urlparse(request.full_url).path.endswith("/10.5555%2Fexample")

    @patch(HTTP_OPEN)
    def test_accepts_doi_prefix(self, mock_urlopen):
        mock_urlopen.return_value = respond({"message": _WORK})

        result = crossref_work("doi:10.5555/example")

        assert result.startswith("Crossref work 10.5555/example:")

    @patch(HTTP_OPEN)
    def test_invalid_doi(self, mock_urlopen):
        with pytest.raises(ToolFailure) as caught:
            crossref_work("bad doi")

        assert caught.value.error.type == "validation_error"
        assert "invalid DOI 'bad doi'" in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.crossref.org/works/10.5555%2Fmissing",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

        with pytest.raises(ToolFailure) as caught:
            crossref_work("10.5555/missing")

        assert caught.value.error.type == "not_found"
        assert caught.value.error.message == (
            "no Crossref work with DOI 10.5555/missing; search with crossref_search."
        )

    @patch(HTTP_OPEN)
    def test_other_statuses_propagate(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        with pytest.raises(ToolFailure) as caught:
            crossref_work("10.5555/example")

        assert caught.value.error.type == "rate_limited"
