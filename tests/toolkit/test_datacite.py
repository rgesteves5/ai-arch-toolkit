"""Tests for toolkit/tools/_datacite.py."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._datacite import datacite_doi, datacite_search
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_DOI_RECORD = {
    "id": "10.5061/dryad.test",
    "attributes": {
        "doi": "10.5061/dryad.test",
        "titles": [{"title": "Example dataset"}],
        "creators": [{"name": "Jane Smith"}],
        "publisher": "Dryad",
        "publicationYear": "2024",
        "types": {"resourceTypeGeneral": "Dataset", "resourceType": "Dataset"},
        "descriptions": [{"description": "Dataset description."}],
        "subjects": [{"subject": "Machine learning"}],
        "url": "https://datadryad.org/example",
        "rightsList": [{"rights": "CC0"}],
        "relatedIdentifiers": [
            {"relationType": "IsSupplementTo", "relatedIdentifier": "10.5555/article"}
        ],
    },
}


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestDataCiteSearch:
    @patch(HTTP_OPEN)
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = respond({"data": [_DOI_RECORD]})

        result = datacite_search("example", resource_type="Dataset", max_results=2, page=3)

        assert "DataCite DOI results for 'example'" in result
        assert "Example dataset" in result
        assert "DOI: 10.5061/dryad.test" in result
        assert "type: Dataset" in result
        assert "year: 2024" in result
        assert "Jane Smith" in result
        assert "Machine learning" in result

        params = _called_params(mock_urlopen)
        assert params["query"] == ["example"]
        assert params["page[size]"] == ["2"]
        assert params["page[number]"] == ["3"]
        assert params["resource-type-id"] == ["dataset"]

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"query": ""}, "query cannot be empty"),
            ({"query": "test", "page": 0}, "page must be greater than or equal to 1"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen, kwargs, words):
        with pytest.raises(ToolFailure) as caught:
            datacite_search(**kwargs)

        assert caught.value.error.type == "validation_error"
        assert words in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_404_is_endpoint_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        with pytest.raises(ToolFailure) as caught:
            datacite_search("test")

        assert caught.value.error.type == "upstream"
        assert "DataCite: endpoint not found (HTTP 404)" in caught.value.error.message


class TestDataCiteDoi:
    @patch(HTTP_OPEN)
    def test_returns_doi(self, mock_urlopen):
        mock_urlopen.return_value = respond({"data": _DOI_RECORD})

        result = datacite_doi("https://doi.org/10.5061/dryad.test")

        assert result.startswith("DataCite DOI 10.5061/dryad.test:")
        assert "Description: Dataset description." in result
        assert "Rights: CC0" in result
        assert "Related: IsSupplementTo: 10.5555/article" in result
        assert urlparse(_called_request(mock_urlopen).full_url).path.endswith(
            "/10.5061%2Fdryad.test"
        )

    @patch(HTTP_OPEN)
    def test_invalid_doi(self, mock_urlopen):
        with pytest.raises(ToolFailure) as caught:
            datacite_doi("not a doi")

        assert caught.value.error.type == "validation_error"
        assert "invalid DOI" in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_rate_limited(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.datacite.org/dois",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

        with pytest.raises(ToolFailure) as caught:
            datacite_search("test")

        assert caught.value.error.type == "rate_limited"
        assert caught.value.error.retryable
        assert "rate limited" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        with pytest.raises(ToolFailure) as caught:
            datacite_doi("10.5061/missing")

        assert caught.value.error.type == "not_found"
        assert "no DataCite record of DOI 10.5061/missing" in caught.value.error.message
        assert "datacite_search" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_other_statuses_propagate(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(500, "Internal Server Error")

        with pytest.raises(ToolFailure) as caught:
            datacite_doi("10.5061/dryad.test")

        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable
