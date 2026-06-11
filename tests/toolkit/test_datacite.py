"""Tests for toolkit/tools/_datacite.py."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._datacite import datacite_doi, datacite_search

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


def _mock_urlopen(data):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestDataCiteSearch:
    @patch("ai_arch_toolkit.toolkit.tools._datacite.urllib.request.urlopen")
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({"data": [_DOI_RECORD]})

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

    @patch("ai_arch_toolkit.toolkit.tools._datacite.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "query cannot be empty" in datacite_search("")
        assert "page must" in datacite_search("test", page=0)
        mock_urlopen.assert_not_called()


class TestDataCiteDoi:
    @patch("ai_arch_toolkit.toolkit.tools._datacite.urllib.request.urlopen")
    def test_returns_doi(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({"data": _DOI_RECORD})

        result = datacite_doi("https://doi.org/10.5061/dryad.test")

        assert result.startswith("DataCite DOI 10.5061/dryad.test:")
        assert "Description: Dataset description." in result
        assert "Rights: CC0" in result
        assert "Related: IsSupplementTo: 10.5555/article" in result
        assert urlparse(_called_request(mock_urlopen).full_url).path.endswith(
            "/10.5061%2Fdryad.test"
        )

    @patch("ai_arch_toolkit.toolkit.tools._datacite.urllib.request.urlopen")
    def test_invalid_doi(self, mock_urlopen):
        result = datacite_doi("not a doi")

        assert "invalid DOI" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._datacite.urllib.request.urlopen")
    def test_rate_limited(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.datacite.org/dois",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

        result = datacite_search("test")

        assert "rate limited" in result
