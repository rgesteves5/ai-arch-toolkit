"""Tests for toolkit/tools/_ror.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._ror import ror_organization, ror_search
from tests.toolkit.http_fakes import HTTP_OPEN, respond

_ORG = {
    "id": "https://ror.org/01c27hj86",
    "names": [
        {"value": "ULisboa", "types": ["acronym"]},
        {"value": "University of Lisbon", "types": ["ror_display", "label"]},
    ],
    "locations": [
        {"geonames_details": {"name": "Lisbon", "country_name": "Portugal", "country_code": "PT"}}
    ],
    "types": ["education", "funder"],
    "status": "active",
    "domains": ["ulisboa.pt"],
    "links": [{"type": "website", "value": "https://www.ulisboa.pt"}],
    "relationships": [
        {"type": "child", "label": "Instituto Superior Técnico", "id": "https://ror.org/03db2by73"}
    ],
}


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestRor:
    @patch(HTTP_OPEN)
    def test_search_and_lookup(self, mock_urlopen):
        mock_urlopen.return_value = respond({"number_of_results": 1, "items": [_ORG]})

        result = ror_search("University of Lisbon", country="PT")

        assert "University of Lisbon | id: https://ror.org/01c27hj86" in result
        assert _params(mock_urlopen)["filter"] == ["country.country_code:pt"]

        mock_urlopen.return_value = respond(_ORG)
        detail = ror_organization("https://ror.org/01c27hj86")
        assert "relationships: child: Instituto Superior Técnico" in detail

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "invalid ror_id" in ror_organization("bad")
        assert "invalid country" in ror_search("x", country="PRT")
        mock_urlopen.assert_not_called()
