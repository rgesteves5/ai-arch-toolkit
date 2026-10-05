"""Tests for toolkit/tools/_ror.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._ror import ror_organization, ror_search
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

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
        for call, args, kwargs, words in (
            (ror_organization, ("bad",), {}, "invalid ror_id"),
            (ror_search, ("x",), {"country": "PRT"}, "invalid country"),
            (ror_search, ("x",), {"org_type": "no spaces"}, "invalid org_type"),
            (ror_search, ("x",), {"page": 0}, "page must be"),
            (ror_search, ("",), {}, "invalid query"),
        ):
            with pytest.raises(ToolFailure) as caught:
                call(*args, **kwargs)
            assert caught.value.error.type == "validation_error"
            assert words in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_no_organization_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond({"number_of_results": 0, "items": []})

        assert ror_search("zzqqxx") == "No ROR organizations found."

    @patch(HTTP_OPEN)
    def test_a_rate_limit_raises_rate_limited(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        with pytest.raises(ToolFailure) as caught:
            ror_search("Lisbon")

        assert caught.value.error.type == "rate_limited"
        assert caught.value.error.retryable
