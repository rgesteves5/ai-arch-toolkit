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


def _failure(fn, *args, **kwargs) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


@patch(HTTP_OPEN)
def test_an_unknown_organization_is_not_found(mock_urlopen):
    body = b'{"errors": ["ROR ID \'https://ror.org/000000000\' does not exist"]}'
    mock_urlopen.side_effect = http_error(404, "Not Found", body=body)

    failure = _failure(ror_organization, "000000000")

    assert failure.error.type == "not_found"
    assert failure.error.message == (
        "ROR has no organization 000000000; find organizations with ror_search"
    )


@patch(HTTP_OPEN)
def test_a_404_on_the_search_is_an_endpoint_not_found(mock_urlopen):
    mock_urlopen.side_effect = http_error(404, "Not Found")

    failure = _failure(ror_search, "Lisbon")

    assert failure.error.type == "upstream"
    assert "ROR: endpoint not found (HTTP 404)" in failure.error.message


@patch(HTTP_OPEN)
def test_a_parameter_ror_refuses_is_a_validation_error(mock_urlopen):
    body = b'{"errors": ["Filter types:zzz is not a valid filter"]}'
    mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)

    failure = _failure(ror_search, "Lisbon", org_type="zzz")

    assert failure.error.type == "validation_error"
    assert not failure.error.retryable
    assert failure.error.message == (
        "ROR refused the request: Filter types:zzz is not a valid filter; correct that parameter"
    )


@patch(HTTP_OPEN)
def test_ror_s_errors_on_a_server_failure_keep_its_words(mock_urlopen):
    body = b'{"errors": ["Search backend unavailable"]}'
    mock_urlopen.side_effect = http_error(503, "Service Unavailable", body=body)

    failure = _failure(ror_search, "Lisbon")

    assert failure.error.type == "upstream"
    assert failure.error.retryable
    assert failure.error.message == "HTTP error 503: Search backend unavailable"


@patch(HTTP_OPEN)
def test_an_answer_without_errors_is_read_as_a_result(mock_urlopen):
    mock_urlopen.return_value = respond({"number_of_results": 0, "items": [], "errors": []})

    assert ror_search("zzqqxx") == "No ROR organizations found."
