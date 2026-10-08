"""Tests for toolkit/tools/_ror.py (T06)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._ror import ror_organization, ror_search
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond
from tests.toolkit.literature_answers import ROR_REFUSED, ror_org, ror_orgs, ror_page


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult) and result.ok, result
    assert isinstance(result.value, str)
    return result.value


def _params(mock_urlopen: MagicMock) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _failure(fn: Any, *args: Any, **kwargs: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


class TestRorSearch:
    @patch(HTTP_OPEN)
    def test_every_row_of_rors_page_shows_and_the_next_page_follows(self, mock_urlopen):
        # ROR answers 20 organizations a page: the old max_results cut the page on our side,
        # so rows 6 to 20 of every page were never shown.
        mock_urlopen.return_value = respond(ror_page(*ror_orgs(20), total=4885))

        result = ror_search("Bath College", page=2)

        text = _text(result)
        lines = text.splitlines()
        assert lines[0] == "ROR organizations that match 'Bath College':"
        assert lines[1] == "21. Organization 0 | id: https://ror.org/000000000"
        assert "40. Organization 19 | id: https://ror.org/000000019" in text
        assert text.endswith("[results 21-40 of 4885 | next: page=3]")
        assert _params(mock_urlopen)["page"] == ["2"]

    @patch(HTTP_OPEN)
    def test_a_row_shows_every_location_the_types_and_the_website(self, mock_urlopen):
        mock_urlopen.return_value = respond(ror_page(ror_org(), total=1))

        text = _text(ror_search("University of Lisbon", country="PT"))

        assert text.splitlines()[1:5] == [
            "1. University of Lisbon | id: https://ror.org/01c27hj86",
            "   locations: Lisbon, Portugal (PT); Porto, Portugal (PT) | types: education, "
            "funder | status: active | established: 2013",
            "   domains: ulisboa.pt",
            "   website: https://www.ulisboa.pt",
        ]
        assert _params(mock_urlopen)["filter"] == ["country.country_code:pt"]

    @patch(HTTP_OPEN)
    def test_past_rors_10000_results_the_rest_cannot_be_read_here(self, mock_urlopen):
        # The API reaches pages 1 to 500 (ror-community/ror-api, rorapi/settings.py: MAX_PAGE).
        mock_urlopen.return_value = respond(ror_page(*ror_orgs(20), total=50_000))

        text = _text(ror_search("university", page=500))

        assert text.endswith("[results 9981-10000 of 50000 | the rest cannot be read here]")

    @patch(HTTP_OPEN)
    def test_zero_results_say_so_with_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond(ror_page(total=0))

        assert _text(ror_search("zzqq")) == "No ROR organizations match 'zzqq'."

    @pytest.mark.parametrize("query", ["AT&T", "Franklin & Marshall College", "C++ Institute!"])
    @patch(HTTP_OPEN)
    def test_names_with_any_character_are_searched(self, mock_urlopen, query):
        # ROR's own documentation searches "Franklin & Marshall College".
        mock_urlopen.return_value = respond(ror_page(total=0))

        ror_search(query)

        assert _params(mock_urlopen)["query"] == [query]

    @pytest.mark.parametrize(("page", "kept"), [(1, True), (500, True), (0, False), (501, False)])
    @patch(HTTP_OPEN)
    def test_the_page_limits_are_the_schemas(self, mock_urlopen, page, kept):
        mock_urlopen.return_value = respond(ror_page(total=0))
        call = ToolCall(id="c1", name="ror_search", input={"query": "x", "page": page})

        assert ToolGroup(ror_search).execute(call).ok is kept

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        for args, kwargs, words in (
            (("x",), {"country": "PRT"}, "invalid country"),
            (("x",), {"org_type": "no spaces"}, "invalid org_type"),
            (("",), {}, "invalid query"),
            (("a\x00b",), {}, "invalid query"),
        ):
            failure = _failure(ror_search, *args, **kwargs)
            assert failure.error.type == "validation_error"
            assert words in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_refused_filter_says_rors_reason(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            400, "Bad Request", body=json.dumps(ROR_REFUSED).encode()
        )

        failure = _failure(ror_search, "x", org_type="colour")

        assert failure.error.type == "validation_error"
        assert failure.error.message == (
            "ROR refused the request: filter key 'colour' is illegal; correct that parameter"
        )


class TestRorOrganization:
    @patch(HTTP_OPEN)
    def test_the_record_is_whole(self, mock_urlopen):
        mock_urlopen.return_value = respond(ror_org())

        text = _text(ror_organization("https://ror.org/01c27hj86"))

        lines = text.splitlines()
        assert lines[:9] == [
            "ROR organization 01c27hj86:",
            "University of Lisbon",
            "id: https://ror.org/01c27hj86 | status: active | types: education, funder | "
            "established: 2013",
            "Locations (2): Lisbon, Lisbon, Portugal (PT), 38.71667, -9.13333 "
            "(GeoNames 2267057); Porto, Portugal (PT), 41.14961, -8.61099 (GeoNames 2735943)",
            "Names: ULisboa (acronym); University of Lisbon (ror_display, label; en); "
            "Universidade de Lisboa (label; pt)",
            "Domains: ulisboa.pt",
            "Links: website https://www.ulisboa.pt; wikipedia "
            "https://en.wikipedia.org/wiki/University_of_Lisbon",
            "External IDs: grid grid.9983.4 (preferred); isni 0000 0001 2181 4263",
            "Relationships (12):",
        ]
        assert "- child: Child 12 (https://ror.org/0child0012)" in text
        assert text.endswith("Record: created 2018-11-14, last modified 2024-12-11\n")

    @patch(HTTP_OPEN)
    def test_a_long_record_reads_on_through_the_window(self, mock_urlopen):
        mock_urlopen.side_effect = [respond(ror_org()) for _ in range(2)]

        first = ror_organization("01c27hj86", max_chars=500)
        last = first.metadata["window"]["last"]
        second = ror_organization("01c27hj86", max_chars=500, offset=last)

        assert _text(first).endswith(f"next: offset={last}]")
        assert second.metadata["window"]["first"] == last

    @patch(HTTP_OPEN)
    def test_invalid_id_does_not_call_api(self, mock_urlopen):
        failure = _failure(ror_organization, "bad")

        assert failure.error.type == "validation_error"
        assert "invalid ror_id" in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_unknown_id_is_not_found(self, mock_urlopen):
        body = b'{"errors": ["ROR ID \'https://ror.org/0abcdefgh\' does not exist"]}'
        mock_urlopen.side_effect = http_error(404, "Not Found", body=body)

        failure = _failure(ror_organization, "0abcdefgh")

        assert failure.error.type == "not_found"
        assert "ror_search" in failure.error.message


@pytest.mark.parametrize(
    "given", ["http://ror.org/01c27hj86", "ror.org/01c27hj86", "https://ror.org/01C27HJ86"]
)
@patch(HTTP_OPEN)
def test_a_ror_id_is_taken_with_any_url_form(mock_urlopen, given):
    mock_urlopen.return_value = respond(ror_org())

    ror_organization(given)

    assert urlparse(mock_urlopen.call_args.args[0].full_url).path.endswith("/01c27hj86")
