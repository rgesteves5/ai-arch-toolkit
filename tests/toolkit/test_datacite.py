"""Tests for toolkit/tools/_datacite.py (T06)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._datacite import datacite_doi, datacite_search
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond
from tests.toolkit.literature_answers import (
    DATACITE_RATE_LIMITED,
    DATACITE_REFUSED,
    datacite_item,
    datacite_list,
    datacite_record,
)


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


class TestDataCiteSearch:
    @patch(HTTP_OPEN)
    def test_a_page_says_the_total_and_the_next_page(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            datacite_list(
                datacite_item("10.5061/a"), datacite_item("10.5061/b"), total=36409, page=3
            )
        )

        result = datacite_search("example", resource_type="Dataset", max_results=2, page=3)

        text = _text(result)
        assert text.splitlines()[:5] == [
            "DataCite DOIs that match 'example':",
            "5. Example dataset",
            "   DOI: 10.5061/a | type: Dataset (Survey data) | year: 2024",
            "   Creators: Smith, Jane, Data Team",
            "   Publisher: Dryad",
        ]
        assert text.endswith("[results 5-6 of 36409 | next: page=4, max_results=2]")
        params = _params(mock_urlopen)
        assert params["query"] == ["example"]
        assert params["page[size]"] == ["2"]
        assert params["page[number]"] == ["3"]
        assert params["resource-type-id"] == ["dataset"]

    @pytest.mark.parametrize(
        ("given", "sent"),
        [
            ("JournalArticle", "journal-article"),
            ("OutputManagementPlan", "output-management-plan"),
            ("journal-article", "journal-article"),
            ("Journal Article", "journal-article"),
            ("Dataset", "dataset"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_a_resource_type_goes_as_the_filter_takes_it(self, mock_urlopen, given, sent):
        # resource-type-id is resourceTypeGeneral in kebab case (https://support.datacite.org/
        # docs/api-queries: "output-management-plan"); lower case alone broke the many-word ones.
        mock_urlopen.return_value = respond(datacite_list(total=0))

        datacite_search("example", resource_type=given)

        assert _params(mock_urlopen)["resource-type-id"] == [sent]

    @patch(HTTP_OPEN)
    def test_long_creator_lists_say_how_many_more_and_where(self, mock_urlopen):
        mock_urlopen.return_value = respond(datacite_list(datacite_item(creators=10), total=1))

        text = _text(datacite_search("example"))

        assert "(+2 more; datacite_doi('10.5061/dryad.test') lists all)" in text
        assert "Subject" not in text and "Related" not in text  # the record lists them all

    @patch(HTTP_OPEN)
    def test_the_last_page_says_end(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            datacite_list(datacite_item(), total=5, page=3, total_pages=3)
        )

        text = _text(datacite_search("example", max_results=2, page=3))

        assert text.endswith("[results 5-5 of 5 | end]")

    @patch(HTTP_OPEN)
    def test_past_the_10000_records_a_page_reaches_the_rest_cannot_be_read_here(
        self, mock_urlopen
    ):
        # Only the first 10,000 records can be paged by number (https://support.datacite.org/
        # docs/pagination).
        mock_urlopen.return_value = respond(
            datacite_list(datacite_item(), datacite_item(), total=36409, page=5000)
        )

        text = _text(datacite_search("example", max_results=2, page=5000))

        assert text.endswith("[results 9999-10000 of 36409 | the rest cannot be read here]")

    @patch(HTTP_OPEN)
    def test_a_page_past_the_10000_records_is_refused_before_the_request(self, mock_urlopen):
        failure = _failure(datacite_search, "example", max_results=20, page=501)

        assert failure.error.type == "validation_error"
        assert "first 10000 records" in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_zero_results_say_so_with_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond(datacite_list(total=0))

        assert _text(datacite_search("zzqq")) == "No DataCite DOIs match 'zzqq'."

    @pytest.mark.parametrize(
        ("max_results", "page", "kept"),
        [(20, 1, True), (21, 1, False), (0, 1, False), (5, 0, False)],
    )
    @patch(HTTP_OPEN)
    def test_the_limits_are_the_schemas(self, mock_urlopen, max_results, page, kept):
        mock_urlopen.return_value = respond(datacite_list(total=0))
        call = ToolCall(
            id="c1",
            name="datacite_search",
            input={"query": "x", "max_results": max_results, "page": page},
        )

        result = ToolGroup(datacite_search).execute(call)

        assert result.ok is kept

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"query": ""}, "query cannot be empty"),
            ({"query": "test", "resource_type": "data;set"}, "invalid resource_type"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen, kwargs, words):
        failure = _failure(datacite_search, **kwargs)

        assert failure.error.type == "validation_error"
        assert words in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_refused_request_says_datacites_reason(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            400, "Bad Request", body=json.dumps(DATACITE_REFUSED).encode()
        )

        failure = _failure(datacite_search, "title:(")

        assert failure.error.type == "validation_error"
        assert failure.error.message.startswith("DataCite refused the request: Bad Request;")

    @patch(HTTP_OPEN)
    def test_rate_limited_in_datacites_words(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            429, "Too Many Requests", body=json.dumps(DATACITE_RATE_LIMITED).encode()
        )

        failure = _failure(datacite_search, "test")

        assert failure.error.type == "rate_limited"
        assert failure.error.retryable
        assert "Your request has been rate limited" in failure.error.message

    @patch(HTTP_OPEN)
    def test_404_is_endpoint_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(datacite_search, "test")

        assert failure.error.type == "upstream"
        assert "DataCite: endpoint not found (HTTP 404)" in failure.error.message


class TestDataCiteDoi:
    @patch(HTTP_OPEN)
    def test_the_record_is_whole(self, mock_urlopen):
        mock_urlopen.return_value = respond(datacite_record(datacite_item(creators=10)))

        text = _text(datacite_doi("https://doi.org/10.5061/dryad.test"))

        lines = text.splitlines()
        assert lines[:7] == [
            "DataCite DOI 10.5061/dryad.test:",
            "Example dataset",
            "DOI: 10.5061/dryad.test | type: Dataset (Survey data) | year: 2024 | version: 2",
            "Also titled: Exemplo (TranslatedTitle)",
            "Publisher: Dryad",
            "URL: https://datadryad.org/example",
            "DataCite: https://commons.datacite.org/doi.org/10.5061/dryad.test",
        ]
        assert "Description (Abstract): Dataset description." in text
        assert "Description (Methods): How it was measured." in text
        assert "Creators (10): Smith, Jane (University of Tests), Data Team, " in text
        assert "Creator 10" in text
        assert "Subject 12" in text
        assert (
            "Rights: Creative Commons Zero v1.0 Universal "
            "(https://creativecommons.org/publicdomain/zero/1.0/legalcode)"
        ) in text
        assert "Related identifiers (7):" in text
        assert "- IsSupplementTo: 10.5555/article7 (DOI)" in text
        assert urlparse(mock_urlopen.call_args.args[0].full_url).path.endswith(
            "/10.5061%2Fdryad.test"
        )

    @patch(HTTP_OPEN)
    def test_a_long_description_reads_on_through_the_window(self, mock_urlopen):
        long = "A sentence of the abstract. " * 100
        mock_urlopen.side_effect = [
            respond(datacite_record(datacite_item(description=long))) for _ in range(2)
        ]

        first = datacite_doi("10.5061/dryad.test", max_chars=1000)
        last = first.metadata["window"]["last"]
        second = datacite_doi("10.5061/dryad.test", max_chars=1000, offset=last)

        assert _text(first).endswith(f"next: offset={last}]")
        assert second.metadata["window"]["first"] == last

    @patch(HTTP_OPEN)
    def test_invalid_doi(self, mock_urlopen):
        failure = _failure(datacite_doi, "not a doi")

        assert failure.error.type == "validation_error"
        assert "invalid DOI" in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(datacite_doi, "10.5061/missing")

        assert failure.error.type == "not_found"
        assert "no DataCite record of DOI 10.5061/missing" in failure.error.message
        assert "datacite_search" in failure.error.message

    @patch(HTTP_OPEN)
    def test_other_statuses_propagate(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(500, "Internal Server Error")

        failure = _failure(datacite_doi, "10.5061/dryad.test")

        assert failure.error.type == "upstream"
        assert failure.error.retryable


@patch(HTTP_OPEN)
def test_without_totalpages_the_total_decides_the_next_page(mock_urlopen):
    answer = datacite_list(datacite_item("10.1/a"), datacite_item("10.1/b"), total=9)
    del answer["meta"]["totalPages"]
    mock_urlopen.return_value = respond(answer)

    text = _text(datacite_search("example", max_results=2))

    assert text.endswith("[results 1-2 of 9 | next: page=2, max_results=2]")
