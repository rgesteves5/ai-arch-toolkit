"""Tests for toolkit/tools/_crossref.py (T06)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._crossref import crossref_search, crossref_work
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond
from tests.toolkit.literature_answers import (
    CROSSREF_REFUSED,
    crossref_item,
    crossref_list,
)
from tests.toolkit.literature_answers import crossref_work as work_answer


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


class TestCrossrefSearch:
    @patch(HTTP_OPEN)
    def test_a_page_says_the_total_and_the_next_start(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            crossref_list(crossref_item("10.5555/a"), crossref_item("10.5555/b"), total=4321)
        )

        result = crossref_search("transformers", max_results=2)

        text = _text(result)
        assert text.splitlines()[:6] == [
            "Crossref works that match 'transformers':",
            "1. Attention Is All You Need: Transformer paper",
            "   DOI: 10.5555/a | type: proceedings-article | published: 2017-06-12",
            "   Authors: Ashish Vaswani, Noam Shazeer",
            "   Venue: Advances in Neural Information Processing Systems | Publisher: NeurIPS",
            "   Cited by: 1234 works in Crossref",
        ]
        assert text.endswith("[results 1-2 of 4321 | next: start=2]")
        params = _params(mock_urlopen)
        assert params["query"] == ["transformers"]
        assert params["rows"] == ["2"]
        assert params["offset"] == ["0"]

    @patch(HTTP_OPEN)
    def test_a_long_author_list_says_how_many_more_and_where(self, mock_urlopen):
        mock_urlopen.return_value = respond(crossref_list(crossref_item(authors=11), total=1))

        text = _text(crossref_search("transformers"))

        assert "(+3 more; crossref_work('10.5555/example') lists all)" in text
        assert "License" not in text and "Links" not in text  # the record lists them all

    @patch(HTTP_OPEN)
    def test_past_crossrefs_offset_limit_the_rest_cannot_be_read_here(self, mock_urlopen):
        # Offsets for /works are limited to 10K (https://github.com/CrossRef/rest-api-doc).
        mock_urlopen.return_value = respond(
            crossref_list(crossref_item("10.5555/a"), crossref_item("10.5555/b"), total=50_000)
        )

        text = _text(crossref_search("transformers", max_results=2, start=10_000))

        assert text.endswith("[results 10001-10002 of 50000 | the rest cannot be read here]")

    @patch(HTTP_OPEN)
    def test_zero_results_say_so_with_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond(crossref_list(total=0))

        assert _text(crossref_search("no such paper")) == (
            "No Crossref works match 'no such paper'."
        )

    @patch(HTTP_OPEN)
    def test_filters_dates_type_and_start(self, mock_urlopen):
        mock_urlopen.return_value = respond(crossref_list(total=0))

        crossref_search(
            "language agents",
            max_results=20,
            start=40,
            from_date="2024-01-01",
            to_date="2024-01-31",
            type_filter="journal-article",
        )

        params = _params(mock_urlopen)
        assert params["rows"] == ["20"]
        assert params["offset"] == ["40"]
        assert params["filter"] == [
            "from-pub-date:2024-01-01,until-pub-date:2024-01-31,type:journal-article"
        ]

    @pytest.mark.parametrize(
        ("max_results", "start", "kept"),
        [(20, 10_000, True), (21, 0, False), (0, 0, False), (5, 10_001, False), (5, -1, False)],
    )
    @patch(HTTP_OPEN)
    def test_the_limits_are_the_schemas(self, mock_urlopen, max_results, start, kept):
        mock_urlopen.return_value = respond(crossref_list(total=0))
        call = ToolCall(
            id="c1",
            name="crossref_search",
            input={"query": "x", "max_results": max_results, "start": start},
        )

        result = ToolGroup(crossref_search).execute(call)

        assert result.ok is kept
        if not kept:
            assert result.error is not None and result.error.type == "validation_error"

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"query": ""}, "query cannot be empty"),
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
        failure = _failure(crossref_search, **kwargs)

        assert failure.error.type == "validation_error"
        assert words in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_refused_filter_says_crossrefs_reason(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            400, "Bad Request", body=json.dumps(CROSSREF_REFUSED).encode()
        )

        failure = _failure(crossref_search, "test", type_filter="journal")

        assert failure.error.type == "validation_error"
        assert failure.error.message == (
            "Crossref refused the request: Type specified as journal but must be one of: "
            "book-section, monograph; correct that parameter"
        )

    @patch(HTTP_OPEN)
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()

        failure = _failure(crossref_search, "test")

        assert failure.error.type == "upstream"
        assert failure.error.retryable

    @patch(HTTP_OPEN)
    def test_404_is_endpoint_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found", body=b"Resource not found.")

        failure = _failure(crossref_search, "test")

        assert failure.error.type == "upstream"
        assert failure.error.message == (
            "Crossref: endpoint not found (HTTP 404); the API may have changed: "
            "Resource not found."
        )

    @patch(HTTP_OPEN)
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond(b"not json")

        failure = _failure(crossref_search, "test")

        assert failure.error.type == "upstream"
        assert "could not parse" in failure.error.message


class TestCrossrefWork:
    @patch(HTTP_OPEN)
    def test_the_record_is_whole(self, mock_urlopen):
        mock_urlopen.return_value = respond(work_answer(crossref_item(authors=11, references=7)))

        text = _text(crossref_work("https://doi.org/10.5555/example"))

        lines = text.splitlines()
        assert lines[:8] == [
            "Crossref work 10.5555/example:",
            "Attention Is All You Need: Transformer paper",
            "DOI: 10.5555/example | type: proceedings-article | published: 2017-06-12",
            "Venue: Advances in Neural Information Processing Systems | Publisher: NeurIPS",
            "Cited by: 1234 works in Crossref | References: 7 deposited",
            "URL: https://doi.org/10.5555/example",
            "Abstract: The dominant sequence transduction model.",
            "Authors (11): Ashish Vaswani (Google Brain; ORCID 0000-0002-1825-0097), "
            "Noam Shazeer, Given Family 3, Given Family 4, Given Family 5, Given Family 6, "
            "Given Family 7, Given Family 8, Given Family 9, Given Family 10, Given Family 11",
        ]
        assert (
            "License: https://creativecommons.org/licenses/by/4.0/ (vor, from 2017-06-12)" in text
        )
        assert "Links: https://content.example/full.pdf (application/pdf, text-mining)" in text
        assert "References (7):" in text
        assert "- Smith; Related Work 7; Journal of Tests; 2016; DOI: 10.5555/ref7" in text
        assert "[chars" not in text
        assert urlparse(mock_urlopen.call_args.args[0].full_url).path.endswith(
            "/10.5555%2Fexample"
        )

    @patch(HTTP_OPEN)
    def test_a_long_record_reads_on_through_the_window(self, mock_urlopen):
        mock_urlopen.side_effect = [
            respond(work_answer(crossref_item(references=80))) for _ in range(2)
        ]

        first = crossref_work("10.5555/example", max_chars=1000)
        last = first.metadata["window"]["last"]
        second = crossref_work("10.5555/example", max_chars=1000, offset=last)

        assert _text(first).endswith(f"next: offset={last}]")
        assert "Abstract:" in _text(first)
        assert second.metadata["window"]["first"] == last
        assert "Abstract:" not in _text(second)

    @patch(HTTP_OPEN)
    def test_deposited_references_the_record_does_not_carry_are_said(self, mock_urlopen):
        item = crossref_item(references=0) | {"reference-count": 45}
        mock_urlopen.return_value = respond(work_answer(item))

        text = _text(crossref_work("10.5555/example"))

        assert "References: 45 deposited, none listed in Crossref's record" in text

    @patch(HTTP_OPEN)
    def test_accepts_doi_prefix(self, mock_urlopen):
        mock_urlopen.return_value = respond(work_answer(crossref_item()))

        text = _text(crossref_work("doi:10.5555/example"))

        assert text.startswith("Crossref work 10.5555/example:")

    @patch(HTTP_OPEN)
    def test_invalid_doi(self, mock_urlopen):
        failure = _failure(crossref_work, "bad doi")

        assert failure.error.type == "validation_error"
        assert "invalid DOI 'bad doi'" in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found", body=b"Resource not found.")

        failure = _failure(crossref_work, "10.5555/missing")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "no Crossref work with DOI 10.5555/missing; search with crossref_search, or look "
            "the DOI up with datacite_doi."
        )

    @patch(HTTP_OPEN)
    def test_other_statuses_propagate(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        failure = _failure(crossref_work, "10.5555/example")

        assert failure.error.type == "rate_limited"


@patch(HTTP_OPEN)
def test_a_null_list_in_one_work_does_not_fail_the_search(mock_urlopen):
    odd = crossref_item("10.1/odd") | {"author": None, "reference": None, "license": None}
    odd["link"] = None
    mock_urlopen.return_value = respond(crossref_list(crossref_item(), odd, total=2))

    text = _text(crossref_search("transformers", max_results=2))

    assert "2. Attention Is All You Need: Transformer paper" in text


@patch(HTTP_OPEN)
def test_a_work_without_a_doi_names_no_call_that_would_fail(mock_urlopen):
    mock_urlopen.return_value = respond(
        crossref_list(crossref_item(authors=11) | {"DOI": None}, total=1)
    )

    text = _text(crossref_search("transformers"))

    assert "Given Family 8 (+3 more)" in text
    assert "crossref_work(''" not in text
