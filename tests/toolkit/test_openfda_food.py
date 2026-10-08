"""Tests for toolkit/tools/_openfda_food.py (T07a).

The error answers are openFDA's own, as its server builds them
(https://github.com/FDA/openfda/blob/master/api/faers/api.js and api_request.js): a search that
matches nothing is a 404 ``NOT_FOUND`` "No matches found!", a refused parameter a 400
``BAD_REQUEST`` with the reason, a failed search a 500 ``SERVER_ERROR``.
"""

from __future__ import annotations

import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._openfda_food import (
    openfda_food_recall,
    openfda_food_recall_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _failure(fn: Any, *args: Any, **kwargs: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _invalid(fn: Any, *args: Any, **kwargs: Any) -> str:
    failure = _failure(fn, *args, **kwargs)
    assert failure.error.type == "validation_error"
    return failure.error.message


def _error(status: int, code: str, message: str) -> urllib.error.HTTPError:
    body = f'{{"error": {{"code": "{code}", "message": "{message}"}}}}'.encode()
    return http_error(status, "Error", body=body)


def _not_found() -> urllib.error.HTTPError:
    """openFDA's answer to a search that matches nothing."""
    return _error(404, "NOT_FOUND", "No matches found!")


_RECALL = {
    "recall_number": "F-2473-2016",
    "status": "Terminated",
    "classification": "Class I",
    "product_type": "Food",
    "product_description": "1.5 oz PEANUT BUTTER COOKIES, 80/case",
    "reason_for_recall": "Potential for contamination with Listeria monocytogenes.",
    "recalling_firm": "Savory Foods, Inc.",
    "city": "Grand Rapids",
    "state": "MI",
    "country": "United States",
    "distribution_pattern": "Domestic distribution.",
    "code_info": "Product Code/Lot Code: 16033",
    "recall_initiation_date": "20160211",
    "report_date": "20161012",
    "termination_date": "20161028",
}


def _results(count: int, *, total: int, skip: int = 0) -> dict[str, Any]:
    recalls = [{**_RECALL, "recall_number": f"F-{2000 + skip + n}-2016"} for n in range(count)]
    return {
        "meta": {"results": {"skip": skip, "limit": count, "total": total}},
        "results": recalls,
    }


def _params(mock_urlopen: MagicMock) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


class TestRecallSearch:
    @patch(HTTP_OPEN)
    def test_a_page_says_the_total_and_the_next_skip(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(_results(2, total=57, skip=3))

        result = openfda_food_recall_search(
            query="peanut butter",
            classification="Class I",
            status="Terminated",
            state="MI",
            from_date="2016-01-01",
            to_date="2016-12-31",
            max_results=2,
            skip=3,
        )

        assert isinstance(result, ToolResult)
        lines = result.value.splitlines()
        assert lines[0].startswith("openFDA food recalls for query 'peanut butter', class")
        assert lines[1] == "4. F-2003-2016 — 1.5 oz PEANUT BUTTER COOKIES, 80/case"
        assert "   recall: F-2003-2016 | class: Class I | status: Terminated" in result.value
        assert "report date: 2016-10-12" in result.value
        assert lines[-1] == "[results 4-5 of 57 | next: skip=5]"
        params = _params(mock_urlopen)
        assert params["limit"] == ["2"]
        assert params["skip"] == ["3"]
        assert 'classification.exact:"Class I"' in params["search"][0]
        assert "report_date:[20160101 TO 20161231]" in params["search"][0]

    @patch(HTTP_OPEN)
    def test_past_the_deepest_skip_the_heading_says_how_to_narrow(self, mock_urlopen):
        mock_urlopen.return_value = respond(_results(20, total=40_000, skip=24_990))

        text = _text(openfda_food_recall_search(query="milk", max_results=20, skip=24_990))

        assert text.endswith("[results 24991-25010 of 40000 | the rest cannot be read here]")
        assert "openFDA pages up to skip=25000; narrow with from_date and to_date" in text

    @pytest.mark.parametrize(("skip", "kept"), [(25_000, True), (25_001, False)])
    @patch(HTTP_OPEN)
    def test_skip_is_refused_past_what_openfda_takes(self, mock_urlopen, skip, kept):
        mock_urlopen.side_effect = _not_found()
        call = ToolCall(
            id="c1", name="openfda_food_recall_search", input={"query": "milk", "skip": skip}
        )

        assert ToolGroup(openfda_food_recall_search).execute(call).ok is kept

    @patch(HTTP_OPEN)
    def test_a_search_that_matches_nothing_says_so_with_the_search(self, mock_urlopen):
        mock_urlopen.side_effect = _not_found()

        assert _text(openfda_food_recall_search(query="missing")) == (
            "No openFDA food recalls match query 'missing'."
        )

    @patch(HTTP_OPEN)
    def test_invalid_search_options_do_not_call_api(self, mock_urlopen: MagicMock):
        assert "provide query" in _invalid(openfda_food_recall_search)
        assert "invalid query" in _invalid(openfda_food_recall_search, query="bad<>")
        assert "invalid reason" in _invalid(openfda_food_recall_search, reason="bad<>")
        assert "invalid from_date" in _invalid(
            openfda_food_recall_search, query="x", from_date="2016"
        )
        assert "invalid to_date" in _invalid(openfda_food_recall_search, query="x", to_date="x")
        assert "from_date must" in _invalid(
            openfda_food_recall_search, query="x", from_date="2017-01-01", to_date="2016-01-01"
        )
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_refused_search_is_a_validation_error_with_openfdas_words(self, mock_urlopen):
        mock_urlopen.side_effect = _error(400, "BAD_REQUEST", "Skip value must 25000 or less.")

        failure = _failure(openfda_food_recall_search, query="x")

        assert failure.error.type == "validation_error"
        assert not failure.error.retryable
        assert "Skip value must 25000 or less." in failure.error.message
        assert "check the filters and the dates" in failure.error.message

    @patch(HTTP_OPEN)
    def test_a_failed_search_is_upstream_and_retryable(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = _error(500, "SERVER_ERROR", "Check your request and try again")

        failure = _failure(openfda_food_recall_search, query="x")

        assert (failure.error.type, failure.error.retryable) == ("upstream", True)
        assert "Check your request and try again" in failure.error.message

    @patch(HTTP_OPEN)
    def test_an_answer_that_is_not_json_is_upstream(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond("not json")

        failure = _failure(openfda_food_recall_search, query="x")

        assert failure.error.type == "upstream"
        assert "could not parse" in failure.error.message


class TestRecall:
    @patch(HTTP_OPEN)
    def test_returns_recall(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(
            {"meta": {"results": {"total": 1}}, "results": [_RECALL]}
        )

        result = openfda_food_recall("f-2473-2016")

        assert result.startswith("openFDA food recall F-2473-2016:")
        assert "Distribution: Domestic distribution." in result
        assert "Code info: Product Code/Lot Code: 16033" in result
        assert "Initiated: 2016-02-11" in result
        assert _params(mock_urlopen)["search"] == ['recall_number:"F-2473-2016"']

    @patch(HTTP_OPEN)
    def test_invalid_recall_number(self, mock_urlopen: MagicMock):
        assert "invalid recall_number 'bad'" in _invalid(openfda_food_recall, "bad")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_unknown_recall_is_not_found(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = _not_found()

        failure = _failure(openfda_food_recall, "F-0000-2016")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "openFDA has no food recall F-0000-2016; find recalls with openfda_food_recall_search"
        )

    @patch(HTTP_OPEN)
    def test_an_answer_without_the_recall_is_not_found(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond({"meta": {"results": {"total": 0}}, "results": []})

        assert _failure(openfda_food_recall, "F-0000-2016").error.type == "not_found"
