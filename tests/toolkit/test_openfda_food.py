"""Tests for toolkit/tools/_openfda_food.py."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._openfda_food import (
    openfda_food_recall,
    openfda_food_recall_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _failure(fn, *args, **kwargs) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _invalid(fn, *args, **kwargs) -> str:
    failure = _failure(fn, *args, **kwargs)
    assert failure.error.type == "validation_error"
    return failure.error.message


def _not_found() -> urllib.error.HTTPError:
    """openFDA's answer to a search that matches nothing."""
    body = b'{"error": {"code": "NOT_FOUND", "message": "No matches found!"}}'
    return http_error(404, "Not Found", body=body)


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


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestOpenFdaFoodRecallSearch:
    @patch(HTTP_OPEN)
    def test_returns_recalls(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"meta": {"results": {"total": 1}}, "results": [_RECALL]}
        )

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

        assert "openFDA food recalls (returned 1, total 1):" in result
        assert "F-2473-2016" in result
        assert "class: Class I" in result
        assert "Reason: Potential for contamination" in result

        request = _called_request(mock_urlopen)
        assert request.headers["User-agent"].startswith("ai-arch-toolkit/")
        params = _called_params(mock_urlopen)
        assert params["limit"] == ["2"]
        assert params["skip"] == ["3"]
        assert 'classification.exact:"Class I"' in params["search"][0]
        assert "report_date:[20160101 TO 20161231]" in params["search"][0]

    @patch(HTTP_OPEN)
    def test_invalid_search_options_do_not_call_api(self, mock_urlopen):
        assert "provide query" in _invalid(openfda_food_recall_search)
        assert "skip must" in _invalid(openfda_food_recall_search, query="x", skip=-1)
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
    def test_a_search_that_matches_nothing_is_an_answer(self, mock_urlopen):
        mock_urlopen.side_effect = _not_found()

        assert openfda_food_recall_search(query="missing") == "No openFDA food recalls found."

    @patch(HTTP_OPEN)
    def test_parse_and_server_failures_are_upstream(self, mock_urlopen):
        mock_urlopen.return_value = respond("not json")
        not_json = _failure(openfda_food_recall_search, query="x")
        assert not_json.error.type == "upstream"
        assert "could not parse" in str(not_json)

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.fda.gov/food/enforcement.json",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=None,
        )
        server = _failure(openfda_food_recall_search, query="x")
        assert server.error.type == "upstream"
        assert server.error.retryable

    @patch(HTTP_OPEN)
    def test_a_bad_request_carries_openfda_s_words(self, mock_urlopen):
        body = b'{"error": {"code": "BAD_REQUEST", "message": "Invalid date format"}}'
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)

        failure = _failure(openfda_food_recall_search, query="x")

        assert failure.error.type == "upstream"
        assert not failure.error.retryable
        assert "Invalid date format" in failure.error.message


class TestOpenFdaFoodRecall:
    @patch(HTTP_OPEN)
    def test_returns_recall(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"meta": {"results": {"total": 1}}, "results": [_RECALL]}
        )

        result = openfda_food_recall("f-2473-2016")

        assert result.startswith("openFDA food recall F-2473-2016:")
        assert "Distribution: Domestic distribution." in result
        assert "Code info: Product Code/Lot Code: 16033" in result
        assert "Initiated: 2016-02-11" in result

        assert _called_params(mock_urlopen)["search"] == ['recall_number:"F-2473-2016"']

    @patch(HTTP_OPEN)
    def test_invalid_recall_number(self, mock_urlopen):
        assert "invalid recall_number 'bad'" in _invalid(openfda_food_recall, "bad")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_unknown_recall_is_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = _not_found()

        failure = _failure(openfda_food_recall, "F-0000-2016")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "openFDA has no food recall F-0000-2016; find recalls with openfda_food_recall_search"
        )

    @patch(HTTP_OPEN)
    def test_an_answer_without_the_recall_is_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond({"meta": {"results": {"total": 0}}, "results": []})

        assert _failure(openfda_food_recall, "F-0000-2016").error.type == "not_found"
