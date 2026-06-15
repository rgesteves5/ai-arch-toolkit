"""Tests for toolkit/tools/_openfda_food.py."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._openfda_food import (
    openfda_food_recall,
    openfda_food_recall_search,
)

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


def _mock_urlopen(data: dict | str):
    resp = MagicMock()
    if isinstance(data, dict):
        resp.read.return_value = json.dumps(data).encode()
    else:
        resp.read.return_value = data.encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestOpenFdaFoodRecallSearch:
    @patch("ai_arch_toolkit.toolkit.tools._openfda_food.urllib.request.urlopen")
    def test_returns_recalls(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
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

    @patch("ai_arch_toolkit.toolkit.tools._openfda_food.urllib.request.urlopen")
    def test_invalid_search_options_do_not_call_api(self, mock_urlopen):
        assert "provide query" in openfda_food_recall_search()
        assert "skip must" in openfda_food_recall_search(query="x", skip=-1)
        assert "invalid query" in openfda_food_recall_search(query="bad<>")
        assert "invalid from_date" in openfda_food_recall_search(query="x", from_date="2016")
        assert "from_date must" in openfda_food_recall_search(
            query="x", from_date="2017-01-01", to_date="2016-01-01"
        )
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._openfda_food.urllib.request.urlopen")
    def test_not_found_and_parse_failure(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.fda.gov/food/enforcement.json",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )
        assert "no matching records" in openfda_food_recall_search(query="missing")

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = _mock_urlopen("not json")
        assert "could not parse" in openfda_food_recall_search(query="x")


class TestOpenFdaFoodRecall:
    @patch("ai_arch_toolkit.toolkit.tools._openfda_food.urllib.request.urlopen")
    def test_returns_recall(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            {"meta": {"results": {"total": 1}}, "results": [_RECALL]}
        )

        result = openfda_food_recall("f-2473-2016")

        assert result.startswith("openFDA food recall F-2473-2016:")
        assert "Distribution: Domestic distribution." in result
        assert "Code info: Product Code/Lot Code: 16033" in result
        assert "Initiated: 2016-02-11" in result

        assert _called_params(mock_urlopen)["search"] == ['recall_number:"F-2473-2016"']

    @patch("ai_arch_toolkit.toolkit.tools._openfda_food.urllib.request.urlopen")
    def test_invalid_recall_number(self, mock_urlopen):
        assert "invalid recall_number" in openfda_food_recall("bad")
        mock_urlopen.assert_not_called()
