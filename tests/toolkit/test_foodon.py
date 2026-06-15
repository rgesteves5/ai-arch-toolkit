"""Tests for toolkit/tools/_foodon.py."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._foodon import foodon_search, foodon_term

_TERM = {
    "iri": "http://purl.obolibrary.org/obo/FOODON_00002473",
    "ontology_name": "foodon",
    "short_form": "FOODON_00002473",
    "description": ["A pome fruit of an apple tree (Malus domestica)."],
    "label": "apple",
    "obo_id": "FOODON:00002473",
    "type": "class",
}


def _payload(docs):
    return {"response": {"docs": docs, "numFound": len(docs), "start": 0}}


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


class TestFoodOnSearch:
    @patch("ai_arch_toolkit.toolkit.tools._foodon.urllib.request.urlopen")
    def test_returns_terms(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_payload([_TERM]))

        result = foodon_search("apple", max_results=2, start=3)

        assert "FoodOn terms for 'apple' (start 3, returned 1, total 1):" in result
        assert "apple" in result
        assert "id: FOODON:00002473" in result
        assert "Definition: A pome fruit" in result
        assert "IRI: http://purl.obolibrary.org/obo/FOODON_00002473" in result

        request = _called_request(mock_urlopen)
        assert request.headers["User-agent"].startswith("ai-arch-toolkit/")
        params = _called_params(mock_urlopen)
        assert params["q"] == ["apple"]
        assert params["ontology"] == ["foodon"]
        assert params["rows"] == ["2"]
        assert params["start"] == ["3"]

    @patch("ai_arch_toolkit.toolkit.tools._foodon.urllib.request.urlopen")
    def test_invalid_search_options_do_not_call_api(self, mock_urlopen):
        assert "query cannot be empty" in foodon_search("")
        assert "start must" in foodon_search("apple", start=-1)
        mock_urlopen.assert_not_called()


class TestFoodOnTerm:
    @patch("ai_arch_toolkit.toolkit.tools._foodon.urllib.request.urlopen")
    def test_returns_term(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_payload([_TERM]))

        result = foodon_term("FOODON_00002473")

        assert result.startswith("FoodOn term FOODON:00002473:")
        assert "apple" in result
        assert "Definition: A pome fruit of an apple tree" in result
        assert _called_params(mock_urlopen)["q"] == ["FOODON:00002473"]

    @patch("ai_arch_toolkit.toolkit.tools._foodon.urllib.request.urlopen")
    def test_invalid_term_and_errors(self, mock_urlopen):
        assert "invalid term_id" in foodon_term("bad")
        mock_urlopen.assert_not_called()

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://www.ebi.ac.uk/ols4/api/search",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )
        assert "rate limited" in foodon_search("apple")

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = _mock_urlopen("not json")
        assert "could not parse" in foodon_search("apple")
