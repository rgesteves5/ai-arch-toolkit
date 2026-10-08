"""Tests for toolkit/tools/_foodon.py (T07a).

Answers are shaped as OLS4's v1 search builds them
(https://github.com/EBISPOT/ols4/blob/dev/backend/src/main/java/uk/ac/ebi/spot/ols/controller/api/v1/V1SearchController.java):
``response.numFound``, ``response.start`` and ``response.docs``; its errors as its exception
handler does, ``{"status": …, "message": …}`` (``GlobalExceptionHandler.java``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._foodon import foodon_search, foodon_term
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_DEFINITION = "A pome fruit of an apple tree (Malus domestica). " + "It is eaten raw. " * 60


def term(n: int = 2473, *, label: str = "apple") -> dict[str, Any]:
    return {
        "iri": f"http://purl.obolibrary.org/obo/FOODON_{n:08d}",
        "ontology_name": "foodon",
        "short_form": f"FOODON_{n:08d}",
        "description": [_DEFINITION],
        "label": label,
        "obo_id": f"FOODON:{n:08d}",
        "type": "class",
        "exact_synonyms": ["eating apple"],
        "related_synonyms": ["pomme"],
    }


def _payload(docs: list[dict[str, Any]], *, total: int | None = None, start: int = 0):
    found = len(docs) if total is None else total
    return {"response": {"docs": docs, "numFound": found, "start": start}}


def _failure(fn: Any, *args: Any, **kwargs: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _params(mock_urlopen: MagicMock) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


class TestSearch:
    @patch(HTTP_OPEN)
    def test_a_page_says_the_total_and_the_next_start(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(
            _payload([term(2473), term(2474, label="apple juice")], total=31, start=3)
        )

        result = foodon_search("apple", max_results=2, start=3)

        assert isinstance(result, ToolResult)
        lines = result.value.splitlines()
        assert lines[0] == "FoodOn terms for 'apple' (details: foodon_term):"
        assert lines[1] == "4. apple | FOODON:00002473 | class | ontology foodon"
        assert lines[2] == f"   Definition: {_DEFINITION.strip()}"  # once cut at 700
        assert lines[3] == "   IRI: http://purl.obolibrary.org/obo/FOODON_00002473"
        assert lines[4] == "5. apple juice | FOODON:00002474 | class | ontology foodon"
        assert lines[-1] == "[results 4-5 of 31 | next: start=5]"
        params = _params(mock_urlopen)
        assert params["q"] == ["apple"]
        assert params["ontology"] == ["foodon"]
        assert params["rows"] == ["2"]
        assert params["start"] == ["3"]

    @patch(HTTP_OPEN)
    def test_no_terms_say_so_with_the_query(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(_payload([]))

        assert _text(foodon_search("zzz")) == "No FoodOn terms match 'zzz'."

    @patch(HTTP_OPEN)
    def test_a_start_past_the_last_term_gives_the_total(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(_payload([], total=57, start=100))

        assert _text(foodon_search("apple", max_results=10, start=100)) == (
            "start=100 is past the end: 57 FoodOn terms match 'apple'; the last page is start=50."
        )

    @pytest.mark.parametrize(("max_results", "kept"), [(20, True), (21, False)])
    @patch(HTTP_OPEN)
    def test_max_results_is_refused_outside_its_limits(self, mock_urlopen, max_results, kept):
        mock_urlopen.return_value = respond(_payload([]))
        call = ToolCall(
            id="c1", name="foodon_search", input={"query": "apple", "max_results": max_results}
        )

        assert ToolGroup(foodon_search).execute(call).ok is kept

    @patch(HTTP_OPEN)
    def test_an_empty_query_fails_before_asking(self, mock_urlopen: MagicMock):
        failure = _failure(foodon_search, "  ")

        assert failure.error.type == "validation_error"
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_ols_error_carries_its_message(self, mock_urlopen: MagicMock):
        body = b'{"status": 400, "message": "Failed to convert value of type \'String\'"}'
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)

        failure = _failure(foodon_search, "apple")

        assert failure.error.type == "upstream"
        assert "Failed to convert value of type 'String'" in failure.error.message

    @patch(HTTP_OPEN)
    def test_a_429_is_rate_limited(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        assert _failure(foodon_search, "apple").error.type == "rate_limited"


class TestTerm:
    @patch(HTTP_OPEN)
    def test_the_term_comes_whole_with_its_synonyms(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(_payload([term(2474, label="juice"), term()]))

        text = foodon_term("FOODON_00002473")

        assert text.splitlines() == [
            "FoodOn term FOODON:00002473: apple",
            "   type: class | ontology: foodon | short form: FOODON_00002473",
            f"   Definition: {_DEFINITION.strip()}",
            "   Exact synonyms: eating apple",
            "   Related synonyms: pomme",
            "   IRI: http://purl.obolibrary.org/obo/FOODON_00002473",
        ]
        params = _params(mock_urlopen)
        assert params["q"] == ["FOODON:00002473"]
        assert params["queryFields"] == ["obo_id"]

    @patch(HTTP_OPEN)
    def test_an_imported_term_reads_by_its_own_prefix(self, mock_urlopen: MagicMock):
        imported = {**term(), "obo_id": "NCBITaxon:3750", "label": "Malus domestica"}
        mock_urlopen.return_value = respond(_payload([imported]))

        assert foodon_term("NCBITaxon_3750").startswith(
            "FoodOn term NCBITaxon:3750: Malus domestica"
        )

    @patch(HTTP_OPEN)
    def test_an_unknown_term_is_not_found(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(_payload([]))

        failure = _failure(foodon_term, "FOODON:09999999")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "no FoodOn term with ID FOODON:09999999; search with foodon_search."
        )

    @patch(HTTP_OPEN)
    def test_an_invalid_id_fails_before_asking(self, mock_urlopen: MagicMock):
        failure = _failure(foodon_term, "bad")

        assert failure.error.type == "validation_error"
        assert "invalid term_id" in failure.error.message
        mock_urlopen.assert_not_called()
