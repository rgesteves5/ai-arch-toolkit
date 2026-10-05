"""Tests for toolkit/tools/_wikidata.py."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._wikidata import (
    wikidata_entity,
    wikidata_search,
    wikidata_sparql,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _failure(call):
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value.error


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestWikidataSearch:
    @patch(HTTP_OPEN)
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "search": [
                    {
                        "id": "Q42",
                        "label": "Douglas Adams",
                        "description": "English writer",
                        "concepturi": "https://www.wikidata.org/wiki/Q42",
                        "match": {"text": "Douglas Adams"},
                    }
                ]
            }
        )

        result = wikidata_search("Douglas Adams", max_results=2)

        assert "Wikidata results for 'Douglas Adams'" in result
        assert "Douglas Adams (Q42)" in result
        assert "Description: English writer" in result
        assert "https://www.wikidata.org/wiki/Q42" in result

        request = _called_request(mock_urlopen)
        assert request.headers["User-agent"].startswith("ai-arch-toolkit/")
        params = _called_params(mock_urlopen)
        assert params["action"] == ["wbsearchentities"]
        assert params["search"] == ["Douglas Adams"]
        assert params["limit"] == ["2"]

    @pytest.mark.parametrize(
        ("call", "words"),
        [
            (lambda: wikidata_search(""), "empty query"),
            (lambda: wikidata_search("test", language="../en"), "invalid language '../en'"),
            (lambda: wikidata_entity("Q42", language="../en"), "invalid language '../en'"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen, call, words):
        error = _failure(call)

        assert error.type == "validation_error"
        assert words in error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_no_results_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond({"search": []})

        assert wikidata_search("zzzz") == "No Wikidata results for: 'zzzz'"

    @patch(HTTP_OPEN)
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond(b"not json")

        error = _failure(lambda: wikidata_search("test"))

        assert error.type == "upstream"
        assert "could not parse" in error.message

    @patch(HTTP_OPEN)
    def test_an_error_the_api_reports_is_the_tools_error(self, mock_urlopen):
        # Sent with HTTP 200 (www.wikidata.org, 2026-09-29); it used to read as no results.
        mock_urlopen.return_value = respond(
            {
                "error": {
                    "code": "badvalue",
                    "info": 'Unrecognized value for parameter "language": xx.',
                    "*": "See https://www.wikidata.org/w/api.php for API usage.",
                },
                "servedby": "mw-api-ext.eqiad.main-79f4dd7c47-fhdrp",
            }
        )

        error = _failure(lambda: wikidata_search("apple", language="xx"))

        assert error.type == "upstream"
        assert error.message == 'badvalue: Unrecognized value for parameter "language": xx.'

    @patch(HTTP_OPEN)
    def test_a_rate_limit_the_api_reports_is_rate_limited(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"error": {"code": "maxlag", "info": "Waiting for 10.64.0.1: 7 seconds lagged."}}
        )

        error = _failure(lambda: wikidata_search("apple"))

        assert error.type == "rate_limited"
        assert error.retryable
        assert "maxlag: Waiting for 10.64.0.1: 7 seconds lagged" in error.message

    @patch(HTTP_OPEN)
    def test_a_404_on_the_search_is_an_endpoint_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        error = _failure(lambda: wikidata_search("apple"))

        assert error.type == "upstream"
        assert error.message.startswith("Wikidata: endpoint not found (HTTP 404)")


class TestWikidataEntity:
    @patch(HTTP_OPEN)
    def test_returns_entity(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "entities": {
                    "Q42": {
                        "labels": {"en": {"value": "Douglas Adams"}},
                        "descriptions": {"en": {"value": "English writer"}},
                        "aliases": {"en": [{"value": "Douglas Noel Adams"}]},
                        "claims": {
                            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}],
                            "P569": [
                                {
                                    "mainsnak": {
                                        "datavalue": {"value": {"time": "+1952-03-11T00:00:00Z"}}
                                    }
                                }
                            ],
                        },
                        "sitelinks": {"enwiki": {"title": "Douglas Adams"}},
                    }
                }
            }
        )

        result = wikidata_entity("q42")

        assert result.startswith("Wikidata entity Q42:")
        assert "Douglas Adams" in result
        assert "Aliases: Douglas Noel Adams" in result
        assert "P31: Q5" in result
        assert "https://en.wikipedia.org/wiki/Douglas_Adams" in result

    @patch(HTTP_OPEN)
    def test_invalid_qid(self, mock_urlopen):
        error = _failure(lambda: wikidata_entity("P31"))

        assert error.type == "validation_error"
        assert "invalid QID 'P31'" in error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_missing_entity_is_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        error = _failure(lambda: wikidata_entity("Q999999999999"))

        assert error.type == "not_found"
        assert not error.retryable
        assert error.message == (
            "Wikidata has no entity Q999999999999; search for it with wikidata_search"
        )

    @patch(HTTP_OPEN)
    def test_an_entity_marked_missing_is_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond({"entities": {"Q1": {"id": "Q1", "missing": ""}}})

        assert _failure(lambda: wikidata_entity("Q1")).type == "not_found"

    @patch(HTTP_OPEN)
    def test_other_statuses_stay_upstream(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(500, "Server Error")

        error = _failure(lambda: wikidata_entity("Q42"))

        assert error.type == "upstream"
        assert error.retryable


class TestWikidataSparql:
    @patch(HTTP_OPEN)
    def test_returns_select_rows_and_appends_limit(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "head": {"vars": ["item", "itemLabel"]},
                "results": {
                    "bindings": [
                        {
                            "item": {"value": "http://www.wikidata.org/entity/Q42"},
                            "itemLabel": {"value": "Douglas Adams"},
                        }
                    ]
                },
            }
        )

        result = wikidata_sparql(
            "SELECT ?item ?itemLabel WHERE { ?item wdt:P31 wd:Q5 . }",
            max_results=2,
        )

        assert "Wikidata SPARQL rows" in result
        assert "itemLabel: Douglas Adams" in result
        assert "LIMIT 2" in _called_params(mock_urlopen)["query"][0]

    @patch(HTTP_OPEN)
    def test_returns_ask_boolean(self, mock_urlopen):
        mock_urlopen.return_value = respond({"boolean": True})

        result = wikidata_sparql("ASK { wd:Q42 wdt:P31 wd:Q5 . }")

        assert result == "Wikidata SPARQL result: True"

    @patch(HTTP_OPEN)
    def test_rejects_unsafe_query(self, mock_urlopen):
        error = _failure(lambda: wikidata_sparql("DELETE WHERE { ?s ?p ?o }"))

        assert error.type == "validation_error"
        assert "read-only" in error.message
        mock_urlopen.assert_not_called()

    @pytest.mark.parametrize(
        ("query", "words"), [("", "empty query"), ("DESCRIBE wd:Q42", "SELECT or ASK")]
    )
    @patch(HTTP_OPEN)
    def test_rejects_other_invalid_queries(self, mock_urlopen, query, words):
        error = _failure(lambda: wikidata_sparql(query))

        assert error.type == "validation_error"
        assert words in error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_no_rows_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond({"head": {"vars": ["x"]}, "results": {"bindings": []}})

        assert wikidata_sparql("SELECT ?x WHERE { }") == "Wikidata SPARQL returned no rows."

    @patch(HTTP_OPEN)
    def test_an_unexpected_answer_is_upstream(self, mock_urlopen):
        mock_urlopen.return_value = respond({"head": {}, "results": {"bindings": {}}})

        assert _failure(lambda: wikidata_sparql("SELECT ?x WHERE { }")).type == "upstream"

    @patch(HTTP_OPEN)
    def test_rate_limited(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://query.wikidata.org/sparql",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

        error = _failure(lambda: wikidata_sparql("ASK { wd:Q42 wdt:P31 wd:Q5 . }"))

        assert error.type == "rate_limited"
        assert "rate limited by Wikidata Query Service (HTTP 429)" in error.message

    @patch(HTTP_OPEN)
    def test_a_query_the_service_rejects_says_why(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            400,
            "Bad Request",
            body=b"SPARQL-QUERY: queryStr=SELECT ?x WHERE {\nMalformedQueryException: "
            b'Encountered "<EOF>"',
        )

        error = _failure(lambda: wikidata_sparql("SELECT ?x WHERE {"))

        assert error.type == "upstream"
        assert not error.retryable
        assert "MalformedQueryException" in error.message


@patch(HTTP_OPEN)
def test_a_merged_qid_reads_as_the_item_it_redirects_to(mock_urlopen):
    # Special:EntityData follows the redirect, so the answer holds only Q48 (2026-09-30); it
    # read as "not found".
    mock_urlopen.return_value = respond(
        {"entities": {"Q48": {"id": "Q48", "labels": {"en": {"value": "Asia"}}}}}
    )

    result = wikidata_entity("Q65439041")

    assert result.startswith("Wikidata entity Q65439041 (redirects to Q48):\nAsia")
    assert "Wikidata: https://www.wikidata.org/wiki/Q48" in result
