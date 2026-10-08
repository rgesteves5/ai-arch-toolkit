"""Tests for toolkit/tools/_wikidata.py."""

from __future__ import annotations

import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolFailure, ToolGroup, ToolResult
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


def _called_request(mock_urlopen, call: int = -1):
    return mock_urlopen.call_args_list[call].args[0]


def _called_params(mock_urlopen, call: int = -1) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen, call).full_url).query)


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult) and result.ok
    assert isinstance(result.value, str)
    return result.value


# --- Search ------------------------------------------------------------------------------------


def _search_answer(count: int, start: int = 1, more: int | None = None) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "search": [
            {
                "id": f"Q{number}",
                "label": f"Item {number}",
                "description": f"thing number {number}",
                "concepturi": f"http://www.wikidata.org/entity/Q{number}",
                "match": {"type": "label", "text": f"Item {number}"},
            }
            for number in range(start, start + count)
        ],
        "success": 1,
    }
    if more is not None:
        answer["search-continue"] = more
    return answer


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
                        "concepturi": "http://www.wikidata.org/entity/Q42",
                        "match": {"type": "alias", "text": "DNA"},
                    }
                ]
            }
        )

        text = _text(wikidata_search("Douglas Adams", max_results=2))

        assert text == (
            "Wikidata items that match 'Douglas Adams' (wikidata_entity reads one):\n"
            "1. Douglas Adams (Q42): English writer\n"
            "   matched: DNA"
        )
        request = _called_request(mock_urlopen)
        assert request.headers["User-agent"].startswith("ai-arch-toolkit/")
        params = _called_params(mock_urlopen)
        assert params["action"] == ["wbsearchentities"]
        assert params["search"] == ["Douglas Adams"]
        assert params["limit"] == ["2"]
        assert params["continue"] == ["0"]

    @patch(HTTP_OPEN)
    def test_the_results_read_on_where_the_api_says(self, mock_urlopen):
        mock_urlopen.return_value = respond(_search_answer(5, start=6, more=10))

        result = wikidata_search("item", max_results=5, offset=5)
        text = _text(result)

        assert _called_params(mock_urlopen)["continue"] == ["5"]
        assert text.splitlines()[1] == "6. Item 6 (Q6): thing number 6"
        assert text.endswith("[results 6-10 | next: offset=10]")
        assert result.metadata["window"]["next_call"] == {"offset": 10}

    @patch(HTTP_OPEN)
    def test_no_results_say_so_with_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond({"search": []})

        assert _text(wikidata_search("zzzz")) == "No Wikidata items match 'zzzz'."

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

    def test_the_limits_are_in_the_schema(self):
        group = ToolGroup(wikidata_search, wikidata_entity, wikidata_sparql)
        for name, args in (
            ("wikidata_search", {"query": "x", "max_results": 51}),
            ("wikidata_search", {"query": "x", "offset": 10001}),
            ("wikidata_entity", {"qid": "Q1", "offset": -1}),
            ("wikidata_sparql", {"query": "ASK {}", "max_results": 101}),
        ):
            result = group.execute(ToolCall(id="c", name=name, input=args))
            assert not result.ok and result.error is not None
            assert result.error.type == "validation_error", args

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


# --- Entity ------------------------------------------------------------------------------------


def _snak(prop: str, datatype: str, value: object) -> dict[str, Any]:
    return {
        "snaktype": "value",
        "property": prop,
        "datavalue": {"type": datatype, "value": value},
    }


def _statement(prop: str, datatype: str, value: object, **extra: Any) -> dict[str, Any]:
    return {
        "mainsnak": _snak(prop, datatype, value),
        "type": "statement",
        "rank": "normal",
        **extra,
    }


def _item(qid: str) -> dict[str, Any]:
    return {"entity-type": "item", "numeric-id": int(qid[1:]), "id": qid}


def _time(time: str, precision: int, calendar: str = "Q1985727") -> dict[str, Any]:
    return {
        "time": time,
        "timezone": 0,
        "before": 0,
        "after": 0,
        "precision": precision,
        "calendarmodel": f"http://www.wikidata.org/entity/{calendar}",
    }


# Shaped as Special:EntityData answers
# (https://doc.wikimedia.org/Wikibase/master/php/docs_topics_json.html).
_ADAMS_CLAIMS = {
    "P31": [_statement("P31", "wikibase-entityid", _item("Q5"))],
    "P569": [_statement("P569", "time", _time("+1952-03-11T00:00:00Z", 11))],
    "P2048": [
        _statement(
            "P2048",
            "quantity",
            {"amount": "+1.96", "unit": "http://www.wikidata.org/entity/Q11573"},
        )
    ],
    "P625": [
        _statement(
            "P625",
            "globecoordinate",
            {
                "latitude": 52.5,
                "longitude": 13.4,
                "precision": 0.1,
                "globe": "http://www.wikidata.org/entity/Q2",
            },
        )
    ],
    "P1477": [
        _statement("P1477", "monolingualtext", {"text": "Douglas Noël Adams", "language": "en"})
    ],
    "P1082": [
        _statement(
            "P1082",
            "quantity",
            {"amount": "+1000", "unit": "1"},
            rank="preferred",
            qualifiers={"P585": [_snak("P585", "time", _time("+2021-00-00T00:00:00Z", 9))]},
        )
    ],
    "P27": [
        _statement("P27", "wikibase-entityid", _item("Q145")),
        _statement("P27", "wikibase-entityid", _item("Q174193"), rank="deprecated"),
    ],
    "P22": [{"mainsnak": {"snaktype": "somevalue", "property": "P22"}, "rank": "normal"}],
    "P570": [_statement("P570", "time", _time("+1001-00-00T00:00:00Z", 7, "Q1985786"))],
}
_ADAMS = {
    "entities": {
        "Q42": {
            "id": "Q42",
            "labels": {"en": {"language": "en", "value": "Douglas Adams"}},
            "descriptions": {"en": {"language": "en", "value": "English writer"}},
            "aliases": {"en": [{"language": "en", "value": "Douglas Noel Adams"}]},
            "claims": _ADAMS_CLAIMS,
            "sitelinks": {"enwiki": {"site": "enwiki", "title": "Douglas Adams"}},
        }
    }
}
_NAMES = {
    "P31": "instance of",
    "Q5": "human",
    "P569": "date of birth",
    "P2048": "height",
    "Q11573": "metre",
    "P625": "coordinate location",
    "P1477": "birth name",
    "P1082": "population",
    "P585": "point in time",
    "P27": "country of citizenship",
    "Q145": "United Kingdom",
    "Q174193": "United Kingdom of Great Britain and Ireland",
    "P22": "father",
    "P570": "date of death",
}


def _labels(names: dict[str, str]) -> dict[str, Any]:
    """A ``wbgetentities`` answer with ``props=labels``."""
    return {
        "entities": {
            code: {"id": code, "labels": {"en": {"language": "en", "value": name}}}
            for code, name in names.items()
        },
        "success": 1,
    }


class TestWikidataEntity:
    @patch(HTTP_OPEN)
    def test_codes_come_with_their_labels_units_and_readable_values(self, mock_urlopen):
        # It showed "P31: Q5", lost units and wrote coordinates as a Python dict.
        mock_urlopen.side_effect = [respond(_ADAMS), respond(_labels(_NAMES))]

        text = _text(wikidata_entity("q42"))

        assert text.splitlines() == [
            "Wikidata entity Q42: Douglas Adams",
            "   description: English writer",
            "   aliases: Douglas Noel Adams",
            "   Wikipedia: https://en.wikipedia.org/wiki/Douglas_Adams",
            "   Wikidata: https://www.wikidata.org/wiki/Q42",
            "Statements (wikidata_entity reads any Q or P code below):",
            "1. instance of (P31): human (Q5)",
            "2. date of birth (P569): 1952-03-11",
            "3. height (P2048): 1.96 metre (Q11573)",
            "4. coordinate location (P625): latitude 52.5, longitude 13.4",
            "5. birth name (P1477): Douglas Noël Adams (en)",
            "6. population (P1082): 1000 (point in time: 2021) [preferred]",
            "7. country of citizenship (P27): United Kingdom (Q145)",
            "8. country of citizenship (P27): United Kingdom of Great Britain and Ireland "
            "(Q174193) [deprecated]",
            "9. father (P22): unknown value",
            "10. date of death (P570): 1001 (to the century, Julian calendar)",
        ]
        labels = _called_params(mock_urlopen, 1)
        assert labels["action"] == ["wbgetentities"]
        assert labels["props"] == ["labels"]
        assert labels["languages"] == ["en"]
        assert set(labels["ids"][0].split("|")) == set(_NAMES)

    @patch(HTTP_OPEN)
    def test_every_statement_reads_on(self, mock_urlopen):
        # It showed 15 claims from the first 20 properties, 3 values each.
        claims = {
            f"P{prop}": [
                _statement(f"P{prop}", "string", f"value {prop}.{value}") for value in range(4)
            ]
            for prop in range(1, 26)
        }
        entity = {"entities": {"Q1": {"id": "Q1", "claims": claims}}}
        mock_urlopen.side_effect = [
            respond(entity),
            respond(_labels({})),
            respond(entity),
            respond(_labels({})),
        ]

        first = _text(wikidata_entity("Q1"))
        second = _text(wikidata_entity("Q1", offset=40))

        assert first.endswith("[results 1-40 of 100 | next: offset=40]")
        assert "40. P10: value 10.3" in first
        assert second.splitlines()[0] == "Wikidata entity Q1: (no label), statements:"
        assert second.splitlines()[1] == "41. P11: value 11.0"
        assert second.endswith("[results 41-80 of 100 | next: offset=80]")

    @patch(HTTP_OPEN)
    def test_labels_are_asked_fifty_codes_at_a_time(self, mock_urlopen):
        # wbgetentities takes at most 50 ids
        # (https://www.wikidata.org/w/api.php?action=help&modules=wbgetentities).
        claims = {
            f"P{1000 + number}": [
                _statement(f"P{1000 + number}", "wikibase-entityid", _item(f"Q{100 + number}"))
            ]
            for number in range(60)
        }
        entity = {"entities": {"Q1": {"id": "Q1", "claims": claims}}}
        mock_urlopen.side_effect = [respond(entity), respond(_labels({})), respond(_labels({}))]

        wikidata_entity("Q1")

        asked = [_called_params(mock_urlopen, call)["ids"][0].split("|") for call in (1, 2)]
        # The 40 statements shown: their 40 properties and 40 values.
        assert [len(ids) for ids in asked] == [50, 30]
        assert mock_urlopen.call_count == 3

    @patch(HTTP_OPEN)
    def test_labels_take_four_requests_at_most(self, mock_urlopen):
        # A page names at most 120 codes, and its time qualifiers' properties; an answer with
        # more cannot make the tool send a request per 50 of them.
        qualifiers = {
            f"P{number}": [_snak(f"P{number}", "time", _time("+2021-00-00T00:00:00Z", 9))]
            for number in range(1000, 1300)
        }
        claims = {"P585": [_statement("P585", "string", "x", qualifiers=qualifiers)]}
        entity = {"entities": {"Q1": {"id": "Q1", "claims": claims}}}
        mock_urlopen.side_effect = [respond(entity), *(respond(_labels({})) for _ in range(4))]

        text = _text(wikidata_entity("Q1"))

        assert mock_urlopen.call_count == 5
        assert "(P1299: 2021)" in text  # unlabelled, still a code wikidata_entity reads

    @patch(HTTP_OPEN)
    def test_labels_in_another_language_fall_back_to_english(self, mock_urlopen):
        entity = {
            "entities": {
                "Q1": {
                    "id": "Q1",
                    "labels": {"pt": {"language": "pt", "value": "coisa"}},
                    "claims": {"P31": [_statement("P31", "wikibase-entityid", _item("Q5"))]},
                }
            }
        }
        labels = {
            "entities": {
                "P31": {
                    "id": "P31",
                    "labels": {"pt": {"language": "pt", "value": "instância de"}},
                },
                "Q5": {"id": "Q5", "labels": {"en": {"language": "en", "value": "human"}}},
            }
        }
        mock_urlopen.side_effect = [respond(entity), respond(labels)]

        text = _text(wikidata_entity("Q1", language="pt"))

        assert "1. instância de (P31): human (Q5)" in text
        assert _called_params(mock_urlopen, 1)["languages"] == ["pt|en"]

    @patch(HTTP_OPEN)
    def test_a_property_is_read_like_an_item(self, mock_urlopen):
        entity = {
            "entities": {
                "P31": {
                    "id": "P31",
                    "datatype": "wikibase-item",
                    "labels": {"en": {"language": "en", "value": "instance of"}},
                }
            }
        }
        mock_urlopen.return_value = respond(entity)

        text = _text(wikidata_entity("p31"))

        assert text.startswith("Wikidata entity P31: instance of")
        assert urlparse(_called_request(mock_urlopen).full_url).path.endswith("/P31.json")

    @patch(HTTP_OPEN)
    def test_invalid_qid(self, mock_urlopen):
        error = _failure(lambda: wikidata_entity("L31"))

        assert error.type == "validation_error"
        assert "invalid ID 'L31'" in error.message
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

    @patch(HTTP_OPEN)
    def test_a_label_request_the_wiki_refuses_says_why(self, mock_urlopen):
        mock_urlopen.side_effect = [
            respond(_ADAMS),
            respond({"error": {"code": "readonly", "info": "The wiki is in read-only mode."}}),
        ]

        error = _failure(lambda: wikidata_entity("Q42"))

        assert error.type == "upstream"
        assert error.message == "readonly: The wiki is in read-only mode."


@patch(HTTP_OPEN)
def test_a_merged_qid_reads_as_the_item_it_redirects_to(mock_urlopen: MagicMock) -> None:
    # Special:EntityData follows the redirect, so the answer holds only Q48 (2026-09-30); it
    # read as "not found".
    mock_urlopen.return_value = respond(
        {"entities": {"Q48": {"id": "Q48", "labels": {"en": {"value": "Asia"}}}}}
    )

    text = _text(wikidata_entity("Q65439041"))

    assert text.startswith("Wikidata entity Q65439041 (redirects to Q48): Asia")
    assert "Wikidata: https://www.wikidata.org/wiki/Q48" in text
    assert text.endswith("Statements: none.")


# --- SPARQL ------------------------------------------------------------------------------------


def _bindings(count: int) -> dict[str, Any]:
    return {
        "head": {"vars": ["item", "itemLabel"]},
        "results": {
            "bindings": [
                {
                    "item": {"type": "uri", "value": f"http://www.wikidata.org/entity/Q{number}"},
                    "itemLabel": {"type": "literal", "xml:lang": "en", "value": f"Item {number}"},
                }
                for number in range(1, count + 1)
            ]
        },
    }


class TestWikidataSparql:
    @patch(HTTP_OPEN)
    def test_a_query_with_its_own_limit_reads_every_row(self, mock_urlopen):
        # A query with LIMIT 100 got 20 rows, with no note.
        query = "SELECT ?item ?itemLabel WHERE { ?item wdt:P31 wd:Q5 . } LIMIT 100"
        mock_urlopen.side_effect = [respond(_bindings(100)), respond(_bindings(100))]

        first = _text(wikidata_sparql(query))
        last = _text(wikidata_sparql(query, offset=80))

        assert _called_params(mock_urlopen)["query"] == [query]  # no LIMIT added
        assert first.splitlines()[1] == "1. item: Q1 | itemLabel: Item 1"
        assert first.endswith("[results 1-20 of 100 | next: offset=20]")
        assert last.splitlines()[-2] == "100. item: Q100 | itemLabel: Item 100"
        assert last.endswith("[results 81-100 of 100 | end]")

    @patch(HTTP_OPEN)
    def test_a_query_without_a_limit_gets_one_and_says_when_it_is_reached(self, mock_urlopen):
        mock_urlopen.return_value = respond(_bindings(1000))

        text = _text(wikidata_sparql("SELECT ?item WHERE { ?item wdt:P31 wd:Q5 }"))

        assert _called_params(mock_urlopen)["query"][0].endswith("\nLIMIT 1000")
        assert text.splitlines()[0] == (
            "Wikidata SPARQL rows (the query had no LIMIT: the tool added LIMIT 1000, and the "
            "answer reached it; to read past it, add ORDER BY, LIMIT and OFFSET to the query):"
        )

    @patch(HTTP_OPEN)
    def test_a_limit_inside_a_subquery_is_not_the_querys(self, mock_urlopen):
        mock_urlopen.return_value = respond(_bindings(1))

        wikidata_sparql("SELECT ?x WHERE { { SELECT ?x WHERE { ?x ?p ?o } LIMIT 5 } }")

        assert _called_params(mock_urlopen)["query"][0].endswith("}\nLIMIT 1000")

    @patch(HTTP_OPEN)
    def test_numbers_read_in_plain_digits(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "head": {"vars": ["pop"]},
                "results": {
                    "bindings": [
                        {
                            "pop": {
                                "type": "literal",
                                "datatype": "http://www.w3.org/2001/XMLSchema#double",
                                "value": "1.0E7",
                            }
                        }
                    ]
                },
            }
        )

        text = _text(wikidata_sparql("SELECT ?pop WHERE { wd:Q1 wdt:P1082 ?pop } LIMIT 1"))

        assert text.splitlines()[1] == "1. pop: 10000000"

    @patch(HTTP_OPEN)
    def test_returns_ask_boolean(self, mock_urlopen):
        mock_urlopen.return_value = respond({"head": {}, "boolean": True})

        assert _text(wikidata_sparql("ASK { wd:Q42 wdt:P31 wd:Q5 . }")) == (
            "Wikidata SPARQL answer: true"
        )

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
    def test_no_rows_say_so_with_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond({"head": {"vars": ["x"]}, "results": {"bindings": []}})

        assert _text(wikidata_sparql("SELECT ?x WHERE {\n  ?x wdt:P31 wd:Q0 }")) == (
            "No rows for the Wikidata query: SELECT ?x WHERE { ?x wdt:P31 wd:Q0 }"
        )

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
    def test_a_query_the_service_cannot_parse_is_a_validation_error(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            400,
            "Bad Request",
            body=b"SPARQL-QUERY: queryStr=SELECT ?x WHERE {\njava.util.concurrent.ExecutionExce"
            b'ption: org.openrdf.query.MalformedQueryException: Encountered "<EOF>" at line 1'
            b".\n\tat java.util.concurrent.FutureTask.report(FutureTask.java:122)",
        )

        error = _failure(lambda: wikidata_sparql("SELECT ?x WHERE {"))

        assert error.type == "validation_error"
        assert not error.retryable
        assert error.message == (
            'the query service could not parse the query (Encountered "<EOF>" at line 1.); '
            "fix the query"
        )

    @patch(HTTP_OPEN)
    def test_a_query_past_the_services_deadline_says_how_to_narrow_it(self, mock_urlopen):
        # The service stops every query at 60 s
        # (https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual#Query_limits).
        mock_urlopen.side_effect = http_error(
            500,
            "Server Error",
            body=b"SPARQL-QUERY: queryStr=SELECT ...\njava.util.concurrent.TimeoutException\n\tat",
        )

        error = _failure(lambda: wikidata_sparql("SELECT ?x WHERE { ?x ?p ?o }"))

        assert error.type == "upstream"
        assert not error.retryable
        assert error.message == (
            "the query ran past the query service's 60 s limit (TimeoutException); narrow it "
            "(fewer patterns, a LIMIT, no label service on many rows) and try again"
        )
