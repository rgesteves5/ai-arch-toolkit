"""Tests for toolkit/tools/_wikidata.py."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._wikidata import (
    wikidata_entity,
    wikidata_search,
    wikidata_sparql,
)
from tests.toolkit.http_fakes import HTTP_OPEN, respond


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

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "query cannot be empty" in wikidata_search("")
        assert "invalid language" in wikidata_search("test", language="../en")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond(b"not json")

        result = wikidata_search("test")

        assert "could not parse" in result


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
        result = wikidata_entity("P31")

        assert "invalid QID" in result
        mock_urlopen.assert_not_called()


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
        result = wikidata_sparql("DELETE WHERE { ?s ?p ?o }")

        assert "read-only" in result
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://query.wikidata.org/sparql",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

        result = wikidata_sparql("ASK { wd:Q42 wdt:P31 wd:Q5 . }")

        assert "rate limited by Wikidata Query Service (HTTP 429)" in result
