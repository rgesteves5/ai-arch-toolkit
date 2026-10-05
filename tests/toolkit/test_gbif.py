"""Tests for toolkit/tools/_gbif.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._gbif import (
    gbif_occurrence_search,
    gbif_species,
    gbif_species_match,
    gbif_species_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _failure(call) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestGbif:
    @patch(HTTP_OPEN)
    def test_species_match(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "usageKey": 5219404,
                "scientificName": "Puma concolor",
                "rank": "SPECIES",
                "status": "ACCEPTED",
                "matchType": "EXACT",
                "kingdom": "Animalia",
                "genus": "Puma",
            }
        )

        result = gbif_species_match("Puma concolor")

        assert "usageKey: 5219404" in result
        assert "classification: Animalia > Puma" in result
        assert _params(mock_urlopen)["name"] == ["Puma concolor"]

    @patch(HTTP_OPEN)
    def test_species_search_and_lookup(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"count": 1, "results": [{"key": 1, "scientificName": "Puma", "rank": "GENUS"}]}
        )
        assert "Puma | key: 1" in gbif_species_search("Puma", rank="GENUS")

        mock_urlopen.return_value = respond(
            {"key": 1, "scientificName": "Puma", "rank": "GENUS", "status": "ACCEPTED"}
        )
        assert "GBIF taxon 1:" in gbif_species("1")

    @patch(HTTP_OPEN)
    def test_occurrence_search_and_validation(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "count": 1,
                "results": [
                    {
                        "key": 10,
                        "scientificName": "Puma concolor",
                        "country": "Portugal",
                        "eventDate": "2024-01-01",
                        "decimalLatitude": 38.7,
                        "decimalLongitude": -9.1,
                    }
                ],
            }
        )

        result = gbif_occurrence_search(taxon_key="5219404", country="PT")

        assert "occurrence key: 10" in result
        assert _params(mock_urlopen)["taxonKey"] == ["5219404"]

    @pytest.mark.parametrize(
        ("call", "words"),
        [
            (lambda: gbif_occurrence_search(), "provide taxon_key"),
            (lambda: gbif_occurrence_search(country="POR"), "invalid country code"),
            (lambda: gbif_occurrence_search(year="20"), "invalid year"),
            (lambda: gbif_species("abc"), "invalid taxon_key"),
            (lambda: gbif_species_match(""), "invalid name"),
            (lambda: gbif_species_search("Puma", offset=-1), "offset must"),
            (lambda: gbif_species_search("Puma", rank="1"), "invalid rank"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_arguments_do_not_call_api(self, mock_urlopen, call, words):
        failure = _failure(call)

        assert failure.error.type == "validation_error"
        assert words in str(failure)
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_zero_results_are_successes(self, mock_urlopen):
        mock_urlopen.return_value = respond({"matchType": "NONE"})
        assert gbif_species_match("Zzzz") == "No GBIF species match found."

        mock_urlopen.return_value = respond({"count": 0, "results": []})
        assert gbif_occurrence_search(country="PT") == "No GBIF occurrences found."

    @patch(HTTP_OPEN)
    def test_a_server_error_is_a_retryable_upstream_failure(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")

        failure = _failure(lambda: gbif_species("1"))

        assert failure.error.type == "upstream"
        assert failure.error.retryable
