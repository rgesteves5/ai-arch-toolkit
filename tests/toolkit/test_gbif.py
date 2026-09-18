"""Tests for toolkit/tools/_gbif.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._gbif import (
    gbif_occurrence_search,
    gbif_species,
    gbif_species_match,
    gbif_species_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, respond


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
        assert "provide taxon_key" in gbif_occurrence_search()
