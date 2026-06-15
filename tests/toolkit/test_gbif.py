"""Tests for toolkit/tools/_gbif.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._gbif import (
    gbif_occurrence_search,
    gbif_species,
    gbif_species_match,
    gbif_species_search,
)


def _mock_urlopen(data: dict | str):
    resp = MagicMock()
    resp.read.return_value = (data if isinstance(data, str) else json.dumps(data)).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestGbif:
    @patch("ai_arch_toolkit.toolkit.tools._gbif.urllib.request.urlopen")
    def test_species_match(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
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

    @patch("ai_arch_toolkit.toolkit.tools._gbif.urllib.request.urlopen")
    def test_species_search_and_lookup(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            {"count": 1, "results": [{"key": 1, "scientificName": "Puma", "rank": "GENUS"}]}
        )
        assert "Puma | key: 1" in gbif_species_search("Puma", rank="GENUS")

        mock_urlopen.return_value = _mock_urlopen(
            {"key": 1, "scientificName": "Puma", "rank": "GENUS", "status": "ACCEPTED"}
        )
        assert "GBIF taxon 1:" in gbif_species("1")

    @patch("ai_arch_toolkit.toolkit.tools._gbif.urllib.request.urlopen")
    def test_occurrence_search_and_validation(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
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
