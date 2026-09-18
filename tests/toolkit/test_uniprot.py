"""Tests for toolkit/tools/_uniprot.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._uniprot import (
    uniprot_crossrefs,
    uniprot_entry,
    uniprot_features,
    uniprot_search,
    uniprot_sequence,
)
from tests.toolkit.http_fakes import HTTP_OPEN, respond

_ENTRY = {
    "primaryAccession": "P01308",
    "entryType": "UniProtKB reviewed (Swiss-Prot)",
    "proteinDescription": {"recommendedName": {"fullName": {"value": "Insulin"}}},
    "organism": {"scientificName": "Homo sapiens"},
    "sequence": {"length": 110},
    "genes": [{"geneName": {"value": "INS"}}],
    "comments": [
        {"commentType": "FUNCTION", "texts": [{"value": "Insulin decreases blood glucose."}]}
    ],
    "features": [
        {
            "type": "Chain",
            "description": "Insulin B chain",
            "location": {"start": {"value": 25}, "end": {"value": 54}},
        }
    ],
    "uniProtKBCrossReferences": [
        {"database": "PDB", "id": "1TRZ", "properties": [{"key": "Method", "value": "X-ray"}]}
    ],
}


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestUniProt:
    @patch(HTTP_OPEN)
    def test_search_and_entry(self, mock_urlopen):
        mock_urlopen.return_value = respond({"totalResults": 1, "results": [_ENTRY]})

        result = uniprot_search("insulin", reviewed="true")

        assert "Insulin | accession: P01308" in result
        assert "reviewed:true" in _params(mock_urlopen)["query"][0]

        mock_urlopen.return_value = respond(_ENTRY)
        assert "Function: Insulin decreases blood glucose." in uniprot_entry("P01308")

    @patch(HTTP_OPEN)
    def test_features_sequence_and_crossrefs(self, mock_urlopen):
        mock_urlopen.return_value = respond(_ENTRY)
        assert "Chain | 25-54" in uniprot_features("P01308")

        mock_urlopen.return_value = respond(">sp|P01308|INS_HUMAN Insulin\nMALWMRLLPLL")
        assert uniprot_sequence("P01308").startswith(">sp|P01308")

        mock_urlopen.return_value = respond(_ENTRY)
        result = uniprot_crossrefs("P01308", database="PDB")
        assert "PDB: 1TRZ" in result
        assert "Method: X-ray" in result

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "invalid accession" in uniprot_entry("bad/id")
        assert "reviewed must" in uniprot_search("insulin", reviewed="maybe")
        mock_urlopen.assert_not_called()
