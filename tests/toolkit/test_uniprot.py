"""Tests for toolkit/tools/_uniprot.py."""

from __future__ import annotations

import json
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.toolkit.tools._uniprot import (
    uniprot_crossrefs,
    uniprot_entry,
    uniprot_features,
    uniprot_search,
    uniprot_sequence,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

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
        # The bare collection path answers with a redirect to plain HTTP on port 8080.
        url = urlparse(mock_urlopen.call_args.args[0].full_url)
        assert url.path == "/uniprotkb/search"

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


# Inactive accessions as UniProt answered them live, with HTTP 200 (2026-09-30).
_DEMERGED = {
    "entryType": "Inactive",
    "primaryAccession": "P00001",
    "uniProtkbId": "CYC_HUMAN",
    "annotationScore": 0.0,
    "inactiveReason": {"inactiveReasonType": "DEMERGED", "mergeDemergeTo": ["P99999", "P99998"]},
    "extraAttributes": {"uniParcId": "UPI0000128BBF"},
}
_DELETED = {
    "entryType": "Inactive",
    "primaryAccession": "A0A008APQ8",
    "uniProtkbId": "A0A008APQ8_STAAU",
    "annotationScore": 0.0,
    "inactiveReason": {
        "inactiveReasonType": "DELETED",
        "deletedReason": "Not part of a reference proteome",
    },
    "extraAttributes": {"uniParcId": "UPI0000054276"},
}


@pytest.mark.parametrize(
    ("fn", "failure"),
    [
        (uniprot_entry, "UniProt entry lookup failed"),
        (uniprot_features, "UniProt features failed"),
        (uniprot_crossrefs, "UniProt cross-references failed"),
    ],
)
@patch(HTTP_OPEN)
def test_an_inactive_accession_says_where_it_went(mock_urlopen, fn, failure):
    # It read as a nameless entry, or as an entry with no features or cross-references.
    mock_urlopen.return_value = respond(_DEMERGED)

    assert fn("P00001") == f"{failure}: P00001 is inactive: demerged into P99999, P99998"


@patch(HTTP_OPEN)
def test_a_deleted_accession_says_why(mock_urlopen):
    mock_urlopen.return_value = respond(_DELETED)

    assert uniprot_entry("A0A008APQ8") == (
        "UniProt entry lookup failed: A0A008APQ8 is inactive: deleted "
        "(Not part of a reference proteome)"
    )


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("nosuchfield:abc", "'nosuchfield' is not a valid search field"),
        ("insulin AND (", "query parameter has an invalid syntax"),
    ],
)
@patch(HTTP_OPEN)
def test_a_search_uniprot_refuses_says_why(mock_urlopen, query, reason):
    # As answered live (2026-09-30); it read as "HTTP error 400: Bad Request".
    body = {"url": "http://rest.uniprot.org/uniprotkb/search", "messages": [reason]}
    mock_urlopen.side_effect = http_error(400, "Bad Request", body=json.dumps(body).encode())

    assert uniprot_search(query) == f"UniProt search failed: HTTP error 400: {reason}"
