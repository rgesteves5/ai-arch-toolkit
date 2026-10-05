"""Tests for toolkit/tools/_uniprot.py."""

from __future__ import annotations

import json
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
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

    @pytest.mark.parametrize(
        ("call", "words"),
        [
            (lambda: uniprot_entry("bad/id"), "invalid accession 'bad/id'"),
            (lambda: uniprot_sequence("x"), "invalid accession 'x'"),
            (lambda: uniprot_search("insulin", reviewed="maybe"), "invalid reviewed 'maybe'"),
            (lambda: uniprot_search("insulin", offset=-1), "invalid offset -1"),
            (lambda: uniprot_search("a;b"), "invalid query"),
            (lambda: uniprot_features("P01308", feature_type="a;b"), "invalid feature_type"),
            (lambda: uniprot_crossrefs("P01308", database="a;b"), "invalid database"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen, call, words):
        with pytest.raises(ToolFailure) as caught:
            call()

        assert caught.value.error.type == "validation_error"
        assert words in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_no_results_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond({"totalResults": 0, "results": []})

        assert uniprot_search("zzzz") == "No UniProt proteins found."

    @patch(HTTP_OPEN)
    def test_a_missing_accession_is_an_upstream_failure(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        with pytest.raises(ToolFailure) as caught:
            uniprot_entry("Q00000")

        assert caught.value.error.type == "upstream"
        assert "no matching records found" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_rate_limiting_is_typed(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        with pytest.raises(ToolFailure) as caught:
            uniprot_search("insulin")

        assert caught.value.error.type == "rate_limited"
        assert caught.value.error.retryable


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


@pytest.mark.parametrize("fn", [uniprot_entry, uniprot_features, uniprot_crossrefs])
@patch(HTTP_OPEN)
def test_an_inactive_accession_says_where_it_went(mock_urlopen, fn):
    # It read as a nameless entry, or as an entry with no features or cross-references.
    mock_urlopen.return_value = respond(_DEMERGED)

    with pytest.raises(ToolFailure) as caught:
        fn("P00001")

    assert caught.value.error.type == "upstream"
    assert caught.value.error.message == "P00001 is inactive: demerged into P99999, P99998"


@patch(HTTP_OPEN)
def test_a_deleted_accession_says_why(mock_urlopen):
    mock_urlopen.return_value = respond(_DELETED)

    with pytest.raises(ToolFailure) as caught:
        uniprot_entry("A0A008APQ8")

    assert caught.value.error.type == "upstream"
    assert caught.value.error.message == (
        "A0A008APQ8 is inactive: deleted (Not part of a reference proteome)"
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

    with pytest.raises(ToolFailure) as caught:
        uniprot_search(query)

    assert caught.value.error.type == "upstream"
    assert caught.value.error.message == f"HTTP error 400: {reason}"


@patch(HTTP_OPEN)
def test_an_empty_sequence_is_not_found(mock_urlopen):
    mock_urlopen.return_value = respond("", content_type="text/plain")

    with pytest.raises(ToolFailure) as caught:
        uniprot_sequence("P01308")

    assert caught.value.error.type == "not_found"
    assert "uniprot_entry" in caught.value.error.message
