"""Tests for toolkit/tools/_pdb.py."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._pdb import (
    pdb_chemical_component,
    pdb_entry,
    pdb_ligands,
    pdb_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


class TestPdb:
    @patch(HTTP_OPEN)
    def test_search(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"total_count": 1, "result_set": [{"identifier": "1A3N", "score": 1.0}]}
        )

        result = pdb_search("hemoglobin")

        assert "1A3N | score: 1.0" in result
        request = mock_urlopen.call_args.args[0]
        assert request.get_method() == "POST"
        payload = json.loads(request.data.decode())
        assert payload["return_type"] == "entry"
        # RCSB answers 400 to a "text" query without an attribute.
        assert payload["query"]["service"] == "full_text"

    @patch(HTTP_OPEN)
    def test_entry_ligands_and_component(self, mock_urlopen):
        entry = {
            "struct": {"title": "Hemoglobin"},
            "rcsb_entry_info": {
                "experimental_method": ["X-ray"],
                "resolution_combined": [2.0],
                "polymer_entity_count": 2,
            },
            "rcsb_entry_container_identifiers": {
                "polymer_entity_ids": ["1"],
                "non_polymer_entity_ids": ["3"],
            },
        }
        mock_urlopen.return_value = respond(entry)
        assert "Hemoglobin" in pdb_entry("1a3n")

        ligand = {
            "pdbx_entity_nonpoly": {"comp_id": "HEM", "name": "HEME"},
            "rcsb_nonpolymer_entity_container_identifiers": {"entity_id": "3"},
        }
        mock_urlopen.side_effect = [respond(entry), respond(ligand)]
        assert "HEM — HEME" in pdb_ligands("1A3N")

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = respond(
            {
                "chem_comp": {"name": "HEME", "type": "non-polymer", "formula": "C34 H32"},
                "rcsb_chem_comp_descriptor": {"SMILES": "C1=CC"},
            }
        )
        assert "RCSB chemical component HEM:" in pdb_chemical_component("hem")

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        for call, arg, words in (
            (pdb_entry, "bad", "invalid pdb_id"),
            (pdb_ligands, "bad", "invalid pdb_id"),
            (pdb_chemical_component, "bad/id", "invalid component_id"),
            (pdb_search, "", "invalid query"),
        ):
            with pytest.raises(ToolFailure) as caught:
                call(arg)
            assert caught.value.error.type == "validation_error"
            assert words in caught.value.error.message
        with pytest.raises(ToolFailure) as caught:
            pdb_search("hemoglobin", start=-1)
        assert caught.value.error.type == "validation_error"
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_failed_request_raises_the_sources_error(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")

        with pytest.raises(ToolFailure) as caught:
            pdb_entry("1A3N")

        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable


@patch(HTTP_OPEN)
def test_a_search_without_hits_is_no_entries_not_a_parse_error(mock_urlopen):
    # RCSB answers it 204 No Content with an empty body (2026-09-30).
    mock_urlopen.return_value = respond(b"", content_type="application/json", status=204)

    assert pdb_search("zzqqxxyyvvww") == "No RCSB PDB entries found for 'zzqqxxyyvvww'."


def _failure(fn, *args) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args)
    return caught.value


_NO_ENTRY = b'{"status": 404, "message": "No data found for entry with ID: 9ZZZ"}'


@pytest.mark.parametrize("tool", [pdb_entry, pdb_ligands])
@patch(HTTP_OPEN)
def test_an_unknown_entry_is_not_found(mock_urlopen, tool):
    mock_urlopen.side_effect = http_error(404, "Not Found", body=_NO_ENTRY)

    failure = _failure(tool, "9zzz")

    assert failure.error.type == "not_found"
    assert failure.error.message == "RCSB PDB has no entry 9ZZZ; find entries with pdb_search"


@patch(HTTP_OPEN)
def test_a_listed_ligand_without_a_record_is_not_found(mock_urlopen):
    entry = {"rcsb_entry_container_identifiers": {"non_polymer_entity_ids": ["3"]}}
    mock_urlopen.side_effect = [respond(entry), http_error(404, "Not Found")]

    failure = _failure(pdb_ligands, "1A3N")

    assert failure.error.type == "not_found"
    assert "lists nonpolymer entity 3 but has no record of it" in failure.error.message


@patch(HTTP_OPEN)
def test_an_unknown_component_is_not_found(mock_urlopen):
    mock_urlopen.side_effect = http_error(404, "Not Found")

    failure = _failure(pdb_chemical_component, "zzz")

    assert failure.error.type == "not_found"
    assert failure.error.message == (
        "RCSB PDB has no chemical component ZZZ; pdb_ligands lists the component IDs of an entry"
    )


@patch(HTTP_OPEN)
def test_a_404_on_the_search_is_an_endpoint_not_found(mock_urlopen):
    mock_urlopen.side_effect = http_error(404, "Not Found")

    failure = _failure(pdb_search, "hemoglobin")

    assert failure.error.type == "upstream"
    assert "RCSB PDB: endpoint not found (HTTP 404)" in failure.error.message


@patch(HTTP_OPEN)
def test_a_bad_search_carries_rcsb_s_words(mock_urlopen):
    body = b'{"status": 400, "message": "JSON schema validation failed for query"}'
    mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)

    failure = _failure(pdb_search, "hemoglobin")

    assert failure.error.type == "upstream"
    assert "JSON schema validation failed for query" in failure.error.message
