"""Tests for toolkit/tools/_pdb.py (T07b).

The Search API answers identifiers (https://search.rcsb.org/); what each entry is comes from the
Data API's GraphQL endpoint, for a whole page in one request, which answers its errors in a 200
(https://data.rcsb.org/index.html#gql-api).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest

from ai_arch_toolkit.core import ToolFailure, ToolResult
from ai_arch_toolkit.toolkit.tools._pdb import (
    pdb_chemical_component,
    pdb_entry,
    pdb_ligands,
    pdb_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _hits(*ids: str, total: int) -> dict[str, Any]:
    """A Search API answer: the page's identifiers and the total."""
    return {
        "query_id": "q",
        "result_type": "entry",
        "total_count": total,
        "result_set": [{"identifier": entry_id, "score": 1.0} for entry_id in ids],
    }


def _entries(*ids: str) -> dict[str, Any]:
    """The GraphQL answer for those entries' titles, methods, resolutions and dates."""
    return {
        "data": {
            "entries": [
                {
                    "rcsb_id": entry_id,
                    "struct": {"title": f"Structure of {entry_id}"},
                    "exptl": [{"method": "X-RAY DIFFRACTION"}],
                    "rcsb_entry_info": {"resolution_combined": [1.74]},
                    "rcsb_accession_info": {"initial_release_date": "1984-07-17T00:00:00Z"},
                }
                for entry_id in ids
            ]
        }
    }


def _sent(mock_urlopen: MagicMock, call: int) -> tuple[str, dict[str, Any]]:
    request = mock_urlopen.call_args_list[call].args[0]
    return request.full_url, json.loads(request.data.decode())


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


def _failure(fn: Any, *args: Any, **kwargs: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


class TestSearch:
    @patch(HTTP_OPEN)
    def test_each_hit_comes_with_its_title_from_one_more_request(
        self, mock_urlopen: MagicMock
    ) -> None:
        # Only identifiers came back: each title cost another call.
        mock_urlopen.side_effect = [
            respond(_hits("4HHB", "1A3N", total=40)),
            respond(_entries("4HHB", "1A3N")),
        ]

        text = _text(pdb_search("hemoglobin", max_results=2))

        assert text.splitlines() == [
            "RCSB PDB entries that match 'hemoglobin', the most relevant first:",
            "1. 4HHB: Structure of 4HHB | X-RAY DIFFRACTION | 1.74 Å | released 1984-07-17",
            "2. 1A3N: Structure of 1A3N | X-RAY DIFFRACTION | 1.74 Å | released 1984-07-17",
            "[results 1-2 of 40 | next: start=2]",
        ]
        url, search = _sent(mock_urlopen, 0)
        assert url == "https://search.rcsb.org/rcsbsearch/v2/query"
        assert search["query"]["service"] == "full_text"  # "text" needs an attribute
        assert search["request_options"]["paginate"] == {"start": 0, "rows": 2}
        url, graphql = _sent(mock_urlopen, 1)
        assert url == "https://data.rcsb.org/graphql"
        assert graphql["variables"] == {"ids": ["4HHB", "1A3N"]}
        assert "entries(entry_ids: $ids)" in graphql["query"]
        assert mock_urlopen.call_count == 2

    @patch(HTTP_OPEN)
    def test_the_next_page_is_numbered_on(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [
            respond(_hits("2HHB", total=3)),
            respond(_entries("2HHB")),
        ]

        text = _text(pdb_search("hemoglobin", max_results=2, start=2))

        assert "3. 2HHB: Structure of 2HHB" in text
        assert text.endswith("[results 3-3 of 3 | end]")
        assert _sent(mock_urlopen, 0)[1]["request_options"]["paginate"]["start"] == 2

    @patch(HTTP_OPEN)
    def test_an_nmr_entry_has_no_resolution_and_a_missing_record_says_so(
        self, mock_urlopen: MagicMock
    ) -> None:
        nmr = {
            "rcsb_id": "1D3Z",
            "struct": {"title": "Ubiquitin"},
            "exptl": [{"method": "SOLUTION NMR"}],
        }
        mock_urlopen.side_effect = [
            respond(_hits("1D3Z", "9ZZZ", total=2)),
            respond({"data": {"entries": [nmr, None]}}),
        ]

        lines = _text(pdb_search("ubiquitin")).splitlines()

        assert lines[1] == "1. 1D3Z: Ubiquitin | SOLUTION NMR"
        assert lines[2] == "2. 9ZZZ: (no record in the Data API; try pdb_entry)"

    @pytest.mark.parametrize("info", ["n/a", [{"resolution_combined": [1.74]}]])
    @patch(HTTP_OPEN)
    def test_a_record_of_another_shape_reads_without_what_it_lacks(
        self, mock_urlopen: MagicMock, info: object
    ) -> None:
        # The lines were built after the parse, and rcsb_entry_info as text or a list raised a
        # bare AttributeError out of the tool.
        odd = {**_entries("4HHB")["data"]["entries"][0], "rcsb_entry_info": info}
        mock_urlopen.side_effect = [
            respond(_hits("4HHB", total=1)),
            respond({"data": {"entries": [odd]}}),
        ]

        assert _text(pdb_search("hemoglobin")).splitlines()[1] == (
            "1. 4HHB: Structure of 4HHB | X-RAY DIFFRACTION | released 1984-07-17"
        )

    @patch(HTTP_OPEN)
    def test_records_the_parse_cannot_read_are_a_typed_failure(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.side_effect = [
            respond(_hits("4HHB", total=1)),
            respond({"data": ["not", "an", "object"]}),
        ]

        failure = _failure(pdb_search, "hemoglobin")

        assert failure.error.type == "upstream"

    @patch(HTTP_OPEN)
    def test_a_page_past_the_hits_does_not_claim_a_total_of_zero(
        self, mock_urlopen: MagicMock
    ) -> None:
        # A 204 has no total_count: it is not known, and the footer said "of 0".
        mock_urlopen.return_value = respond(b"", content_type="application/json", status=204)

        assert _text(pdb_search("hemoglobin", start=10)).endswith("[no results from 11 | end]")

    @patch(HTTP_OPEN)
    def test_a_search_without_hits_is_a_success_that_names_the_query(
        self, mock_urlopen: MagicMock
    ) -> None:
        # RCSB answers it 204 No Content with an empty body (https://search.rcsb.org/, Empty
        # results); it read as a parse failure before 6a34668.
        mock_urlopen.return_value = respond(b"", content_type="application/json", status=204)

        assert _text(pdb_search("zzqqxxyyvvww")) == ("No RCSB PDB entries match 'zzqqxxyyvvww'.")
        assert mock_urlopen.call_count == 1

    @patch(HTTP_OPEN)
    def test_a_search_rcsb_refuses_is_the_callers_to_fix_in_its_words(
        self, mock_urlopen: MagicMock
    ) -> None:
        body = b'{"status": 400, "message": "JSON schema validation failed for query"}'
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)

        failure = _failure(pdb_search, "hemoglobin")

        assert failure.error.type == "validation_error"
        assert failure.error.message == (
            "RCSB PDB refused the request: JSON schema validation failed for query; "
            "rephrase the query in plain words"
        )

    @pytest.mark.parametrize(
        ("tool", "first"),
        [
            (pdb_search, _hits("4HHB", total=1)),
            (pdb_ligands, {"rcsb_entry_container_identifiers": {"non_polymer_entity_ids": ["3"]}}),
        ],
    )
    @patch(HTTP_OPEN)
    def test_an_error_the_graphql_endpoint_reports_in_a_200_is_a_failure(
        self, mock_urlopen: MagicMock, tool: Any, first: dict[str, Any]
    ) -> None:
        # GraphQL answers its errors in a 200 (https://data.rcsb.org/index.html#gql-api). The
        # message named the "Data API" on the search too, and gave no next step.
        mock_urlopen.side_effect = [
            respond(first),
            respond({"errors": [{"message": "Field 'x' in type 'CoreEntry' is undefined"}]}),
        ]

        failure = _failure(tool, "4HHB")

        assert failure.error.type == "upstream"
        assert failure.error.message == (
            "RCSB PDB's GraphQL endpoint said: Field 'x' in type 'CoreEntry' is undefined; "
            "pdb_entry reads an entry without it, or try again later"
        )

    @patch(HTTP_OPEN)
    def test_the_search_api_does_not_read_graphqls_errors(self, mock_urlopen: MagicMock) -> None:
        # Only the GraphQL endpoint answers errors in a 200; the search's own answer is its hits.
        mock_urlopen.side_effect = [
            respond({**_hits("4HHB", total=1), "errors": [{"message": "not the search's"}]}),
            respond(_entries("4HHB")),
        ]

        assert "1. 4HHB: Structure of 4HHB" in _text(pdb_search("hemoglobin"))

    @patch(HTTP_OPEN)
    def test_a_404_on_the_search_is_an_endpoint_not_found(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(pdb_search, "hemoglobin")

        assert failure.error.type == "upstream"
        assert "RCSB PDB: endpoint not found (HTTP 404)" in failure.error.message


class TestEntry:
    @patch(HTTP_OPEN)
    def test_an_entry_reads_with_units_dates_and_ids_other_tools_take(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = respond(
            {
                "struct": {"title": "Hemoglobin"},
                "exptl": [{"method": "X-RAY DIFFRACTION"}],
                "rcsb_entry_info": {
                    "resolution_combined": [1.74],
                    "polymer_entity_count": 2,
                    "nonpolymer_entity_count": 2,
                },
                "rcsb_accession_info": {
                    "deposit_date": "1984-03-07T00:00:00Z",
                    "initial_release_date": "1984-07-17T00:00:00Z",
                },
                "rcsb_primary_citation": {
                    "title": "The crystal structure of human deoxyhaemoglobin",
                    "rcsb_journal_abbrev": "J Mol Biol",
                    "year": 1984,
                    "pdbx_database_id_DOI": "10.1016/0022-2836(84)90472-8",
                    "pdbx_database_id_PubMed": 6726807,
                },
            }
        )

        assert _text(pdb_entry("4hhb")).splitlines() == [
            "RCSB PDB entry 4HHB:",
            "Hemoglobin",
            "Method: X-RAY DIFFRACTION | resolution: 1.74 Å",
            "Deposited: 1984-03-07 | released: 1984-07-17",
            "Entities: 2 polymer, 2 non-polymer (list them with pdb_ligands)",
            "Citation: The crystal structure of human deoxyhaemoglobin. J Mol Biol, 1984. "
            "DOI 10.1016/0022-2836(84)90472-8, PubMed 6726807",
        ]

    @patch(HTTP_OPEN)
    def test_a_request_the_data_api_refuses_names_a_step_that_fits(
        self, mock_urlopen: MagicMock
    ) -> None:
        # The search's step ("rephrase the query") came with an entry's 400 too.
        body = b'{"status": 400, "message": "Invalid entry ID"}'
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)

        failure = _failure(pdb_entry, "4HHB")

        assert failure.error.type == "validation_error"
        assert failure.error.message == (
            "RCSB PDB refused the request: Invalid entry ID; check the ID, or find one with "
            "pdb_search"
        )

    @pytest.mark.parametrize("tool", [pdb_entry, pdb_ligands])
    @patch(HTTP_OPEN)
    def test_an_unknown_entry_is_not_found(self, mock_urlopen: MagicMock, tool: Any) -> None:
        body = b'{"status": 404, "message": "No data found for entry with ID: 9ZZZ"}'
        mock_urlopen.side_effect = http_error(404, "Not Found", body=body)

        failure = _failure(tool, "9zzz")

        assert failure.error.type == "not_found"
        assert failure.error.message == "RCSB PDB has no entry 9ZZZ; find entries with pdb_search"


class TestLigands:
    @patch(HTTP_OPEN)
    def test_every_ligand_comes_from_one_more_request(self, mock_urlopen: MagicMock) -> None:
        # One request per ligand, for as many ligands as the entry has.
        entry = {"rcsb_entry_container_identifiers": {"non_polymer_entity_ids": ["3", "4"]}}
        ligands = {
            "data": {
                "nonpolymer_entities": [
                    {
                        "rcsb_id": "4HHB_3",
                        "pdbx_entity_nonpoly": {"comp_id": "PO4", "name": "PHOSPHATE ION"},
                        "rcsb_nonpolymer_entity": {"pdbx_number_of_molecules": 2},
                    },
                    {
                        "rcsb_id": "4HHB_4",
                        "pdbx_entity_nonpoly": {"comp_id": "HEM", "name": "HEME"},
                        "rcsb_nonpolymer_entity": {"pdbx_number_of_molecules": 4},
                    },
                ]
            }
        }
        mock_urlopen.side_effect = [respond(entry), respond(ligands)]

        assert _text(pdb_ligands("4HHB")).splitlines() == [
            "Ligands of RCSB PDB entry 4HHB (read one with pdb_chemical_component):",
            "1. PO4: PHOSPHATE ION | entity 3 | 2 copies",
            "2. HEM: HEME | entity 4 | 4 copies",
        ]
        assert _sent(mock_urlopen, 1)[1]["variables"] == {"ids": ["4HHB_3", "4HHB_4"]}
        assert mock_urlopen.call_count == 2

    @patch(HTTP_OPEN)
    def test_an_entry_without_ligands_says_so_in_one_request(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = respond({"rcsb_entry_container_identifiers": {}})

        assert _text(pdb_ligands("1A3N")) == "RCSB PDB entry 1A3N has no ligands."
        assert mock_urlopen.call_count == 1

    @patch(HTTP_OPEN)
    def test_a_listed_ligand_without_a_record_says_so(self, mock_urlopen: MagicMock) -> None:
        entry = {"rcsb_entry_container_identifiers": {"non_polymer_entity_ids": ["3"]}}
        mock_urlopen.side_effect = [
            respond(entry),
            respond({"data": {"nonpolymer_entities": [None]}}),
        ]

        assert _text(pdb_ligands("1A3N")).splitlines()[1] == (
            "1. entity 3: no record in the Data API"
        )


class TestComponent:
    @patch(HTTP_OPEN)
    def test_a_component_reads_with_its_weight_in_daltons(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(
            {
                "chem_comp": {
                    "name": "PROTOPORPHYRIN IX CONTAINING FE",
                    "type": "NON-POLYMER",
                    "formula": "C34 H32 Fe N4 O4",
                    "formula_weight": 616.487,
                },
                "rcsb_chem_comp_descriptor": {
                    "SMILES": "C1=CC",
                    "InChIKey": "KABFMIBPWCXCRK-RGGAHWMASA-L",
                },
            }
        )

        assert _text(pdb_chemical_component("hem")).splitlines() == [
            "RCSB chemical component HEM:",
            "PROTOPORPHYRIN IX CONTAINING FE",
            "Type: NON-POLYMER | formula: C34 H32 Fe N4 O4 | weight: 616.487 Da",
            "SMILES: C1=CC",
            "InChIKey: KABFMIBPWCXCRK-RGGAHWMASA-L",
        ]
        assert urlparse(mock_urlopen.call_args.args[0].full_url).path == (
            "/rest/v1/core/chemcomp/HEM"
        )

    @patch(HTTP_OPEN)
    def test_an_unknown_component_is_not_found(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(pdb_chemical_component, "zzz")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "RCSB PDB has no chemical component ZZZ; pdb_ligands lists the component IDs of an "
            "entry"
        )


@pytest.mark.parametrize(
    ("call", "words"),
    [
        (lambda: pdb_entry("bad"), "invalid pdb_id"),
        (lambda: pdb_ligands("bad"), "invalid pdb_id"),
        (lambda: pdb_chemical_component("bad/id"), "invalid component_id"),
        (lambda: pdb_search(" "), "query cannot be empty"),
        (lambda: pdb_search("a\x00b"), "invalid query"),
    ],
)
@patch(HTTP_OPEN)
def test_invalid_arguments_do_not_call_the_api(mock_urlopen: MagicMock, call: Any, words: str):
    with pytest.raises(ToolFailure) as caught:
        call()

    assert caught.value.error.type == "validation_error"
    assert words in caught.value.error.message
    mock_urlopen.assert_not_called()


@patch(HTTP_OPEN)
def test_a_failed_request_raises_the_sources_error(mock_urlopen: MagicMock) -> None:
    mock_urlopen.side_effect = http_error(503, "Service Unavailable")

    failure = _failure(pdb_entry, "1A3N")

    assert failure.error.type == "upstream"
    assert failure.error.retryable
