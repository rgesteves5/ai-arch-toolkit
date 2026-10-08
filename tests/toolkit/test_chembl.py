"""Tests for toolkit/tools/_chembl.py (T07b).

Lists page by ``limit`` and ``offset`` and count in ``page_meta``
(https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services);
a refused request says why in ``error_message`` (the web services' source,
https://github.com/chembl/chembl_webservices_py3, ``core/resource.py``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolFailure, ToolResult
from ai_arch_toolkit.toolkit.tools._chembl import (
    chembl_activity_search,
    chembl_molecule,
    chembl_molecule_search,
    chembl_target,
    chembl_target_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_MOLECULE = {
    "molecule_chembl_id": "CHEMBL25",
    "pref_name": "ASPIRIN",
    "molecule_type": "Small molecule",
    "max_phase": "4.0",
    "first_approval": 1950,
    "molecule_properties": {
        "full_mwt": "180.16",
        "alogp": "1.31",
        "hba": 3,
        "hbd": 1,
        "psa": "63.60",
        "num_ro5_violations": 0,
    },
    "molecule_structures": {
        "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "standard_inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
    },
}
_TARGET = {
    "target_chembl_id": "CHEMBL203",
    "pref_name": "Epidermal growth factor receptor erbB1",
    "target_type": "SINGLE PROTEIN",
    "organism": "Homo sapiens",
    "tax_id": 9606,
}
_ACTIVITY = {
    "molecule_chembl_id": "CHEMBL25",
    "molecule_pref_name": "ASPIRIN",
    "target_chembl_id": "CHEMBL203",
    "target_pref_name": "Epidermal growth factor receptor erbB1",
    "target_organism": "Homo sapiens",
    "standard_type": "IC50",
    "standard_relation": "=",
    "standard_value": "10.0",
    "standard_units": "nM",
    "pchembl_value": "8.00",
    "assay_chembl_id": "CHEMBL1217643",
    "assay_description": "Inhibition of EGFR",
    "document_journal": "J Med Chem",
    "document_year": 2010,
}


def _page(key: str, items: list[dict[str, Any]], *, total: int, offset: int = 0) -> Any:
    """A list answer with its ``page_meta`` (``next`` is null on the last page)."""
    more = offset + len(items) < total
    meta = {"limit": len(items), "offset": offset, "total_count": total}
    meta["next"] = f"/chembl/api/data/{key}?offset={offset + len(items)}" if more else None
    return respond({"page_meta": meta, key: items})


def _query(mock_urlopen: MagicMock) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


def _failure(call: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


class TestSearches:
    @patch(HTTP_OPEN)
    def test_a_molecule_search_pages_with_the_total(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [
            _page(
                "molecules", [_MOLECULE, {**_MOLECULE, "molecule_chembl_id": "CHEMBL2"}], total=3
            ),
            _page(
                "molecules", [{**_MOLECULE, "molecule_chembl_id": "CHEMBL3"}], total=3, offset=2
            ),
        ]

        first = _text(chembl_molecule_search("aspirin", max_results=2))
        second = _text(chembl_molecule_search("aspirin", max_results=2, offset=2))

        assert first.splitlines() == [
            "ChEMBL molecules that match 'aspirin':",
            "1. ASPIRIN | CHEMBL25 | Small molecule | max phase: 4 (approved)",
            "2. ASPIRIN | CHEMBL2 | Small molecule | max phase: 4 (approved)",
            "[results 1-2 of 3 | next: offset=2]",
        ]
        assert second.endswith("[results 3-3 of 3 | end]")
        assert _query(mock_urlopen) == {"q": ["aspirin"], "limit": ["2"], "offset": ["2"]}

    @patch(HTTP_OPEN)
    def test_a_target_search_names_type_and_organism(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _page("targets", [_TARGET], total=1)

        assert _text(chembl_target_search("EGFR")).splitlines()[1] == (
            "1. Epidermal growth factor receptor erbB1 | CHEMBL203 | SINGLE PROTEIN | Homo sapiens"
        )

    @patch(HTTP_OPEN)
    def test_an_activity_reads_with_its_labels_and_units(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _page("activities", [_ACTIVITY], total=1)

        text = _text(chembl_activity_search(molecule_chembl_id="chembl25", standard_type="IC50"))

        assert text.splitlines() == [
            "ChEMBL activities of molecule CHEMBL25, type IC50:",
            "1. CHEMBL25 (ASPIRIN) -> CHEMBL203 (Epidermal growth factor receptor erbB1, "
            "Homo sapiens) | IC50: 10.0 nM | pChEMBL 8.00 | assay CHEMBL1217643: Inhibition of "
            "EGFR | J Med Chem, 2010",
        ]
        query = _query(mock_urlopen)
        assert query["molecule_chembl_id"] == ["CHEMBL25"]
        assert query["standard_type"] == ["IC50"]

    @pytest.mark.parametrize(
        ("call", "said"),
        [
            (lambda: chembl_molecule_search("zzqq"), "No ChEMBL molecules match 'zzqq'."),
            (lambda: chembl_target_search("zzqq"), "No ChEMBL targets match 'zzqq'."),
            (
                lambda: chembl_activity_search(target_chembl_id="CHEMBL1"),
                "No ChEMBL activities of target CHEMBL1.",
            ),
        ],
    )
    @patch(HTTP_OPEN)
    def test_no_results_is_a_success_that_names_the_search(
        self, mock_urlopen: MagicMock, call: Any, said: str
    ) -> None:
        mock_urlopen.return_value = respond({"page_meta": {"total_count": 0, "next": None}})

        assert _text(call()) == said

    @pytest.mark.parametrize(
        "call",
        [
            lambda: chembl_molecule_search("aspirin"),
            lambda: chembl_target_search("EGFR"),
            lambda: chembl_activity_search(molecule_chembl_id="CHEMBL25"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_404_on_a_search_is_endpoint_not_found(self, mock_urlopen: MagicMock, call: Any):
        # It read as "no matching records found." (upstream), like an empty search.
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(call)

        assert failure.error.type == "upstream"
        assert "ChEMBL: endpoint not found (HTTP 404)" in failure.error.message

    @patch(HTTP_OPEN)
    def test_a_request_chembl_refuses_is_the_callers_to_fix_in_its_words(
        self, mock_urlopen: MagicMock
    ) -> None:
        body = b'{"error_message": "Search query too short"}'
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)

        failure = _failure(lambda: chembl_molecule_search("aspirin"))

        assert failure.error.type == "validation_error"
        assert failure.error.message == (
            "ChEMBL refused the request: Search query too short; change what it names"
        )


class TestRecords:
    @patch(HTTP_OPEN)
    def test_a_molecule_reads_with_phase_labels_and_units(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(_MOLECULE)

        assert _text(chembl_molecule("chembl25")).splitlines() == [
            "ChEMBL molecule CHEMBL25:",
            "ASPIRIN | Small molecule",
            "Max phase: 4 (approved) | first approval: 1950",
            "Properties: MW 180.16 Da | AlogP 1.31 | HBA 3 | HBD 1 | PSA 63.60 Å² | "
            "Ro5 violations 0",
            "SMILES: CC(=O)Oc1ccccc1C(=O)O",
            "InChIKey: BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        ]

    @pytest.mark.parametrize(
        ("phase", "label"),
        [("3.0", "3 (phase 3)"), ("0.5", "0.5 (early phase 1)"), ("-1.0", "-1 (unknown)")],
    )
    @patch(HTTP_OPEN)
    def test_every_phase_has_its_label(self, mock_urlopen: MagicMock, phase: str, label: str):
        # https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/drug-and-compound-questions
        mock_urlopen.return_value = respond({**_MOLECULE, "max_phase": phase})

        assert f"Max phase: {label}" in _text(chembl_molecule("CHEMBL25"))

    @patch(HTTP_OPEN)
    def test_a_molecule_without_a_phase_is_preclinical(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond({**_MOLECULE, "max_phase": None})

        assert "Max phase: none (preclinical)" in _text(chembl_molecule("CHEMBL25"))

    @patch(HTTP_OPEN)
    def test_a_target_lists_its_uniprot_accessions(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(
            {**_TARGET, "target_components": [{"accession": "P00533"}, {"accession": "Q9UHC1"}]}
        )

        assert _text(chembl_target("CHEMBL203")).splitlines() == [
            "ChEMBL target CHEMBL203:",
            "Epidermal growth factor receptor erbB1 | SINGLE PROTEIN | Homo sapiens (taxon 9606)",
            "Components (UniProt; read with uniprot_entry): P00533, Q9UHC1",
        ]

    @pytest.mark.parametrize(
        ("call", "words"),
        [
            (
                lambda: chembl_molecule("CHEMBL999999999"),
                "no ChEMBL molecule with ID CHEMBL999999999",
            ),
            (lambda: chembl_target("CHEMBL25"), "no ChEMBL target with ID CHEMBL25"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_unknown_id_is_not_found(self, mock_urlopen: MagicMock, call: Any, words: str):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(call)

        assert failure.error.type == "not_found"
        assert not failure.error.retryable
        assert words in failure.error.message
        assert "_search" in failure.error.message

    @patch(HTTP_OPEN)
    def test_upstream_failure_propagates(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = http_error(500, "Internal Server Error")

        failure = _failure(lambda: chembl_molecule("CHEMBL25"))

        assert failure.error.type == "upstream"
        assert failure.error.retryable


@pytest.mark.parametrize(
    ("call", "words"),
    [
        (lambda: chembl_activity_search(), "provide molecule_chembl_id"),
        (lambda: chembl_activity_search(target_chembl_id="X1"), "invalid target_chembl_id"),
        (lambda: chembl_molecule("bad"), "invalid chembl_id 'bad'"),
        (lambda: chembl_target("CHEMBL"), "invalid chembl_id 'CHEMBL'"),
        # ChEMBL refuses a search under 3 characters ("Search query too short").
        (lambda: chembl_molecule_search("ab"), "invalid query 'ab'"),
        (lambda: chembl_target_search("a\x00b"), "invalid query"),
        (
            lambda: chembl_activity_search(molecule_chembl_id="CHEMBL25", standard_type="a;b"),
            "invalid standard_type",
        ),
    ],
)
@patch(HTTP_OPEN)
def test_invalid_options_do_not_call_api(mock_urlopen: MagicMock, call: Any, words: str) -> None:
    failure = _failure(call)

    assert failure.error.type == "validation_error"
    assert words in failure.error.message
    mock_urlopen.assert_not_called()
