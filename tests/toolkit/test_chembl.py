"""Tests for toolkit/tools/_chembl.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
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
    "max_phase": 4,
    "first_approval": 1950,
    "molecule_properties": {"full_mwt": "180.16", "alogp": "1.31", "hba": "3", "hbd": "1"},
    "molecule_structures": {"canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
}
_TARGET = {
    "target_chembl_id": "CHEMBL2094253",
    "pref_name": "Cyclooxygenase",
    "target_type": "PROTEIN FAMILY",
    "organism": "Homo sapiens",
    "tax_id": 9606,
}


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestChembl:
    @patch(HTTP_OPEN)
    def test_molecule_tools(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"page_meta": {"total_count": 1}, "molecules": [_MOLECULE]}
        )
        assert "ASPIRIN | id: CHEMBL25" in chembl_molecule_search("aspirin")

        mock_urlopen.return_value = respond(_MOLECULE)
        result = chembl_molecule("chembl25")
        assert "SMILES:" in result
        assert "MW: 180.16" in result

    @patch(HTTP_OPEN)
    def test_target_and_activity_tools(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"page_meta": {"total_count": 1}, "targets": [_TARGET]}
        )
        assert "Cyclooxygenase | id: CHEMBL2094253" in chembl_target_search("COX")

        mock_urlopen.return_value = respond(
            {**_TARGET, "target_components": [{"accession": "P23219"}]}
        )
        assert "components: P23219" in chembl_target("CHEMBL2094253")

        mock_urlopen.return_value = respond(
            {
                "page_meta": {"total_count": 1},
                "activities": [
                    {
                        "molecule_chembl_id": "CHEMBL25",
                        "target_chembl_id": "CHEMBL2094253",
                        "standard_type": "IC50",
                        "standard_value": "10",
                        "standard_units": "nM",
                        "assay_chembl_id": "CHEMBL1",
                    }
                ],
            }
        )
        result = chembl_activity_search(molecule_chembl_id="CHEMBL25", standard_type="IC50")
        assert "IC50: 10 nM" in result
        assert _params(mock_urlopen)["standard_type"] == ["IC50"]

    @pytest.mark.parametrize(
        ("call", "words"),
        [
            (lambda: chembl_activity_search(), "provide molecule_chembl_id"),
            (lambda: chembl_activity_search(target_chembl_id="X1"), "invalid target_chembl_id"),
            (
                lambda: chembl_activity_search(molecule_chembl_id="CHEMBL25", offset=-1),
                "offset must be greater than or equal to 0",
            ),
            (lambda: chembl_molecule("bad"), "invalid chembl_id 'bad'"),
            (lambda: chembl_target("CHEMBL"), "invalid chembl_id 'CHEMBL'"),
            (lambda: chembl_molecule_search("<script>"), "invalid query"),
            (lambda: chembl_target_search("EGFR", offset=-1), "offset must be"),
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
    def test_upstream_failure_propagates(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(500, "Internal Server Error")

        with pytest.raises(ToolFailure) as caught:
            chembl_molecule("CHEMBL25")

        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable
