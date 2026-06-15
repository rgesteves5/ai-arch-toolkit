"""Tests for toolkit/tools/_chembl.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._chembl import (
    chembl_activity_search,
    chembl_molecule,
    chembl_molecule_search,
    chembl_target,
    chembl_target_search,
)

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


def _mock_urlopen(data: dict | str):
    resp = MagicMock()
    resp.read.return_value = (data if isinstance(data, str) else json.dumps(data)).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestChembl:
    @patch("ai_arch_toolkit.toolkit.tools._chembl.urllib.request.urlopen")
    def test_molecule_tools(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            {"page_meta": {"total_count": 1}, "molecules": [_MOLECULE]}
        )
        assert "ASPIRIN | id: CHEMBL25" in chembl_molecule_search("aspirin")

        mock_urlopen.return_value = _mock_urlopen(_MOLECULE)
        result = chembl_molecule("chembl25")
        assert "SMILES:" in result
        assert "MW: 180.16" in result

    @patch("ai_arch_toolkit.toolkit.tools._chembl.urllib.request.urlopen")
    def test_target_and_activity_tools(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            {"page_meta": {"total_count": 1}, "targets": [_TARGET]}
        )
        assert "Cyclooxygenase | id: CHEMBL2094253" in chembl_target_search("COX")

        mock_urlopen.return_value = _mock_urlopen(
            {**_TARGET, "target_components": [{"accession": "P23219"}]}
        )
        assert "components: P23219" in chembl_target("CHEMBL2094253")

        mock_urlopen.return_value = _mock_urlopen(
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

    @patch("ai_arch_toolkit.toolkit.tools._chembl.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "provide molecule" in chembl_activity_search()
        assert "invalid chembl_id" in chembl_molecule("bad")
        mock_urlopen.assert_not_called()
