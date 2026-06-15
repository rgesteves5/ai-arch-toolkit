"""Tests for toolkit/tools/_pdb.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.toolkit.tools._pdb import (
    pdb_chemical_component,
    pdb_entry,
    pdb_ligands,
    pdb_search,
)


def _mock_urlopen(data: dict | str):
    resp = MagicMock()
    resp.read.return_value = (data if isinstance(data, str) else json.dumps(data)).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestPdb:
    @patch("ai_arch_toolkit.toolkit.tools._pdb.urllib.request.urlopen")
    def test_search(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            {"total_count": 1, "result_set": [{"identifier": "1A3N", "score": 1.0}]}
        )

        result = pdb_search("hemoglobin")

        assert "1A3N | score: 1.0" in result
        request = mock_urlopen.call_args.args[0]
        assert request.get_method() == "POST"
        assert json.loads(request.data.decode())["return_type"] == "entry"

    @patch("ai_arch_toolkit.toolkit.tools._pdb.urllib.request.urlopen")
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
        mock_urlopen.return_value = _mock_urlopen(entry)
        assert "Hemoglobin" in pdb_entry("1a3n")

        ligand = {
            "pdbx_entity_nonpoly": {"comp_id": "HEM", "name": "HEME"},
            "rcsb_nonpolymer_entity_container_identifiers": {"entity_id": "3"},
        }
        mock_urlopen.side_effect = [_mock_urlopen(entry), _mock_urlopen(ligand)]
        assert "HEM — HEME" in pdb_ligands("1A3N")

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = _mock_urlopen(
            {
                "chem_comp": {"name": "HEME", "type": "non-polymer", "formula": "C34 H32"},
                "rcsb_chem_comp_descriptor": {"SMILES": "C1=CC"},
            }
        )
        assert "RCSB chemical component HEM:" in pdb_chemical_component("hem")

    @patch("ai_arch_toolkit.toolkit.tools._pdb.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "invalid pdb_id" in pdb_entry("bad")
        assert "invalid component_id" in pdb_chemical_component("bad/id")
        mock_urlopen.assert_not_called()
