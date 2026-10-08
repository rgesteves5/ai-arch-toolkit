"""Answers of UniProt, RCSB PDB, ChEMBL and GBIF, shaped as each documents them, for the
contract cases (``contract_cases.py``, T07b).

- UniProt pages a search by the cursor in its ``Link`` header and counts in ``x-total-results``
  (https://www.uniprot.org/help/pagination).
- RCSB's Search API answers identifiers and ``total_count``; the Data API's GraphQL endpoint what
  they are (https://search.rcsb.org/, https://data.rcsb.org/index.html#gql-api).
- ChEMBL counts in ``page_meta`` (https://chembl.gitbook.io/chembl-interface-documentation/).
- GBIF counts in ``count`` and says ``endOfRecords`` (https://techdocs.gbif.org/en/openapi/).
"""

from __future__ import annotations

from typing import Any


def uniprot_entry(accession: str = "P69905", *, items: int = 12) -> dict[str, Any]:
    """An entry with ``items`` features and cross-references, and enough annotation to need
    several windows."""
    return {
        "primaryAccession": accession,
        "uniProtkbId": "HBA_HUMAN",
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Hemoglobin alpha"}}},
        "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
        "comments": [
            {"commentType": "FUNCTION", "texts": [{"value": f"Carries oxygen, part {n}."}]}
            for n in range(60)
        ],
        "features": [
            {
                "type": "Helix",
                "location": {"start": {"value": n}, "end": {"value": n + 5}},
            }
            for n in range(1, items + 1)
        ],
        "uniProtKBCrossReferences": [
            {"database": "PDB", "id": f"{n}HHB", "properties": []} for n in range(1, items + 1)
        ],
    }


def uniprot_hits(*accessions: str) -> dict[str, Any]:
    return {"results": [uniprot_entry(accession) for accession in accessions]}


def uniprot_headers(total: int, cursor: str = "") -> dict[str, str]:
    headers = {"x-total-results": str(total)}
    if cursor:
        headers["Link"] = (
            "<https://rest.uniprot.org/uniprotkb/search?query=insulin"
            f'&cursor={cursor}&size=2>; rel="next"'
        )
    return headers


def pdb_hits(*ids: str, total: int) -> dict[str, Any]:
    return {
        "query_id": "q",
        "result_type": "entry",
        "total_count": total,
        "result_set": [{"identifier": entry_id, "score": 1.0} for entry_id in ids],
    }


def pdb_entries(*ids: str) -> dict[str, Any]:
    return {
        "data": {
            "entries": [
                {"rcsb_id": entry_id, "struct": {"title": f"Structure {entry_id}"}}
                for entry_id in ids
            ]
        }
    }


CHEMBL_ITEMS: dict[str, dict[str, Any]] = {
    "molecules": {"molecule_chembl_id": "CHEMBL25", "pref_name": "ASPIRIN"},
    "targets": {"target_chembl_id": "CHEMBL203", "pref_name": "EGFR"},
    "activities": {
        "molecule_chembl_id": "CHEMBL25",
        "target_chembl_id": "CHEMBL204",
        "standard_type": "IC50",
        "standard_value": "10",
        "standard_units": "nM",
    },
}


def chembl_page(key: str, *, count: int, offset: int, total: int) -> dict[str, Any]:
    """A page of ``count`` items of a ChEMBL list, with its ``page_meta``."""
    more = offset + count < total
    return {
        "page_meta": {
            "limit": count,
            "offset": offset,
            "total_count": total,
            "next": f"/chembl/api/data/{key}?offset={offset + count}" if more else None,
        },
        key: [CHEMBL_ITEMS[key]] * count,
    }


GBIF_ITEMS: dict[str, dict[str, Any]] = {
    "taxa": {"key": 2435099, "scientificName": "Puma concolor", "rank": "SPECIES"},
    "occurrences": {"key": 4011775102, "scientificName": "Puma concolor"},
}


def gbif_page(kind: str, *, count: int, offset: int, total: int) -> dict[str, Any]:
    """A page of ``count`` items of a GBIF list, with its ``count`` and ``endOfRecords``."""
    return {
        "offset": offset,
        "limit": count,
        "endOfRecords": offset + count >= total,
        "count": total,
        "results": [GBIF_ITEMS[kind]] * count,
    }
