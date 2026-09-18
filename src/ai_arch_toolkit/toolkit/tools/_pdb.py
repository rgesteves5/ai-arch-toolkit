"""RCSB PDB tools — public biomolecular structure lookup."""

from __future__ import annotations

import re
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_DATA = Api(
    base="https://data.rcsb.org/rest/v1/core",
    name="RCSB PDB",
    timeout_s=20,
    status_messages={404: "no matching records found."},
)
_SEARCH = Api(
    base="https://search.rcsb.org/rcsbsearch/v2/query",
    name="RCSB PDB",
    timeout_s=20,
    status_messages={404: "no matching records found."},
)
_MAX_LIMIT = 25
_PDB_ID_RE = re.compile(r"^[A-Za-z0-9]{4}$")
_CHEM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,12}$")
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)


@tool(capability="network")
def pdb_search(query: str, max_results: int = 10, start: int = 0) -> str:
    """Search RCSB PDB structures by free text.

    Args:
        query: Text query, e.g. protein name, organism, ligand, or method.
        max_results: Number of PDB entries to return (1-25). Defaults to 10.
        start: Zero-based result offset. Defaults to 0.
    """
    if not _valid_text(query):
        return "RCSB PDB search failed: invalid query."
    if start < 0:
        return "RCSB PDB search failed: start must be greater than or equal to 0."
    payload = {
        "query": {"type": "terminal", "service": "text", "parameters": {"value": query.strip()}},
        "return_type": "entry",
        "request_options": {"paginate": {"start": start, "rows": _bounded(max_results)}},
    }
    try:
        return _SEARCH.post_json(
            payload=payload, parse=lambda data: _search_text(data, query, start)
        )
    except HttpError as e:
        return f"RCSB PDB search failed: {e}"


@tool(capability="network")
def pdb_entry(pdb_id: str) -> str:
    """Get RCSB PDB entry metadata.

    Args:
        pdb_id: Four-character PDB ID, e.g. "1A3N".
    """
    normalized = pdb_id.strip().upper()
    if not _PDB_ID_RE.fullmatch(normalized):
        return f"RCSB PDB entry lookup failed: invalid pdb_id: {pdb_id!r}"
    try:
        return _DATA.get_json(
            "entry", normalized, parse=lambda data: _entry_text(data, normalized)
        )
    except HttpError as e:
        return f"RCSB PDB entry lookup failed: {e}"


@tool(capability="network")
def pdb_ligands(pdb_id: str) -> str:
    """List non-polymer ligands for a PDB entry.

    Args:
        pdb_id: Four-character PDB ID, e.g. "1A3N".
    """
    normalized = pdb_id.strip().upper()
    if not _PDB_ID_RE.fullmatch(normalized):
        return f"RCSB PDB ligands failed: invalid pdb_id: {pdb_id!r}"
    try:
        ids = _DATA.get_json("entry", normalized, parse=_nonpolymer_ids)
        ligands = [
            _DATA.get_json("nonpolymer_entity", normalized, entity_id, parse=_ligand)
            for entity_id in ids
        ]
    except HttpError as e:
        return f"RCSB PDB ligands failed: {e}"

    if not ligands:
        return f"No RCSB PDB ligands found for {normalized}."
    lines = [f"RCSB PDB ligands for {normalized}:"]
    lines.extend(f"{index}. {ligand}" for index, ligand in enumerate(ligands, start=1))
    return "\n".join(lines)


@tool(capability="network")
def pdb_chemical_component(component_id: str) -> str:
    """Get RCSB chemical component metadata for a ligand/residue.

    Args:
        component_id: Chemical component ID, e.g. "ATP", "HEM", or "NAG".
    """
    normalized = component_id.strip().upper()
    if not _CHEM_ID_RE.fullmatch(normalized):
        return f"RCSB PDB chemical component failed: invalid component_id: {component_id!r}"
    try:
        return _DATA.get_json(
            "chemcomp", normalized, parse=lambda data: _component_text(data, normalized)
        )
    except HttpError as e:
        return f"RCSB PDB chemical component failed: {e}"


def _search_text(data: dict[str, Any], query: str, start: int) -> str:
    results = data.get("result_set", [])
    if not isinstance(results, list) or not results:
        return "No RCSB PDB entries found."
    total = _string(data.get("total_count")) or "?"
    lines = [
        f"RCSB PDB entries for {query!r} (returned {len(results)}, total {total}, start {start}):"
    ]
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        identifier = _string(item.get("identifier"))
        score = _string(item.get("score"))
        lines.append(f"{index}. {identifier} | score: {score or '?'}")
    return "\n".join(lines)


def _entry_text(data: dict[str, Any], pdb_id: str) -> str:
    title = _nested(data, "struct", "title")
    info = data.get("rcsb_entry_info", {})
    ids = data.get("rcsb_entry_container_identifiers", {})
    lines = [f"RCSB PDB entry {pdb_id}:", title or "(no title)"]
    lines.append(
        "   "
        + " | ".join(
            [
                f"method: {_join(info.get('experimental_method')) or '?'}",
                f"resolution: {_string(info.get('resolution_combined')) or '?'}",
                f"polymer entities: {_string(info.get('polymer_entity_count')) or '?'}",
            ]
        )
    )
    polymer_ids = _list_text(ids.get("polymer_entity_ids"))
    nonpolymer_ids = _list_text(ids.get("non_polymer_entity_ids"))
    assembly_ids = _list_text(ids.get("assembly_ids"))
    if polymer_ids:
        lines.append(f"   polymer entity IDs: {polymer_ids}")
    if nonpolymer_ids:
        lines.append(f"   non-polymer entity IDs: {nonpolymer_ids}")
    if assembly_ids:
        lines.append(f"   assembly IDs: {assembly_ids}")
    citation = _first(data.get("citation"))
    if isinstance(citation, dict):
        citation_title = _string(citation.get("title"))
        year = _string(citation.get("year"))
        if citation_title:
            lines.append(f"   citation: {citation_title} ({year or '?'})")
    return "\n".join(lines)


def _nonpolymer_ids(entry: dict[str, Any]) -> list[str]:
    ids = entry.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids")
    return [str(entity_id) for entity_id in _as_list(ids)]


def _ligand(ligand: dict[str, Any]) -> str:
    comp = _nested(ligand, "pdbx_entity_nonpoly", "comp_id")
    name = _nested(ligand, "pdbx_entity_nonpoly", "name")
    entity_id = _nested(ligand, "rcsb_nonpolymer_entity_container_identifiers", "entity_id")
    return f"{comp} — {name} | entity_id: {entity_id}"


def _component_text(data: dict[str, Any], component_id: str) -> str:
    chem = data.get("chem_comp", {})
    desc = data.get("rcsb_chem_comp_descriptor", {})
    lines = [f"RCSB chemical component {component_id}:"]
    lines.append(f"{_string(chem.get('name')) or '(no name)'}")
    lines.append(
        "   "
        + " | ".join(
            [
                f"type: {_string(chem.get('type')) or '?'}",
                f"formula: {_string(chem.get('formula')) or '?'}",
                f"weight: {_string(chem.get('formula_weight')) or '?'}",
            ]
        )
    )
    smiles = _string(desc.get("SMILES_stereo")) or _string(desc.get("SMILES"))
    if smiles:
        lines.append(f"   SMILES: {smiles}")
    return "\n".join(lines)


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _nested(data: dict[str, Any], *keys: str) -> str:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return _string(current)


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _list_text(value: Any) -> str:
    return ", ".join(_string(item) for item in _as_list(value) if _string(item))


def _join(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_string(item) for item in value if _string(item))
    return _string(value)


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(_string(item) for item in value)
    return " ".join(str(value).split())
