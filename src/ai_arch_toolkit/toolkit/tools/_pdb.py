"""RCSB PDB tools — public biomolecular structure lookup."""

from __future__ import annotations

import re
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api

# The Data API answers a record it does not have with a 404, so each of its calls declares the
# record it asks for (``missing=``); both APIs give their errors' text in ``message``, which the
# door quotes (https://data.rcsb.org/redoc/index.html, https://search.rcsb.org/#return-codes).
_DATA = Api(base="https://data.rcsb.org/rest/v1/core", name="RCSB PDB", timeout_s=20)
_SEARCH = Api(base="https://search.rcsb.org/rcsbsearch/v2/query", name="RCSB PDB", timeout_s=20)
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

    Raises:
        ToolFailure: validation_error when the query is empty, too long or has characters the
            search does not take, or ``start`` is negative.
    """
    if not _valid_text(query):
        raise ToolFailure(
            "validation_error",
            f"invalid query {query!r}; give 1-180 characters of words, digits and basic "
            "punctuation, e.g. 'hemoglobin human'.",
        )
    if start < 0:
        raise ToolFailure(
            "validation_error", f"start must be greater than or equal to 0 (got {start})."
        )
    # "text" searches one attribute and needs its name; free text is "full_text":
    # https://search.rcsb.org/#search-services
    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query.strip()},
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": start, "rows": _bounded(max_results)}},
    }
    # A query that matches nothing is answered 204 No Content:
    # https://search.rcsb.org/#empty-results
    return _SEARCH.post_json(
        payload=payload, parse=lambda data: _search_text(data, query, start), allow_empty=True
    )


@tool(capability="network")
def pdb_entry(pdb_id: str) -> str:
    """Get RCSB PDB entry metadata.

    Args:
        pdb_id: Four-character PDB ID, e.g. "1A3N".

    Raises:
        ToolFailure: validation_error when ``pdb_id`` is not four letters or digits; not_found
            when the PDB has no entry with it.
    """
    normalized = _pdb_id(pdb_id)
    return _DATA.get_json(
        "entry",
        normalized,
        parse=lambda data: _entry_text(data, normalized),
        missing=_no_entry(normalized),
    )


@tool(capability="network")
def pdb_ligands(pdb_id: str) -> str:
    """List non-polymer ligands for a PDB entry.

    Args:
        pdb_id: Four-character PDB ID, e.g. "1A3N".

    Raises:
        ToolFailure: validation_error when ``pdb_id`` is not four letters or digits; not_found
            when the PDB has no entry with it, or no record of a ligand the entry lists.
    """
    normalized = _pdb_id(pdb_id)
    ids = _DATA.get_json("entry", normalized, parse=_nonpolymer_ids, missing=_no_entry(normalized))
    ligands = [
        _DATA.get_json(
            "nonpolymer_entity",
            normalized,
            entity_id,
            parse=_ligand,
            missing=(
                f"RCSB PDB entry {normalized} lists nonpolymer entity {entity_id} but has no "
                f"record of it; see the entry with pdb_entry"
            ),
        )
        for entity_id in ids
    ]
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

    Raises:
        ToolFailure: validation_error when ``component_id`` is not 1-12 letters, digits, ``_``
            or ``-``; not_found when the PDB has no chemical component with it.
    """
    normalized = component_id.strip().upper()
    if not _CHEM_ID_RE.fullmatch(normalized):
        raise ToolFailure(
            "validation_error",
            f"invalid component_id {component_id!r}; a chemical component ID is 1-12 letters "
            "or digits, e.g. 'ATP' or 'HEM'.",
        )
    return _DATA.get_json(
        "chemcomp",
        normalized,
        parse=lambda data: _component_text(data, normalized),
        missing=(
            f"RCSB PDB has no chemical component {normalized}; pdb_ligands lists the component "
            "IDs of an entry"
        ),
    )


def _no_entry(pdb_id: str) -> str:
    return f"RCSB PDB has no entry {pdb_id}; find entries with pdb_search"


def _pdb_id(pdb_id: str) -> str:
    """``pdb_id`` upper-cased, or a validation_error when it is not a PDB ID."""
    normalized = pdb_id.strip().upper()
    if not _PDB_ID_RE.fullmatch(normalized):
        raise ToolFailure(
            "validation_error",
            f"invalid pdb_id {pdb_id!r}; a PDB ID is four letters or digits, e.g. '1A3N'; "
            "find one with pdb_search.",
        )
    return normalized


def _search_text(data: dict[str, Any], query: str, start: int) -> str:
    results = data.get("result_set", [])
    if not isinstance(results, list) or not results:
        return f"No RCSB PDB entries found for {query!r}."
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
