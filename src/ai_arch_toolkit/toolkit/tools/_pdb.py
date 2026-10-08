"""RCSB PDB tools: search structures, and read an entry, its ligands and a chemical component
(T07b; D39).

The Search API answers identifiers and a total, and a query that matches nothing with
``204 No Content`` (https://search.rcsb.org/, "Empty results"). What each entry or ligand is comes
from the Data API's GraphQL endpoint, one request for a whole page of identifiers
(https://data.rcsb.org/index.html#gql-api), so a search costs two requests whatever its size,
and an entry's ligands two.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._window import list_window


def _refused(reply: Reply, step: str) -> ToolFailure | None:
    """A 400's ``message``: a request RCSB refused, ``validation_error`` in its words with
    ``step``; ``None`` for anything else.

    The REST and Search APIs explain an error status in ``message`` (RCSB's own agent tools
    read it so: https://github.com/rcsb/rcsb-mcp, ``src/rcsb_mcp/client.py``).
    """
    body = reply.body
    message = _string(body.get("message")) if isinstance(body, dict) else ""
    if message and reply.status == 400:
        return ToolFailure("validation_error", f"RCSB PDB refused the request: {message}; {step}")
    return None


def _search_error(reply: Reply) -> ToolFailure | None:
    """The error a Search API answer explains; ``None`` for a result."""
    return _refused(reply, "rephrase the query in plain words")


def _data_error(reply: Reply) -> ToolFailure | None:
    """The error a Data API (REST) answer explains; ``None`` for a result."""
    return _refused(reply, "check the ID, or find one with pdb_search")


def _graphql_error(reply: Reply) -> ToolFailure | str | None:
    """The errors a GraphQL answer reports; ``None`` for a result.

    The endpoint answers 200 and puts its errors in ``errors``
    (https://data.rcsb.org/index.html#gql-api; RCSB's client reads ``errors[].message``,
    https://github.com/rcsb/py-rcsb-api, ``rcsbapi/data/data_query.py``). The query is the
    tool's, not the caller's, so the step is the tool that reads an entry without it.
    """
    body = reply.body
    errors = body.get("errors") if isinstance(body, dict) else None
    if not isinstance(errors, list) or not errors:
        return _refused(reply, "pdb_entry reads an entry without it")
    said = "; ".join(_string(e.get("message")) for e in errors if isinstance(e, dict))
    return (
        f"RCSB PDB's GraphQL endpoint said: {said or 'the query failed'}; pdb_entry reads an "
        "entry without it, or try again later"
    )


# The Data API answers a record it does not have with a 404, so each REST call declares the
# record it asks for (``missing=``).
_DATA = Api(
    base="https://data.rcsb.org/rest/v1/core",
    name="RCSB PDB",
    timeout_s=20,
    error_reader=_data_error,
)
_GRAPHQL = Api(
    base="https://data.rcsb.org/graphql",
    name="RCSB PDB",
    timeout_s=20,
    error_reader=_graphql_error,
)
_SEARCH = Api(
    base="https://search.rcsb.org/rcsbsearch/v2/query",
    name="RCSB PDB",
    timeout_s=20,
    error_reader=_search_error,
)
_PDB_ID_RE = re.compile(r"[A-Za-z0-9]{4}")
_CHEM_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,12}")
_QUERY_CHARS = 500
# The fields of the entries a search lists: the ones RCSB's own agent tools ask for
# (https://github.com/rcsb/rcsb-mcp, ``src/rcsb_mcp/report/tables.py``).
_ENTRIES_QUERY = """query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    exptl { method }
    rcsb_entry_info { resolution_combined }
    rcsb_accession_info { initial_release_date }
  }
}"""
_LIGANDS_QUERY = """query($ids: [String!]!) {
  nonpolymer_entities(entity_ids: $ids) {
    rcsb_id
    pdbx_entity_nonpoly { comp_id name }
    rcsb_nonpolymer_entity { pdbx_number_of_molecules }
  }
}"""


@tool(capability="network")
def pdb_search(
    query: str,
    max_results: Annotated[int, Range(1, 25)] = 10,
    start: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Search RCSB PDB structures by free text, the most relevant first: each entry with its
    title, method, resolution and release date.

    Args:
        query: Words to look for, e.g. a protein, an organism, a ligand or a method.
        max_results: How many entries to list.
        start: How many entries to skip; the footer gives the next start.

    Raises:
        ToolFailure: validation_error when the query is empty or RCSB refuses it (with its
            reason).
    """
    text = query.strip()
    if not text:
        raise ToolFailure("validation_error", "query cannot be empty; say what to search for")
    if len(text) > _QUERY_CHARS or any(ord(char) < 32 for char in text):
        raise ToolFailure(
            "validation_error",
            f"invalid query {query[:100]!r}; give up to {_QUERY_CHARS} characters on one line",
        )
    # "text" searches one attribute and needs its name; free text is "full_text":
    # https://search.rcsb.org/#search-services
    payload = {
        "query": {"type": "terminal", "service": "full_text", "parameters": {"value": text}},
        "return_type": "entry",
        "request_options": {"paginate": {"start": start, "rows": max_results}},
    }
    hits = _SEARCH.post_json(payload=payload, parse=_hits, allow_empty=True)
    if not hits.ids and start == 0:
        return ToolResult.success(f"No RCSB PDB entries match {text!r}.")
    described = _described(_ENTRIES_QUERY, "entries", hits.ids, _hit_line) if hits.ids else []
    lines = [f"{start + number}. {line}" for number, line in enumerate(described, start=1)]
    shown = start + len(lines)
    more = hits.total is not None and shown < hits.total
    next_call = {"start": shown} if lines and more else None
    window = list_window(lines, first=start + 1, total=hits.total, next_call=next_call)
    return window.result(heading=f"RCSB PDB entries that match {text!r}, the most relevant first:")


@tool(capability="network")
def pdb_entry(pdb_id: str) -> str:
    """Read an RCSB PDB entry: its title, method, resolution, dates, entities and citation.

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
    """List the ligands (non-polymer entities) of an RCSB PDB entry, with their chemical
    component IDs.

    Args:
        pdb_id: Four-character PDB ID, e.g. "1A3N".

    Raises:
        ToolFailure: validation_error when ``pdb_id`` is not four letters or digits; not_found
            when the PDB has no entry with it.
    """
    normalized = _pdb_id(pdb_id)
    entities = _DATA.get_json(
        "entry", normalized, parse=_nonpolymer_ids, missing=_no_entry(normalized)
    )
    if not entities:
        return f"RCSB PDB entry {normalized} has no ligands."
    entity_of = {f"{normalized}_{entity}": entity for entity in entities}
    described = _described(
        _LIGANDS_QUERY,
        "nonpolymer_entities",
        tuple(entity_of),
        lambda entity_id, record: _ligand_line(entity_of[entity_id], record),
    )
    lines = [
        f"Ligands of RCSB PDB entry {normalized} (read one with pdb_chemical_component):",
        *(f"{number}. {line}" for number, line in enumerate(described, start=1)),
    ]
    return "\n".join(lines)


@tool(capability="network")
def pdb_chemical_component(component_id: str) -> str:
    """Read an RCSB chemical component (a ligand or residue): name, type, formula, weight and
    structure descriptors.

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
            f"invalid component_id {component_id[:100]!r}; a chemical component ID is 1-12 "
            "letters or digits, e.g. 'ATP' or 'HEM'",
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


# --- Requests ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Hits:
    """A page of search hits: their identifiers, in order, and the total (``None`` when the
    answer gives none: a 204)."""

    ids: tuple[str, ...]
    total: int | None


def _hits(data: dict[str, Any]) -> _Hits:
    """The identifiers and total of a search answer (``{}`` for a 204: none)."""
    ids = tuple(
        _string(item.get("identifier"))
        for item in data.get("result_set") or []
        if isinstance(item, dict) and _string(item.get("identifier"))
    )
    total = data.get("total_count")
    known = isinstance(total, int) and not isinstance(total, bool)
    return _Hits(ids=ids, total=total if known else None)


def _described(
    query: str,
    field: str,
    ids: tuple[str, ...],
    describe: Callable[[str, dict[str, Any] | None], str],
) -> list[str]:
    """Each of ``ids`` described from the GraphQL endpoint's record of it (``None`` when it has
    none), in one request.

    The lines are built inside the parse, so a record of a shape nobody expected is the door's
    parse failure, never a crash of the tool.
    """

    def read(data: dict[str, Any]) -> list[str]:
        found = (data.get("data") or {}).get(field) or []
        by_id = {
            _string(item.get("rcsb_id")): item
            for item in found
            if isinstance(item, dict) and _string(item.get("rcsb_id"))
        }
        return [describe(entry_id, by_id.get(entry_id)) for entry_id in ids]

    payload = {"query": query, "variables": {"ids": list(ids)}}
    return _GRAPHQL.post_json(payload=payload, parse=read)


def _nonpolymer_ids(entry: dict[str, Any]) -> list[str]:
    ids = (entry.get("rcsb_entry_container_identifiers") or {}).get("non_polymer_entity_ids")
    return [_string(entity) for entity in ids or [] if _string(entity)]


def _no_entry(pdb_id: str) -> str:
    return f"RCSB PDB has no entry {pdb_id}; find entries with pdb_search"


def _pdb_id(pdb_id: str) -> str:
    """``pdb_id`` upper-cased, or a validation_error when it is not a PDB ID."""
    normalized = pdb_id.strip().upper()
    if not _PDB_ID_RE.fullmatch(normalized):
        raise ToolFailure(
            "validation_error",
            f"invalid pdb_id {pdb_id[:100]!r}; a PDB ID is four letters or digits, e.g. "
            "'1A3N'; find one with pdb_search",
        )
    return normalized


# --- Answers -----------------------------------------------------------------------------------


def _hit_line(entry_id: str, record: dict[str, Any] | None) -> str:
    if record is None:
        return f"{entry_id}: (no record in the Data API; try pdb_entry)"
    parts = [
        f"{entry_id}: {_nested(record, 'struct', 'title') or '(no title)'}",
        _methods(record),
        _resolution(record),
    ]
    released = _date(_nested(record, "rcsb_accession_info", "initial_release_date"))
    parts.append(f"released {released}" if released else "")
    return " | ".join(part for part in parts if part)


def _ligand_line(entity: str, record: dict[str, Any] | None) -> str:
    if record is None:
        return f"entity {entity}: no record in the Data API"
    component = _nested(record, "pdbx_entity_nonpoly", "comp_id") or "?"
    name = _nested(record, "pdbx_entity_nonpoly", "name") or "(no name)"
    copies = _nested(record, "rcsb_nonpolymer_entity", "pdbx_number_of_molecules")
    return f"{component}: {name} | entity {entity}" + (f" | {copies} copies" if copies else "")


def _entry_text(data: dict[str, Any], pdb_id: str) -> str:
    info = _object(data.get("rcsb_entry_info"))
    method = _methods(data) or _join(info.get("experimental_method")) or "?"
    resolution = _resolution(data)
    deposited = _date(_nested(data, "rcsb_accession_info", "deposit_date")) or "?"
    released = _date(_nested(data, "rcsb_accession_info", "initial_release_date")) or "?"
    polymers = _string(info.get("polymer_entity_count")) or "?"
    ligands = _string(info.get("nonpolymer_entity_count")) or "?"
    lines = [
        f"RCSB PDB entry {pdb_id}:",
        _nested(data, "struct", "title") or "(no title)",
        f"Method: {method}" + (f" | resolution: {resolution}" if resolution else ""),
        f"Deposited: {deposited} | released: {released}",
        f"Entities: {polymers} polymer, {ligands} non-polymer (list them with pdb_ligands)",
    ]
    citation = _citation(data.get("rcsb_primary_citation"))
    if citation:
        lines.append(f"Citation: {citation}")
    return "\n".join(lines)


def _citation(citation: object) -> str:
    """The primary citation, with the DOI and the PubMed ID other tools read."""
    if not isinstance(citation, dict):
        return ""
    title = _string(citation.get("title")).rstrip(".")
    where = ", ".join(
        part
        for part in (_string(citation.get("rcsb_journal_abbrev")), _string(citation.get("year")))
        if part
    )
    doi = _string(citation.get("pdbx_database_id_DOI"))
    pubmed = _string(citation.get("pdbx_database_id_PubMed"))
    ids = ", ".join(part for part in (doi and f"DOI {doi}", pubmed and f"PubMed {pubmed}") if part)
    return ". ".join(part for part in (title, where, ids) if part)


def _component_text(data: dict[str, Any], component_id: str) -> str:
    chem = _object(data.get("chem_comp"))
    descriptors = _object(data.get("rcsb_chem_comp_descriptor"))
    weight = _string(chem.get("formula_weight"))
    lines = [
        f"RCSB chemical component {component_id}:",
        _string(chem.get("name")) or "(no name)",
        f"Type: {_string(chem.get('type')) or '?'} | formula: "
        f"{_string(chem.get('formula')) or '?'} | weight: {f'{weight} Da' if weight else '?'}",
    ]
    smiles = _string(descriptors.get("SMILES_stereo")) or _string(descriptors.get("SMILES"))
    if smiles:
        lines.append(f"SMILES: {smiles}")
    if inchikey := _string(descriptors.get("InChIKey")):
        lines.append(f"InChIKey: {inchikey}")
    return "\n".join(lines)


def _methods(record: dict[str, Any]) -> str:
    methods = (_string(item.get("method")) for item in _dicts(record.get("exptl")))
    return ", ".join(dict.fromkeys(method for method in methods if method))


def _resolution(record: dict[str, Any]) -> str:
    """The resolution in ångströms; empty when there is none (NMR, most EM maps)."""
    values = _object(record.get("rcsb_entry_info")).get("resolution_combined")
    first = values[0] if isinstance(values, list) and values else values
    if isinstance(first, bool) or not isinstance(first, int | float):
        return ""
    return f"{f'{first:.2f}'.rstrip('0').rstrip('.')} Å"


def _date(value: str) -> str:
    """The day of an RCSB timestamp (``1984-07-17T00:00:00Z``), in ISO 8601."""
    return value[:10] if re.match(r"\d{4}-\d{2}-\d{2}", value) else value


def _object(value: object) -> dict[str, Any]:
    """``value`` when it is an object; ``{}`` for anything else (none, text, a list)."""
    return value if isinstance(value, dict) else {}


def _dicts(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _nested(data: dict[str, Any], *keys: str) -> str:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return _string(current)


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
