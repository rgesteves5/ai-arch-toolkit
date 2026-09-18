"""ChEMBL tools — public chemistry, target, and bioactivity lookup."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_API = Api(
    base="https://www.ebi.ac.uk/chembl/api/data",
    name="ChEMBL",
    timeout_s=20,
    status_messages={404: "no matching records found."},
)
_MAX_LIMIT = 25
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_CHEMBL_RE = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)


@tool(capability="network")
def chembl_molecule_search(query: str, max_results: int = 10, offset: int = 0) -> str:
    """Search ChEMBL molecules by name, synonym, or text.

    Args:
        query: Molecule search text, e.g. "aspirin".
        max_results: Number of molecules to return (1-25). Defaults to 10.
        offset: Zero-based result offset. Defaults to 0.
    """
    if not _valid_text(query):
        return "ChEMBL molecule search failed: invalid query."
    if offset < 0:
        return "ChEMBL molecule search failed: offset must be greater than or equal to 0."
    params = {"q": query.strip(), "limit": str(_bounded(max_results)), "offset": str(offset)}
    header = f"ChEMBL molecules for {query!r}"
    return _fetch(
        "ChEMBL molecule search failed",
        ("molecule", "search.json"),
        params,
        lambda data: _page(data, "molecules", header, offset, _compact_molecule),
    )


@tool(capability="network")
def chembl_molecule(chembl_id: str) -> str:
    """Get ChEMBL molecule metadata.

    Args:
        chembl_id: ChEMBL molecule ID, e.g. "CHEMBL25".
    """
    normalized = chembl_id.strip().upper()
    if not _CHEMBL_RE.fullmatch(normalized):
        return f"ChEMBL molecule lookup failed: invalid chembl_id: {chembl_id!r}"
    return _fetch(
        "ChEMBL molecule lookup failed",
        ("molecule", f"{normalized}.json"),
        {},
        lambda data: "\n".join(
            [f"ChEMBL molecule {normalized}:", *_format_molecule(data, index=None, compact=False)]
        ),
    )


@tool(capability="network")
def chembl_target_search(query: str, max_results: int = 10, offset: int = 0) -> str:
    """Search ChEMBL biological targets.

    Args:
        query: Target search text, e.g. gene/protein name.
        max_results: Number of targets to return (1-25). Defaults to 10.
        offset: Zero-based result offset. Defaults to 0.
    """
    if not _valid_text(query):
        return "ChEMBL target search failed: invalid query."
    if offset < 0:
        return "ChEMBL target search failed: offset must be greater than or equal to 0."
    params = {"q": query.strip(), "limit": str(_bounded(max_results)), "offset": str(offset)}
    header = f"ChEMBL targets for {query!r}"
    return _fetch(
        "ChEMBL target search failed",
        ("target", "search.json"),
        params,
        lambda data: _page(data, "targets", header, offset, _format_target),
    )


@tool(capability="network")
def chembl_target(chembl_id: str) -> str:
    """Get ChEMBL target metadata.

    Args:
        chembl_id: ChEMBL target ID, e.g. "CHEMBL203".
    """
    normalized = chembl_id.strip().upper()
    if not _CHEMBL_RE.fullmatch(normalized):
        return f"ChEMBL target lookup failed: invalid chembl_id: {chembl_id!r}"
    return _fetch(
        "ChEMBL target lookup failed",
        ("target", f"{normalized}.json"),
        {},
        lambda data: _target_text(data, normalized),
    )


@tool(capability="network")
def chembl_activity_search(
    molecule_chembl_id: str = "",
    target_chembl_id: str = "",
    standard_type: str = "",
    max_results: int = 10,
    offset: int = 0,
) -> str:
    """Search ChEMBL bioactivity measurements.

    Args:
        molecule_chembl_id: Optional molecule ChEMBL ID.
        target_chembl_id: Optional target ChEMBL ID.
        standard_type: Optional measurement type, e.g. "IC50", "Ki", or "EC50".
        max_results: Number of activities to return (1-25). Defaults to 10.
        offset: Zero-based result offset. Defaults to 0.
    """
    problem = _activity_problem(molecule_chembl_id, target_chembl_id, standard_type, offset)
    if problem:
        return f"ChEMBL activity search failed: {problem}"
    params = {"limit": str(_bounded(max_results)), "offset": str(offset)}
    filters = {
        "molecule_chembl_id": molecule_chembl_id.strip().upper(),
        "target_chembl_id": target_chembl_id.strip().upper(),
        "standard_type": standard_type.strip(),
    }
    params.update({key: value for key, value in filters.items() if value})
    return _fetch(
        "ChEMBL activity search failed",
        ("activity.json",),
        params,
        lambda data: _page(data, "activities", "ChEMBL activities", offset, _format_activity),
    )


def _fetch(
    failure: str,
    segments: tuple[str, ...],
    params: dict[str, str],
    render: Callable[[dict[str, Any]], str],
) -> str:
    try:
        return _API.get_json(*segments, params=params, parse=render)
    except HttpError as e:
        return f"{failure}: {e}"


def _activity_problem(molecule_id: str, target_id: str, standard_type: str, offset: int) -> str:
    if not any((molecule_id.strip(), target_id.strip())):
        return "provide molecule_chembl_id or target_chembl_id."
    if molecule_id and not _CHEMBL_RE.fullmatch(molecule_id.strip()):
        return "invalid molecule_chembl_id."
    if target_id and not _CHEMBL_RE.fullmatch(target_id.strip()):
        return "invalid target_chembl_id."
    if standard_type and not _valid_text(standard_type):
        return "invalid standard_type."
    if offset < 0:
        return "offset must be greater than or equal to 0."
    return ""


_NOTHING_FOUND = {
    "molecules": "No ChEMBL molecules found.",
    "targets": "No ChEMBL targets found.",
    "activities": "No ChEMBL activities found.",
}


def _page(
    data: dict[str, Any],
    key: str,
    header: str,
    offset: int,
    format_item: Callable[..., list[str]],
) -> str:
    items = data.get(key, [])
    if not isinstance(items, list) or not items:
        return _NOTHING_FOUND[key]
    total = _string(data.get("page_meta", {}).get("total_count")) or "?"
    lines = [f"{header} (returned {len(items)}, total {total}, offset {offset}):"]
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            lines.extend(format_item(item, index=index))
    return "\n".join(lines)


def _compact_molecule(item: dict[str, Any], *, index: int) -> list[str]:
    return _format_molecule(item, index=index, compact=True)


def _target_text(data: dict[str, Any], chembl_id: str) -> str:
    lines = [f"ChEMBL target {chembl_id}:", *_format_target(data, index=None)]
    components = data.get("target_components", [])
    if isinstance(components, list) and components:
        names = [
            _string(component.get("accession")) or _string(component.get("component_id"))
            for component in components
            if isinstance(component, dict)
        ]
        lines.append(f"   components: {', '.join(name for name in names if name) or '?'}")
    return "\n".join(lines)


def _format_activity(item: dict[str, Any], *, index: int) -> list[str]:
    value = " ".join(
        part
        for part in (
            _string(item.get("standard_relation")),
            _string(item.get("standard_value")),
            _string(item.get("standard_units")),
        )
        if part
    )
    return [
        f"{index}. {_string(item.get('molecule_chembl_id'))} -> "
        f"{_string(item.get('target_chembl_id'))} | "
        f"{_string(item.get('standard_type'))}: {value or '?'}",
        f"   assay: {_string(item.get('assay_chembl_id')) or '?'} | "
        f"document: {_string(item.get('document_chembl_id')) or '?'}",
    ]


def _format_molecule(item: dict[str, Any], *, index: int | None, compact: bool) -> list[str]:
    prefix = f"{index}. " if index is not None else ""
    chembl_id = _string(item.get("molecule_chembl_id"))
    name = _string(item.get("pref_name")) or "(no preferred name)"
    lines = [f"{prefix}{name} | id: {chembl_id}"]
    lines.append(
        "   "
        + " | ".join(
            [
                f"type: {_string(item.get('molecule_type')) or '?'}",
                f"max phase: {_string(item.get('max_phase')) or '?'}",
                f"first approval: {_string(item.get('first_approval')) or '?'}",
            ]
        )
    )
    props = item.get("molecule_properties", {})
    if isinstance(props, dict) and not compact:
        lines.append(
            "   "
            + " | ".join(
                [
                    f"MW: {_string(props.get('full_mwt')) or '?'}",
                    f"alogP: {_string(props.get('alogp')) or '?'}",
                    (
                        f"HBA/HBD: {_string(props.get('hba')) or '?'}/"
                        f"{_string(props.get('hbd')) or '?'}"
                    ),
                ]
            )
        )
    structures = item.get("molecule_structures", {})
    if isinstance(structures, dict) and not compact:
        smiles = _string(structures.get("canonical_smiles"))
        if smiles:
            lines.append(f"   SMILES: {smiles}")
    return lines


def _format_target(item: dict[str, Any], *, index: int | None) -> list[str]:
    prefix = f"{index}. " if index is not None else ""
    target_id = _string(item.get("target_chembl_id"))
    name = _string(item.get("pref_name")) or "(no preferred name)"
    lines = [f"{prefix}{name} | id: {target_id}"]
    lines.append(
        "   "
        + " | ".join(
            [
                f"type: {_string(item.get('target_type')) or '?'}",
                f"organism: {_string(item.get('organism')) or '?'}",
                f"tax_id: {_string(item.get('tax_id')) or '?'}",
            ]
        )
    )
    return lines


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
