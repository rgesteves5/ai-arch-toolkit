"""ChEMBL tools: search molecules, targets and bioactivity measurements, and read a molecule or a
target (T07b; D39).

Lists page by ``limit`` and ``offset``, with the total in ``page_meta.total_count``
(https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services).
A refused request answers 400 and says why in ``error_message``; an unknown ID answers 404 (the
web services' source: https://github.com/chembl/chembl_webservices_py3,
``src/chembl_webservices/core/resource.py``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._values import decimal_text, plain
from ai_arch_toolkit.toolkit.tools._window import list_window


def _chembl_error(reply: Reply) -> ToolFailure | str | None:
    """The error a ChEMBL answer explains in ``error_message``; ``None`` for a result. A 400 is
    a request ChEMBL refused ("Search query too short"), the caller's to change."""
    body = reply.body
    said = body.get("error_message") if isinstance(body, dict) else None
    text = "; ".join(map(_string, said)) if isinstance(said, list) else _string(said)
    if text and reply.status == 400:
        return ToolFailure(
            "validation_error",
            f"ChEMBL refused the request: {text.rstrip('.')}; change what it names",
        )
    return text or None


_API = Api(
    base="https://www.ebi.ac.uk/chembl/api/data",
    name="ChEMBL",
    timeout_s=20,
    error_reader=_chembl_error,
)
_TEXT_RE = re.compile(r"[\w\s,.'()/%:+-]{1,180}")
_CHEMBL_RE = re.compile(r"CHEMBL\d+", re.IGNORECASE)
# ChEMBL refuses a search under 3 characters ("Search query too short", ``core/resource.py``).
_QUERY_MIN, _QUERY_MAX = 3, 200
# The phases ``max_phase`` stands for
# (https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/drug-and-compound-questions).
_PHASES = {
    4.0: "approved",
    3.0: "phase 3",
    2.0: "phase 2",
    1.0: "phase 1",
    0.5: "early phase 1",
    -1.0: "unknown",
}


@tool(capability="network")
def chembl_molecule_search(
    query: str,
    max_results: Annotated[int, Range(1, 25)] = 10,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Search ChEMBL molecules by name, synonym or text.

    Args:
        query: What to look for, e.g. "aspirin".
        max_results: How many molecules to list.
        offset: How many molecules to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when the query is invalid or ChEMBL refuses it.
    """
    text = _query(query)
    return _API.get_json(
        "molecule",
        "search.json",
        params={"q": text, "limit": str(max_results), "offset": str(offset)},
        parse=lambda data: _list_answer(
            data,
            "molecules",
            offset,
            _molecule_line,
            heading=f"ChEMBL molecules that match {text!r}:",
            nothing=f"No ChEMBL molecules match {text!r}.",
        ),
    )


@tool(capability="network")
def chembl_molecule(chembl_id: str) -> str:
    """Read a ChEMBL molecule: name, type, development phase, properties and structure.

    Args:
        chembl_id: ChEMBL molecule ID, e.g. "CHEMBL25".

    Raises:
        ToolFailure: validation_error when the ID is malformed; not_found when ChEMBL has no
            molecule with it.
    """
    normalized = _chembl_id("chembl_id", chembl_id)
    return _API.get_json(
        "molecule",
        f"{normalized}.json",
        missing=(
            f"no ChEMBL molecule with ID {normalized} (a target ID is read with chembl_target); "
            "search with chembl_molecule_search"
        ),
        parse=lambda data: _molecule_text(data, normalized),
    )


@tool(capability="network")
def chembl_target_search(
    query: str,
    max_results: Annotated[int, Range(1, 25)] = 10,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Search ChEMBL biological targets: proteins, complexes, organisms, cell lines.

    Args:
        query: What to look for, e.g. a gene or protein name.
        max_results: How many targets to list.
        offset: How many targets to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when the query is invalid or ChEMBL refuses it.
    """
    text = _query(query)
    return _API.get_json(
        "target",
        "search.json",
        params={"q": text, "limit": str(max_results), "offset": str(offset)},
        parse=lambda data: _list_answer(
            data,
            "targets",
            offset,
            _target_line,
            heading=f"ChEMBL targets that match {text!r}:",
            nothing=f"No ChEMBL targets match {text!r}.",
        ),
    )


@tool(capability="network")
def chembl_target(chembl_id: str) -> str:
    """Read a ChEMBL target: name, type, organism and the UniProt accessions of its components.

    Args:
        chembl_id: ChEMBL target ID, e.g. "CHEMBL203".

    Raises:
        ToolFailure: validation_error when the ID is malformed; not_found when ChEMBL has no
            target with it.
    """
    normalized = _chembl_id("chembl_id", chembl_id)
    return _API.get_json(
        "target",
        f"{normalized}.json",
        missing=(
            f"no ChEMBL target with ID {normalized} (a molecule ID is read with chembl_molecule); "
            "search with chembl_target_search"
        ),
        parse=lambda data: _target_text(data, normalized),
    )


@tool(capability="network")
def chembl_activity_search(
    molecule_chembl_id: str = "",
    target_chembl_id: str = "",
    standard_type: str = "",
    max_results: Annotated[int, Range(1, 25)] = 10,
    offset: Annotated[int, Range(0)] = 0,
    assay_chembl_id: str = "",
) -> ToolResult:
    """Search ChEMBL bioactivity measurements of a molecule, on a target, in an assay, or any
    of them together.

    Args:
        molecule_chembl_id: The molecule's ChEMBL ID.
        target_chembl_id: The target's ChEMBL ID.
        standard_type: Only this measurement type, e.g. "IC50", "Ki" or "EC50".
        max_results: How many measurements to list.
        offset: How many measurements to skip; the footer gives the next offset.
        assay_chembl_id: The assay's ChEMBL ID, as each measurement names it.

    Raises:
        ToolFailure: validation_error when no ID is given, an argument is invalid, or ChEMBL
            refuses the filters.
    """
    molecule = _optional_id("molecule_chembl_id", molecule_chembl_id)
    target = _optional_id("target_chembl_id", target_chembl_id)
    assay = _optional_id("assay_chembl_id", assay_chembl_id)
    if not (molecule or target or assay):
        msg = "provide molecule_chembl_id, target_chembl_id or assay_chembl_id, e.g. CHEMBL25"
        raise ToolFailure("validation_error", msg)
    kind = standard_type.strip()
    if kind and not _TEXT_RE.fullmatch(kind):
        msg = f"invalid standard_type {standard_type[:100]!r}; use a type such as IC50, Ki or EC50"
        raise ToolFailure("validation_error", msg)
    filters = (
        ("molecule_chembl_id", "molecule", molecule),
        ("target_chembl_id", "target", target),
        ("assay_chembl_id", "assay", assay),
        ("standard_type", "type", kind),
    )
    params = {"limit": str(max_results), "offset": str(offset)}
    params.update({param: value for param, _, value in filters if value})
    label = ", ".join(f"{word} {value}" for _, word, value in filters if value)
    return _API.get_json(
        "activity.json",
        params=params,
        parse=lambda data: _list_answer(
            data,
            "activities",
            offset,
            _activity_line,
            heading=f"ChEMBL activities of {label}:",
            nothing=f"No ChEMBL activities of {label}.",
        ),
    )


# --- Arguments ---------------------------------------------------------------------------------


def _query(query: str) -> str:
    """The search text ChEMBL takes: 3 to 200 characters on one line."""
    text = query.strip()
    if not _QUERY_MIN <= len(text) <= _QUERY_MAX or any(ord(char) < 32 for char in text):
        raise ToolFailure(
            "validation_error",
            f"invalid query {query[:100]!r}; ChEMBL searches {_QUERY_MIN} to {_QUERY_MAX} "
            "characters on one line, e.g. 'aspirin'",
        )
    return text


def _chembl_id(name: str, value: str) -> str:
    """``value`` as a ChEMBL ID in upper case; raises when it is not one."""
    normalized = value.strip().upper()
    if not _CHEMBL_RE.fullmatch(normalized):
        msg = f"invalid {name} {value[:100]!r}; a ChEMBL ID looks like CHEMBL25"
        raise ToolFailure("validation_error", msg)
    return normalized


def _optional_id(name: str, value: str) -> str:
    """An optional ChEMBL ID: empty when not given."""
    return _chembl_id(name, value) if value.strip() else ""


# --- Answers -----------------------------------------------------------------------------------


def _list_answer(
    data: dict[str, Any],
    key: str,
    offset: int,
    line: Callable[[int, dict[str, Any]], str],
    *,
    heading: str,
    nothing: str,
) -> ToolResult:
    """A page of a ChEMBL list, numbered from ``offset``, with the total ``page_meta`` gives."""
    items = [item for item in data.get(key) or [] if isinstance(item, dict)]
    if not items and offset == 0:
        return ToolResult.success(nothing)
    meta = data.get("page_meta") or {}
    total = meta.get("total_count")
    total = total if isinstance(total, int) and not isinstance(total, bool) else None
    lines = [line(offset + number, item) for number, item in enumerate(items, start=1)]
    shown = offset + len(lines)
    more = shown < total if total is not None else bool(meta.get("next"))
    next_call = {"offset": shown} if lines and more else None
    window = list_window(lines, first=offset + 1, total=total, next_call=next_call)
    return window.result(heading=heading)


def _molecule_line(number: int, item: dict[str, Any]) -> str:
    return " | ".join(
        [
            f"{number}. {_string(item.get('pref_name')) or '(no preferred name)'}",
            _string(item.get("molecule_chembl_id")) or "?",
            _string(item.get("molecule_type")) or "?",
            f"max phase: {_phase(item.get('max_phase'))}",
        ]
    )


def _molecule_text(data: dict[str, Any], chembl_id: str) -> str:
    approval = _string(data.get("first_approval"))
    lines = [
        f"ChEMBL molecule {chembl_id}:",
        f"{_string(data.get('pref_name')) or '(no preferred name)'} | "
        f"{_string(data.get('molecule_type')) or '?'}",
        f"Max phase: {_phase(data.get('max_phase'))}"
        + (f" | first approval: {approval}" if approval else ""),
    ]
    if properties := _properties(data.get("molecule_properties")):
        lines.append(f"Properties: {properties}")
    structures = data.get("molecule_structures")
    structures = structures if isinstance(structures, dict) else {}
    if smiles := _string(structures.get("canonical_smiles")):
        lines.append(f"SMILES: {smiles}")
    if inchikey := _string(structures.get("standard_inchi_key")):
        lines.append(f"InChIKey: {inchikey}")
    return "\n".join(lines)


# Each property ChEMBL computes that a molecule's text shows: label, key, unit.
_PROPERTIES = (
    ("MW", "full_mwt", " Da"),
    ("AlogP", "alogp", ""),
    ("HBA", "hba", ""),
    ("HBD", "hbd", ""),
    ("PSA", "psa", " Å²"),
    ("Ro5 violations", "num_ro5_violations", ""),
)


def _properties(properties: object) -> str:
    if not isinstance(properties, dict):
        return ""
    shown = ((label, _string(properties.get(key)), unit) for label, key, unit in _PROPERTIES)
    return " | ".join(f"{label} {value}{unit}" for label, value, unit in shown if value)


def _phase(value: object) -> str:
    """``max_phase`` with what it stands for: ``4 (approved)``; none is preclinical."""
    if value is None or value == "":
        return "none (preclinical)"
    try:
        number = float(str(value))
    except ValueError:
        return _string(value)
    shown = f"{number:.1f}".removesuffix(".0")
    label = _PHASES.get(number)
    return f"{shown} ({label})" if label else shown


def _target_line(number: int, item: dict[str, Any]) -> str:
    return " | ".join(
        [
            f"{number}. {_string(item.get('pref_name')) or '(no preferred name)'}",
            _string(item.get("target_chembl_id")) or "?",
            _string(item.get("target_type")) or "?",
            _string(item.get("organism")) or "?",
        ]
    )


def _target_text(data: dict[str, Any], chembl_id: str) -> str:
    taxon = _string(data.get("tax_id"))
    organism = (_string(data.get("organism")) or "?") + (f" (taxon {taxon})" if taxon else "")
    lines = [
        f"ChEMBL target {chembl_id}:",
        f"{_string(data.get('pref_name')) or '(no preferred name)'} | "
        f"{_string(data.get('target_type')) or '?'} | {organism}",
    ]
    components = data.get("target_components")
    accessions = [
        _string(component.get("accession"))
        for component in components or []
        if isinstance(component, dict) and _string(component.get("accession"))
    ]
    if accessions:
        lines.append(f"Components (UniProt; read with uniprot_entry): {', '.join(accessions)}")
    return "\n".join(lines)


def _activity_line(number: int, item: dict[str, Any]) -> str:
    molecule = _named(item.get("molecule_chembl_id"), item.get("molecule_pref_name"))
    target_names = ", ".join(
        part
        for part in (_string(item.get("target_pref_name")), _string(item.get("target_organism")))
        if part
    )
    target = _named(item.get("target_chembl_id"), target_names)
    parts = [f"{number}. {molecule} -> {target}", _measure(item)]
    if pchembl := _number(item.get("pchembl_value")):
        parts.append(f"pChEMBL {pchembl}")
    assay = _string(item.get("assay_chembl_id"))
    description = _string(item.get("assay_description"))
    if assay:
        parts.append(f"assay {assay}" + (f": {description}" if description else ""))
    source = ", ".join(
        part
        for part in (_string(item.get("document_journal")), _string(item.get("document_year")))
        if part
    )
    parts.append(source)
    if flag := _string(item.get("data_validity_comment")):
        parts.append(f"flagged: {flag}")
    return " | ".join(part for part in parts if part)


def _named(identifier: object, name: object) -> str:
    """``CHEMBL25 (ASPIRIN)``, or the ID alone when it has no name."""
    ident, label = _string(identifier) or "?", _string(name)
    return f"{ident} ({label})" if label else ident


def _measure(item: dict[str, Any]) -> str:
    """``IC50: 10.0 nM``; a relation other than ``=`` before the value (``> 10.0 nM``)."""
    relation = _string(item.get("standard_relation"))
    value = " ".join(
        part
        for part in (
            relation if relation != "=" else "",
            _number(item.get("standard_value")),
            _string(item.get("standard_units")),
        )
        if part
    )
    return f"{_string(item.get('standard_type')) or '?'}: {value or '?'}"


def _number(value: object) -> str:
    """A ChEMBL number as its digits: ChEMBL sends its decimals as text, a small one in
    scientific notation ("1E-7"), and a number as JSON now and then."""
    return decimal_text(value) if isinstance(value, str) else plain(value)


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
