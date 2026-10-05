"""UniProt tools — public protein search and annotation lookup."""

from __future__ import annotations

import re
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api


def _uniprot_error(answer: object) -> str | None:
    """The error a UniProt answer explains; ``None`` for a result.

    A request UniProt refuses says why in ``messages``, with the error status
    (https://www.uniprot.org/help/rest-api-headers).
    """
    messages = answer.get("messages") if isinstance(answer, dict) else None
    if isinstance(messages, list) and messages:
        return "; ".join(_string(message) for message in messages)
    return _inactive(answer)


def _inactive(answer: object) -> str | None:
    """Why an accession has no entry, when UniProt says it is inactive; ``None`` otherwise.

    An accession merged, demerged or deleted (https://www.uniprot.org/help/deleted_accessions)
    still answers HTTP 200: ``{"entryType": "Inactive", "inactiveReason": {...}}`` (seen
    2026-09-30), which read as an entry with no name, features or cross-references.
    """
    if not isinstance(answer, dict) or answer.get("entryType") != "Inactive":
        return None
    accession = _string(answer.get("primaryAccession")) or "the accession"
    reason = answer.get("inactiveReason")
    reason = reason if isinstance(reason, dict) else {}
    kind = _string(reason.get("inactiveReasonType")).lower()
    targets = reason.get("mergeDemergeTo")
    if kind and isinstance(targets, list) and targets:
        return f"{accession} is inactive: {kind} into {', '.join(map(_string, targets))}"
    why = _string(reason.get("deletedReason"))
    detail = (f": {kind}" if kind else "") + (f" ({why})" if why else "")
    return f"{accession} is inactive{detail}"


_API = Api(
    base="https://rest.uniprot.org/uniprotkb",
    name="UniProt",
    timeout_s=20,
    status_messages={404: "no matching records found."},
    body_error=_uniprot_error,
)
_MAX_LIMIT = 25
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_ACCESSION_RE = re.compile(r"^[A-Z0-9]{6,10}(?:-\d+)?$", re.IGNORECASE)
_TEXT_HINT = "use 1-180 letters, digits, spaces and ,.'()/%:+-"


@tool(capability="network")
def uniprot_search(
    query: str,
    organism: str = "",
    reviewed: str = "",
    max_results: int = 10,
    offset: int = 0,
) -> str:
    """Search UniProtKB proteins.

    Args:
        query: UniProt query text, e.g. protein, gene, accession, or function.
        organism: Optional organism name or taxonomy ID filter.
        reviewed: Optional reviewed filter: "true" for Swiss-Prot, "false" for TrEMBL.
        max_results: Number of proteins to return (1-25). Defaults to 10.
        offset: Zero-based result offset. Defaults to 0.

    Raises:
        ToolFailure: validation_error when the query, the organism, ``reviewed`` or the offset is
            invalid.
    """
    if not _valid_text(query):
        raise ToolFailure("validation_error", f"invalid query {query!r}; {_TEXT_HINT}")
    if organism and not _valid_text(organism):
        raise ToolFailure("validation_error", f"invalid organism {organism!r}; {_TEXT_HINT}")
    if reviewed and reviewed.lower() not in {"true", "false"}:
        raise ToolFailure(
            "validation_error", f"invalid reviewed {reviewed!r}; use 'true', 'false', or ''"
        )
    if offset < 0:
        raise ToolFailure("validation_error", f"invalid offset {offset}; use 0 or more")

    params = {
        "query": _search_query(query, organism, reviewed),
        "format": "json",
        "size": str(_bounded(max_results)),
        "offset": str(offset),
        "fields": "accession,protein_name,gene_names,organism_name,reviewed,length",
    }
    return _API.get_json(
        "search", params=params, parse=lambda data: _search_text(data, query, offset)
    )


@tool(capability="network")
def uniprot_entry(accession: str) -> str:
    """Get UniProtKB entry metadata by accession.

    Args:
        accession: UniProt accession, e.g. "P01308".

    Raises:
        ToolFailure: validation_error when the accession is malformed.
    """
    normalized = _accession(accession)
    return _API.get_json(
        normalized, params={"format": "json"}, parse=lambda data: _entry_text(data, normalized)
    )


@tool(capability="network")
def uniprot_features(accession: str, feature_type: str = "", max_results: int = 20) -> str:
    """List UniProtKB sequence features.

    Args:
        accession: UniProt accession, e.g. "P01308".
        feature_type: Optional feature type filter, e.g. "Domain" or "Active site".
        max_results: Number of features to return (1-25). Defaults to 20.

    Raises:
        ToolFailure: validation_error when the accession or the feature type is malformed.
    """
    normalized = _accession(accession)
    if feature_type and not _valid_text(feature_type):
        raise ToolFailure(
            "validation_error", f"invalid feature_type {feature_type!r}; e.g. 'Domain'"
        )
    return _API.get_json(
        normalized,
        params={"format": "json"},
        parse=lambda data: _features_text(data, normalized, feature_type, max_results),
    )


@tool(capability="network")
def uniprot_sequence(accession: str) -> str:
    """Get a UniProtKB protein sequence in FASTA form.

    Args:
        accession: UniProt accession, e.g. "P01308".

    Raises:
        ToolFailure: validation_error when the accession is malformed; not_found when UniProt
            returns no sequence for it.
    """
    normalized = _accession(accession)
    fasta = _API.get_text(f"{normalized}.fasta", parse=str.strip)
    if not fasta:
        msg = f"UniProt has no sequence for {normalized}; check the entry with uniprot_entry"
        raise ToolFailure("not_found", msg)
    return fasta


@tool(capability="network")
def uniprot_crossrefs(accession: str, database: str = "", max_results: int = 25) -> str:
    """List UniProtKB database cross-references.

    Args:
        accession: UniProt accession, e.g. "P01308".
        database: Optional database filter, e.g. "PDB", "Reactome", or "ChEMBL".
        max_results: Number of cross-references to return (1-25). Defaults to 25.

    Raises:
        ToolFailure: validation_error when the accession or the database is malformed.
    """
    normalized = _accession(accession)
    if database and not _valid_text(database):
        raise ToolFailure("validation_error", f"invalid database {database!r}; e.g. 'PDB'")
    return _API.get_json(
        normalized,
        params={"format": "json"},
        parse=lambda data: _crossrefs_text(data, normalized, database, max_results),
    )


def _accession(accession: str) -> str:
    """``accession`` upper-cased; a malformed one raises ``ToolFailure`` (validation_error)."""
    normalized = accession.strip().upper()
    if not _ACCESSION_RE.fullmatch(normalized):
        raise ToolFailure(
            "validation_error",
            f"invalid accession {accession!r}; a UniProt accession looks like P01308 "
            "(find one with uniprot_search)",
        )
    return normalized


def _search_query(query: str, organism: str, reviewed: str) -> str:
    search = query.strip()
    if organism.strip():
        org = organism.strip()
        search += (
            f" AND (organism_id:{org} OR organism_name:{org})"
            if org.isdigit()
            else f" AND organism_name:{org}"
        )
    if reviewed.strip():
        search += f" AND reviewed:{reviewed.strip().lower()}"
    return search


def _search_text(data: dict[str, Any], query: str, offset: int) -> str:
    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        return "No UniProt proteins found."
    total = _string(data.get("totalResults"))
    lines = [
        (
            f"UniProt proteins for {query!r} "
            f"(returned {len(results)}, total {total or '?'}, offset {offset}):"
        )
    ]
    for index, item in enumerate(results, start=1):
        if isinstance(item, dict):
            lines.extend(_format_entry(item, index=index, compact=True))
    return "\n".join(lines)


def _entry_text(data: dict[str, Any], accession: str) -> str:
    lines = [f"UniProt entry {accession}:"]
    lines.extend(_format_entry(data, index=None, compact=False))
    function = _comment_text(data, "FUNCTION")
    if function:
        lines.append(f"   Function: {_trim(function, 500)}")
    return "\n".join(lines)


def _features_text(
    data: dict[str, Any], accession: str, feature_type: str, max_results: int
) -> str:
    features = data.get("features", [])
    if not isinstance(features, list):
        features = []
    wanted = feature_type.strip().lower()
    if wanted:
        features = [
            feature
            for feature in features
            if isinstance(feature, dict) and _string(feature.get("type")).lower() == wanted
        ]
    features = features[: _bounded(max_results)]
    if not features:
        return f"No UniProt features found for {accession}."
    lines = [f"UniProt features for {accession}:"]
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            continue
        location = _feature_location(feature.get("location"))
        lines.append(
            f"{index}. {_string(feature.get('type')) or '?'} | {location or '?'} | "
            f"{_string(feature.get('description')) or '(no description)'}"
        )
    return "\n".join(lines)


def _crossrefs_text(data: dict[str, Any], accession: str, database: str, max_results: int) -> str:
    refs = data.get("uniProtKBCrossReferences", [])
    if not isinstance(refs, list):
        refs = []
    wanted = database.strip().lower()
    if wanted:
        refs = [
            ref
            for ref in refs
            if isinstance(ref, dict) and _string(ref.get("database")).lower() == wanted
        ]
    refs = refs[: _bounded(max_results)]
    if not refs:
        return f"No UniProt cross-references found for {accession}."
    lines = [f"UniProt cross-references for {accession}:"]
    for index, ref in enumerate(refs, start=1):
        if not isinstance(ref, dict):
            continue
        lines.append(
            f"{index}. {_string(ref.get('database')) or '?'}: {_string(ref.get('id')) or '?'}"
        )
        props = ref.get("properties", [])
        if isinstance(props, list) and props:
            prop_text = ", ".join(
                f"{_string(prop.get('key'))}: {_string(prop.get('value'))}"
                for prop in props[:3]
                if isinstance(prop, dict)
            )
            if prop_text:
                lines.append(f"   {prop_text}")
    return "\n".join(lines)


def _format_entry(item: dict[str, Any], *, index: int | None, compact: bool) -> list[str]:
    prefix = f"{index}. " if index is not None else ""
    accession = _string(item.get("primaryAccession"))
    protein = _protein_name(item)
    organism = _nested(item, "organism", "scientificName")
    reviewed = _string(item.get("entryType"))
    length = _nested(item, "sequence", "length")
    genes = _gene_names(item)
    lines = [f"{prefix}{protein or '(no protein name)'} | accession: {accession}"]
    lines.append(
        f"   organism: {organism or '?'} | entry: {reviewed or '?'} | length: {length or '?'}"
    )
    if genes:
        lines.append(f"   genes: {genes}")
    if not compact:
        created = _string(item.get("entryAudit", {}).get("firstPublicDate"))
        modified = _string(item.get("entryAudit", {}).get("lastAnnotationUpdateDate"))
        if created or modified:
            lines.append(
                f"   first public: {created or '?'} | annotation update: {modified or '?'}"
            )
    return lines


def _protein_name(item: dict[str, Any]) -> str:
    description = item.get("proteinDescription", {})
    recommended = description.get("recommendedName", {}) if isinstance(description, dict) else {}
    full_name = recommended.get("fullName", {}) if isinstance(recommended, dict) else {}
    return _string(full_name.get("value")) if isinstance(full_name, dict) else ""


def _gene_names(item: dict[str, Any]) -> str:
    genes = item.get("genes", [])
    names: list[str] = []
    if isinstance(genes, list):
        for gene in genes:
            if not isinstance(gene, dict):
                continue
            name = gene.get("geneName", {})
            if isinstance(name, dict) and _string(name.get("value")):
                names.append(_string(name.get("value")))
    return ", ".join(names)


def _comment_text(item: dict[str, Any], comment_type: str) -> str:
    comments = item.get("comments", [])
    if not isinstance(comments, list):
        return ""
    for comment in comments:
        if not isinstance(comment, dict) or _string(comment.get("commentType")) != comment_type:
            continue
        texts = comment.get("texts", [])
        if isinstance(texts, list):
            return " ".join(_string(text.get("value")) for text in texts if isinstance(text, dict))
    return ""


def _feature_location(location: Any) -> str:
    if not isinstance(location, dict):
        return ""
    start = _position(location.get("start"))
    end = _position(location.get("end"))
    return f"{start}-{end}" if start or end else ""


def _position(value: Any) -> str:
    if isinstance(value, dict):
        return _string(value.get("value"))
    return _string(value)


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


def _trim(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
