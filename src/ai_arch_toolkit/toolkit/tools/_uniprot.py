"""UniProt tools — public protein search and annotation lookup."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://rest.uniprot.org/uniprotkb"
_TIMEOUT = 20
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_LIMIT = 25
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_ACCESSION_RE = re.compile(r"^[A-Z0-9]{6,10}(?:-\d+)?$", re.IGNORECASE)


@tool
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
    """
    if not _valid_text(query):
        return "UniProt search failed: invalid query."
    if organism and not _valid_text(organism):
        return "UniProt search failed: invalid organism."
    if reviewed and reviewed.lower() not in {"true", "false"}:
        return "UniProt search failed: reviewed must be true, false, or empty."
    if offset < 0:
        return "UniProt search failed: offset must be greater than or equal to 0."

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
    params = {
        "query": search,
        "format": "json",
        "size": str(_bounded(max_results)),
        "offset": str(offset),
        "fields": "accession,protein_name,gene_names,organism_name,reviewed,length",
    }
    try:
        data = _fetch_json("", params)
        results = data.get("results", [])
    except urllib.error.HTTPError as e:
        return _http_error("UniProt search failed", e)
    except urllib.error.URLError as e:
        return f"UniProt search failed: URL error: {e.reason}"
    except TimeoutError:
        return "UniProt search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"UniProt search failed: could not parse API response: {e}"

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


@tool
def uniprot_entry(accession: str) -> str:
    """Get UniProtKB entry metadata by accession.

    Args:
        accession: UniProt accession, e.g. "P01308".
    """
    normalized = accession.strip().upper()
    if not _ACCESSION_RE.fullmatch(normalized):
        return f"UniProt entry lookup failed: invalid accession: {accession!r}"
    try:
        data = _fetch_json(f"/{normalized}", {"format": "json"})
    except urllib.error.HTTPError as e:
        return _http_error("UniProt entry lookup failed", e)
    except urllib.error.URLError as e:
        return f"UniProt entry lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "UniProt entry lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"UniProt entry lookup failed: could not parse API response: {e}"

    lines = [f"UniProt entry {normalized}:"]
    lines.extend(_format_entry(data, index=None, compact=False))
    function = _comment_text(data, "FUNCTION")
    if function:
        lines.append(f"   Function: {_trim(function, 500)}")
    return "\n".join(lines)


@tool
def uniprot_features(accession: str, feature_type: str = "", max_results: int = 20) -> str:
    """List UniProtKB sequence features.

    Args:
        accession: UniProt accession, e.g. "P01308".
        feature_type: Optional feature type filter, e.g. "Domain" or "Active site".
        max_results: Number of features to return (1-25). Defaults to 20.
    """
    normalized = accession.strip().upper()
    if not _ACCESSION_RE.fullmatch(normalized):
        return f"UniProt features failed: invalid accession: {accession!r}"
    if feature_type and not _valid_text(feature_type):
        return "UniProt features failed: invalid feature_type."
    try:
        data = _fetch_json(f"/{normalized}", {"format": "json"})
        features = data.get("features", [])
    except urllib.error.HTTPError as e:
        return _http_error("UniProt features failed", e)
    except urllib.error.URLError as e:
        return f"UniProt features failed: URL error: {e.reason}"
    except TimeoutError:
        return "UniProt features failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"UniProt features failed: could not parse API response: {e}"

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
        return f"No UniProt features found for {normalized}."
    lines = [f"UniProt features for {normalized}:"]
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            continue
        location = _feature_location(feature.get("location"))
        lines.append(
            f"{index}. {_string(feature.get('type')) or '?'} | {location or '?'} | "
            f"{_string(feature.get('description')) or '(no description)'}"
        )
    return "\n".join(lines)


@tool
def uniprot_sequence(accession: str) -> str:
    """Get a UniProtKB protein sequence in FASTA form.

    Args:
        accession: UniProt accession, e.g. "P01308".
    """
    normalized = accession.strip().upper()
    if not _ACCESSION_RE.fullmatch(normalized):
        return f"UniProt sequence failed: invalid accession: {accession!r}"
    try:
        text = _fetch_text(f"/{normalized}.fasta", {})
    except urllib.error.HTTPError as e:
        return _http_error("UniProt sequence failed", e)
    except urllib.error.URLError as e:
        return f"UniProt sequence failed: URL error: {e.reason}"
    except TimeoutError:
        return "UniProt sequence failed: request timed out."

    return text.strip() or f"No UniProt sequence found for {normalized}."


@tool
def uniprot_crossrefs(accession: str, database: str = "", max_results: int = 25) -> str:
    """List UniProtKB database cross-references.

    Args:
        accession: UniProt accession, e.g. "P01308".
        database: Optional database filter, e.g. "PDB", "Reactome", or "ChEMBL".
        max_results: Number of cross-references to return (1-25). Defaults to 25.
    """
    normalized = accession.strip().upper()
    if not _ACCESSION_RE.fullmatch(normalized):
        return f"UniProt cross-references failed: invalid accession: {accession!r}"
    if database and not _valid_text(database):
        return "UniProt cross-references failed: invalid database."
    try:
        data = _fetch_json(f"/{normalized}", {"format": "json"})
        refs = data.get("uniProtKBCrossReferences", [])
    except urllib.error.HTTPError as e:
        return _http_error("UniProt cross-references failed", e)
    except urllib.error.URLError as e:
        return f"UniProt cross-references failed: URL error: {e.reason}"
    except TimeoutError:
        return "UniProt cross-references failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"UniProt cross-references failed: could not parse API response: {e}"

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
        return f"No UniProt cross-references found for {normalized}."
    lines = [f"UniProt cross-references for {normalized}:"]
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


def _fetch_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    return json.loads(_fetch_text(path, params))


def _fetch_text(path: str, params: dict[str, str]) -> str:
    url = f"{_BASE_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


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


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 404:
        return f"{prefix}: no matching records found."
    if error.code == 429:
        return f"{prefix}: rate limited by UniProt (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


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
