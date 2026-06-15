"""ChEMBL tools — public chemistry, target, and bioactivity lookup."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
_TIMEOUT = 20
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_LIMIT = 25
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_CHEMBL_RE = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)


@tool
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
    try:
        data = _fetch_json(
            "/molecule/search.json",
            {"q": query.strip(), "limit": str(_bounded(max_results)), "offset": str(offset)},
        )
        items = data.get("molecules", [])
    except urllib.error.HTTPError as e:
        return _http_error("ChEMBL molecule search failed", e)
    except urllib.error.URLError as e:
        return f"ChEMBL molecule search failed: URL error: {e.reason}"
    except TimeoutError:
        return "ChEMBL molecule search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"ChEMBL molecule search failed: could not parse API response: {e}"

    if not isinstance(items, list) or not items:
        return "No ChEMBL molecules found."
    total = _string(data.get("page_meta", {}).get("total_count")) or "?"
    lines = [
        f"ChEMBL molecules for {query!r} (returned {len(items)}, total {total}, offset {offset}):"
    ]
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            lines.extend(_format_molecule(item, index=index, compact=True))
    return "\n".join(lines)


@tool
def chembl_molecule(chembl_id: str) -> str:
    """Get ChEMBL molecule metadata.

    Args:
        chembl_id: ChEMBL molecule ID, e.g. "CHEMBL25".
    """
    normalized = chembl_id.strip().upper()
    if not _CHEMBL_RE.fullmatch(normalized):
        return f"ChEMBL molecule lookup failed: invalid chembl_id: {chembl_id!r}"
    try:
        data = _fetch_json(f"/molecule/{normalized}.json", {})
    except urllib.error.HTTPError as e:
        return _http_error("ChEMBL molecule lookup failed", e)
    except urllib.error.URLError as e:
        return f"ChEMBL molecule lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "ChEMBL molecule lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"ChEMBL molecule lookup failed: could not parse API response: {e}"

    lines = [f"ChEMBL molecule {normalized}:"]
    lines.extend(_format_molecule(data, index=None, compact=False))
    return "\n".join(lines)


@tool
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
    try:
        data = _fetch_json(
            "/target/search.json",
            {"q": query.strip(), "limit": str(_bounded(max_results)), "offset": str(offset)},
        )
        items = data.get("targets", [])
    except urllib.error.HTTPError as e:
        return _http_error("ChEMBL target search failed", e)
    except urllib.error.URLError as e:
        return f"ChEMBL target search failed: URL error: {e.reason}"
    except TimeoutError:
        return "ChEMBL target search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"ChEMBL target search failed: could not parse API response: {e}"

    if not isinstance(items, list) or not items:
        return "No ChEMBL targets found."
    total = _string(data.get("page_meta", {}).get("total_count")) or "?"
    lines = [
        f"ChEMBL targets for {query!r} (returned {len(items)}, total {total}, offset {offset}):"
    ]
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            lines.extend(_format_target(item, index=index))
    return "\n".join(lines)


@tool
def chembl_target(chembl_id: str) -> str:
    """Get ChEMBL target metadata.

    Args:
        chembl_id: ChEMBL target ID, e.g. "CHEMBL203".
    """
    normalized = chembl_id.strip().upper()
    if not _CHEMBL_RE.fullmatch(normalized):
        return f"ChEMBL target lookup failed: invalid chembl_id: {chembl_id!r}"
    try:
        data = _fetch_json(f"/target/{normalized}.json", {})
    except urllib.error.HTTPError as e:
        return _http_error("ChEMBL target lookup failed", e)
    except urllib.error.URLError as e:
        return f"ChEMBL target lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "ChEMBL target lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"ChEMBL target lookup failed: could not parse API response: {e}"

    lines = [f"ChEMBL target {normalized}:"]
    lines.extend(_format_target(data, index=None))
    components = data.get("target_components", [])
    if isinstance(components, list) and components:
        names = [
            _string(component.get("accession")) or _string(component.get("component_id"))
            for component in components
            if isinstance(component, dict)
        ]
        lines.append(f"   components: {', '.join(name for name in names if name) or '?'}")
    return "\n".join(lines)


@tool
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
    if not any((molecule_chembl_id.strip(), target_chembl_id.strip())):
        return "ChEMBL activity search failed: provide molecule_chembl_id or target_chembl_id."
    if molecule_chembl_id and not _CHEMBL_RE.fullmatch(molecule_chembl_id.strip()):
        return "ChEMBL activity search failed: invalid molecule_chembl_id."
    if target_chembl_id and not _CHEMBL_RE.fullmatch(target_chembl_id.strip()):
        return "ChEMBL activity search failed: invalid target_chembl_id."
    if standard_type and not _valid_text(standard_type):
        return "ChEMBL activity search failed: invalid standard_type."
    if offset < 0:
        return "ChEMBL activity search failed: offset must be greater than or equal to 0."

    params = {"limit": str(_bounded(max_results)), "offset": str(offset)}
    if molecule_chembl_id.strip():
        params["molecule_chembl_id"] = molecule_chembl_id.strip().upper()
    if target_chembl_id.strip():
        params["target_chembl_id"] = target_chembl_id.strip().upper()
    if standard_type.strip():
        params["standard_type"] = standard_type.strip()
    try:
        data = _fetch_json("/activity.json", params)
        items = data.get("activities", [])
    except urllib.error.HTTPError as e:
        return _http_error("ChEMBL activity search failed", e)
    except urllib.error.URLError as e:
        return f"ChEMBL activity search failed: URL error: {e.reason}"
    except TimeoutError:
        return "ChEMBL activity search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"ChEMBL activity search failed: could not parse API response: {e}"

    if not isinstance(items, list) or not items:
        return "No ChEMBL activities found."
    total = _string(data.get("page_meta", {}).get("total_count")) or "?"
    lines = [f"ChEMBL activities (returned {len(items)}, total {total}, offset {offset}):"]
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        value = " ".join(
            part
            for part in (
                _string(item.get("standard_relation")),
                _string(item.get("standard_value")),
                _string(item.get("standard_units")),
            )
            if part
        )
        lines.append(
            f"{index}. {_string(item.get('molecule_chembl_id'))} -> "
            f"{_string(item.get('target_chembl_id'))} | "
            f"{_string(item.get('standard_type'))}: {value or '?'}"
        )
        lines.append(
            f"   assay: {_string(item.get('assay_chembl_id')) or '?'} | "
            f"document: {_string(item.get('document_chembl_id')) or '?'}"
        )
    return "\n".join(lines)


def _fetch_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{_BASE_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


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


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 404:
        return f"{prefix}: no matching records found."
    if error.code == 429:
        return f"{prefix}: rate limited by ChEMBL (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
