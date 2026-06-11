"""DataCite tools — public DOI metadata search and lookup."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool

_API_URL = "https://api.datacite.org/dois"
_TIMEOUT = 15
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_RESULTS_LIMIT = 20
_DESCRIPTION_MAX_CHARS = 900


@dataclass(frozen=True, slots=True, kw_only=True)
class _DataCiteDoi:
    """Normalized DataCite DOI metadata."""

    doi: str
    title: str
    creators: tuple[str, ...]
    publisher: str
    publication_year: int | None
    resource_type: str
    descriptions: tuple[str, ...]
    subjects: tuple[str, ...]
    url: str
    rights: tuple[str, ...]
    related_identifiers: tuple[str, ...]


@tool
def datacite_search(
    query: str,
    resource_type: str = "",
    max_results: int = 5,
    page: int = 1,
) -> str:
    """Search DataCite DOI metadata using the public DataCite REST API.

    Args:
        query: Metadata search text.
        resource_type: Optional resourceTypeGeneral filter, e.g. Dataset, Software, Text.
        max_results: Number of DOI records to return (1-20). Defaults to 5.
        page: One-based result page. Defaults to 1.
    """
    query = query.strip()
    if not query:
        return "DataCite search failed: query cannot be empty."
    if page < 1:
        return "DataCite search failed: page must be greater than or equal to 1."

    params = {
        "query": query,
        "page[size]": str(max(1, min(max_results, _MAX_RESULTS_LIMIT))),
        "page[number]": str(page),
    }
    if resource_type.strip():
        params["resource-type-id"] = resource_type.strip().lower()

    try:
        data = _fetch_json(_API_URL, params)
        items = data.get("data", [])
        dois = [_parse_doi(item) for item in items if isinstance(item, dict)]
        dois = [doi for doi in dois if doi is not None]
    except urllib.error.HTTPError as e:
        return _http_error("DataCite search failed", e)
    except urllib.error.URLError as e:
        return f"DataCite search failed: URL error: {e.reason}"
    except TimeoutError:
        return "DataCite search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"DataCite search failed: could not parse API response: {e}"

    if not dois:
        return f"No DataCite DOI records found for: {query!r}"

    return f"DataCite DOI results for {query!r}:\n" + _format_dois(dois, include_description=False)


@tool
def datacite_doi(doi: str) -> str:
    """Fetch DataCite metadata for a specific DOI.

    Args:
        doi: DOI string or DOI URL.
    """
    normalized = _normalize_doi(doi)
    if not normalized:
        return f"DataCite DOI lookup failed: invalid DOI: {doi!r}"

    try:
        data = _fetch_json(f"{_API_URL}/{urllib.parse.quote(normalized, safe='')}", {})
        item = data.get("data")
        record = _parse_doi(item) if isinstance(item, dict) else None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"DataCite DOI not found: {normalized}"
        return _http_error("DataCite DOI lookup failed", e)
    except urllib.error.URLError as e:
        return f"DataCite DOI lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "DataCite DOI lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"DataCite DOI lookup failed: could not parse API response: {e}"

    if record is None:
        return f"DataCite DOI not found: {normalized}"

    return f"DataCite DOI {normalized}:\n" + _format_dois(
        [record],
        include_index=False,
        include_description=True,
    )


def _fetch_json(url: str, params: dict[str, str]) -> dict[str, Any]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _parse_doi(data: dict[str, Any]) -> _DataCiteDoi | None:
    attrs = data.get("attributes")
    if not isinstance(attrs, dict):
        return None
    doi = str(attrs.get("doi", "") or data.get("id", "") or "").strip()
    if not doi:
        return None
    return _DataCiteDoi(
        doi=doi,
        title=_first_title(attrs.get("titles")),
        creators=_creators(attrs.get("creators")),
        publisher=str(attrs.get("publisher", "") or "").strip(),
        publication_year=_int_or_none(attrs.get("publicationYear")),
        resource_type=_resource_type(attrs),
        descriptions=_descriptions(attrs.get("descriptions")),
        subjects=_subjects(attrs.get("subjects")),
        url=str(attrs.get("url", "") or "").strip(),
        rights=_rights(attrs.get("rightsList")),
        related_identifiers=_related_identifiers(attrs.get("relatedIdentifiers")),
    )


def _format_dois(
    dois: list[_DataCiteDoi],
    *,
    include_index: bool = True,
    include_description: bool = False,
) -> str:
    blocks: list[str] = []
    for index, doi in enumerate(dois, start=1):
        title = f"{index}. {doi.title}" if include_index else doi.title
        lines = [title]
        meta = [f"DOI: {doi.doi}"]
        if doi.resource_type:
            meta.append(f"type: {doi.resource_type}")
        if doi.publication_year is not None:
            meta.append(f"year: {doi.publication_year}")
        lines.append("   " + " | ".join(meta))
        if doi.creators:
            lines.append("   Creators: " + ", ".join(doi.creators[:8]))
        if doi.publisher:
            lines.append(f"   Publisher: {doi.publisher}")
        if doi.subjects:
            lines.append("   Subjects: " + ", ".join(doi.subjects[:10]))
        if include_description and doi.descriptions:
            lines.append(
                f"   Description: {_truncate(doi.descriptions[0], _DESCRIPTION_MAX_CHARS)}"
            )
        if doi.rights:
            lines.append("   Rights: " + " | ".join(doi.rights[:5]))
        if doi.related_identifiers:
            lines.append("   Related: " + " | ".join(doi.related_identifiers[:5]))
        if doi.url:
            lines.append(f"   URL: {doi.url}")
        lines.append(f"   DataCite: https://commons.datacite.org/doi.org/{doi.doi}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _normalize_doi(value: str) -> str:
    doi = value.strip()
    if not doi:
        return ""
    lower = doi.lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if lower.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    if doi.lower().startswith("doi:"):
        doi = doi[4:]
    doi = doi.strip()
    if not doi.lower().startswith("10.") or "/" not in doi or " " in doi:
        return ""
    return doi


def _first_title(value: Any) -> str:
    for item in value or []:
        if isinstance(item, dict):
            title = str(item.get("title", "") or "").strip()
            if title:
                return title
    return "(untitled)"


def _creators(value: Any) -> tuple[str, ...]:
    creators: list[str] = []
    for item in value or []:
        if isinstance(item, dict):
            name = str(item.get("name", "") or "").strip()
            if name:
                creators.append(name)
    return tuple(creators)


def _resource_type(attrs: dict[str, Any]) -> str:
    types = attrs.get("types")
    if isinstance(types, dict):
        for key in ("resourceTypeGeneral", "resourceType"):
            value = str(types.get(key, "") or "").strip()
            if value:
                return value
    return ""


def _descriptions(value: Any) -> tuple[str, ...]:
    descriptions: list[str] = []
    for item in value or []:
        if isinstance(item, dict):
            text = str(item.get("description", "") or "").strip()
            if text:
                descriptions.append(" ".join(text.split()))
    return tuple(descriptions)


def _subjects(value: Any) -> tuple[str, ...]:
    subjects: list[str] = []
    for item in value or []:
        if isinstance(item, dict):
            subject = str(item.get("subject", "") or "").strip()
            if subject:
                subjects.append(subject)
    return tuple(subjects)


def _rights(value: Any) -> tuple[str, ...]:
    rights: list[str] = []
    for item in value or []:
        if isinstance(item, dict):
            text = str(item.get("rights", "") or item.get("rightsUri", "") or "").strip()
            if text:
                rights.append(text)
    return tuple(rights)


def _related_identifiers(value: Any) -> tuple[str, ...]:
    related: list[str] = []
    for item in value or []:
        if isinstance(item, dict):
            identifier = str(item.get("relatedIdentifier", "") or "").strip()
            relation = str(item.get("relationType", "") or "").strip()
            if identifier and relation:
                related.append(f"{relation}: {identifier}")
            elif identifier:
                related.append(identifier)
    return tuple(related)


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 429:
        return f"{prefix}: rate limited by DataCite (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    try:
        if value is not None and str(value).strip():
            return int(value)
    except ValueError:
        return None
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"
