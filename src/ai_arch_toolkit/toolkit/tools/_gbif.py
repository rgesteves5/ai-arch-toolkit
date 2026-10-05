"""GBIF tools — public biodiversity taxonomy and occurrence lookup."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, NoReturn

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api

_API = Api(base="https://api.gbif.org/v1", name="GBIF", timeout_s=15)
_MAX_LIMIT = 50
_TEXT_RE = re.compile(r"^[\w\s,.'()/-]{1,160}$", re.UNICODE)
_KEY_RE = re.compile(r"^\d+$")
_CODE_RE = re.compile(r"^[A-Za-z_ -]{0,80}$")
_TEXT_HINT = "pass 1-160 characters of letters, digits and basic punctuation"
_RANK_HINT = "pass a rank name such as SPECIES, GENUS or FAMILY."
_KEY_HINT = "a GBIF taxon key is a number; gbif_species_match finds it."


@tool(capability="network")
def gbif_species_match(name: str, rank: str = "", kingdom: str = "") -> str:
    """Resolve a scientific name to the best GBIF taxon match.

    Args:
        name: Scientific or common taxon name to resolve.
        rank: Optional taxonomic rank hint, e.g. "species" or "genus".
        kingdom: Optional kingdom hint, e.g. "Animalia" or "Plantae".

    Raises:
        ToolFailure: validation_error when ``name``, ``rank`` or ``kingdom`` is malformed.
    """
    if not _valid_text(name):
        _invalid(f"invalid name {name!r}; {_TEXT_HINT}, e.g. 'Puma concolor'.")
    if rank and not _CODE_RE.fullmatch(rank):
        _invalid(f"invalid rank {rank!r}; {_RANK_HINT}")
    if kingdom and not _valid_text(kingdom):
        _invalid(f"invalid kingdom {kingdom!r}; pass a kingdom name, e.g. 'Animalia'.")

    params = {"name": name.strip()}
    if rank.strip():
        params["rank"] = rank.strip().upper()
    if kingdom.strip():
        params["kingdom"] = kingdom.strip()
    return _API.get_json(
        "species", "match", params=params, parse=lambda data: _match_text(data, name)
    )


@tool(capability="network")
def gbif_species_search(
    query: str,
    rank: str = "",
    highertaxon_key: str = "",
    max_results: int = 10,
    offset: int = 0,
) -> str:
    """Search GBIF taxa.

    Args:
        query: Taxon name search text.
        rank: Optional rank filter, e.g. "SPECIES", "GENUS", or "FAMILY".
        highertaxon_key: Optional parent taxon key filter.
        max_results: Number of taxa to return (1-50). Defaults to 10.
        offset: Zero-based result offset. Defaults to 0.

    Raises:
        ToolFailure: validation_error when ``query``, ``rank`` or ``highertaxon_key`` is
            malformed, or ``offset`` is negative.
    """
    if not _valid_text(query):
        _invalid(f"invalid query {query!r}; {_TEXT_HINT}, e.g. 'Puma'.")
    if offset < 0:
        _invalid(f"offset must be greater than or equal to 0, got {offset}.")
    if rank and not _CODE_RE.fullmatch(rank):
        _invalid(f"invalid rank {rank!r}; {_RANK_HINT}")
    if highertaxon_key and not _KEY_RE.fullmatch(highertaxon_key.strip()):
        _invalid(f"invalid highertaxon_key {highertaxon_key!r}; {_KEY_HINT}")

    params = {
        "q": query.strip(),
        "limit": str(_bounded(max_results)),
        "offset": str(offset),
    }
    if rank.strip():
        params["rank"] = rank.strip().upper()
    if highertaxon_key.strip():
        params["highertaxonKey"] = highertaxon_key.strip()
    header = f"GBIF taxa for {query!r}"
    return _API.get_json(
        "species",
        "search",
        params=params,
        parse=lambda data: _page(data, header, offset, "No GBIF taxa found.", _format_taxon),
    )


@tool(capability="network")
def gbif_species(taxon_key: str) -> str:
    """Get GBIF taxon metadata by taxon key.

    Args:
        taxon_key: GBIF taxon key, usually from gbif_species_match or gbif_species_search.

    Raises:
        ToolFailure: validation_error when ``taxon_key`` is not a number; not_found when GBIF
            has no taxon with that key.
    """
    key = taxon_key.strip()
    if not _KEY_RE.fullmatch(key):
        _invalid(f"invalid taxon_key {taxon_key!r}; {_KEY_HINT}")
    return _API.get_json(
        "species",
        key,
        parse=lambda data: "\n".join([f"GBIF taxon {key}:", *_format_taxon(data, index=None)]),
        missing=f"GBIF has no taxon {key}; find its key with gbif_species_match",
    )


@tool(capability="network")
def gbif_occurrence_search(
    taxon_key: str = "",
    country: str = "",
    year: str = "",
    has_coordinate: bool = True,
    max_results: int = 10,
    offset: int = 0,
) -> str:
    """Search GBIF species occurrence records.

    Args:
        taxon_key: Optional GBIF taxon key.
        country: Optional ISO 3166-1 alpha-2 country code, e.g. "PT".
        year: Optional collection year or range accepted by GBIF, e.g. "2020" or "2020,2024".
        has_coordinate: Whether to restrict to georeferenced records. Defaults to True.
        max_results: Number of occurrences to return (1-50). Defaults to 10.
        offset: Zero-based result offset. Defaults to 0.

    Raises:
        ToolFailure: validation_error when an option is malformed or none of ``taxon_key``,
            ``country`` and ``year`` is given.
    """
    if offset < 0:
        _invalid(f"offset must be greater than or equal to 0, got {offset}.")
    if taxon_key and not _KEY_RE.fullmatch(taxon_key.strip()):
        _invalid(f"invalid taxon_key {taxon_key!r}; {_KEY_HINT}")
    if country and not re.fullmatch(r"^[A-Za-z]{2}$", country.strip()):
        _invalid(f"invalid country code {country!r}; use ISO 3166-1 alpha-2, e.g. 'PT'.")
    if year and not re.fullmatch(r"^\d{4}(,\d{4})?$", year.strip()):
        _invalid(f"invalid year {year!r}; use YYYY or YYYY,YYYY, e.g. '2020,2024'.")
    if not any((taxon_key.strip(), country.strip(), year.strip())):
        _invalid("provide taxon_key, country, or year to narrow the occurrences.")

    params = {
        "limit": str(_bounded(max_results)),
        "offset": str(offset),
        "hasCoordinate": str(has_coordinate).lower(),
    }
    filters = {
        "taxonKey": taxon_key.strip(),
        "country": country.strip().upper(),
        "year": year.strip(),
    }
    params.update({key: value for key, value in filters.items() if value})
    return _API.get_json(
        "occurrence",
        "search",
        params=params,
        parse=lambda data: _page(
            data, "GBIF occurrences", offset, "No GBIF occurrences found.", _format_occurrence
        ),
    )


def _match_text(data: dict[str, Any], name: str) -> str:
    usage_key = _string(data.get("usageKey"))
    if not usage_key:
        return "No GBIF species match found."
    status = _string(data.get("status"))
    match_type = _string(data.get("matchType"))
    lines = [
        f"GBIF species match for {name!r}:",
        f"{_string(data.get('scientificName')) or _string(data.get('canonicalName'))}",
        f"   usageKey: {usage_key} | status: {status} | match: {match_type}",
    ]
    rank_text = _string(data.get("rank"))
    confidence = _string(data.get("confidence"))
    if rank_text or confidence:
        lines.append(f"   rank: {rank_text or '?'} | confidence: {confidence or '?'}")
    classification = _classification(data)
    if classification:
        lines.append(f"   classification: {classification}")
    return "\n".join(lines)


def _page(
    data: dict[str, Any],
    header: str,
    offset: int,
    nothing_found: str,
    format_item: Callable[..., list[str]],
) -> str:
    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        return nothing_found
    total = _string(data.get("count")) or "?"
    lines = [f"{header} (returned {len(results)}, total {total}, offset {offset}):"]
    for index, item in enumerate(results, start=1):
        if isinstance(item, dict):
            lines.extend(format_item(item, index=index))
    return "\n".join(lines)


def _format_occurrence(item: dict[str, Any], *, index: int) -> list[str]:
    name = _string(item.get("scientificName")) or _string(item.get("species"))
    key = _string(item.get("key"))
    place = ", ".join(
        part
        for part in (
            _string(item.get("locality")),
            _string(item.get("stateProvince")),
            _string(item.get("country")),
        )
        if part
    )
    coords = _coords(item)
    event_date = _string(item.get("eventDate")) or _string(item.get("year"))
    lines = [
        f"{index}. {name} | occurrence key: {key}",
        f"   date: {event_date or '?'} | place: {place or '?'} | coords: {coords or '?'}",
    ]
    dataset = _string(item.get("datasetName"))
    if dataset:
        lines.append(f"   dataset: {dataset}")
    return lines


def _format_taxon(item: dict[str, Any], *, index: int | None) -> list[str]:
    prefix = f"{index}. " if index is not None else ""
    key = _string(item.get("key")) or _string(item.get("usageKey"))
    name = _string(item.get("scientificName")) or _string(item.get("canonicalName"))
    lines = [f"{prefix}{name} | key: {key}"]
    meta = [
        f"rank: {_string(item.get('rank')) or '?'}",
        f"status: {_string(item.get('taxonomicStatus')) or _string(item.get('status')) or '?'}",
    ]
    accepted = _string(item.get("acceptedKey"))
    if accepted:
        meta.append(f"acceptedKey: {accepted}")
    lines.append("   " + " | ".join(meta))
    classification = _classification(item)
    if classification:
        lines.append(f"   classification: {classification}")
    return lines


def _classification(data: dict[str, Any]) -> str:
    parts = []
    for field in ("kingdom", "phylum", "class", "order", "family", "genus", "species"):
        value = _string(data.get(field))
        if value:
            parts.append(value)
    return " > ".join(parts)


def _coords(item: dict[str, Any]) -> str:
    lat = _string(item.get("decimalLatitude"))
    lon = _string(item.get("decimalLongitude"))
    return f"{lat}, {lon}" if lat and lon else ""


def _invalid(message: str) -> NoReturn:
    raise ToolFailure("validation_error", message)


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
