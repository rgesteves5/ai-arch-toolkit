"""GBIF tools — public biodiversity taxonomy and occurrence lookup."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://api.gbif.org/v1"
_TIMEOUT = 15
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_LIMIT = 50
_TEXT_RE = re.compile(r"^[\w\s,.'()/-]{1,160}$", re.UNICODE)
_KEY_RE = re.compile(r"^\d+$")
_CODE_RE = re.compile(r"^[A-Za-z_ -]{0,80}$")


@tool
def gbif_species_match(name: str, rank: str = "", kingdom: str = "") -> str:
    """Resolve a scientific name to the best GBIF taxon match.

    Args:
        name: Scientific or common taxon name to resolve.
        rank: Optional taxonomic rank hint, e.g. "species" or "genus".
        kingdom: Optional kingdom hint, e.g. "Animalia" or "Plantae".
    """
    if not _valid_text(name):
        return "GBIF species match failed: invalid name."
    if rank and not _CODE_RE.fullmatch(rank):
        return "GBIF species match failed: invalid rank."
    if kingdom and not _valid_text(kingdom):
        return "GBIF species match failed: invalid kingdom."

    params = {"name": name.strip()}
    if rank.strip():
        params["rank"] = rank.strip().upper()
    if kingdom.strip():
        params["kingdom"] = kingdom.strip()
    try:
        data = _fetch_json("/species/match", params)
    except urllib.error.HTTPError as e:
        return _http_error("GBIF species match failed", e)
    except urllib.error.URLError as e:
        return f"GBIF species match failed: URL error: {e.reason}"
    except TimeoutError:
        return "GBIF species match failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"GBIF species match failed: could not parse API response: {e}"

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


@tool
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
    """
    if not _valid_text(query):
        return "GBIF species search failed: invalid query."
    if offset < 0:
        return "GBIF species search failed: offset must be greater than or equal to 0."
    if rank and not _CODE_RE.fullmatch(rank):
        return "GBIF species search failed: invalid rank."
    if highertaxon_key and not _KEY_RE.fullmatch(highertaxon_key.strip()):
        return "GBIF species search failed: invalid highertaxon_key."

    params = {
        "q": query.strip(),
        "limit": str(_bounded(max_results)),
        "offset": str(offset),
    }
    if rank.strip():
        params["rank"] = rank.strip().upper()
    if highertaxon_key.strip():
        params["highertaxonKey"] = highertaxon_key.strip()
    try:
        data = _fetch_json("/species/search", params)
        results = data.get("results", [])
    except urllib.error.HTTPError as e:
        return _http_error("GBIF species search failed", e)
    except urllib.error.URLError as e:
        return f"GBIF species search failed: URL error: {e.reason}"
    except TimeoutError:
        return "GBIF species search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"GBIF species search failed: could not parse API response: {e}"

    if not isinstance(results, list) or not results:
        return "No GBIF taxa found."
    total = _string(data.get("count")) or "?"
    lines = [f"GBIF taxa for {query!r} (returned {len(results)}, total {total}, offset {offset}):"]
    for index, item in enumerate(results, start=1):
        if isinstance(item, dict):
            lines.extend(_format_taxon(item, index=index))
    return "\n".join(lines)


@tool
def gbif_species(taxon_key: str) -> str:
    """Get GBIF taxon metadata by taxon key.

    Args:
        taxon_key: GBIF taxon key, usually from gbif_species_match or gbif_species_search.
    """
    key = taxon_key.strip()
    if not _KEY_RE.fullmatch(key):
        return f"GBIF species lookup failed: invalid taxon_key: {taxon_key!r}"
    try:
        data = _fetch_json(f"/species/{key}", {})
    except urllib.error.HTTPError as e:
        return _http_error("GBIF species lookup failed", e)
    except urllib.error.URLError as e:
        return f"GBIF species lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "GBIF species lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"GBIF species lookup failed: could not parse API response: {e}"

    lines = [f"GBIF taxon {key}:"]
    lines.extend(_format_taxon(data, index=None))
    return "\n".join(lines)


@tool
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
    """
    if offset < 0:
        return "GBIF occurrence search failed: offset must be greater than or equal to 0."
    if taxon_key and not _KEY_RE.fullmatch(taxon_key.strip()):
        return "GBIF occurrence search failed: invalid taxon_key."
    if country and not re.fullmatch(r"^[A-Za-z]{2}$", country.strip()):
        return "GBIF occurrence search failed: invalid country code."
    if year and not re.fullmatch(r"^\d{4}(,\d{4})?$", year.strip()):
        return "GBIF occurrence search failed: invalid year."
    if not any((taxon_key.strip(), country.strip(), year.strip())):
        return "GBIF occurrence search failed: provide taxon_key, country, or year."

    params = {
        "limit": str(_bounded(max_results)),
        "offset": str(offset),
        "hasCoordinate": str(has_coordinate).lower(),
    }
    if taxon_key.strip():
        params["taxonKey"] = taxon_key.strip()
    if country.strip():
        params["country"] = country.strip().upper()
    if year.strip():
        params["year"] = year.strip()
    try:
        data = _fetch_json("/occurrence/search", params)
        results = data.get("results", [])
    except urllib.error.HTTPError as e:
        return _http_error("GBIF occurrence search failed", e)
    except urllib.error.URLError as e:
        return f"GBIF occurrence search failed: URL error: {e.reason}"
    except TimeoutError:
        return "GBIF occurrence search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"GBIF occurrence search failed: could not parse API response: {e}"

    if not isinstance(results, list) or not results:
        return "No GBIF occurrences found."
    total = _string(data.get("count")) or "?"
    lines = [f"GBIF occurrences (returned {len(results)}, total {total}, offset {offset}):"]
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        name = _string(item.get("scientificName")) or _string(item.get("species"))
        key = _string(item.get("key"))
        lines.append(f"{index}. {name} | occurrence key: {key}")
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
        lines.append(
            f"   date: {event_date or '?'} | place: {place or '?'} | coords: {coords or '?'}"
        )
        dataset = _string(item.get("datasetName"))
        if dataset:
            lines.append(f"   dataset: {dataset}")
    return "\n".join(lines)


def _fetch_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{_BASE_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


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


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 404:
        return f"{prefix}: no matching records found."
    if error.code == 429:
        return f"{prefix}: rate limited by GBIF (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
