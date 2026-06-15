"""Overpass tools — public OpenStreetMap object queries."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ai_arch_toolkit.core import tool

_API_URL = "https://overpass-api.de/api/interpreter"
_TIMEOUT = 35
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_LIMIT = 50
_TAG_RE = re.compile(r"^[A-Za-z0-9_:-]{1,80}$")
_VALUE_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,120}$", re.UNICODE)


@tool
def overpass_query(query: str, max_results: int = 25) -> str:
    """Run a bounded Overpass QL query and summarize returned OSM elements.

    Args:
        query: Complete Overpass QL query. It should include output format and timeout.
        max_results: Number of elements to return (1-50). Defaults to 25.
    """
    if not query.strip() or len(query) > 4000:
        return "Overpass query failed: query must be 1-4000 characters."
    if "[out:" not in query or "out" not in query:
        return "Overpass query failed: include [out:json] and an out statement."
    try:
        data = _fetch_overpass(query)
        elements = _elements(data)
    except urllib.error.HTTPError as e:
        return _http_error("Overpass query failed", e)
    except urllib.error.URLError as e:
        return f"Overpass query failed: URL error: {e.reason}"
    except TimeoutError:
        return "Overpass query failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Overpass query failed: could not parse API response: {e}"

    if not elements:
        return "No Overpass elements found."
    page = elements[: _bounded(max_results)]
    return _format_elements(
        page, header=f"Overpass elements (returned {len(page)} of {len(elements)}):"
    )


@tool
def overpass_pois(
    tag_key: str,
    tag_value: str = "",
    bbox: str = "",
    latitude: float | None = None,
    longitude: float | None = None,
    radius_m: int = 1000,
    max_results: int = 25,
) -> str:
    """Search OpenStreetMap points/ways/relations by tag in a bbox or radius.

    Args:
        tag_key: OSM tag key, e.g. "amenity", "shop", or "tourism".
        tag_value: Optional exact tag value, e.g. "hospital" or "cafe".
        bbox: Optional south,west,north,east bounding box.
        latitude: Optional center latitude for radius search.
        longitude: Optional center longitude for radius search.
        radius_m: Radius in meters when latitude/longitude are provided. Defaults to 1000.
        max_results: Number of elements to return (1-50). Defaults to 25.
    """
    if not _TAG_RE.fullmatch(tag_key.strip()):
        return "Overpass POI search failed: invalid tag_key."
    if tag_value and not _VALUE_RE.fullmatch(tag_value.strip()):
        return "Overpass POI search failed: invalid tag_value."
    area = _area_clause(bbox, latitude, longitude, radius_m)
    if area.startswith("error:"):
        return f"Overpass POI search failed: {area.removeprefix('error:')}"
    tag = (
        f'["{tag_key.strip()}"="{tag_value.strip()}"]'
        if tag_value.strip()
        else f'["{tag_key.strip()}"]'
    )
    selector = f"{tag}{area}"
    query = (
        "[out:json][timeout:25];"
        "("
        f"node{selector};"
        f"way{selector};"
        f"relation{selector};"
        ");"
        "out center tags;"
    )
    try:
        data = _fetch_overpass(query)
        elements = _elements(data)
    except urllib.error.HTTPError as e:
        return _http_error("Overpass POI search failed", e)
    except urllib.error.URLError as e:
        return f"Overpass POI search failed: URL error: {e.reason}"
    except TimeoutError:
        return "Overpass POI search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Overpass POI search failed: could not parse API response: {e}"

    if not elements:
        return "No Overpass POIs found."
    page = elements[: _bounded(max_results)]
    return _format_elements(
        page, header=f"Overpass POIs (returned {len(page)} of {len(elements)}):"
    )


def _fetch_overpass(query: str) -> dict[str, Any]:
    body = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        _API_URL,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _area_clause(
    bbox: str,
    latitude: float | None,
    longitude: float | None,
    radius_m: int,
) -> str:
    if bbox.strip():
        parts = [part.strip() for part in bbox.split(",")]
        if len(parts) != 4:
            return "error:bbox must be south,west,north,east."
        try:
            south, west, north, east = [float(part) for part in parts]
        except ValueError:
            return "error:bbox values must be numeric."
        if not (-90 <= south <= north <= 90 and -180 <= west <= east <= 180):
            return "error:invalid bbox coordinate order or range."
        return f"({south},{west},{north},{east})"
    if latitude is None or longitude is None:
        return "error:provide bbox or latitude/longitude."
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return "error:invalid latitude/longitude."
    if radius_m <= 0 or radius_m > 50000:
        return "error:radius_m must be between 1 and 50000."
    return f"(around:{radius_m},{latitude},{longitude})"


def _elements(data: dict[str, Any]) -> list[dict[str, Any]]:
    value = data.get("elements")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _format_elements(elements: list[dict[str, Any]], *, header: str) -> str:
    lines = [header]
    for index, item in enumerate(elements, start=1):
        tags = item.get("tags", {}) if isinstance(item.get("tags"), dict) else {}
        name = _string(tags.get("name")) or "(unnamed)"
        element_id = _string(item.get("id"))
        element_type = _string(item.get("type"))
        lat, lon = _coords(item)
        lines.append(
            f"{index}. {name} | {element_type}/{element_id} | coords: {lat or '?'}, {lon or '?'}"
        )
        interesting = []
        for key in ("amenity", "shop", "tourism", "leisure", "website", "phone", "opening_hours"):
            if _string(tags.get(key)):
                interesting.append(f"{key}={_string(tags.get(key))}")
        if interesting:
            lines.append(f"   tags: {'; '.join(interesting)}")
    return "\n".join(lines)


def _coords(item: dict[str, Any]) -> tuple[str, str]:
    lat = _string(item.get("lat"))
    lon = _string(item.get("lon"))
    if not lat and isinstance(item.get("center"), dict):
        lat = _string(item["center"].get("lat"))
        lon = _string(item["center"].get("lon"))
    return lat, lon


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 429:
        return f"{prefix}: rate limited by Overpass (HTTP 429). Try again later."
    if error.code == 504:
        return f"{prefix}: Overpass query timed out upstream (HTTP 504)."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
