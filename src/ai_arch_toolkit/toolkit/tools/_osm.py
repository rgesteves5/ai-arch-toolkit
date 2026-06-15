"""OpenStreetMap Nominatim tools — public place search and reverse geocoding."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://nominatim.openstreetmap.org"
_TIMEOUT = 15
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit; research tool)"
_MIN_REQUEST_INTERVAL_SECONDS = 1.1
_LAST_REQUEST_AT = 0.0
_MAX_RESULTS_LIMIT = 10
_COUNTRY_CODES_RE = re.compile(r"^[a-zA-Z]{2}(,[a-zA-Z]{2})*$")
_LAYERS = {"address", "poi", "railway", "natural", "manmade"}


@dataclass(frozen=True, slots=True, kw_only=True)
class _OsmPlace:
    """Normalized Nominatim place result."""

    place_id: str
    osm_type: str
    osm_id: str
    display_name: str
    category: str
    type: str
    importance: float | None
    latitude: str
    longitude: str
    bounding_box: tuple[str, ...]
    address: tuple[str, ...]
    extra_tags: tuple[str, ...]


@tool
def osm_search_place(
    query: str,
    max_results: int = 5,
    country_codes: str = "",
    layer: str = "",
    accept_language: str = "en",
    include_extra_tags: bool = False,
) -> str:
    """Search places with OpenStreetMap Nominatim.

    Args:
        query: Free-form place or address query. Do not use for autocomplete or bulk geocoding.
        max_results: Number of places to return (1-10). Defaults to 5.
        country_codes: Optional comma-separated ISO 3166-1 alpha-2 filters, e.g. "pt,es".
        layer: Optional layer filter: address, poi, railway, natural, manmade, or comma-separated.
        accept_language: Preferred result language. Defaults to "en".
        include_extra_tags: Include selected OSM extra tags when available. Defaults to False.
    """
    query = query.strip()
    if not query:
        return "OSM place search failed: query cannot be empty."
    country_codes = country_codes.strip().lower()
    if country_codes and not _COUNTRY_CODES_RE.fullmatch(country_codes):
        return "OSM place search failed: country_codes must be comma-separated 2-letter codes."
    parsed_layers = _parse_layers(layer)
    if isinstance(parsed_layers, str):
        return f"OSM place search failed: {parsed_layers}"

    params = {
        "format": "jsonv2",
        "q": query,
        "limit": str(max(1, min(max_results, _MAX_RESULTS_LIMIT))),
        "addressdetails": "1",
        "extratags": "1" if include_extra_tags else "0",
        "accept-language": accept_language.strip() or "en",
    }
    if country_codes:
        params["countrycodes"] = country_codes
    if parsed_layers:
        params["layer"] = ",".join(parsed_layers)

    try:
        data = _fetch_json("/search", params)
        places = [_parse_place(item) for item in data if isinstance(item, dict)]
        places = [place for place in places if place is not None]
    except urllib.error.HTTPError as e:
        return _http_error("OSM place search failed", e)
    except urllib.error.URLError as e:
        return f"OSM place search failed: URL error: {e.reason}"
    except TimeoutError:
        return "OSM place search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"OSM place search failed: could not parse API response: {e}"

    if not places:
        return f"No OSM places found for: {query!r}"
    return f"OSM places for {query!r}:\n" + _format_places(places)


@tool
def osm_reverse_geocode(
    latitude: float,
    longitude: float,
    zoom: int = 18,
    layer: str = "address,poi",
    accept_language: str = "en",
    include_extra_tags: bool = False,
) -> str:
    """Reverse geocode coordinates with OpenStreetMap Nominatim.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        zoom: Address detail zoom level (3-18). Defaults to 18.
        layer: Layer filter: address, poi, railway, natural, manmade, or comma-separated.
        accept_language: Preferred result language. Defaults to "en".
        include_extra_tags: Include selected OSM extra tags when available. Defaults to False.
    """
    validation = _validate_location(latitude, longitude)
    if validation:
        return f"OSM reverse geocode failed: {validation}"
    zoom = max(3, min(zoom, 18))
    parsed_layers = _parse_layers(layer)
    if isinstance(parsed_layers, str):
        return f"OSM reverse geocode failed: {parsed_layers}"

    try:
        data = _fetch_json(
            "/reverse",
            {
                "format": "jsonv2",
                "lat": str(latitude),
                "lon": str(longitude),
                "zoom": str(zoom),
                "addressdetails": "1",
                "extratags": "1" if include_extra_tags else "0",
                "accept-language": accept_language.strip() or "en",
                "layer": ",".join(parsed_layers),
            },
        )
        place = _parse_place(data) if isinstance(data, dict) else None
    except urllib.error.HTTPError as e:
        return _http_error("OSM reverse geocode failed", e)
    except urllib.error.URLError as e:
        return f"OSM reverse geocode failed: URL error: {e.reason}"
    except TimeoutError:
        return "OSM reverse geocode failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"OSM reverse geocode failed: could not parse API response: {e}"

    if place is None:
        return f"No OSM reverse geocode result for: {latitude}, {longitude}"
    return f"OSM reverse geocode for {latitude}, {longitude}:\n" + _format_places(
        [place],
        include_index=False,
    )


def _fetch_json(path: str, params: dict[str, str]) -> Any:
    url = f"{_BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    _throttle()
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _throttle() -> None:
    global _LAST_REQUEST_AT

    now = time.monotonic()
    elapsed = now - _LAST_REQUEST_AT
    if elapsed < _MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    _LAST_REQUEST_AT = time.monotonic()


def _parse_place(data: dict[str, Any]) -> _OsmPlace | None:
    display_name = _string(data.get("display_name") or data.get("name"))
    if not display_name:
        return None
    return _OsmPlace(
        place_id=_string(data.get("place_id")),
        osm_type=_string(data.get("osm_type")),
        osm_id=_string(data.get("osm_id")),
        display_name=display_name,
        category=_string(data.get("category") or data.get("class")),
        type=_string(data.get("type")),
        importance=_float_or_none(data.get("importance")),
        latitude=_string(data.get("lat")),
        longitude=_string(data.get("lon")),
        bounding_box=_string_tuple(data.get("boundingbox")),
        address=_address_parts(data.get("address")),
        extra_tags=_extra_tags(data.get("extratags")),
    )


def _format_places(places: list[_OsmPlace], *, include_index: bool = True) -> str:
    blocks: list[str] = []
    for index, place in enumerate(places, start=1):
        title = f"{index}. {place.display_name}" if include_index else place.display_name
        lines = [title]
        meta = []
        if place.osm_type or place.osm_id:
            meta.append(f"OSM: {place.osm_type}/{place.osm_id}")
        if place.category or place.type:
            meta.append(f"type: {place.category}/{place.type}")
        if place.importance is not None:
            meta.append(f"importance: {place.importance:.4g}")
        if place.latitude and place.longitude:
            meta.append(f"coords: {place.latitude}, {place.longitude}")
        if meta:
            lines.append("   " + " | ".join(meta))
        if place.address:
            lines.append("   Address: " + " | ".join(place.address[:10]))
        if place.bounding_box:
            lines.append("   Bounding box: " + ", ".join(place.bounding_box))
        if place.extra_tags:
            lines.append("   Extra tags: " + " | ".join(place.extra_tags[:8]))
        lines.append("   Data: © OpenStreetMap contributors, ODbL")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parse_layers(value: str) -> tuple[str, ...] | str:
    layers = tuple(
        dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip())
    )
    invalid = [layer for layer in layers if layer not in _LAYERS]
    if invalid:
        return f"invalid layer(s): {', '.join(invalid)}"
    return layers


def _validate_location(latitude: float, longitude: float) -> str:
    if not -90 <= latitude <= 90:
        return "latitude must be between -90 and 90."
    if not -180 <= longitude <= 180:
        return "longitude must be between -180 and 180."
    return ""


def _address_parts(value: Any) -> tuple[str, ...]:
    if not isinstance(value, dict):
        return ()
    parts = []
    for key in (
        "house_number",
        "road",
        "neighbourhood",
        "suburb",
        "city",
        "town",
        "village",
        "county",
        "state",
        "postcode",
        "country",
    ):
        text = _string(value.get(key))
        if text:
            parts.append(f"{key}: {text}")
    return tuple(parts)


def _extra_tags(value: Any) -> tuple[str, ...]:
    if not isinstance(value, dict):
        return ()
    tags = []
    for key in sorted(value)[:12]:
        text = _string(value.get(key))
        if text:
            tags.append(f"{key}: {text}")
    return tuple(tags)


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 429:
        return f"{prefix}: rate limited by Nominatim (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(_string(item) for item in value if _string(item))
    text = _string(value)
    return (text,) if text else ()


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if value is not None and str(value).strip():
            return float(value)
    except ValueError:
        return None
    return None
