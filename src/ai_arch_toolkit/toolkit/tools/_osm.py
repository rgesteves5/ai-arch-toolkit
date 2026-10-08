"""OpenStreetMap Nominatim tools: places by name or address, and the place at a point (free, no
key; https://nominatim.org/release-docs/latest/api/Overview/)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._first_results import first_results_window
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._values import plain

# What a reverse lookup answers, with HTTP 200, where it finds nothing: no failure, but nothing.
_NOTHING_HERE = "Unable to geocode"


def _nominatim_error(reply: Reply) -> ToolFailure | str | None:
    """The error a Nominatim answer reports; ``None`` for a result.

    Nominatim refuses a request with an HTTP error and ``{"error": {"code": 400, "message":
    "..."}}``: a parameter it cannot read ("Parameter 'lat' must be a number.") is a 400, the
    caller's to correct (``_format_error`` in src/nominatim_api/v1/format.py and ``raise_error``
    in src/nominatim_api/server/asgi_adaptor.py, https://github.com/osm-search/Nominatim). A
    reverse lookup that finds nothing answers HTTP 200 with ``{"error": "Unable to geocode"}``
    (https://nominatim.org/release-docs/latest/api/Reverse/: "exactly one result or an error";
    the text is in Nominatim's test/bdd/features/api/reverse/v1_json.feature), which the tool
    reports as nothing found.
    """
    body = reply.body
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = _string(error.get("message")) or "no reason given"
    elif isinstance(error, str) and _string(error):
        if _string(error).rstrip(".") == _NOTHING_HERE:
            return None
        message = _string(error)
    else:
        return None
    if reply.status == 400:
        return ToolFailure(
            "validation_error",
            f"Nominatim refused the request: {message.rstrip('.')}; correct that argument",
        )
    return message


# Nominatim's usage policy allows one request per second:
# https://operations.osmfoundation.org/policies/nominatim/
_NOMINATIM = Api(
    base="https://nominatim.openstreetmap.org",
    name="Nominatim",
    timeout_s=15,
    min_interval_s=1.1,
    error_reader=_nominatim_error,
)
# A search returns at most 40 places, and has no offset
# (https://nominatim.org/release-docs/latest/api/Search/, "limit"). The tool always asks for the
# 40: Nominatim's first results change with the limit (``_first_results``).
_DEPTH = 40
_COUNTRY_CODES_RE = re.compile(r"^[a-zA-Z]{2}(,[a-zA-Z]{2})*$")
_LAYERS = {"address", "poi", "railway", "natural", "manmade"}
_ATTRIBUTION = "Nominatim; data © OpenStreetMap contributors, ODbL"


@dataclass(frozen=True, slots=True, kw_only=True)
class _OsmPlace:
    """A Nominatim place, as the tools show it."""

    osm_type: str
    osm_id: str
    display_name: str
    category: str
    type: str
    latitude: str
    longitude: str
    bounding_box: tuple[str, ...]
    address: tuple[str, ...]
    extra_tags: tuple[str, ...]


@tool(capability="network")
def osm_search_place(
    query: str,
    max_results: Annotated[int, Range(1, _DEPTH)] = 5,
    offset: Annotated[int, Range(0, _DEPTH - 1)] = 0,
    country_codes: str = "",
    layer: str = "",
    accept_language: str = "en",
    include_extra_tags: bool = False,
) -> ToolResult:
    """Search places and addresses in OpenStreetMap, with Nominatim, in its order of relevance.

    Args:
        query: A place or an address, free-form, e.g. "Belém Tower, Lisbon". Not for
            autocomplete or bulk geocoding (Nominatim's usage policy).
        max_results: How many places to list.
        offset: How many places to skip; the footer gives the next offset.
        country_codes: Comma-separated ISO 3166-1 alpha-2 codes to search in, e.g. "pt,es".
        layer: Comma-separated themes to search: address, poi, railway, natural, manmade.
        accept_language: The language of the names, e.g. "en" or "pt".
        include_extra_tags: Add OSM's extra tags (website, opening hours, Wikidata…).

    Raises:
        ToolFailure: validation_error when the query is empty, or a country code or a layer is
            invalid.
    """
    query = query.strip()
    if not query:
        raise ToolFailure("validation_error", "query cannot be empty; give a place or address.")
    country_codes = country_codes.strip().lower()
    if country_codes and not _COUNTRY_CODES_RE.fullmatch(country_codes):
        raise ToolFailure(
            "validation_error",
            f"invalid country_codes {country_codes!r}; give comma-separated ISO 3166-1 alpha-2 "
            "codes, e.g. 'pt,es'.",
        )
    parsed_layers = _parse_layers(layer)
    params = {
        "format": "jsonv2",
        "q": query,
        "limit": str(_DEPTH),
        "addressdetails": "1",
        "extratags": "1" if include_extra_tags else "0",
        "accept-language": accept_language.strip() or "en",
    }
    if country_codes:
        params["countrycodes"] = country_codes
    if parsed_layers:
        params["layer"] = ",".join(parsed_layers)
    return _NOMINATIM.get_json_list(
        "search",
        params=params,
        parse=lambda data: _search_answer(_places(data), query, offset, max_results),
    )


@tool(capability="network")
def osm_reverse_geocode(
    latitude: Annotated[float, Range(-90, 90)],
    longitude: Annotated[float, Range(-180, 180)],
    zoom: Annotated[int, Range(0, 18)] = 18,
    layer: str = "address,poi",
    accept_language: str = "en",
    include_extra_tags: bool = False,
) -> str:
    """Find the OpenStreetMap place at a point, with Nominatim: its name, full address and OSM
    object.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        zoom: How detailed the place is: 18 a building, 17 a street, 14 a neighbourhood, 10 a
            city, 8 a county, 5 a state, 3 a country.
        layer: Comma-separated themes to look in: address, poi, railway, natural, manmade.
        accept_language: The language of the names, e.g. "en" or "pt".
        include_extra_tags: Add OSM's extra tags (website, opening hours, Wikidata…).

    Raises:
        ToolFailure: validation_error when a layer is invalid, or Nominatim refuses an argument.
    """
    parsed_layers = _parse_layers(layer)
    params = {
        "format": "jsonv2",
        "lat": plain(latitude),
        "lon": plain(longitude),
        "zoom": str(zoom),
        "addressdetails": "1",
        "extratags": "1" if include_extra_tags else "0",
        "accept-language": accept_language.strip() or "en",
        "layer": ",".join(parsed_layers),
    }
    return _NOMINATIM.get_json(
        "reverse",
        params=params,
        parse=lambda data: _reverse_text(data, latitude, longitude, zoom),
    )


def _search_answer(places: list[_OsmPlace], query: str, offset: int, limit: int) -> ToolResult:
    if not places:
        return ToolResult.success(
            f"No OpenStreetMap places match {query!r}; try fewer words, another spelling, or "
            "no country or layer filter."
        )
    blocks = [_block(place, f"{number}. ") for number, place in enumerate(places, start=1)]
    window = first_results_window(
        blocks,
        offset=offset,
        limit=limit,
        requested=_DEPTH,
        depth=_DEPTH,
        narrow="add country_codes or layer, or a more precise query",
    )
    return window.result(heading=f"OpenStreetMap places for {query!r} ({_ATTRIBUTION}):")


def _reverse_text(data: dict[str, Any], latitude: float, longitude: float, zoom: int) -> str:
    point = f"latitude {plain(latitude)}, longitude {plain(longitude)}"
    place = _parse_place(data)
    if place is None:
        return (
            f"No OpenStreetMap place at {point} (zoom {zoom}): Nominatim has no data there, "
            "e.g. open sea."
        )
    return f"OpenStreetMap place at {point}, zoom {zoom} ({_ATTRIBUTION}):\n" + _block(place)


def _places(data: list[Any]) -> list[_OsmPlace]:
    places = [_parse_place(item) for item in data if isinstance(item, dict)]
    return [place for place in places if place is not None]


def _parse_place(data: dict[str, Any]) -> _OsmPlace | None:
    display_name = _string(data.get("display_name") or data.get("name"))
    if not display_name:
        return None
    return _OsmPlace(
        osm_type=_string(data.get("osm_type")),
        osm_id=_string(data.get("osm_id")),
        display_name=display_name,
        category=_string(data.get("category") or data.get("class")),
        type=_string(data.get("type")),
        latitude=_string(data.get("lat")),
        longitude=_string(data.get("lon")),
        bounding_box=_string_tuple(data.get("boundingbox")),
        address=_labelled(data.get("address")),
        extra_tags=_labelled(data.get("extratags"), ordered=True),
    )


def _block(place: _OsmPlace, prefix: str = "") -> str:
    """The place's lines: its name, then its OSM object (an ``osm_type/osm_id`` that
    ``overpass_query`` reads, e.g. ``relation(540);out;``), position, address and tags."""
    lines = [f"{prefix}{place.display_name}"]
    meta = []
    if place.osm_type or place.osm_id:
        meta.append(f"OSM: {place.osm_type}/{place.osm_id}")
    if place.category or place.type:
        meta.append(f"type: {place.category}/{place.type}")
    if place.latitude and place.longitude:
        meta.append(f"latitude {place.latitude}, longitude {place.longitude}")
    if meta:
        lines.append("   " + " | ".join(meta))
    if place.address:
        lines.append("   Address: " + " | ".join(place.address))
    if len(place.bounding_box) == len(_BOX_SIDES):
        sides = (
            f"{side} {value}" for side, value in zip(_BOX_SIDES, place.bounding_box, strict=True)
        )
        lines.append("   Bounding box: " + ", ".join(sides))
    if place.extra_tags:
        lines.append("   Extra tags: " + " | ".join(place.extra_tags))
    return "\n".join(lines)


# Nominatim's bounding box: min latitude, max latitude, min longitude, max longitude
# (https://nominatim.org/release-docs/latest/api/Output/).
_BOX_SIDES = ("south", "north", "west", "east")


def _parse_layers(value: str) -> tuple[str, ...]:
    layers = tuple(
        dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip())
    )
    invalid = [layer for layer in layers if layer not in _LAYERS]
    if invalid:
        raise ToolFailure(
            "validation_error",
            f"invalid layer(s): {', '.join(invalid)}; use {', '.join(sorted(_LAYERS))}.",
        )
    return layers


def _labelled(value: Any, *, ordered: bool = False) -> tuple[str, ...]:
    """Every ``key: value`` of an object, in the source's order (an address goes from the most
    precise part to the country) or by key."""
    if not isinstance(value, dict):
        return ()
    keys = sorted(value, key=str) if ordered else list(value)
    pairs = ((_string(key), _string(value.get(key))) for key in keys)
    return tuple(f"{key}: {text}" for key, text in pairs if key and text)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(_string(item) for item in value if _string(item))
    text = _string(value)
    return (text,) if text else ()


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
