"""Overpass tools — public OpenStreetMap object queries."""

from __future__ import annotations

import re
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api


def _runtime_error(data: object) -> str | None:
    """The error an Overpass answer reports in its ``remark``; ``None`` for a result.

    A query that fails while running, on a timeout or out of memory
    (https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL), still answers HTTP 200: the
    elements found so far and a ``remark`` that starts "runtime error" (seen 2026-09-29). Other
    remarks are notes.
    """
    remark = _string(data.get("remark")) if isinstance(data, dict) else ""
    return remark if remark.startswith("runtime error") else None


_API = Api(
    base="https://overpass-api.de/api/interpreter",
    name="Overpass",
    timeout_s=35,
    status_messages={504: "Overpass query timed out upstream (HTTP 504)."},
    body_error=_runtime_error,
)
_MAX_LIMIT = 50
_TAG_RE = re.compile(r"^[A-Za-z0-9_:-]{1,80}$")
_VALUE_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,120}$", re.UNICODE)


@tool(capability="network")
def overpass_query(query: str, max_results: int = 25) -> str:
    """Run a bounded Overpass QL query and summarize returned OSM elements.

    Args:
        query: Complete Overpass QL query. It should include output format and timeout.
        max_results: Number of elements to return (1-50). Defaults to 25.

    Raises:
        ToolFailure: validation_error when the query is empty, too long, or has no output
            format; upstream when Overpass reports a runtime error (a timeout, out of memory).
    """
    if not query.strip() or len(query) > 4000:
        raise ToolFailure(
            "validation_error", f"query must be 1-4000 characters (got {len(query)})."
        )
    if "[out:" not in query or "out" not in query:
        raise ToolFailure(
            "validation_error",
            "the query has no output format; start it with [out:json] and end it with an out "
            "statement, e.g. '[out:json][timeout:25];node[amenity=cafe](38,-10,39,-9);out;'.",
        )
    return _run(
        query,
        max_results,
        label="Overpass elements",
        nothing="No Overpass elements found.",
    )


@tool(capability="network")
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

    Raises:
        ToolFailure: validation_error when the tag or the area is invalid or missing; upstream
            when Overpass reports a runtime error (a timeout, out of memory).
    """
    if not _TAG_RE.fullmatch(tag_key.strip()):
        raise ToolFailure(
            "validation_error",
            f"invalid tag_key {tag_key!r}; give an OSM tag key such as 'amenity' or 'shop'.",
        )
    if tag_value and not _VALUE_RE.fullmatch(tag_value.strip()):
        raise ToolFailure(
            "validation_error",
            f"invalid tag_value {tag_value!r}; give a plain OSM tag value such as 'cafe'.",
        )
    area = _area_clause(bbox, latitude, longitude, radius_m)
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
    return _run(
        query,
        max_results,
        label="Overpass POIs",
        nothing="No Overpass POIs found.",
    )


def _run(query: str, max_results: int, *, label: str, nothing: str) -> str:
    return _API.post_form(
        form={"data": query},
        parse=lambda data: _summary(data, max_results, label=label, nothing=nothing),
    )


def _summary(data: dict[str, Any], max_results: int, *, label: str, nothing: str) -> str:
    elements = _elements(data)
    if not elements:
        return nothing
    page = elements[: _bounded(max_results)]
    return _format_elements(page, header=f"{label} (returned {len(page)} of {len(elements)}):")


def _area_clause(
    bbox: str,
    latitude: float | None,
    longitude: float | None,
    radius_m: int,
) -> str:
    if bbox.strip():
        parts = [part.strip() for part in bbox.split(",")]
        if len(parts) != 4:
            raise ToolFailure(
                "validation_error",
                f"invalid bbox {bbox!r}; give south,west,north,east, e.g. '38.6,-9.3,38.8,-9.0'.",
            )
        try:
            south, west, north, east = [float(part) for part in parts]
        except ValueError:
            raise ToolFailure(
                "validation_error", f"invalid bbox {bbox!r}; its four values must be numbers."
            ) from None
        if not (-90 <= south <= north <= 90 and -180 <= west <= east <= 180):
            raise ToolFailure(
                "validation_error",
                f"invalid bbox {bbox!r}; need -90 <= south <= north <= 90 and "
                "-180 <= west <= east <= 180.",
            )
        return f"({south},{west},{north},{east})"
    if latitude is None or longitude is None:
        raise ToolFailure(
            "validation_error", "no area to search; provide bbox or latitude and longitude."
        )
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ToolFailure(
            "validation_error",
            f"invalid coordinates {latitude}, {longitude}; latitude must be between -90 and 90 "
            "and longitude between -180 and 180.",
        )
    if radius_m <= 0 or radius_m > 50000:
        raise ToolFailure(
            "validation_error", f"radius_m must be between 1 and 50000 (got {radius_m})."
        )
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


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
