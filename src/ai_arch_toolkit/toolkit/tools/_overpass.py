"""Overpass tools: OpenStreetMap objects by query, or by tag in an area (free, no key;
https://wiki.openstreetmap.org/wiki/Overpass_API).

Overpass answers a query whole, with no paging of its own, so a page of its elements is cut from
the answer and the footer gives the offset of the next (the query runs again for it).
"""

from __future__ import annotations

import html
import re
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._values import plain
from ai_arch_toolkit.toolkit.tools._window import page_window

# Overpass explains a failed request in an HTML page, one "<strong>Error</strong>: ..." paragraph
# per error (https://dev.overpass-api.de/overpass-doc/en/preface/commons.html): a 400 for a query
# it cannot parse ("line 1: parse error: ..."), a 429 or 504 for a quota or load it cannot serve.
_PAGE_ERROR_RE = re.compile(r"<strong[^>]*>\s*Error\s*</strong>\s*:?(.*?)</p>", re.DOTALL)
_TAG_MARKUP_RE = re.compile(r"<[^>]+>")


def _overpass_error(reply: Reply) -> ToolFailure | str | None:
    """The error an Overpass answer reports; ``None`` for a result.

    A query that fails while running, on a timeout or out of memory
    (https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL), still answers HTTP 200: the
    elements found so far and a ``remark`` that starts "runtime error" (seen 2026-09-29). Other
    remarks are notes. An error status's page says what went wrong: a 400 is a query Overpass
    could not read (``validation_error``); any other status keeps the page's words, typed by it.
    """
    body = reply.body
    if isinstance(body, dict):
        remark = _string(body.get("remark"))
        if remark.startswith("runtime error"):
            said = remark.rstrip(".")
            return f"{said}; narrow the query (a smaller area, fewer elements)"
        return None
    errors = _page_errors(body) if isinstance(body, str) else ""
    if reply.status == 400 and errors:
        return ToolFailure(
            "validation_error",
            f"Overpass could not read the query (HTTP 400): {errors}; correct the Overpass QL "
            "(overpass_query) or the tag and area (overpass_pois)",
        )
    if errors:
        return errors
    if reply.status == 504:
        return "Overpass is overloaded or the query timed out; retry later or narrow the query"
    return None


def _page_errors(page: str) -> str:
    """The text of the error paragraphs of an Overpass HTML page, joined; empty without any."""
    found = (
        _string(html.unescape(_TAG_MARKUP_RE.sub("", match)))
        for match in _PAGE_ERROR_RE.findall(page)
    )
    return "; ".join(error for error in found if error)


_API = Api(
    base="https://overpass-api.de/api/interpreter",
    name="Overpass",
    timeout_s=35,
    error_reader=_overpass_error,
)
_MAX_RESULTS = 50
_MAX_QUERY_CHARS = 4000
_MAX_RADIUS_M = 50_000
_TAG_RE = re.compile(r"^[A-Za-z0-9_:-]{1,80}$")
_VALUE_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,120}$", re.UNICODE)
# What overpass_query needs of a query to read its answer: the JSON output setting, and an out
# statement, which may follow a statement, open a block or name its input set (``.a out;``;
# https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL, "Settings" and "Out"). Without
# one, Overpass returns no elements at all.
_JSON_OUTPUT_RE = re.compile(r"\[\s*out\s*:\s*json\s*\]")
_OUT_STATEMENT_RE = re.compile(r"[;{]\s*(?:\.\w+\s+)?out\b")
_QUERY_EXAMPLE = "'[out:json][timeout:25];node[amenity=cafe](38,-10,39,-9);out;'"
_ATTRIBUTION = "Overpass API; data © OpenStreetMap contributors, ODbL"
# The tags worth a line under each element, besides its name.
_SHOWN_TAGS = ("amenity", "shop", "tourism", "leisure", "website", "phone", "opening_hours")


@tool(capability="network")
def overpass_query(
    query: str,
    max_results: Annotated[int, Range(1, _MAX_RESULTS)] = 25,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Run an Overpass QL query and list the OpenStreetMap elements it returns.

    Args:
        query: A complete Overpass QL query with JSON output and a timeout, e.g.
            '[out:json][timeout:25];node["amenity"="cafe"](38.7,-9.2,38.8,-9.1);out;'.
        max_results: How many elements to list.
        offset: How many elements to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when the query is empty, too long, does not ask for JSON
            output, has no out statement, or Overpass cannot read it (HTTP 400); upstream when
            Overpass reports a runtime error (a timeout, out of memory).
    """
    _check_query(query)
    return _run(
        query,
        offset,
        max_results,
        heading=f"OpenStreetMap elements the Overpass query returned ({_ATTRIBUTION}):",
        nothing=f"No OpenStreetMap elements match the Overpass query {query!r}.",
    )


@tool(capability="network")
def overpass_pois(
    tag_key: str,
    tag_value: str = "",
    bbox: str = "",
    latitude: Annotated[float | None, Range(-90, 90)] = None,
    longitude: Annotated[float | None, Range(-180, 180)] = None,
    radius_m: Annotated[int, Range(1, _MAX_RADIUS_M)] = 1000,
    max_results: Annotated[int, Range(1, _MAX_RESULTS)] = 25,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """List the OpenStreetMap nodes, ways and relations with a tag, in a box or around a point.

    Args:
        tag_key: An OSM tag key, e.g. "amenity", "shop" or "tourism".
        tag_value: The tag's exact value, e.g. "hospital" or "cafe"; any value when empty.
        bbox: A box as south,west,north,east in degrees, e.g. "38.6,-9.3,38.8,-9.0".
        latitude: The center's latitude, for a search around a point (without ``bbox``).
        longitude: The center's longitude.
        radius_m: The radius around the point, in meters.
        max_results: How many elements to list.
        offset: How many elements to skip; the footer gives the next offset.

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
    area, where = _area_clause(bbox, latitude, longitude, radius_m)
    key, value = tag_key.strip(), tag_value.strip()
    tag = f'["{key}"="{value}"]' if value else f'["{key}"]'
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
    tagged = f"{key}={value}" if value else f"{key}=*"
    return _run(
        query,
        offset,
        max_results,
        heading=f"OpenStreetMap elements tagged {tagged} {where} ({_ATTRIBUTION}):",
        nothing=f"No OpenStreetMap elements tagged {tagged} {where}.",
    )


def _check_query(query: str) -> None:
    """Refuse a query whose answer the tool could not read: empty or too long, without JSON
    output (XML or CSV would not parse), or without an out statement (no elements would come,
    and nothing would read as "no match")."""
    if not query.strip() or len(query) > _MAX_QUERY_CHARS:
        raise ToolFailure(
            "validation_error",
            f"query must be 1-{_MAX_QUERY_CHARS} characters (got {len(query)}).",
        )
    if not _JSON_OUTPUT_RE.search(query):
        raise ToolFailure(
            "validation_error",
            "the query does not ask for JSON output, the only one this tool reads; start it "
            f"with [out:json], e.g. {_QUERY_EXAMPLE}.",
        )
    if not _OUT_STATEMENT_RE.search(query):
        raise ToolFailure(
            "validation_error",
            "the query has no out statement, so Overpass would return no elements; end it with "
            f"one (out; or out center tags;), e.g. {_QUERY_EXAMPLE}.",
        )


def _run(query: str, offset: int, limit: int, *, heading: str, nothing: str) -> ToolResult:
    return _API.post_form(
        form={"data": query},
        parse=lambda data: _answer(data, offset, limit, heading=heading, nothing=nothing),
    )


def _answer(
    data: dict[str, Any], offset: int, limit: int, *, heading: str, nothing: str
) -> ToolResult:
    elements = _elements(data)
    if not elements:
        return ToolResult.success(nothing)
    lines = [_element(number, item) for number, item in enumerate(elements, start=1)]
    return page_window(lines, offset=offset, limit=limit).result(heading=heading)


def _area_clause(
    bbox: str,
    latitude: float | None,
    longitude: float | None,
    radius_m: int,
) -> tuple[str, str]:
    """The area filter of the query, and how the answer names it."""
    if bbox.strip():
        south, west, north, east = (plain(side) for side in _bbox(bbox))
        where = f"in the box south {south}, west {west}, north {north}, east {east}"
        return f"({south},{west},{north},{east})", where
    if latitude is None or longitude is None:
        raise ToolFailure(
            "validation_error", "no area to search; provide bbox or latitude and longitude."
        )
    center = f"{plain(latitude)},{plain(longitude)}"
    where = f"within {radius_m} m of latitude {plain(latitude)}, longitude {plain(longitude)}"
    return f"(around:{radius_m},{center})", where


def _bbox(bbox: str) -> tuple[float, float, float, float]:
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
    return south, west, north, east


def _elements(data: dict[str, Any]) -> list[dict[str, Any]]:
    value = data.get("elements")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _element(number: int, item: dict[str, Any]) -> str:
    """An element's lines: its name, its ``type/id`` (``overpass_query`` reads it back, e.g.
    ``node(1);out;``), its position (a way's or relation's center) and the tags worth showing."""
    tags = _mapping(item.get("tags"))
    name = _string(tags.get("name")) or "(unnamed)"
    reference = f"{_string(item.get('type'))}/{_string(item.get('id'))}"
    line = f"{number}. {name} | {reference} | {_position(item)}"
    shown = [f"{key}={_string(tags.get(key))}" for key in _SHOWN_TAGS if _string(tags.get(key))]
    return f"{line}\n   tags: {'; '.join(shown)}" if shown else line


def _position(item: dict[str, Any]) -> str:
    point = item if "lat" in item else item.get("center")
    if not isinstance(point, dict) or point.get("lat") is None or point.get("lon") is None:
        return "no position"
    return f"latitude {plain(point.get('lat'))}, longitude {plain(point.get('lon'))}"


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
