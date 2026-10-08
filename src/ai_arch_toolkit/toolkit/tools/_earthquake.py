"""USGS earthquake tools: the ComCat catalog through its FDSN event web service (free, no key;
https://earthquake.usgs.gov/fdsnws/event/1/).

A search asks the service's ``count`` method for the total, with the same filters, then the page
(``limit``, ``offset``): the GeoJSON ``metadata.count`` of a page is not documented as a total.
Times arrive in milliseconds since the epoch and are shown in ISO 8601 UTC.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Any, NoReturn

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._values import plain, utc
from ai_arch_toolkit.toolkit.tools._window import list_window


def _fdsn_error(reply: Reply) -> ToolFailure | str | None:
    """The error a USGS error status reports; ``None`` for a result.

    FDSN services explain an error status in plain text: ``Error <code>: <description>``, a
    blank line, the detail, then "Usage details are available from …"
    (https://www.fdsn.org/webservices/FDSN-WS-Specification-Commonalities-1.2.pdf, "Errors
    messages"). A 400 is a query the service cannot take and a 413 one that would return too
    much ("Result set limitations"), both the caller's to change; a 409 is an event USGS deleted
    (https://earthquake.usgs.gov/fdsnws/event/1/, ``includedeleted``).
    """
    if reply.status < 400:
        return None
    detail = _fdsn_detail(reply.body)
    said = f": {detail}" if detail else ""
    if reply.status == 400:
        return ToolFailure(
            "validation_error",
            f"USGS rejected the query (HTTP 400){said}; correct the dates (YYYY-MM-DD), the "
            "magnitudes, the depths or the area",
        )
    if reply.status == 413:
        return ToolFailure(
            "validation_error",
            f"USGS would return too many events (HTTP 413){said}; narrow the dates, the "
            "magnitudes or the area",
        )
    if reply.status == 409:
        return ToolFailure(
            "not_found",
            f"USGS deleted this event (HTTP 409){said}; find the event that replaced it with "
            "earthquake_search",
        )
    return detail or None


def _fdsn_detail(body: object) -> str:
    """The detail of an FDSN error text: the lines between the status line and the usage line
    (the status's description when there are none); empty for any other body."""
    if not isinstance(body, str) or not body.lstrip().startswith("Error "):
        return ""
    first, _, rest = body.strip().partition("\n")
    detail = rest.split("Usage details are available from", 1)[0]
    text = " ".join(detail.split()) or first.partition(":")[2].strip()
    return text[:300].rstrip(".")


_API = Api(
    base="https://earthquake.usgs.gov/fdsnws/event/1",
    name="USGS",
    timeout_s=15,
    error_reader=_fdsn_error,
)
_MAX_RESULTS = 50
_EVENT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_ORDER_BY = {"time", "time-asc", "magnitude", "magnitude-asc"}


@tool(capability="network")
def earthquake_search(
    start_time: str = "",
    end_time: str = "",
    min_magnitude: float = 0.0,
    max_magnitude: float = 10.0,
    latitude: float | None = None,
    longitude: float | None = None,
    max_radius_km: float | None = None,
    min_depth_km: float | None = None,
    max_depth_km: float | None = None,
    order_by: str = "time",
    max_results: Annotated[int, Range(1, _MAX_RESULTS)] = 10,
    offset: Annotated[int, Range(1)] = 1,
) -> ToolResult:
    """Search the USGS earthquake catalog: each event's ID, magnitude, time and place, with the
    number of events that match.

    Args:
        start_time: The first day, YYYY-MM-DD, in UTC; USGS starts 30 days ago without one.
        end_time: The last day, YYYY-MM-DD, in UTC; now without one.
        min_magnitude: Lowest magnitude.
        max_magnitude: Highest magnitude.
        latitude: Center latitude, for a search around a point.
        longitude: Center longitude, for a search around a point.
        max_radius_km: Radius around the point, in kilometers.
        min_depth_km: Shallowest depth, in kilometers.
        max_depth_km: Deepest depth, in kilometers.
        order_by: time (newest first), time-asc, magnitude (largest first) or magnitude-asc.
        max_results: How many events to list.
        offset: The number of the first event to list, from 1; the footer gives the next one.

    Raises:
        ToolFailure: validation_error when an argument is invalid, or USGS rejects the query.
    """
    filters = _search_filters(
        start_time,
        end_time,
        min_magnitude,
        max_magnitude,
        latitude,
        longitude,
        max_radius_km,
        min_depth_km,
        max_depth_km,
    )
    if order_by not in _ORDER_BY:
        _invalid(f"invalid order_by {order_by!r}; use time, time-asc, magnitude or magnitude-asc.")
    total = _API.get_text("count", params=filters, parse=_count)
    asked = ", ".join(f"{name}={value}" for name, value in filters.items())
    if total == 0:
        return ToolResult.success(
            f"No USGS earthquakes match {asked}; widen the dates, the magnitudes or the area."
        )
    query = {
        **filters,
        "format": "geojson",
        "orderby": order_by,
        "limit": str(max_results),
        "offset": str(offset),
    }
    return _API.get_json(
        "query",
        params=query,
        allow_empty=True,  # no event there: 204 No Content, the FDSN default ("nodata")
        parse=lambda data: _search_answer(data, asked, total, offset),
    )


@tool(capability="network")
def earthquake_event(event_id: str) -> str:
    """Get a USGS earthquake event by ID: magnitude, time, place, depth, felt reports, alert
    level and the USGS event page.

    Args:
        event_id: A USGS event ID from earthquake_search, e.g. "us7000m9gq".

    Raises:
        ToolFailure: validation_error when the ID is malformed; not_found when USGS has no
            event with it, or deleted it.
    """
    if not _EVENT_RE.fullmatch(event_id.strip()):
        msg = (
            f"invalid event_id {event_id!r}; a USGS event ID is letters, digits, '_', '.' "
            "or '-', e.g. us7000m9gq."
        )
        raise ToolFailure("validation_error", msg)
    params = {"format": "geojson", "eventid": event_id.strip()}
    return _API.get_json(
        "query",
        params=params,
        parse=lambda data: _event_text(data, event_id),
        missing=_no_event(event_id),
        allow_empty=True,
    )


@tool(capability="network")
def earthquake_count(
    start_time: str = "",
    end_time: str = "",
    min_magnitude: float = 0.0,
    max_magnitude: float = 10.0,
) -> str:
    """Count the USGS earthquakes in a period and a magnitude range.

    Args:
        start_time: The first day, YYYY-MM-DD, in UTC; USGS starts 30 days ago without one.
        end_time: The last day, YYYY-MM-DD, in UTC; now without one.
        min_magnitude: Lowest magnitude.
        max_magnitude: Highest magnitude.

    Raises:
        ToolFailure: validation_error when a date or the magnitude range is invalid, or USGS
            rejects the query.
    """
    params = _period(start_time, end_time, min_magnitude, max_magnitude)
    total = _API.get_text("count", params=params, parse=_count)
    asked = ", ".join(f"{name}={value}" for name, value in params.items())
    return f"USGS earthquake count for {asked}: {total}"


# --- Answers -----------------------------------------------------------------------------------


def _count(text: str) -> int:
    """The ``count`` method's plain-text answer: one integer (nothing, for a 204)."""
    return int(text.strip() or "0")


def _search_answer(data: dict[str, Any], asked: str, total: int, offset: int) -> ToolResult:
    features = [item for item in data.get("features") or [] if isinstance(item, dict)]
    lines = [
        "\n".join(_event_lines(feature, number=number))
        for number, feature in enumerate(features, start=offset)
    ]
    last = offset + len(lines) - 1
    window = list_window(
        lines,
        first=offset,
        total=total,
        next_call={"offset": last + 1} if lines and last < total else None,
    )
    return window.result(heading=f"USGS earthquakes for {asked}:")


def _event_text(data: dict[str, Any], event_id: str) -> str:
    if not data:  # 204 No Content: no such event
        raise ToolFailure("not_found", _no_event(event_id))
    lines = [f"USGS earthquake {event_id.strip()}:", *_event_lines(data), *_details(data)]
    return "\n".join(lines)


def _no_event(event_id: str) -> str:
    return f"USGS has no earthquake with ID {event_id.strip()}; search with earthquake_search."


def _event_lines(feature: dict[str, Any], *, number: int | None = None) -> list[str]:
    props = _mapping(feature.get("properties"))
    title = _string(props.get("title")) or _string(props.get("place")) or "(untitled)"
    prefix = f"{number}. " if number is not None else ""
    magnitude = " ".join(
        part for part in (plain(props.get("mag")), _string(props.get("magType"))) if part
    )
    facts = [
        f"time: {_time(props.get('time'))}",
        f"magnitude {magnitude or 'not reported'}",
        f"type: {_string(props.get('type')) or '?'}",
    ]
    lines = [f"{prefix}{title} | id: {_string(feature.get('id'))}", "   " + " | ".join(facts)]
    if where := _where(feature.get("geometry")):
        lines.append(f"   {where}")
    return lines


def _details(feature: dict[str, Any]) -> list[str]:
    props = _mapping(feature.get("properties"))
    facts = [
        f"{label}: {plain(props.get(key))}"
        for key, label in _DETAILS
        if props.get(key) is not None and plain(props.get(key))
    ]
    lines = ["   " + " | ".join(facts)] if facts else []
    if props.get("updated") is not None:
        lines.append(f"   updated: {_time(props.get('updated'))}")
    if url := _string(props.get("url")):
        lines.append(f"   USGS event page: {url}")
    return lines


# The detail properties an event shows, by name (https://earthquake.usgs.gov/data/comcat/).
_DETAILS = (
    ("status", "review status"),
    ("felt", "felt reports"),
    ("alert", "PAGER alert"),
    ("tsunami", "tsunami flag"),
    ("sig", "significance"),
)


def _time(value: object) -> str:
    """A ComCat time, milliseconds since the epoch, in ISO 8601 UTC."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "not reported"
    return utc(value / 1000)


def _where(geometry: object) -> str:
    """A GeoJSON point (longitude, latitude, depth in km), labelled."""
    coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if not isinstance(coords, list) or len(coords) < 2:
        return ""
    where = f"latitude {plain(coords[1])}, longitude {plain(coords[0])}"
    return f"{where}, depth {plain(coords[2])} km" if len(coords) > 2 else where


# --- Arguments ---------------------------------------------------------------------------------


def _search_filters(
    start_time: str,
    end_time: str,
    min_magnitude: float,
    max_magnitude: float,
    latitude: float | None,
    longitude: float | None,
    max_radius_km: float | None,
    min_depth_km: float | None,
    max_depth_km: float | None,
) -> dict[str, str]:
    """The filters of a search, the same for its count and its page; raises when invalid."""
    params = _period(start_time, end_time, min_magnitude, max_magnitude)
    _validate_circle(latitude, longitude, max_radius_km)
    if latitude is not None and longitude is not None and max_radius_km is not None:
        params["latitude"] = str(latitude)
        params["longitude"] = str(longitude)
        params["maxradiuskm"] = str(max_radius_km)
    if min_depth_km is not None:
        params["mindepth"] = str(min_depth_km)
    if max_depth_km is not None:
        params["maxdepth"] = str(max_depth_km)
    return params


def _validate_circle(
    latitude: float | None, longitude: float | None, max_radius_km: float | None
) -> None:
    radius_values = [latitude is not None, longitude is not None, max_radius_km is not None]
    if any(radius_values) and not all(radius_values):
        _invalid("latitude, longitude, and max_radius_km must be provided together.")
    if latitude is not None and not -90 <= latitude <= 90:
        _invalid(f"latitude must be between -90 and 90, got {latitude}.")
    if longitude is not None and not -180 <= longitude <= 180:
        _invalid(f"longitude must be between -180 and 180, got {longitude}.")
    if max_radius_km is not None and max_radius_km <= 0:
        _invalid(f"max_radius_km must be greater than 0, got {max_radius_km}.")


def _period(
    start_time: str,
    end_time: str,
    min_magnitude: float,
    max_magnitude: float,
) -> dict[str, str]:
    """The period's and the magnitudes' parameters, the days as YYYY-MM-DD; raises when
    invalid."""
    start = _parse_date(start_time.strip()) if start_time.strip() else None
    end = _parse_date(end_time.strip()) if end_time.strip() else None
    if start_time.strip() and start is None:
        _invalid(f"invalid start_time {start_time!r}; use YYYY-MM-DD.")
    if end_time.strip() and end is None:
        _invalid(f"invalid end_time {end_time!r}; use YYYY-MM-DD.")
    if start and end and start > end:
        _invalid(f"start_time {start} must be before or equal to end_time {end}.")
    if min_magnitude > max_magnitude:
        _invalid(
            f"min_magnitude {min_magnitude} must be less than or equal to "
            f"max_magnitude {max_magnitude}."
        )
    params = {"minmagnitude": str(min_magnitude), "maxmagnitude": str(max_magnitude)}
    if start:
        params["starttime"] = start.isoformat()
    if end:
        params["endtime"] = end.isoformat()
    return params


def _invalid(message: str) -> NoReturn:
    raise ToolFailure("validation_error", message)


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
