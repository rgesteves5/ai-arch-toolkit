"""USGS earthquake tools — public seismic event search."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, NoReturn

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api

_API = Api(
    base="https://earthquake.usgs.gov/fdsnws/event/1",
    name="USGS",
    timeout_s=15,
)
_MAX_LIMIT = 50
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
    max_results: int = 10,
    offset: int = 1,
) -> str:
    """Search earthquakes from the USGS event catalog.

    Args:
        start_time: Optional start date as YYYY-MM-DD.
        end_time: Optional end date as YYYY-MM-DD.
        min_magnitude: Minimum magnitude. Defaults to 0.
        max_magnitude: Maximum magnitude. Defaults to 10.
        latitude: Optional center latitude for radius search.
        longitude: Optional center longitude for radius search.
        max_radius_km: Optional radius in kilometers when latitude/longitude are provided.
        min_depth_km: Optional minimum depth in kilometers.
        max_depth_km: Optional maximum depth in kilometers.
        order_by: Sort order: time, time-asc, magnitude, or magnitude-asc.
        max_results: Number of events to return (1-50). Defaults to 10.
        offset: One-based result offset. Defaults to 1.

    Raises:
        ToolFailure: validation_error when an argument is invalid.
    """
    params = _search_params(
        start_time,
        end_time,
        min_magnitude,
        max_magnitude,
        latitude,
        longitude,
        max_radius_km,
        min_depth_km,
        max_depth_km,
        order_by,
        max_results,
        offset,
    )
    return _API.get_json("query", params=params, parse=lambda data: _events_text(data, offset))


@tool(capability="network")
def earthquake_event(event_id: str) -> str:
    """Get a USGS earthquake event by ID.

    Args:
        event_id: USGS event ID, e.g. "us7000m9gq".

    Raises:
        ToolFailure: validation_error when the ID is malformed; not_found when USGS has no
            event with it.
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
    )


@tool(capability="network")
def earthquake_count(
    start_time: str = "",
    end_time: str = "",
    min_magnitude: float = 0.0,
    max_magnitude: float = 10.0,
) -> str:
    """Count USGS earthquakes for a date and magnitude range.

    Args:
        start_time: Optional start date as YYYY-MM-DD.
        end_time: Optional end date as YYYY-MM-DD.
        min_magnitude: Minimum magnitude. Defaults to 0.
        max_magnitude: Maximum magnitude. Defaults to 10.

    Raises:
        ToolFailure: validation_error when a date or the magnitude range is invalid.
    """
    _validate_dates_and_magnitude(start_time, end_time, min_magnitude, max_magnitude)
    params = {
        "format": "text",
        "minmagnitude": str(min_magnitude),
        "maxmagnitude": str(max_magnitude),
    }
    if start_time.strip():
        params["starttime"] = start_time.strip()
    if end_time.strip():
        params["endtime"] = end_time.strip()
    return _API.get_text("count", params=params, parse=_count_text)


def _events_text(data: dict[str, Any], offset: int) -> str:
    features = data.get("features", [])
    if not isinstance(features, list) or not features:
        return "No USGS earthquakes found."
    total = _string(data.get("metadata", {}).get("count")) or "?"
    lines = [f"USGS earthquakes (returned {len(features)}, count {total}, offset {offset}):"]
    for index, feature in enumerate(features, start=1):
        if isinstance(feature, dict):
            lines.extend(_format_event(feature, index=index, details=False))
    return "\n".join(lines)


def _event_text(data: dict[str, Any], event_id: str) -> str:
    if not data:
        raise ToolFailure("not_found", _no_event(event_id))
    lines = [f"USGS earthquake {event_id.strip()}:"]
    lines.extend(_format_event(data, index=None, details=True))
    return "\n".join(lines)


def _no_event(event_id: str) -> str:
    return f"USGS has no earthquake with ID {event_id.strip()}; search with earthquake_search."


def _count_text(text: str) -> str:
    return f"USGS earthquake count: {_string(text) or '0'}"


def _search_params(
    start_time: str,
    end_time: str,
    min_magnitude: float,
    max_magnitude: float,
    latitude: float | None,
    longitude: float | None,
    max_radius_km: float | None,
    min_depth_km: float | None,
    max_depth_km: float | None,
    order_by: str,
    max_results: int,
    offset: int,
) -> dict[str, str]:
    """The query of a search; raises when its arguments are invalid."""
    _validate_search(
        start_time,
        end_time,
        min_magnitude,
        max_magnitude,
        latitude,
        longitude,
        max_radius_km,
        order_by,
        offset,
    )
    params = {
        "format": "geojson",
        "minmagnitude": str(min_magnitude),
        "maxmagnitude": str(max_magnitude),
        "orderby": order_by,
        "limit": str(max(1, min(max_results, _MAX_LIMIT))),
        "offset": str(offset),
    }
    if start_time.strip():
        params["starttime"] = start_time.strip()
    if end_time.strip():
        params["endtime"] = end_time.strip()
    if latitude is not None and longitude is not None and max_radius_km is not None:
        params["latitude"] = str(latitude)
        params["longitude"] = str(longitude)
        params["maxradiuskm"] = str(max_radius_km)
    if min_depth_km is not None:
        params["mindepth"] = str(min_depth_km)
    if max_depth_km is not None:
        params["maxdepth"] = str(max_depth_km)
    return params


def _validate_search(
    start_time: str,
    end_time: str,
    min_magnitude: float,
    max_magnitude: float,
    latitude: float | None,
    longitude: float | None,
    max_radius_km: float | None,
    order_by: str,
    offset: int,
) -> None:
    if offset < 1:
        _invalid(f"offset must be greater than or equal to 1, got {offset}.")
    if order_by not in _ORDER_BY:
        _invalid(f"invalid order_by {order_by!r}; use time, time-asc, magnitude or magnitude-asc.")
    _validate_dates_and_magnitude(start_time, end_time, min_magnitude, max_magnitude)
    radius_values = [latitude is not None, longitude is not None, max_radius_km is not None]
    if any(radius_values) and not all(radius_values):
        _invalid("latitude, longitude, and max_radius_km must be provided together.")
    if latitude is not None and not -90 <= latitude <= 90:
        _invalid(f"latitude must be between -90 and 90, got {latitude}.")
    if longitude is not None and not -180 <= longitude <= 180:
        _invalid(f"longitude must be between -180 and 180, got {longitude}.")
    if max_radius_km is not None and max_radius_km <= 0:
        _invalid(f"max_radius_km must be greater than 0, got {max_radius_km}.")


def _validate_dates_and_magnitude(
    start_time: str,
    end_time: str,
    min_magnitude: float,
    max_magnitude: float,
) -> None:
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


def _invalid(message: str) -> NoReturn:
    raise ToolFailure("validation_error", message)


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _format_event(feature: dict[str, Any], *, index: int | None, details: bool) -> list[str]:
    props = feature.get("properties", {})
    geometry = feature.get("geometry", {})
    if not isinstance(props, dict):
        props = {}
    if not isinstance(geometry, dict):
        geometry = {}
    event_id = _string(feature.get("id"))
    title = _string(props.get("title")) or _string(props.get("place"))
    prefix = f"{index}. " if index is not None else ""
    lines = [f"{prefix}{title} | id: {event_id}"]
    lines.append(
        "   "
        + " | ".join(
            [
                f"mag: {_string(props.get('mag')) or '?'}",
                f"type: {_string(props.get('type')) or '?'}",
                f"time: {_string(props.get('time')) or '?'}",
            ]
        )
    )
    coords = geometry.get("coordinates")
    if isinstance(coords, list) and len(coords) >= 3:
        lines.append(f"   coordinates: {coords[1]}, {coords[0]} | depth_km: {coords[2]}")
    if details:
        url = _string(props.get("url"))
        tsunami = _string(props.get("tsunami"))
        if tsunami:
            lines.append(f"   tsunami flag: {tsunami}")
        if url:
            lines.append(f"   USGS: {url}")
    return lines


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
