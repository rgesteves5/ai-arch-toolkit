"""USGS earthquake tools — public seismic event search."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1"
_TIMEOUT = 15
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_LIMIT = 50
_EVENT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_ORDER_BY = {"time", "time-asc", "magnitude", "magnitude-asc"}


@tool
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
    """
    validation = _validate_search(
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
    if validation:
        return f"USGS earthquake search failed: {validation}"

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
    try:
        data = _fetch_json("/query", params)
        features = data.get("features", [])
    except urllib.error.HTTPError as e:
        return _http_error("USGS earthquake search failed", e)
    except urllib.error.URLError as e:
        return f"USGS earthquake search failed: URL error: {e.reason}"
    except TimeoutError:
        return "USGS earthquake search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"USGS earthquake search failed: could not parse API response: {e}"

    if not isinstance(features, list) or not features:
        return "No USGS earthquakes found."
    total = _string(data.get("metadata", {}).get("count")) or "?"
    lines = [f"USGS earthquakes (returned {len(features)}, count {total}, offset {offset}):"]
    for index, feature in enumerate(features, start=1):
        if isinstance(feature, dict):
            lines.extend(_format_event(feature, index=index, details=False))
    return "\n".join(lines)


@tool
def earthquake_event(event_id: str) -> str:
    """Get a USGS earthquake event by ID.

    Args:
        event_id: USGS event ID, e.g. "us7000m9gq".
    """
    if not _EVENT_RE.fullmatch(event_id.strip()):
        return f"USGS earthquake lookup failed: invalid event_id: {event_id!r}"
    try:
        data = _fetch_json("/query", {"format": "geojson", "eventid": event_id.strip()})
    except urllib.error.HTTPError as e:
        return _http_error("USGS earthquake lookup failed", e)
    except urllib.error.URLError as e:
        return f"USGS earthquake lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "USGS earthquake lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"USGS earthquake lookup failed: could not parse API response: {e}"

    if not data:
        return f"USGS earthquake not found: {event_id}"
    lines = [f"USGS earthquake {event_id.strip()}:"]
    lines.extend(_format_event(data, index=None, details=True))
    return "\n".join(lines)


@tool
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
    """
    validation = _validate_dates_and_magnitude(start_time, end_time, min_magnitude, max_magnitude)
    if validation:
        return f"USGS earthquake count failed: {validation}"
    params = {
        "format": "text",
        "minmagnitude": str(min_magnitude),
        "maxmagnitude": str(max_magnitude),
    }
    if start_time.strip():
        params["starttime"] = start_time.strip()
    if end_time.strip():
        params["endtime"] = end_time.strip()
    try:
        text = _fetch_text("/count", params)
    except urllib.error.HTTPError as e:
        return _http_error("USGS earthquake count failed", e)
    except urllib.error.URLError as e:
        return f"USGS earthquake count failed: URL error: {e.reason}"
    except TimeoutError:
        return "USGS earthquake count failed: request timed out."

    return f"USGS earthquake count: {_string(text) or '0'}"


def _fetch_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    return json.loads(_fetch_text(path, params))


def _fetch_text(path: str, params: dict[str, str]) -> str:
    url = f"{_BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


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
) -> str:
    if offset < 1:
        return "offset must be greater than or equal to 1."
    if order_by not in _ORDER_BY:
        return "order_by must be one of time, time-asc, magnitude, magnitude-asc."
    date_error = _validate_dates_and_magnitude(start_time, end_time, min_magnitude, max_magnitude)
    if date_error:
        return date_error
    radius_values = [latitude is not None, longitude is not None, max_radius_km is not None]
    if any(radius_values) and not all(radius_values):
        return "latitude, longitude, and max_radius_km must be provided together."
    if latitude is not None and not -90 <= latitude <= 90:
        return "latitude must be between -90 and 90."
    if longitude is not None and not -180 <= longitude <= 180:
        return "longitude must be between -180 and 180."
    if max_radius_km is not None and max_radius_km <= 0:
        return "max_radius_km must be greater than 0."
    return ""


def _validate_dates_and_magnitude(
    start_time: str,
    end_time: str,
    min_magnitude: float,
    max_magnitude: float,
) -> str:
    start = _parse_date(start_time.strip()) if start_time.strip() else None
    end = _parse_date(end_time.strip()) if end_time.strip() else None
    if start_time.strip() and start is None:
        return "invalid start_time. Use YYYY-MM-DD."
    if end_time.strip() and end is None:
        return "invalid end_time. Use YYYY-MM-DD."
    if start and end and start > end:
        return "start_time must be before or equal to end_time."
    if min_magnitude > max_magnitude:
        return "min_magnitude must be less than or equal to max_magnitude."
    return ""


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


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 404:
        return f"{prefix}: no matching records found."
    if error.code == 429:
        return f"{prefix}: rate limited by USGS (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
