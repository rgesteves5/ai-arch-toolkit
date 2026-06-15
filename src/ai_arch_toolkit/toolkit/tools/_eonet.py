"""NASA EONET tools — public natural events tracker."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://eonet.gsfc.nasa.gov/api/v3"
_TIMEOUT = 20
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_LIMIT = 50
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)


@tool
def eonet_categories() -> str:
    """List NASA EONET event categories."""
    try:
        data = _fetch_json("/categories", {})
        categories = data.get("categories", [])
    except urllib.error.HTTPError as e:
        return _http_error("NASA EONET categories failed", e)
    except urllib.error.URLError as e:
        return f"NASA EONET categories failed: URL error: {e.reason}"
    except TimeoutError:
        return "NASA EONET categories failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"NASA EONET categories failed: could not parse API response: {e}"

    if not isinstance(categories, list) or not categories:
        return "No NASA EONET categories found."
    lines = ["NASA EONET categories:"]
    for index, category in enumerate(categories, start=1):
        if isinstance(category, dict):
            lines.append(
                f"{index}. {_string(category.get('id'))} — {_string(category.get('title'))}"
            )
            description = _string(category.get("description"))
            if description:
                lines.append(f"   {description}")
    return "\n".join(lines)


@tool
def eonet_events(
    category: str = "",
    status: str = "open",
    source: str = "",
    bbox: str = "",
    days: int = 30,
    start_date: str = "",
    end_date: str = "",
    max_results: int = 10,
) -> str:
    """Search NASA EONET natural events.

    Args:
        category: Optional EONET category ID, e.g. "wildfires" or "severeStorms".
        status: Event status: "open", "closed", or "all". Defaults to "open".
        source: Optional EONET source ID.
        bbox: Optional west,south,east,north bounding box.
        days: Number of recent days when start/end dates are not provided. Defaults to 30.
        start_date: Optional start date as YYYY-MM-DD.
        end_date: Optional end date as YYYY-MM-DD.
        max_results: Number of events to return (1-50). Defaults to 10.
    """
    validation = _validate_events(category, status, source, bbox, days, start_date, end_date)
    if validation:
        return f"NASA EONET events failed: {validation}"
    params = {"limit": str(_bounded(max_results)), "status": status.strip().lower()}
    if category.strip():
        params["category"] = category.strip()
    if source.strip():
        params["source"] = source.strip()
    if bbox.strip():
        params["bbox"] = bbox.strip()
    if start_date.strip() or end_date.strip():
        if start_date.strip():
            params["start"] = start_date.strip()
        if end_date.strip():
            params["end"] = end_date.strip()
    else:
        params["days"] = str(max(1, min(days, 365)))

    try:
        data = _fetch_json("/events", params)
        events = data.get("events", [])
    except urllib.error.HTTPError as e:
        return _http_error("NASA EONET events failed", e)
    except urllib.error.URLError as e:
        return f"NASA EONET events failed: URL error: {e.reason}"
    except TimeoutError:
        return "NASA EONET events failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"NASA EONET events failed: could not parse API response: {e}"

    if not isinstance(events, list) or not events:
        return "No NASA EONET events found."
    lines = [f"NASA EONET events (returned {len(events)}):"]
    for index, event in enumerate(events, start=1):
        if isinstance(event, dict):
            lines.extend(_format_event(event, index=index, details=False))
    return "\n".join(lines)


@tool
def eonet_event(event_id: str) -> str:
    """Get a NASA EONET event by ID.

    Args:
        event_id: EONET event ID, e.g. "EONET_12345".
    """
    if not _ID_RE.fullmatch(event_id.strip()):
        return f"NASA EONET event failed: invalid event_id: {event_id!r}"
    try:
        data = _fetch_json(f"/events/{urllib.parse.quote(event_id.strip())}", {})
    except urllib.error.HTTPError as e:
        return _http_error("NASA EONET event failed", e)
    except urllib.error.URLError as e:
        return f"NASA EONET event failed: URL error: {e.reason}"
    except TimeoutError:
        return "NASA EONET event failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"NASA EONET event failed: could not parse API response: {e}"

    lines = [f"NASA EONET event {event_id.strip()}:"]
    lines.extend(_format_event(data, index=None, details=True))
    return "\n".join(lines)


def _fetch_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{_BASE_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _format_event(event: dict[str, Any], *, index: int | None, details: bool) -> list[str]:
    prefix = f"{index}. " if index is not None else ""
    lines = [f"{prefix}{_string(event.get('title'))} | id: {_string(event.get('id'))}"]
    categories = event.get("categories", [])
    category_text = ", ".join(
        _string(category.get("id")) or _string(category.get("title"))
        for category in categories
        if isinstance(category, dict)
    )
    geometry = event.get("geometry", [])
    latest = geometry[-1] if isinstance(geometry, list) and geometry else {}
    date_text = _string(latest.get("date")) if isinstance(latest, dict) else ""
    coords = _coords(latest) if isinstance(latest, dict) else ""
    lines.append(
        f"   categories: {category_text or '?'} | "
        f"latest: {date_text or '?'} | coords: {coords or '?'}"
    )
    if details:
        sources = event.get("sources", [])
        source_text = ", ".join(
            f"{_string(source.get('id'))}: {_string(source.get('url'))}"
            for source in sources[:5]
            if isinstance(source, dict)
        )
        if source_text:
            lines.append(f"   sources: {source_text}")
        if _string(event.get("description")):
            lines.append(f"   description: {_string(event.get('description'))}")
    return lines


def _coords(geometry: dict[str, Any]) -> str:
    coords = geometry.get("coordinates")
    if isinstance(coords, list) and len(coords) >= 2:
        return ", ".join(_string(value) for value in coords[:2])
    return ""


def _validate_events(
    category: str,
    status: str,
    source: str,
    bbox: str,
    days: int,
    start_date: str,
    end_date: str,
) -> str:
    if category and not _ID_RE.fullmatch(category.strip()):
        return "invalid category."
    if source and not _ID_RE.fullmatch(source.strip()):
        return "invalid source."
    if status.strip().lower() not in {"open", "closed", "all"}:
        return "status must be open, closed, or all."
    if days < 1:
        return "days must be greater than or equal to 1."
    if bbox.strip() and not _valid_bbox(bbox):
        return "bbox must be west,south,east,north."
    start = _parse_date(start_date.strip()) if start_date.strip() else None
    end = _parse_date(end_date.strip()) if end_date.strip() else None
    if start_date.strip() and start is None:
        return "invalid start_date. Use YYYY-MM-DD."
    if end_date.strip() and end is None:
        return "invalid end_date. Use YYYY-MM-DD."
    if start and end and start > end:
        return "start_date must be before or equal to end_date."
    return ""


def _valid_bbox(value: str) -> bool:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        return False
    try:
        west, south, east, north = [float(part) for part in parts]
    except ValueError:
        return False
    return -180 <= west <= east <= 180 and -90 <= south <= north <= 90


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 404:
        return f"{prefix}: no matching records found."
    if error.code == 429:
        return f"{prefix}: rate limited by NASA EONET (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
