"""NASA EONET tools: natural events (wildfires, storms, volcanoes…) and their tracks (free, no key;
https://eonet.gsfc.nasa.gov/docs/v3).

EONET lists events with a ``limit`` and no offset, so ``eonet_events`` pages the first results
it returns (``_first_results``). An event's track is every position EONET dated, read a page at a
time with ``eonet_event``. EONET's dates are ISO 8601 UTC already; its coordinates are GeoJSON,
longitude first, and are shown labelled.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Any, Literal, NoReturn, get_args

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._first_results import asked, first_results_window
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._values import plain
from ai_arch_toolkit.toolkit.tools._window import page_window


def _unknown_event(reply: Reply) -> ToolFailure | str | None:
    """What an EONET event answer's error means; ``None`` for any other.

    EONET documents no error answers (https://eonet.gsfc.nasa.gov/docs/v3), and answers an event
    ID it does not know with an HTTP 500 page (seen 2026-09-30). A 500 may be an outage too, so it
    stays the source's failure, retryable, saying what else it may mean.
    """
    if reply.status != 500:
        return None
    return (
        "NASA EONET failed, or has no event with that ID (it answers an unknown ID this way); "
        "check the ID with eonet_events, or try again later"
    )


_API = Api(base="https://eonet.gsfc.nasa.gov/api/v3", name="NASA EONET", timeout_s=20)
_EVENT_API = Api(
    base="https://eonet.gsfc.nasa.gov/api/v3/events",
    name="NASA EONET",
    timeout_s=20,
    error_reader=_unknown_event,
)
_MAX_RESULTS = 50
# EONET's limit has no stated maximum and there is no offset: the tool asks for the first events
# up to this depth, and past it the period or the filters narrow the list.
_DEPTH = 1000
_MAX_DAYS = 365
_MAX_POINTS = 200
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
# Categories and sources go as one ID or several, comma-separated, read as "any of them"
# (https://eonet.gsfc.nasa.gov/docs/v3, "source" and "category").
_IDS_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}(?:,[A-Za-z0-9_.:-]{1,80})*$")
type Status = Literal["open", "closed", "all"]
_STATUSES = frozenset(get_args(Status.__value__))


@tool(capability="network")
def eonet_categories() -> str:
    """List NASA EONET's event categories, with the ID ``eonet_events`` takes."""
    return _API.get_json("categories", parse=_categories_text)


@tool(capability="network")
def eonet_events(
    category: str = "",
    status: Status = "open",
    source: str = "",
    bbox: str = "",
    days: Annotated[int, Range(1, _MAX_DAYS)] = 30,
    start_date: str = "",
    end_date: str = "",
    max_results: Annotated[int, Range(1, _MAX_RESULTS)] = 10,
    offset: Annotated[int, Range(0, _DEPTH - 1)] = 0,
) -> ToolResult:
    """Search NASA EONET's natural events: each with its categories, status and where its
    track ends.

    Args:
        category: A category ID from eonet_categories, e.g. "wildfires", or several separated by
            commas for events in any of them.
        status: Open (still going), closed or all.
        source: An EONET source ID, e.g. "InciWeb", or several separated by commas.
        bbox: A box as west,south,east,north in degrees, e.g. "-125,32,-114,42".
        days: The last days to search, today included, without dates; for a longer period give
            start_date and end_date.
        start_date: The period's first day, YYYY-MM-DD.
        end_date: The period's last day, YYYY-MM-DD.
        max_results: How many events to list.
        offset: How many events to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when an option is invalid.
    """
    _validate_events(category, status, source, bbox, start_date, end_date)
    filters = {
        "status": status,
        "category": category.strip(),
        "source": source.strip(),
        **_period(days, start_date, end_date),
    }
    shown = {name: value for name, value in filters.items() if value}
    params = {**shown, "limit": str(asked(offset, max_results, _DEPTH))}
    if bbox.strip():
        params["bbox"] = _eonet_box(bbox)
        shown["bbox"] = bbox.strip()
    return _API.get_json(
        "events",
        params=params,
        parse=lambda data: _events_answer(data, shown, offset, max_results),
    )


@tool(capability="network")
def eonet_event(
    event_id: str,
    offset: Annotated[int, Range(0)] = 0,
    max_points: Annotated[int, Range(1, _MAX_POINTS)] = 50,
) -> ToolResult:
    """Get a NASA EONET event by ID: its categories, status, sources and its whole track, one
    dated position per line.

    Args:
        event_id: An EONET event ID from eonet_events, e.g. "EONET_12345".
        offset: How many track points to skip; the footer gives the next offset.
        max_points: How many track points to show.

    Raises:
        ToolFailure: validation_error when ``event_id`` is malformed; not_found when EONET
            answers that it has no event with it (HTTP 404); upstream when EONET fails, its
            HTTP 500 saying the ID may be unknown (EONET sends it for one).
    """
    event_id = event_id.strip()
    if not _ID_RE.fullmatch(event_id):
        msg = f"invalid event_id {event_id!r}; an EONET event ID looks like EONET_12345."
        raise ToolFailure("validation_error", msg)
    missing = (
        f"NASA EONET has no event with ID {event_id!r}; list the current IDs with eonet_events."
    )
    return _EVENT_API.get_json(
        event_id,
        parse=lambda data: _event_answer(data, event_id, offset, max_points),
        missing=missing,
    )


def _period(days: int, start_date: str, end_date: str) -> dict[str, str]:
    """The period's parameters: the dates given, as YYYY-MM-DD, or the last ``days``."""
    if start_date.strip() or end_date.strip():
        start, end = _parse_date(start_date.strip()), _parse_date(end_date.strip())
        return {
            "start": start.isoformat() if start else "",
            "end": end.isoformat() if end else "",
        }
    return {"days": str(days)}


def _eonet_box(bbox: str) -> str:
    """The tool's west,south,east,north as EONET's order: the upper-left corner, then the
    lower-right (min lon, max lat, max lon, min lat; https://eonet.gsfc.nasa.gov/docs/v3)."""
    west, south, east, north = (part.strip() for part in bbox.split(","))
    return f"{west},{north},{east},{south}"


# --- Answers -----------------------------------------------------------------------------------


def _categories_text(data: dict[str, Any]) -> str:
    categories = [item for item in data.get("categories") or [] if isinstance(item, dict)]
    if not categories:
        return "NASA EONET lists no categories."
    lines = ["NASA EONET categories (the ID first, as eonet_events takes it):"]
    for number, category in enumerate(categories, start=1):
        lines.append(f"{number}. {_string(category.get('id'))}: {_string(category.get('title'))}")
        if description := _string(category.get("description")):
            lines.append(f"   {description}")
    return "\n".join(lines)


def _events_answer(
    data: dict[str, Any], filters: dict[str, str], offset: int, limit: int
) -> ToolResult:
    events = [item for item in data.get("events") or [] if isinstance(item, dict)]
    asked_for = ", ".join(f"{name}={value}" for name, value in filters.items())
    if not events:
        return ToolResult.success(
            f"No NASA EONET events match {asked_for}; widen the period (days, start_date) or "
            "drop a filter."
        )
    lines = [_event_line(number, event) for number, event in enumerate(events, start=1)]
    window = first_results_window(
        lines,
        offset=offset,
        limit=limit,
        requested=asked(offset, limit, _DEPTH),
        depth=_DEPTH,
        narrow="narrow the period (start_date, end_date), the category or the bbox",
    )
    return window.result(heading=f"NASA EONET events for {asked_for}:")


def _event_line(number: int, event: dict[str, Any]) -> str:
    points = _track(event)
    head = f"{number}. {_string(event.get('title'))} | id: {_string(event.get('id'))}"
    head += f" | {_categories(event)} | {_status(event)}"
    if not points:
        return f"{head}\n   no track"
    span = f"{points[0][0]} to {points[-1][0]}" if len(points) > 1 else points[0][0]
    latest = points[-1]
    return (
        f"{head}\n   track: {len(points)} points, {span}; "
        f"latest at {latest[1]}{_magnitude(latest[2])} (eonet_event reads the track)"
    )


def _event_answer(data: dict[str, Any], event_id: str, offset: int, limit: int) -> ToolResult:
    if not _string(data.get("id")):
        msg = f"NASA EONET answered without an event for {event_id!r}; try again later."
        raise ToolFailure("upstream", msg)
    lines = [
        f"NASA EONET event {_string(data.get('id'))}: {_string(data.get('title'))}",
        f"  Categories: {_categories(data)}",
        f"  Status: {_status(data)}",
    ]
    if description := _string(data.get("description")):
        lines.append(f"  Description: {description}")
    sources = [item for item in data.get("sources") or [] if isinstance(item, dict)]
    if sources:
        listed = (f"{_string(s.get('id'))}: {_string(s.get('url'))}" for s in sources)
        lines.append(f"  Sources: {'; '.join(listed)}")
    points = _track(data)
    if not points:
        return ToolResult.success("\n".join([*lines, "  Track: no positions"]))
    rows = [f"{when} | {where}{_magnitude(size)}" for when, where, size in points]
    lines.append(f"  Track ({len(points)} points; date | position | magnitude):")
    return page_window(rows, offset=offset, limit=limit).result(heading="\n".join(lines))


def _track(event: dict[str, Any]) -> list[tuple[str, str, str]]:
    """The event's dated positions, in EONET's order: (date, position, magnitude)."""
    geometry = event.get("geometry")
    if not isinstance(geometry, list):
        return []
    points = [item for item in geometry if isinstance(item, dict)]
    return [
        (_string(item.get("date")) or "undated", _position(item), _size(item)) for item in points
    ]


def _position(geometry: dict[str, Any]) -> str:
    """A GeoJSON point or polygon, labelled: a point's latitude and longitude, a polygon's span."""
    coordinates = geometry.get("coordinates")
    if _string(geometry.get("type")) == "Polygon" and isinstance(coordinates, list):
        corners = [c for ring in coordinates if isinstance(ring, list) for c in ring]
        pairs = [c for c in corners if isinstance(c, list) and len(c) >= 2]
        if pairs:
            lons = [pair[0] for pair in pairs]
            lats = [pair[1] for pair in pairs]
            return (
                f"area latitude {plain(min(lats))} to {plain(max(lats))}, "
                f"longitude {plain(min(lons))} to {plain(max(lons))}"
            )
    if isinstance(coordinates, list) and len(coordinates) >= 2:
        return f"latitude {plain(coordinates[1])}, longitude {plain(coordinates[0])}"
    return "no position"


def _size(geometry: dict[str, Any]) -> str:
    value = geometry.get("magnitudeValue")
    if value is None:
        return ""
    unit = _string(geometry.get("magnitudeUnit"))
    return f"{plain(value)} {unit}".strip()


def _magnitude(size: str) -> str:
    return f", magnitude {size}" if size else ""


def _categories(event: dict[str, Any]) -> str:
    categories = [item for item in event.get("categories") or [] if isinstance(item, dict)]
    named = (
        f"{_string(item.get('title'))} ({_string(item.get('id'))})"
        if _string(item.get("title"))
        else _string(item.get("id"))
        for item in categories
    )
    return ", ".join(name for name in named if name) or "no category"


def _status(event: dict[str, Any]) -> str:
    closed = _string(event.get("closed"))
    return f"closed {closed}" if closed else "open"


# --- Arguments ---------------------------------------------------------------------------------


def _validate_events(
    category: str,
    status: str,
    source: str,
    bbox: str,
    start_date: str,
    end_date: str,
) -> None:
    if category and not _IDS_RE.fullmatch(category.strip()):
        _invalid(
            f"invalid category {category!r}; eonet_categories lists the category IDs, given "
            "alone or separated by commas."
        )
    if source and not _IDS_RE.fullmatch(source.strip()):
        _invalid(
            f"invalid source {source!r}; pass EONET source IDs such as InciWeb or InciWeb,EO."
        )
    if status not in _STATUSES:
        _invalid(f"status must be open, closed, or all, got {status!r}.")
    if bbox.strip() and not _valid_bbox(bbox):
        _invalid(f"bbox must be west,south,east,north in degrees, got {bbox!r}.")
    start = _parse_date(start_date.strip()) if start_date.strip() else None
    end = _parse_date(end_date.strip()) if end_date.strip() else None
    if start_date.strip() and start is None:
        _invalid(f"invalid start_date {start_date!r}; use YYYY-MM-DD.")
    if end_date.strip() and end is None:
        _invalid(f"invalid end_date {end_date!r}; use YYYY-MM-DD.")
    if start and end and start > end:
        _invalid("start_date must be before or equal to end_date.")


def _invalid(message: str) -> NoReturn:
    raise ToolFailure("validation_error", message)


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


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
