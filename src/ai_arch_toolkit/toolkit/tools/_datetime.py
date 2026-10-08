"""Date & time tools — current time, arithmetic, and timezone conversion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo, available_timezones

from ai_arch_toolkit.core import Range, tool
from ai_arch_toolkit.core._tools._result import ToolFailure

_DATE_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d")
# The calendar's span, years 1 to 9999: no shift longer than it lands on a date.
_SPAN = datetime.max - datetime.min
_DAYS = _SPAN.days
_HOURS = _SPAN // timedelta(hours=1)
_MINUTES = _SPAN // timedelta(minutes=1)


@tool(capability="compute")
def datetime_now(tz: str = "UTC") -> str:
    """Get the current date and time in a given timezone.

    Args:
        tz: IANA timezone name, e.g. "America/New_York", "Asia/Tokyo", "Europe/London".
            Defaults to UTC.

    Raises:
        ToolFailure: validation_error when the timezone is unknown.
    """
    if tz.upper() == "UTC":
        zone = UTC
    else:
        matches = [t for t in available_timezones() if t.lower() == tz.lower()]
        if not matches:
            msg = f"unknown timezone {tz!r}; use an IANA name such as 'America/New_York'."
            raise ToolFailure("validation_error", msg)
        zone = ZoneInfo(matches[0])

    now = datetime.now(zone)
    return f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')} ({now.strftime('%A')})"


@tool(capability="compute")
def timezone_convert(time_str: str, from_tz: str, to_tz: str) -> str:
    """Convert a time from one timezone to another.

    Args:
        time_str: Time in "HH:MM" or "YYYY-MM-DD HH:MM" format.
        from_tz: Source IANA timezone, e.g. "America/New_York".
        to_tz: Target IANA timezone, e.g. "Asia/Tokyo".

    Raises:
        ToolFailure: validation_error when a timezone or the time is invalid, or the result
            falls outside the calendar (years 1-9999).
    """
    from_zone = _zone("from_tz", from_tz)
    to_zone = _zone("to_tz", to_tz)

    try:
        if " " in time_str:
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        else:
            today = datetime.now(from_zone).date()
            t = datetime.strptime(time_str, "%H:%M").time()
            dt = datetime.combine(today, t)
    except ValueError as e:
        msg = f"invalid time {time_str!r}; use 'HH:MM' or 'YYYY-MM-DD HH:MM'."
        raise ToolFailure("validation_error", msg) from e

    localized = dt.replace(tzinfo=from_zone)
    try:
        converted = localized.astimezone(to_zone)
    except OverflowError as e:
        raise _out_of_range(e) from e
    return f"{localized.strftime('%Y-%m-%d %H:%M %Z')} → {converted.strftime('%Y-%m-%d %H:%M %Z')}"


@tool(capability="compute")
def date_add(
    date_str: str,
    days: Annotated[int, Range(-_DAYS, _DAYS)] = 0,
    hours: Annotated[int, Range(-_HOURS, _HOURS)] = 0,
    minutes: Annotated[int, Range(-_MINUTES, _MINUTES)] = 0,
) -> str:
    """Add days, hours, and minutes to a date/time string.

    Args:
        date_str: Date/time in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" format.
        days: Number of days to add (negative to subtract). Defaults to 0.
        hours: Number of hours to add. Defaults to 0.
        minutes: Number of minutes to add. Defaults to 0.

    Raises:
        ToolFailure: validation_error when the date is malformed or the result falls outside
            the calendar (years 1-9999).
    """
    dt, input_format = _parse_datetime(date_str)

    try:
        result = dt + timedelta(days=days, hours=hours, minutes=minutes)
    except OverflowError as e:
        raise _out_of_range(e) from e
    return _format_datetime(result, input_format, force_datetime=hours != 0 or minutes != 0)


@tool(capability="compute")
def date_diff(start: str, end: str, unit: str = "seconds") -> str:
    """Calculate the difference between two date/time strings.

    Args:
        start: Start date/time in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" format.
        end: End date/time in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" format.
        unit: Output unit: "seconds", "minutes", "hours", or "days".

    Raises:
        ToolFailure: validation_error when a date is malformed or the unit is unknown.
    """
    start_dt, _ = _parse_datetime(start, "start")
    end_dt, _ = _parse_datetime(end, "end")

    seconds = (end_dt - start_dt).total_seconds()
    scales = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600,
        "days": 86400,
    }
    if unit not in scales:
        msg = f"invalid unit {unit!r}; use 'seconds', 'minutes', 'hours', or 'days'."
        raise ToolFailure("validation_error", msg)

    value = seconds / scales[unit]
    value_str = str(int(value)) if value == int(value) else f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{start} → {end} = {value_str} {unit}"


@tool(capability="compute")
def date_format(date_str: str, format_out: str) -> str:
    """Format a date/time string using strftime syntax.

    Args:
        date_str: Date/time in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" format.
        format_out: strftime output format, e.g. "%d/%m/%Y" or "%A, %b %d".

    Raises:
        ToolFailure: validation_error when the date is malformed.
    """
    dt, _ = _parse_datetime(date_str)
    return dt.strftime(format_out)


def _zone(name: str, value: str) -> ZoneInfo:
    """The IANA timezone ``value``; raises when there is none by that name."""
    try:
        return ZoneInfo(value)
    except (KeyError, ValueError) as e:
        msg = f"invalid {name} {value!r}; use an IANA name such as 'America/New_York'."
        raise ToolFailure("validation_error", msg) from e


def _out_of_range(error: OverflowError) -> ToolFailure:
    msg = f"the result falls outside the calendar (years 1-9999): {error}."
    return ToolFailure("validation_error", msg)


def _parse_datetime(value: str, which: str = "") -> tuple[datetime, str]:
    """Parse supported date/time formats and return the datetime plus matching format.

    Raises:
        ToolFailure: validation_error when ``value`` matches no supported format.
    """
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt), fmt
        except ValueError:
            continue
    label = f"{which} date/time" if which else "date/time"
    msg = f"invalid {label} {value!r}; use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'."
    raise ToolFailure("validation_error", msg)


def _format_datetime(dt: datetime, input_format: str, force_datetime: bool = False) -> str:
    """Format a datetime while preserving date-only inputs when possible."""
    if input_format == "%Y-%m-%d" and not force_datetime and dt.time() == datetime.min.time():
        return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d %H:%M")
