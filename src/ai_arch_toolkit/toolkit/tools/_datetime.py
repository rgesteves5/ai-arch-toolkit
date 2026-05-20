"""Date & time tools — current time, arithmetic, and timezone conversion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, available_timezones

from ai_arch_toolkit.core import tool

_DATE_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d")


@tool
def datetime_now(tz: str = "UTC") -> str:
    """Get the current date and time in a given timezone.

    Args:
        tz: IANA timezone name, e.g. "America/New_York", "Asia/Tokyo", "Europe/London".
            Defaults to UTC.
    """
    if tz.upper() == "UTC":
        zone = UTC
    else:
        matches = [t for t in available_timezones() if t.lower() == tz.lower()]
        if not matches:
            return f"Unknown timezone: {tz!r}. Use IANA names like 'America/New_York'."
        zone = ZoneInfo(matches[0])

    now = datetime.now(zone)
    return f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')} ({now.strftime('%A')})"


@tool
def timezone_convert(time_str: str, from_tz: str, to_tz: str) -> str:
    """Convert a time from one timezone to another.

    Args:
        time_str: Time in "HH:MM" or "YYYY-MM-DD HH:MM" format.
        from_tz: Source IANA timezone, e.g. "America/New_York".
        to_tz: Target IANA timezone, e.g. "Asia/Tokyo".
    """
    try:
        from_zone = ZoneInfo(from_tz)
        to_zone = ZoneInfo(to_tz)
    except (KeyError, ValueError) as e:
        return f"Invalid timezone: {e}"

    try:
        if " " in time_str:
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        else:
            today = datetime.now(from_zone).date()
            t = datetime.strptime(time_str, "%H:%M").time()
            dt = datetime.combine(today, t)
    except ValueError:
        return f"Invalid time format: {time_str!r}. Use 'HH:MM' or 'YYYY-MM-DD HH:MM'."

    localized = dt.replace(tzinfo=from_zone)
    converted = localized.astimezone(to_zone)
    return f"{localized.strftime('%Y-%m-%d %H:%M %Z')} → {converted.strftime('%Y-%m-%d %H:%M %Z')}"


@tool
def date_add(date_str: str, days: int = 0, hours: int = 0, minutes: int = 0) -> str:
    """Add days, hours, and minutes to a date/time string.

    Args:
        date_str: Date/time in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" format.
        days: Number of days to add. Defaults to 0.
        hours: Number of hours to add. Defaults to 0.
        minutes: Number of minutes to add. Defaults to 0.
    """
    parsed = _parse_datetime(date_str)
    if isinstance(parsed, str):
        return parsed
    dt, input_format = parsed

    result = dt + timedelta(days=days, hours=hours, minutes=minutes)
    return _format_datetime(result, input_format, force_datetime=hours != 0 or minutes != 0)


@tool
def date_diff(start: str, end: str, unit: str = "seconds") -> str:
    """Calculate the difference between two date/time strings.

    Args:
        start: Start date/time in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" format.
        end: End date/time in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" format.
        unit: Output unit: "seconds", "minutes", "hours", or "days".
    """
    parsed_start = _parse_datetime(start)
    if isinstance(parsed_start, str):
        return parsed_start.replace("date/time", "start date/time")
    parsed_end = _parse_datetime(end)
    if isinstance(parsed_end, str):
        return parsed_end.replace("date/time", "end date/time")

    start_dt, _ = parsed_start
    end_dt, _ = parsed_end

    seconds = (end_dt - start_dt).total_seconds()
    scales = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600,
        "days": 86400,
    }
    if unit not in scales:
        return f"Invalid unit: {unit!r}. Use 'seconds', 'minutes', 'hours', or 'days'."

    value = seconds / scales[unit]
    value_str = str(int(value)) if value == int(value) else f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{start} → {end} = {value_str} {unit}"


@tool
def date_format(date_str: str, format_out: str) -> str:
    """Format a date/time string using strftime syntax.

    Args:
        date_str: Date/time in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" format.
        format_out: strftime output format, e.g. "%d/%m/%Y" or "%A, %b %d".
    """
    parsed = _parse_datetime(date_str)
    if isinstance(parsed, str):
        return parsed
    dt, _ = parsed
    return dt.strftime(format_out)


def _parse_datetime(value: str) -> tuple[datetime, str] | str:
    """Parse supported date/time formats and return the datetime plus matching format."""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt), fmt
        except ValueError:
            continue
    return f"Invalid date/time format: {value!r}. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'."


def _format_datetime(dt: datetime, input_format: str, force_datetime: bool = False) -> str:
    """Format a datetime while preserving date-only inputs when possible."""
    if input_format == "%Y-%m-%d" and not force_datetime and dt.time() == datetime.min.time():
        return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d %H:%M")
