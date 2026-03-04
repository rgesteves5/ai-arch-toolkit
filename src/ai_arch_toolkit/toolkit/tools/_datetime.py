"""Date & time tools — current time and timezone conversion."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, available_timezones

from ai_arch_toolkit.core import tool


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
