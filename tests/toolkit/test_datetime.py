"""Tests for toolkit/tools/_datetime.py."""

from __future__ import annotations

import re

from ai_arch_toolkit.toolkit.tools._datetime import (
    date_add,
    date_diff,
    date_format,
    datetime_now,
    timezone_convert,
)


class TestDatetimeNow:
    def test_utc_default(self):
        result = datetime_now()
        assert "UTC" in result

    def test_utc_explicit(self):
        result = datetime_now("UTC")
        assert "UTC" in result

    def test_named_timezone(self):
        result = datetime_now("America/New_York")
        # Should contain a day name in parentheses
        assert re.search(r"\(\w+\)$", result)

    def test_case_insensitive(self):
        result = datetime_now("asia/tokyo")
        assert "JST" in result or "Asia/Tokyo" in result or "20" in result

    def test_unknown_timezone(self):
        result = datetime_now("Mars/Olympus")
        assert "Unknown timezone" in result

    def test_format_contains_date(self):
        result = datetime_now("UTC")
        # Should have YYYY-MM-DD
        assert re.search(r"\d{4}-\d{2}-\d{2}", result)


class TestTimezoneConvert:
    def test_hhmm_format(self):
        result = timezone_convert("12:00", "America/New_York", "Europe/London")
        assert "12:00" in result
        assert "→" in result or "->" in result or "New_York" in result

    def test_full_datetime_format(self):
        result = timezone_convert("2026-01-15 09:00", "UTC", "Asia/Tokyo")
        assert "2026-01-15" in result
        assert "18:00" in result  # UTC+9

    def test_invalid_timezone(self):
        result = timezone_convert("12:00", "Fake/Zone", "UTC")
        assert "Invalid timezone" in result

    def test_invalid_time_format(self):
        result = timezone_convert("not-a-time", "UTC", "UTC")
        assert "Invalid time format" in result


class TestDateAdd:
    def test_add_days_to_date(self):
        result = date_add("2026-01-15", days=2)
        assert result == "2026-01-17"

    def test_add_time_to_date_forces_datetime_output(self):
        result = date_add("2026-01-15", hours=2, minutes=30)
        assert result == "2026-01-15 02:30"

    def test_invalid_input(self):
        result = date_add("15/01/2026", days=1)
        assert "Invalid date/time format" in result


class TestDateDiff:
    def test_diff_in_days(self):
        result = date_diff("2026-01-15", "2026-01-17", unit="days")
        assert result.endswith("= 2 days")

    def test_diff_in_hours(self):
        result = date_diff("2026-01-15 09:00", "2026-01-15 12:30", unit="hours")
        assert result.endswith("= 3.5 hours")

    def test_invalid_unit(self):
        result = date_diff("2026-01-15", "2026-01-17", unit="weeks")
        assert "Invalid unit" in result


class TestDateFormat:
    def test_reformats_date(self):
        result = date_format("2026-01-15", "%d/%m/%Y")
        assert result == "15/01/2026"

    def test_reformats_datetime(self):
        result = date_format("2026-01-15 09:30", "%H:%M on %A")
        assert result.startswith("09:30 on ")
