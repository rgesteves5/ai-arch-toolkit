"""Tests for toolkit/tools/_datetime.py."""

from __future__ import annotations

import re

from ai_arch_toolkit.toolkit.tools._datetime import datetime_now, timezone_convert


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
