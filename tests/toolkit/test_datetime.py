"""Tests for toolkit/tools/_datetime.py."""

from __future__ import annotations

import re

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._datetime import (
    date_add,
    date_diff,
    date_format,
    datetime_now,
    timezone_convert,
)


class TestDateAddLimits:
    """The shifts ``date_add`` takes are bounded by the calendar's span, years 1 to 9999."""

    def test_the_schema_bounds_each_shift_by_the_calendars_span(self):
        properties = date_add.__tool_definition__.schema.input_schema["properties"]

        bounds = {
            name: (properties[name]["minimum"], properties[name]["maximum"])
            for name in ("days", "hours", "minutes")
        }

        assert bounds == {
            "days": (-3_652_058, 3_652_058),
            "hours": (-87_649_415, 87_649_415),
            "minutes": (-5_258_964_959, 5_258_964_959),
        }

    def test_the_whole_span_is_reachable(self):
        assert date_add("0001-01-01", days=3_652_058) == "9999-12-31"
        assert date_add("9999-12-31 23:59", minutes=-5_258_964_959) == "0001-01-01 00:00"

    def test_the_executor_refuses_a_shift_past_the_span(self):
        call = ToolCall(id="c", name="date_add", input={"date_str": "2024-01-01", "days": 10**9})

        result = ToolGroup(date_add).execute(call)

        assert result.error is not None and result.error.type == "validation_error"
        assert "from -3652058 to 3652058" in result.error.message


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
        with pytest.raises(ToolFailure) as caught:
            datetime_now("Mars/Olympus")

        assert caught.value.error.type == "validation_error"
        assert "unknown timezone 'Mars/Olympus'" in caught.value.error.message

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
        with pytest.raises(ToolFailure) as caught:
            timezone_convert("12:00", "Fake/Zone", "UTC")

        assert caught.value.error.type == "validation_error"
        assert "invalid from_tz 'Fake/Zone'" in caught.value.error.message

    def test_invalid_target_timezone(self):
        with pytest.raises(ToolFailure) as caught:
            timezone_convert("12:00", "UTC", "Fake/Zone")

        assert caught.value.error.type == "validation_error"
        assert "invalid to_tz 'Fake/Zone'" in caught.value.error.message

    def test_invalid_time_format(self):
        with pytest.raises(ToolFailure) as caught:
            timezone_convert("not-a-time", "UTC", "UTC")

        assert caught.value.error.type == "validation_error"
        assert "invalid time 'not-a-time'" in caught.value.error.message


class TestDateAdd:
    def test_add_days_to_date(self):
        result = date_add("2026-01-15", days=2)
        assert result == "2026-01-17"

    def test_add_time_to_date_forces_datetime_output(self):
        result = date_add("2026-01-15", hours=2, minutes=30)
        assert result == "2026-01-15 02:30"

    def test_invalid_input(self):
        with pytest.raises(ToolFailure) as caught:
            date_add("15/01/2026", days=1)

        assert caught.value.error.type == "validation_error"
        assert "invalid date/time '15/01/2026'" in caught.value.error.message


class TestDateDiff:
    def test_diff_in_days(self):
        result = date_diff("2026-01-15", "2026-01-17", unit="days")
        assert result.endswith("= 2 days")

    def test_diff_in_hours(self):
        result = date_diff("2026-01-15 09:00", "2026-01-15 12:30", unit="hours")
        assert result.endswith("= 3.5 hours")

    def test_invalid_unit(self):
        with pytest.raises(ToolFailure) as caught:
            date_diff("2026-01-15", "2026-01-17", unit="weeks")

        assert caught.value.error.type == "validation_error"
        assert "invalid unit 'weeks'" in caught.value.error.message

    def test_invalid_start_and_end(self):
        with pytest.raises(ToolFailure) as caught:
            date_diff("bad", "2026-01-17")

        assert caught.value.error.type == "validation_error"
        assert "invalid start date/time 'bad'" in caught.value.error.message

        with pytest.raises(ToolFailure) as caught:
            date_diff("2026-01-15", "bad")

        assert caught.value.error.type == "validation_error"
        assert "invalid end date/time 'bad'" in caught.value.error.message


class TestDateFormat:
    def test_reformats_date(self):
        result = date_format("2026-01-15", "%d/%m/%Y")
        assert result == "15/01/2026"

    def test_reformats_datetime(self):
        result = date_format("2026-01-15 09:30", "%H:%M on %A")
        assert result.startswith("09:30 on ")

    def test_invalid_input(self):
        with pytest.raises(ToolFailure) as caught:
            date_format("2026/01/15", "%Y")

        assert caught.value.error.type == "validation_error"
        assert "invalid date/time" in caught.value.error.message


class TestRange:
    @pytest.mark.parametrize(("date_str", "days"), [("2024-01-01", 10**9), ("9999-12-31", 1)])
    def test_arithmetic_past_the_calendar_is_a_validation_error(self, date_str, days):
        with pytest.raises(ToolFailure) as caught:
            date_add(date_str, days=days)

        assert caught.value.error.type == "validation_error"
        assert "outside the calendar" in caught.value.error.message
        assert caught.value.error.message.endswith(
            "; use a smaller shift or a date further from the calendar's ends."
        )

    def test_a_conversion_past_the_calendar_is_a_validation_error(self):
        with pytest.raises(ToolFailure) as caught:
            timezone_convert("9999-12-31 23:59", "America/New_York", "Asia/Tokyo")

        assert caught.value.error.type == "validation_error"
        assert "outside the calendar" in caught.value.error.message
