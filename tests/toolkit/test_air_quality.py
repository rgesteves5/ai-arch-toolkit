"""Tests for toolkit/tools/_air_quality.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._air_quality import (
    air_quality_current,
    air_quality_forecast,
)
from tests.toolkit import geo_answers
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_CURRENT = {
    "latitude": 38.75,
    "longitude": -9.15,
    "timezone": "Europe/Lisbon",
    "current_units": {"time": "unixtime", "european_aqi": "EAQI", "pm2_5": "μg/m³"},
    "current": {"time": 1781280000, "european_aqi": 35, "pm2_5": 8.3},
}


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


class TestAirQualityCurrent:
    @patch(HTTP_OPEN)
    def test_returns_current_values_with_their_units_at_a_utc_time(self, mock_urlopen):
        mock_urlopen.return_value = respond(_CURRENT)

        result = air_quality_current(38.75, -9.15, variables="european_aqi,pm2_5")

        assert result == (
            "Current air quality at latitude 38.75, longitude -9.15 (time zone Europe/Lisbon) "
            "(Open-Meteo Air Quality API, CAMS data):\n"
            "Time: 2026-06-12T16:00:00Z\n"
            "european_aqi: 35 EAQI\n"
            "pm2_5: 8.3 μg/m³"
        )
        params = _called_params(mock_urlopen)
        assert params["latitude"] == ["38.75"]
        assert params["longitude"] == ["-9.15"]
        assert params["current"] == ["european_aqi,pm2_5"]
        assert params["timezone"] == ["auto"]
        assert params["timeformat"] == ["unixtime"]

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"latitude": -91, "longitude": 0}, "latitude must"),
            ({"latitude": 0, "longitude": 181}, "longitude must"),
            ({"latitude": 0, "longitude": 0, "variables": ""}, "variables cannot be empty"),
            ({"latitude": 0, "longitude": 0, "variables": "bad"}, "invalid variables: bad"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_current_options_do_not_call_api(self, mock_urlopen, kwargs, words):
        with pytest.raises(ToolFailure) as caught:
            air_quality_current(**kwargs)

        assert caught.value.error.type == "validation_error"
        assert words in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_answer_without_current_values_is_upstream(self, mock_urlopen):
        mock_urlopen.return_value = respond({"latitude": 0, "longitude": 0})

        with pytest.raises(ToolFailure) as caught:
            air_quality_current(0, 0)

        assert caught.value.error.type == "upstream"
        assert "without current values" in caught.value.error.message


class TestAirQualityForecast:
    @patch(HTTP_OPEN)
    def test_every_hour_is_reachable_through_the_window(self, mock_urlopen):
        # Up to 72 of 336 rows (7 days back, 7 ahead) were shown, and the rest were lost.
        mock_urlopen.side_effect = lambda *_: respond(geo_answers.air_quality_hours(336))

        first = air_quality_forecast(38.75, -9.15, forecast_days=7, past_days=7, max_hours=72)
        last = air_quality_forecast(
            38.75, -9.15, forecast_days=7, past_days=7, max_hours=72, offset=288
        )

        assert isinstance(first, ToolResult)
        assert _text(first).endswith("[results 1-72 of 336 | next: offset=72]")
        assert first.metadata["window"]["next_call"] == {"offset": 72}
        assert (
            _text(last)
            .splitlines()[-2]
            .startswith("2026-06-26T15:00:00Z | european_aqi: 355 EAQI")
        )
        assert _text(last).endswith("[results 289-336 of 336 | end]")
        params = _called_params(mock_urlopen)
        assert params["forecast_days"] == ["7"]
        assert params["past_days"] == ["7"]
        assert params["timeformat"] == ["unixtime"]

    @patch(HTTP_OPEN)
    def test_each_hour_is_a_utc_time_with_its_values_and_units(self, mock_urlopen):
        mock_urlopen.return_value = respond(geo_answers.air_quality_hours(2))

        text = _text(air_quality_forecast(38.75, -9.15, variables="pm10,european_aqi"))

        assert text.splitlines() == [
            "Hourly air quality at latitude 38.75, longitude -9.15 (time zone Europe/Lisbon), "
            "times in UTC (Open-Meteo Air Quality API, CAMS data):",
            "2026-06-12T16:00:00Z | pm10: 10.5 μg/m³ | european_aqi: 20 EAQI",
            "2026-06-12T17:00:00Z | pm10: 11.5 μg/m³ | european_aqi: 21 EAQI",
        ]

    def test_the_days_and_hours_are_bounded_in_the_schema_as_open_meteo_allows(self):
        properties = air_quality_forecast.__tool_definition__.schema.input_schema["properties"]
        bounds = {
            name: (properties[name].get("minimum"), properties[name].get("maximum"))
            for name in ("forecast_days", "past_days", "max_hours", "offset")
        }
        assert bounds == {
            "forecast_days": (1, 7),
            "past_days": (0, 92),
            "max_hours": (1, 72),
            "offset": (0, None),
        }

    @patch(HTTP_OPEN)
    def test_rejected_parameter_is_validation_error(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            400, "Bad Request", body=b'{"error": true, "reason": "Invalid timezone"}'
        )

        with pytest.raises(ToolFailure) as caught:
            air_quality_forecast(0, 0, timezone="Mars/Olympus")

        assert caught.value.error.type == "validation_error"
        assert not caught.value.error.retryable
        assert caught.value.error.message == (
            "Open-Meteo rejected the request: Invalid timezone; correct the argument it names "
            "and call again"
        )

    @patch(HTTP_OPEN)
    def test_server_error_reason_is_retryable_upstream(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            500, "Internal Server Error", body=b'{"error": true, "reason": "backend down"}'
        )

        with pytest.raises(ToolFailure) as caught:
            air_quality_forecast(0, 0)

        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable
        assert "HTTP error 500: backend down" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_error_reported_in_a_success_is_upstream(self, mock_urlopen):
        mock_urlopen.return_value = respond({"error": True, "reason": "data unavailable"})

        with pytest.raises(ToolFailure) as caught:
            air_quality_current(0, 0)

        assert caught.value.error.type == "upstream"
        assert caught.value.error.message == "data unavailable"

    @patch(HTTP_OPEN)
    def test_404_is_endpoint_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        with pytest.raises(ToolFailure) as caught:
            air_quality_current(0, 0)

        assert caught.value.error.type == "upstream"
        assert "endpoint not found (HTTP 404)" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_rate_limited(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        with pytest.raises(ToolFailure) as caught:
            air_quality_forecast(0, 0)

        assert caught.value.error.type == "rate_limited"
        assert caught.value.error.retryable

    @patch(HTTP_OPEN)
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond("not json")

        with pytest.raises(ToolFailure) as caught:
            air_quality_current(0, 0)

        assert caught.value.error.type == "upstream"
        assert "could not parse" in caught.value.error.message
