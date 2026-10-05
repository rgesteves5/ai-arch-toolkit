"""Tests for toolkit/tools/_air_quality.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._air_quality import (
    air_quality_current,
    air_quality_forecast,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_CURRENT = {
    "latitude": 38.75,
    "longitude": -9.15,
    "timezone": "Europe/Lisbon",
    "current_units": {"time": "iso8601", "european_aqi": "EAQI", "pm2_5": "µg/m³"},
    "current": {"time": "2026-06-12T12:00", "european_aqi": 35, "pm2_5": 8.3},
}
_FORECAST = {
    "latitude": 38.75,
    "longitude": -9.15,
    "timezone": "Europe/Lisbon",
    "hourly_units": {"time": "iso8601", "pm10": "µg/m³", "ozone": "µg/m³"},
    "hourly": {
        "time": ["2026-06-12T12:00", "2026-06-12T13:00"],
        "pm10": [12.5, 13.0],
        "ozone": [78, 80],
    },
}


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestAirQualityCurrent:
    @patch(HTTP_OPEN)
    def test_returns_current_values(self, mock_urlopen):
        mock_urlopen.return_value = respond(_CURRENT)

        result = air_quality_current(38.75, -9.15, variables="european_aqi,pm2_5")

        assert "Open-Meteo air quality current for 38.75, -9.15" in result
        assert "Time: 2026-06-12T12:00" in result
        assert "european_aqi: 35 EAQI" in result
        assert "pm2_5: 8.3 µg/m³" in result
        assert "Attribution: Open-Meteo" in result

        params = _called_params(mock_urlopen)
        assert params["latitude"] == ["38.75"]
        assert params["longitude"] == ["-9.15"]
        assert params["current"] == ["european_aqi,pm2_5"]
        assert params["timezone"] == ["auto"]

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
    def test_returns_forecast_values_and_caps_options(self, mock_urlopen):
        mock_urlopen.return_value = respond(_FORECAST)

        result = air_quality_forecast(
            38.75,
            -9.15,
            variables="pm10,ozone",
            forecast_days=99,
            past_days=99,
            max_hours=1,
        )

        assert "Open-Meteo air quality forecast (1 hours)" in result
        assert "2026-06-12T12:00 | pm10: 12.5 µg/m³ | ozone: 78 µg/m³" in result
        assert "2026-06-12T13:00" not in result

        params = _called_params(mock_urlopen)
        assert params["hourly"] == ["pm10,ozone"]
        assert params["forecast_days"] == ["7"]
        assert params["past_days"] == ["7"]

    @patch(HTTP_OPEN)
    def test_rejected_parameter_is_validation_error(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            400, "Bad Request", body=b'{"error": true, "reason": "Invalid timezone"}'
        )

        with pytest.raises(ToolFailure) as caught:
            air_quality_forecast(0, 0, timezone="Mars/Olympus")

        assert caught.value.error.type == "validation_error"
        assert not caught.value.error.retryable
        assert "Open-Meteo rejected the request: Invalid timezone" in caught.value.error.message
        assert "'auto'" in caught.value.error.message

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
        assert "Open-Meteo: endpoint not found (HTTP 404)" in caught.value.error.message

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
