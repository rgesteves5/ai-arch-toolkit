"""Tests for toolkit/tools/_air_quality.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

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

    @patch(HTTP_OPEN)
    def test_invalid_current_options_do_not_call_api(self, mock_urlopen):
        assert "latitude must" in air_quality_current(-91, 0)
        assert "longitude must" in air_quality_current(0, 181)
        assert "variables cannot be empty" in air_quality_current(0, 0, variables="")
        assert "invalid variables" in air_quality_current(0, 0, variables="bad")
        mock_urlopen.assert_not_called()


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
    def test_api_error_body(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            400, "Bad Request", body=b'{"reason": "invalid variable"}'
        )

        result = air_quality_forecast(0, 0)

        assert "HTTP error 400: invalid variable" in result

    @patch(HTTP_OPEN)
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond("not json")

        result = air_quality_current(0, 0)

        assert "could not parse" in result
