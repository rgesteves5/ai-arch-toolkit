"""Tests for toolkit/tools/_weather.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.toolkit.tools._weather import (
    get_forecast,
    get_forecast_by_coords,
    get_weather,
    get_weather_by_coords,
    weather_units,
)


def _mock_urlopen(data: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


_GEOCODE_RESPONSE = {
    "results": [{"latitude": 35.6762, "longitude": 139.6503, "name": "Tokyo", "country": "Japan"}]
}

_CURRENT_WEATHER = {
    "current": {
        "temperature_2m": 22.5,
        "apparent_temperature": 21.0,
        "relative_humidity_2m": 65,
        "weather_code": 1,
        "wind_speed_10m": 12.0,
        "wind_direction_10m": 180,
    },
    "timezone": "Asia/Tokyo",
}

_FORECAST = {
    "daily": {
        "time": ["2026-02-27", "2026-02-28"],
        "temperature_2m_max": [15.0, 17.0],
        "temperature_2m_min": [5.0, 7.0],
        "weather_code": [0, 3],
        "precipitation_sum": [0.0, 2.5],
        "wind_speed_10m_max": [10.0, 15.0],
    },
}


class TestGetWeather:
    @patch("ai_arch_toolkit.toolkit.tools._weather.urllib.request.urlopen")
    def test_returns_weather(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _mock_urlopen(_GEOCODE_RESPONSE),
            _mock_urlopen(_CURRENT_WEATHER),
        ]
        result = get_weather("Tokyo")
        assert "Tokyo" in result
        assert "22.5" in result
        assert "Mainly clear" in result
        assert "65%" in result

    @patch("ai_arch_toolkit.toolkit.tools._weather.urllib.request.urlopen")
    def test_city_not_found(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({"results": None})
        result = get_weather("Nonexistentville")
        assert "not found" in result.lower()

    @patch("ai_arch_toolkit.toolkit.tools._weather.urllib.request.urlopen")
    def test_api_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = get_weather("Tokyo")
        assert "failed" in result.lower()


class TestGetWeatherByCoords:
    @patch("ai_arch_toolkit.toolkit.tools._weather.urllib.request.urlopen")
    def test_returns_weather(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_CURRENT_WEATHER)
        result = get_weather_by_coords(35.6762, 139.6503)
        assert "35.6762, 139.6503" in result
        assert "22.5" in result
        assert "Mainly clear" in result


class TestWeatherUnits:
    @patch("ai_arch_toolkit.toolkit.tools._weather.urllib.request.urlopen")
    def test_converts_to_fahrenheit(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _mock_urlopen(_GEOCODE_RESPONSE),
            _mock_urlopen(_CURRENT_WEATHER),
        ]
        result = weather_units("Tokyo", unit="f")
        assert "72.5" in result
        assert "69.8" in result
        assert "mph" in result

    @patch("ai_arch_toolkit.toolkit.tools._weather.urllib.request.urlopen")
    def test_invalid_unit(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _mock_urlopen(_GEOCODE_RESPONSE),
            _mock_urlopen(_CURRENT_WEATHER),
        ]
        result = weather_units("Tokyo", unit="k")
        assert "Invalid unit" in result


class TestGetForecast:
    @patch("ai_arch_toolkit.toolkit.tools._weather.urllib.request.urlopen")
    def test_returns_forecast(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _mock_urlopen(_GEOCODE_RESPONSE),
            _mock_urlopen(_FORECAST),
        ]
        result = get_forecast("Tokyo", days=2)
        assert "Tokyo" in result
        assert "2026-02-27" in result
        assert "2026-02-28" in result
        assert "Clear sky" in result
        assert "Overcast" in result

    @patch("ai_arch_toolkit.toolkit.tools._weather.urllib.request.urlopen")
    def test_clamps_days(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _mock_urlopen(_GEOCODE_RESPONSE),
            _mock_urlopen(_FORECAST),
        ]
        # days=99 should be clamped to 7
        result = get_forecast("Tokyo", days=99)
        assert "7-day forecast" in result


class TestGetForecastByCoords:
    @patch("ai_arch_toolkit.toolkit.tools._weather.urllib.request.urlopen")
    def test_returns_forecast(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_FORECAST)
        result = get_forecast_by_coords(35.6762, 139.6503, days=2)
        assert "35.6762, 139.6503" in result
        assert "2026-02-27" in result
        assert "Overcast" in result
