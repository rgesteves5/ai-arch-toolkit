"""Tests for toolkit/tools/_weather.py: one tool for the current weather and one for the forecast,
each at a city or a point (T08b, D41)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools import _weather
from ai_arch_toolkit.toolkit.tools._weather import get_forecast, get_weather
from tests.toolkit import geo_answers
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_CURRENT = {
    "latitude": 39.8,
    "longitude": -89.6,
    "timezone": "America/Chicago",
    "utc_offset_seconds": -18000,
    "current_units": {
        "time": "unixtime",
        "temperature_2m": "°C",
        "apparent_temperature": "°C",
        "relative_humidity_2m": "%",
        "precipitation": "mm",
        "weather_code": "wmo code",
        "wind_speed_10m": "km/h",
        "wind_direction_10m": "°",
    },
    "current": {
        "time": 1781280000,
        "temperature_2m": 22.5,
        "apparent_temperature": 21.0,
        "relative_humidity_2m": 65,
        "precipitation": 0.0,
        "weather_code": 1,
        "wind_speed_10m": 12.0,
        "wind_direction_10m": 180,
    },
}
_FORECAST = {
    "timezone": "America/Chicago",
    "utc_offset_seconds": -18000,
    "daily_units": {
        "time": "iso8601",
        "temperature_2m_max": "°F",
        "temperature_2m_min": "°F",
        "precipitation_sum": "inch",
        "wind_speed_10m_max": "mph",
    },
    "daily": {
        "time": ["2026-06-12", "2026-06-13"],
        "temperature_2m_max": [75.2, 80.1],
        "temperature_2m_min": [60.0, 62.6],
        "weather_code": [0, 63],
        "precipitation_sum": [0.0, 0.25],
        "wind_speed_10m_max": [10.0, 15.3],
    },
}


def _failure(fn, *args, **kwargs) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _query(mock_urlopen: MagicMock, call: int = -1) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args_list[call].args[0].full_url).query)


class TestTheCity:
    @patch(HTTP_OPEN)
    def test_a_city_shared_by_several_places_says_which_one_and_how_to_pick_another(
        self, mock_urlopen
    ):
        # The tools took the first match without a word.
        mock_urlopen.side_effect = [respond(geo_answers.geocoding(2)), respond(_CURRENT)]

        text = get_weather("Springfield")

        assert text.splitlines()[0] == (
            "Current weather at Springfield, State 1, United States (US) (latitude 39.80172, "
            "longitude -89.64371), time zone America/Chicago (UTC-05:00), as of "
            "2026-06-12T16:00:00Z:"
        )
        assert text.splitlines()[-1] == (
            "Springfield, State 1, United States (US) is the first of several places named "
            "'Springfield'; for another, list them with geocode('Springfield') and pass its "
            "latitude and longitude."
        )
        assert _query(mock_urlopen, 0)["count"] == ["2"]
        assert _query(mock_urlopen)["latitude"] == ["39.80172"]

    @patch(HTTP_OPEN)
    def test_a_city_with_one_place_needs_no_note(self, mock_urlopen):
        mock_urlopen.side_effect = [respond(geo_answers.geocoding(1)), respond(_CURRENT)]

        assert "first of several" not in get_weather("Springfield")

    @patch(HTTP_OPEN)
    def test_an_unknown_city_is_not_found_with_the_next_step(self, mock_urlopen):
        mock_urlopen.side_effect = lambda *_: respond(geo_answers.NO_PLACES)

        for tool_fn in (get_weather, get_forecast):
            failure = _failure(tool_fn, "Nowhereville")
            assert failure.error.type == "not_found"
            assert failure.error.message == (
                "Open-Meteo knows no place named 'Nowhereville'; check the spelling, find it "
                "with osm_search_place, or give its latitude and longitude"
            )

    def test_neither_a_city_nor_coordinates_is_refused(self):
        failure = _failure(get_weather)
        assert failure.error.type == "validation_error"
        assert "give a city" in failure.error.message


class TestThePoint:
    @patch(HTTP_OPEN)
    def test_coordinates_need_no_geocoding_and_name_the_point(self, mock_urlopen):
        mock_urlopen.return_value = respond(_CURRENT)

        text = get_weather(latitude=35.6762, longitude=139.6503)

        assert text == (
            "Current weather at latitude 35.6762, longitude 139.6503, time zone America/Chicago "
            "(UTC-05:00), as of 2026-06-12T16:00:00Z:\n"
            "  Conditions: Mainly clear (WMO code 1)\n"
            "  Temperature: 22.5 °C, feels like 21.0 °C\n"
            "  Humidity: 65 %\n"
            "  Precipitation: 0.0 mm\n"
            "  Wind: 12.0 km/h from 180 °"
        )
        assert mock_urlopen.call_count == 1
        query = _query(mock_urlopen)
        assert query["timeformat"] == ["unixtime"]
        assert query["current"][0].split(",")[0] == "temperature_2m"

    @patch(HTTP_OPEN)
    def test_with_coordinates_a_city_only_names_the_place(self, mock_urlopen):
        mock_urlopen.return_value = respond(_CURRENT)

        text = get_weather("Home", latitude=38.7, longitude=-9.1)

        assert text.startswith("Current weather at Home (latitude 38.7, longitude -9.1),")
        assert mock_urlopen.call_count == 1

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"latitude": 91.0, "longitude": 0.0}, "latitude must be between -90 and 90"),
            ({"latitude": 0.0, "longitude": 181.0}, "longitude must be between -180 and 180"),
            ({"latitude": 10.0}, "give both latitude and longitude"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_bad_coordinates_are_refused_before_any_request(self, mock_urlopen, kwargs, words):
        # get_weather_by_coords sent them unchecked.
        for tool_fn in (get_weather, get_forecast):
            failure = _failure(tool_fn, **kwargs)
            assert failure.error.type == "validation_error"
            assert words in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_open_meteos_reason_for_a_refusal_reaches_the_agent(self, mock_urlopen):
        # get_weather_by_coords lost it: Open-Meteo's 400 read as a bare HTTP error.
        body = b'{"error": true, "reason": "Latitude must be in range of -90 to 90. Given: 95."}'
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)

        failure = _failure(get_weather, latitude=10.0, longitude=10.0)

        assert failure.error.type == "validation_error"
        assert failure.error.message == (
            "Open-Meteo rejected the request: Latitude must be in range of -90 to 90. Given: 95; "
            "correct the argument it names and call again"
        )


class TestUnitsAndForecast:
    @patch(HTTP_OPEN)
    def test_imperial_units_are_asked_of_open_meteo_and_shown_as_it_names_them(self, mock_urlopen):
        mock_urlopen.return_value = respond(_FORECAST)

        text = get_forecast(latitude=39.8, longitude=-89.6, days=2, units="imperial")

        assert text.splitlines() == [
            "Daily forecast at latitude 39.8, longitude -89.6, time zone America/Chicago "
            "(UTC-05:00):",
            "  2026-06-12: Clear sky (WMO code 0); 60.0 °F to 75.2 °F; precipitation 0.0 inch; "
            "wind up to 10.0 mph",
            "  2026-06-13: Moderate rain (WMO code 63); 62.6 °F to 80.1 °F; precipitation "
            "0.25 inch; wind up to 15.3 mph",
        ]
        query = _query(mock_urlopen)
        assert query["forecast_days"] == ["2"]
        assert query["temperature_unit"] == ["fahrenheit"]
        assert query["wind_speed_unit"] == ["mph"]
        assert query["precipitation_unit"] == ["inch"]

    def test_unknown_units_are_refused(self):
        failure = _failure(get_weather, "Tokyo", units="kelvin")
        assert failure.error.type == "validation_error"
        assert "'metric' or 'imperial'" in failure.error.message

    def test_the_forecast_days_are_bounded_in_the_schema_as_open_meteo_allows(self):
        days = get_forecast.__tool_definition__.schema.input_schema["properties"]["days"]
        assert (days["minimum"], days["maximum"]) == (1, 16)

    def test_more_days_than_open_meteo_forecasts_are_refused_by_the_executor(self):
        # They were cut to 7 without a word.
        result = ToolGroup(get_forecast).execute(
            ToolCall(id="c", name="get_forecast", input={"city": "Tokyo", "days": 30})
        )
        assert result.error is not None
        assert result.error.type == "validation_error"

    @patch(HTTP_OPEN)
    def test_an_answer_without_values_is_upstream(self, mock_urlopen):
        mock_urlopen.side_effect = lambda *_: respond({"timezone": "UTC"})

        assert _failure(get_weather, latitude=0.0, longitude=0.0).error.type == "upstream"
        assert _failure(get_forecast, latitude=0.0, longitude=0.0).error.type == "upstream"

    @patch(HTTP_OPEN)
    def test_an_unlisted_weather_code_shows_the_code(self, mock_urlopen):
        mock_urlopen.return_value = respond({**_CURRENT, "current": {"weather_code": 42}})

        text = get_weather(latitude=0.0, longitude=0.0)

        assert "Conditions: WMO code 42" in text
        assert "Temperature: not reported, feels like not reported" in text


def test_the_three_current_weather_tools_are_one():
    # D41: get_weather_by_coords and weather_units were the same job with other arguments.
    assert not hasattr(_weather, "get_weather_by_coords")
    assert not hasattr(_weather, "weather_units")
    assert not hasattr(_weather, "get_forecast_by_coords")
