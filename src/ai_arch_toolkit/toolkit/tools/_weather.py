"""Weather tools: the current weather and the daily forecast at a place, from Open-Meteo (free, no
key; https://open-meteo.com/en/docs).

A place is a city, which Open-Meteo's geocoding resolves to its first match (the answer says when
others share the name), or the coordinates of any point. Open-Meteo converts the units itself
(``temperature_unit``, ``wind_speed_unit``, ``precipitation_unit``) and names each one in the
answer (``current_units``, ``daily_units``), so every value is shown with the unit it came in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from ai_arch_toolkit.core import Range, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._open_meteo import (
    FORECAST,
    GEOCODING,
    Place,
    geocoding_params,
    measured,
    places,
)
from ai_arch_toolkit.toolkit.tools._values import plain, utc

type Units = Literal["metric", "imperial"]

# https://open-meteo.com/en/docs: "temperature_unit", "wind_speed_unit", "precipitation_unit".
_UNITS: dict[str, dict[str, str]] = {
    "metric": {},
    "imperial": {
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    },
}
_CURRENT_FIELDS = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,"
    "wind_speed_10m,wind_direction_10m"
)
_DAILY_FIELDS = (
    "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
)
# Up to 16 days of forecast (https://open-meteo.com/en/docs, "forecast_days").
_MAX_DAYS = 16

# The WMO weather interpretation codes Open-Meteo returns (https://open-meteo.com/en/docs,
# "WMO Weather interpretation codes").
_WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@tool(capability="network")
def get_weather(
    city: str = "",
    latitude: float | None = None,
    longitude: float | None = None,
    units: Units = "metric",
) -> str:
    """Get the current weather at a city or a point, from Open-Meteo: conditions, temperature
    and how it feels, humidity, precipitation and wind.

    Args:
        city: A place name, e.g. "Tokyo"; Open-Meteo's first match is used, and the answer says
            when other places share the name.
        latitude: The point's latitude in decimal degrees, with ``longitude``; then ``city`` only
            names the place.
        longitude: The point's longitude in decimal degrees.
        units: "metric" (°C, km/h, mm) or "imperial" (°F, mph, inch).

    Raises:
        ToolFailure: validation_error when neither a city nor both coordinates are given, a
            coordinate is out of range or ``units`` is unknown; not_found when Open-Meteo knows
            no place by that name.
    """
    unit_params = _unit_params(units)
    where = _where(city, latitude, longitude)
    params = {
        "latitude": where.latitude,
        "longitude": where.longitude,
        "current": _CURRENT_FIELDS,
        "timezone": "auto",
        "timeformat": "unixtime",
        **unit_params,
    }
    return FORECAST.get_json(
        "forecast", params=params, parse=lambda data: _current_text(data, where)
    )


@tool(capability="network")
def get_forecast(
    city: str = "",
    latitude: float | None = None,
    longitude: float | None = None,
    days: Annotated[int, Range(1, _MAX_DAYS)] = 3,
    units: Units = "metric",
) -> str:
    """Get the daily weather forecast at a city or a point, from Open-Meteo: conditions, lowest
    and highest temperature, precipitation and strongest wind for each day, from today.

    Args:
        city: A place name, e.g. "Tokyo"; Open-Meteo's first match is used, and the answer says
            when other places share the name.
        latitude: The point's latitude in decimal degrees, with ``longitude``; then ``city`` only
            names the place.
        longitude: The point's longitude in decimal degrees.
        days: How many days to forecast.
        units: "metric" (°C, km/h, mm) or "imperial" (°F, mph, inch).

    Raises:
        ToolFailure: validation_error when neither a city nor both coordinates are given, a
            coordinate is out of range or ``units`` is unknown; not_found when Open-Meteo knows
            no place by that name.
    """
    unit_params = _unit_params(units)
    where = _where(city, latitude, longitude)
    params = {
        "latitude": where.latitude,
        "longitude": where.longitude,
        "daily": _DAILY_FIELDS,
        "timezone": "auto",
        "forecast_days": days,
        **unit_params,
    }
    return FORECAST.get_json(
        "forecast", params=params, parse=lambda data: _forecast_text(data, where)
    )


# --- The place ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class _Where:
    """The point a forecast is for: its coordinates, how to name it, and a note on how it was
    chosen (empty when the caller gave it)."""

    latitude: float
    longitude: float
    name: str
    note: str = ""


def _unit_params(units: str) -> dict[str, str]:
    if units not in _UNITS:
        raise ToolFailure(
            "validation_error", f"invalid units {units!r}; use 'metric' or 'imperial'"
        )
    return _UNITS[units]


def _where(city: str, latitude: float | None, longitude: float | None) -> _Where:
    """The point the caller gave, or the first place Open-Meteo finds for ``city``.

    Raises:
        ToolFailure: validation_error without a city or both coordinates, or for a coordinate out
            of range; not_found when no place has that name.
    """
    name = " ".join(city.split())
    if latitude is not None or longitude is not None:
        if latitude is None or longitude is None:
            raise ToolFailure(
                "validation_error",
                "give both latitude and longitude, or a city instead of them",
            )
        _validate_coords(latitude, longitude)
        position = f"latitude {plain(latitude)}, longitude {plain(longitude)}"
        return _Where(
            latitude=latitude,
            longitude=longitude,
            name=f"{name} ({position})" if name else position,
        )
    if not name:
        raise ToolFailure(
            "validation_error", "give a city, e.g. 'Lisbon', or its latitude and longitude"
        )
    found = GEOCODING.get_json("search", params=geocoding_params(name, 2), parse=places)
    if not found:
        raise ToolFailure(
            "not_found",
            f"Open-Meteo knows no place named {name!r}; check the spelling, find it with "
            "osm_search_place, or give its latitude and longitude",
        )
    return _chosen(found, name)


def _chosen(found: Sequence[Place], name: str) -> _Where:
    first = found[0]
    note = (
        f"{first.label()} is the first of several places named {name!r}; for another, list "
        f"them with geocode({name!r}) and pass its latitude and longitude."
        if len(found) > 1
        else ""
    )
    return _Where(
        latitude=first.latitude,
        longitude=first.longitude,
        name=f"{first.label()} ({first.position()})",
        note=note,
    )


def _validate_coords(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise ToolFailure(
            "validation_error", f"latitude must be between -90 and 90, got {latitude}"
        )
    if not -180 <= longitude <= 180:
        raise ToolFailure(
            "validation_error", f"longitude must be between -180 and 180, got {longitude}"
        )


# --- Answers -----------------------------------------------------------------------------------


def _current_text(data: dict[str, Any], where: _Where) -> str:
    current = data.get("current")
    if not isinstance(current, dict):
        raise ToolFailure(
            "upstream", "Open-Meteo answered without current values; try again later."
        )
    units = data.get("current_units") or {}

    def value(field: str) -> str:
        return measured(current.get(field), units.get(field))

    when = current.get("time")
    at = f", as of {utc(when)}" if isinstance(when, int | float) else ""
    lines = [
        f"Current weather at {where.name}{_zone(data)}{at}:",
        f"  Conditions: {_condition(current.get('weather_code'))}",
        f"  Temperature: {value('temperature_2m')}, feels like {value('apparent_temperature')}",
        f"  Humidity: {value('relative_humidity_2m')}",
        f"  Precipitation: {value('precipitation')}",
        f"  Wind: {value('wind_speed_10m')} from {value('wind_direction_10m')}",
    ]
    return "\n".join([*lines, where.note] if where.note else lines)


def _forecast_text(data: dict[str, Any], where: _Where) -> str:
    daily = data.get("daily")
    if not isinstance(daily, dict) or not isinstance(daily.get("time"), list):
        raise ToolFailure("upstream", "Open-Meteo answered without a forecast; try again later.")
    units = data.get("daily_units") or {}

    def value(field: str, index: int) -> str:
        values = daily.get(field)
        found = values[index] if isinstance(values, list) and index < len(values) else None
        return measured(found, units.get(field))

    lines = [f"Daily forecast at {where.name}{_zone(data)}:"]
    for index, day in enumerate(daily["time"]):
        codes = daily.get("weather_code")
        code = codes[index] if isinstance(codes, list) and index < len(codes) else None
        lines.append(
            f"  {plain(day)}: {_condition(code)}; "
            f"{value('temperature_2m_min', index)} to {value('temperature_2m_max', index)}; "
            f"precipitation {value('precipitation_sum', index)}; "
            f"wind up to {value('wind_speed_10m_max', index)}"
        )
    return "\n".join([*lines, where.note] if where.note else lines)


def _zone(data: dict[str, Any]) -> str:
    """The time zone Open-Meteo resolved for the point, with its UTC offset now."""
    timezone = _string(data.get("timezone"))
    seconds = data.get("utc_offset_seconds")
    if not timezone:
        return ""
    if not isinstance(seconds, int):
        return f", time zone {timezone}"
    sign = "+" if seconds >= 0 else "-"
    hours, minutes = divmod(abs(seconds) // 60, 60)
    return f", time zone {timezone} (UTC{sign}{hours:02d}:{minutes:02d})"


def _condition(code: object) -> str:
    """The WMO code with its label."""
    if not isinstance(code, int):
        return "not reported"
    label = _WMO_CODES.get(code)
    return f"{label} (WMO code {code})" if label else f"WMO code {code}"


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
