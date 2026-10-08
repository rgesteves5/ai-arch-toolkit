"""Air quality tools: Open-Meteo's air quality, now and hour by hour, for any point (free, no key;
https://open-meteo.com/en/docs/air-quality-api, CAMS data).

Times are asked as Unix time (``timeformat=unixtime``, always UTC) and shown in ISO 8601 UTC: a
local time would hide the clock changes inside a forecast. The point's time zone, which sets where
each day starts, is named in the heading.
"""

from __future__ import annotations

from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._open_meteo import AIR_QUALITY, measured
from ai_arch_toolkit.toolkit.tools._values import plain, utc
from ai_arch_toolkit.toolkit.tools._window import page_window

# Up to 7 days of forecast and 92 past days (https://open-meteo.com/en/docs/air-quality-api).
_MAX_FORECAST_DAYS = 7
_MAX_PAST_DAYS = 92
_MAX_HOURS = 72
_ATTRIBUTION = "Open-Meteo Air Quality API, CAMS data"
_DEFAULT_VARIABLES = "european_aqi,us_aqi,pm10,pm2_5,ozone,nitrogen_dioxide"
_VALID_VARIABLES = {
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "carbon_dioxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "aerosol_optical_depth",
    "dust",
    "uv_index",
    "uv_index_clear_sky",
    "ammonia",
    "methane",
    "alder_pollen",
    "birch_pollen",
    "grass_pollen",
    "mugwort_pollen",
    "olive_pollen",
    "ragweed_pollen",
    "european_aqi",
    "european_aqi_pm2_5",
    "european_aqi_pm10",
    "european_aqi_nitrogen_dioxide",
    "european_aqi_ozone",
    "european_aqi_sulphur_dioxide",
    "us_aqi",
    "us_aqi_pm2_5",
    "us_aqi_pm10",
    "us_aqi_nitrogen_dioxide",
    "us_aqi_ozone",
    "us_aqi_sulphur_dioxide",
    "us_aqi_carbon_monoxide",
    "formaldehyde",
    "glyoxal",
    "non_methane_volatile_organic_compounds",
    "pm10_wildfires",
    "peroxyacyl_nitrates",
    "secondary_inorganic_aerosol",
    "residential_elementary_carbon",
    "total_elementary_carbon",
    "pm2_5_total_organic_matter",
    "sea_salt_aerosol",
    "nitrogen_monoxide",
}


@tool(capability="network")
def air_quality_current(
    latitude: float,
    longitude: float,
    variables: str = _DEFAULT_VARIABLES,
    timezone: str = "auto",
) -> str:
    """Get the current air quality at a point, from Open-Meteo: AQI and pollutant values, each
    with its unit.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        variables: Comma-separated variables, e.g. "european_aqi,pm2_5,ozone".
        timezone: An IANA time zone, e.g. "Europe/Lisbon", or "auto" for the point's own.

    Raises:
        ToolFailure: validation_error when the coordinates, the variables or the timezone are
            invalid (Open-Meteo rejects the request); upstream when Open-Meteo fails or answers
            without the values.
    """
    _validate_location(latitude, longitude)
    parsed = _parse_variables(variables)
    params = {
        "latitude": str(latitude),
        "longitude": str(longitude),
        "current": ",".join(parsed),
        "timezone": timezone.strip() or "auto",
        "timeformat": "unixtime",
    }
    return AIR_QUALITY.get_json(params=params, parse=lambda data: _current_text(data, parsed))


@tool(capability="network")
def air_quality_forecast(
    latitude: float,
    longitude: float,
    variables: str = _DEFAULT_VARIABLES,
    forecast_days: Annotated[int, Range(1, _MAX_FORECAST_DAYS)] = 3,
    past_days: Annotated[int, Range(0, _MAX_PAST_DAYS)] = 0,
    timezone: str = "auto",
    max_hours: Annotated[int, Range(1, _MAX_HOURS)] = 24,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Get the hourly air quality at a point, from Open-Meteo: the past days asked for, then
    the forecast, one hour per line.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        variables: Comma-separated variables, e.g. "european_aqi,pm2_5,ozone".
        forecast_days: Days of forecast, from today.
        past_days: Past days before today, to include.
        timezone: An IANA time zone, e.g. "Europe/Lisbon", or "auto" for the point's own; it
            sets where each day starts.
        max_hours: How many hours to show.
        offset: How many hours to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when the coordinates, the variables or the timezone are
            invalid (Open-Meteo rejects the request); upstream when Open-Meteo fails or answers
            without the values.
    """
    _validate_location(latitude, longitude)
    parsed = _parse_variables(variables)
    params = {
        "latitude": str(latitude),
        "longitude": str(longitude),
        "hourly": ",".join(parsed),
        "forecast_days": str(forecast_days),
        "past_days": str(past_days),
        "timezone": timezone.strip() or "auto",
        "timeformat": "unixtime",
    }
    return AIR_QUALITY.get_json(
        params=params, parse=lambda data: _forecast_answer(data, parsed, offset, max_hours)
    )


def _current_text(data: dict[str, Any], variables: tuple[str, ...]) -> str:
    current = data.get("current")
    if not isinstance(current, dict):
        raise ToolFailure(
            "upstream", "Open-Meteo answered without current values; try again later."
        )
    units = data.get("current_units") or {}
    lines = [f"Current air quality at {_point(data)} ({_ATTRIBUTION}):"]
    when = current.get("time")
    if isinstance(when, int | float):
        lines.append(f"Time: {utc(when)}")
    lines += [f"{name}: {measured(current.get(name), units.get(name))}" for name in variables]
    return "\n".join(lines)


def _forecast_answer(
    data: dict[str, Any], variables: tuple[str, ...], offset: int, limit: int
) -> ToolResult:
    hourly = data.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise ToolFailure(
            "upstream", "Open-Meteo answered without hourly values; try again later."
        )
    units = data.get("hourly_units") or {}
    rows = [
        " | ".join([utc(when), *(_cell(hourly, units, name, index) for name in variables)])
        for index, when in enumerate(hourly["time"])
    ]
    heading = f"Hourly air quality at {_point(data)}, times in UTC ({_ATTRIBUTION}):"
    if not rows:
        return ToolResult.success(f"{heading}\nOpen-Meteo returned no hours for that period.")
    return page_window(rows, offset=offset, limit=limit).result(heading=heading)


def _cell(hourly: dict[str, Any], units: dict[str, Any], name: str, index: int) -> str:
    values = hourly.get(name)
    value = values[index] if isinstance(values, list) and index < len(values) else None
    return f"{name}: {measured(value, units.get(name))}"


def _parse_variables(value: str) -> tuple[str, ...]:
    variables = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not variables:
        msg = (
            f"variables cannot be empty; pass comma-separated names, e.g. {_DEFAULT_VARIABLES!r}."
        )
        raise ToolFailure("validation_error", msg)
    invalid = [variable for variable in variables if variable not in _VALID_VARIABLES]
    if invalid:
        msg = (
            f"invalid variables: {', '.join(invalid)}; "
            f"valid ones include {_DEFAULT_VARIABLES!r}, dust and uv_index."
        )
        raise ToolFailure("validation_error", msg)
    return variables


def _validate_location(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        msg = f"latitude must be between -90 and 90, got {latitude}."
        raise ToolFailure("validation_error", msg)
    if not -180 <= longitude <= 180:
        msg = f"longitude must be between -180 and 180, got {longitude}."
        raise ToolFailure("validation_error", msg)


def _point(data: dict[str, Any]) -> str:
    """The grid point Open-Meteo answered for, and its time zone."""
    where = f"latitude {plain(data.get('latitude')) or '?'}, "
    where += f"longitude {plain(data.get('longitude')) or '?'}"
    timezone = _string(data.get("timezone"))
    return f"{where} (time zone {timezone})" if timezone else where


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
