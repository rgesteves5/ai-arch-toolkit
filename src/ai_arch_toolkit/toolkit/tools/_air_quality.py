"""Air quality tools — Open-Meteo air quality forecasts with no API key required."""

from __future__ import annotations

from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply


def _error_reason(reply: Reply) -> ToolFailure | str | None:
    """The error Open-Meteo reports in a JSON body; ``None`` when there is none.

    An invalid parameter gets HTTP 400 and ``{"error": true, "reason": "..."}``
    (https://open-meteo.com/en/docs/air-quality-api, "Errors"): the request's arguments were
    rejected, a ``validation_error``. Any other answer carrying a ``reason`` is the source
    failing, in its words.
    """
    body = reply.body
    reason = _string(body.get("reason")) if isinstance(body, dict) else ""
    if not reason:
        return None
    if reply.status == 400:
        msg = (
            f"Open-Meteo rejected the request: {reason}; check the coordinates, the variables "
            "and the timezone (an IANA name such as 'Europe/Lisbon', or 'auto')."
        )
        return ToolFailure("validation_error", msg)
    return reason


_API = Api(
    base="https://air-quality-api.open-meteo.com/v1/air-quality",
    name="Open-Meteo",
    timeout_s=15,
    error_reader=_error_reason,
)
_MAX_HOURS_LIMIT = 72
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
    """Get current air quality values for coordinates using Open-Meteo.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        variables: Comma-separated current variables. Defaults to common AQI and pollutant values.
        timezone: Timezone name or "auto". Defaults to "auto".

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
    }
    return _API.get_json(params=params, parse=lambda data: _current_text(data, parsed))


@tool(capability="network")
def air_quality_forecast(
    latitude: float,
    longitude: float,
    variables: str = _DEFAULT_VARIABLES,
    forecast_days: int = 3,
    past_days: int = 0,
    timezone: str = "auto",
    max_hours: int = 24,
) -> str:
    """Get an hourly air quality forecast for coordinates using Open-Meteo.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        variables: Comma-separated hourly variables. Defaults to common AQI and pollutant values.
        forecast_days: Forecast days to request (1-7). Defaults to 3.
        past_days: Past forecast days to include (0-7). Defaults to 0.
        timezone: Timezone name or "auto". Defaults to "auto".
        max_hours: Maximum hourly rows to return (1-72). Defaults to 24.

    Raises:
        ToolFailure: validation_error when the coordinates, the variables or the timezone are
            invalid (Open-Meteo rejects the request); upstream when Open-Meteo fails or answers
            without the values.
    """
    _validate_location(latitude, longitude)
    parsed = _parse_variables(variables)

    forecast_days = max(1, min(forecast_days, 7))
    past_days = max(0, min(past_days, 7))
    max_hours = max(1, min(max_hours, _MAX_HOURS_LIMIT))

    params = {
        "latitude": str(latitude),
        "longitude": str(longitude),
        "hourly": ",".join(parsed),
        "forecast_days": str(forecast_days),
        "past_days": str(past_days),
        "timezone": timezone.strip() or "auto",
    }
    return _API.get_json(params=params, parse=lambda data: _forecast_text(data, parsed, max_hours))


def _current_text(data: dict[str, Any], variables: tuple[str, ...]) -> str:
    if not isinstance(data.get("current"), dict):
        raise ToolFailure("upstream", "Open-Meteo answered without current values.")
    return _format_current(data, variables)


def _forecast_text(data: dict[str, Any], variables: tuple[str, ...], max_hours: int) -> str:
    if not isinstance(data.get("hourly"), dict):
        raise ToolFailure("upstream", "Open-Meteo answered without hourly values.")
    return _format_forecast(data, variables, max_hours)


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


def _format_current(data: dict[str, Any], variables: tuple[str, ...]) -> str:
    current = data.get("current", {})
    units = data.get("current_units", {})
    header = _location_header("Open-Meteo air quality current", data)
    lines = [header]
    time = _string(current.get("time"))
    if time:
        lines.append(f"Time: {time}")
    for variable in variables:
        value = current.get(variable)
        unit = _string(units.get(variable))
        lines.append(f"{variable}: {_format_value(value)}{_unit_suffix(unit)}")
    lines.append("Attribution: Open-Meteo Air Quality API / CAMS data providers")
    return "\n".join(lines)


def _format_forecast(data: dict[str, Any], variables: tuple[str, ...], max_hours: int) -> str:
    hourly = data.get("hourly", {})
    units = data.get("hourly_units", {})
    times = hourly.get("time", [])
    if not isinstance(times, list):
        times = []
    lines = [
        _location_header(
            f"Open-Meteo air quality forecast ({min(len(times), max_hours)} hours)", data
        )
    ]
    for index, timestamp in enumerate(times[:max_hours], start=1):
        parts = [str(timestamp)]
        for variable in variables:
            values = hourly.get(variable, [])
            value = (
                values[index - 1] if isinstance(values, list) and index - 1 < len(values) else None
            )
            unit = _string(units.get(variable))
            parts.append(f"{variable}: {_format_value(value)}{_unit_suffix(unit)}")
        lines.append(f"{index}. " + " | ".join(parts))
    lines.append("Attribution: Open-Meteo Air Quality API / CAMS data providers")
    return "\n".join(lines)


def _location_header(label: str, data: dict[str, Any]) -> str:
    lat = _format_value(data.get("latitude"))
    lon = _format_value(data.get("longitude"))
    timezone = _string(data.get("timezone"))
    suffix = f" ({timezone})" if timezone else ""
    return f"{label} for {lat}, {lon}{suffix}:"


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _format_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _unit_suffix(unit: str) -> str:
    return f" {unit}" if unit else ""
