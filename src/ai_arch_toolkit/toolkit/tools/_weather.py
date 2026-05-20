"""Weather tools — real weather data via Open-Meteo (no API key required)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from ai_arch_toolkit.core import tool

_TIMEOUT = 10

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
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
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


def _geocode(city: str) -> tuple[float, float, str] | str:
    """Geocode a city name → (lat, lon, display_name) or error string."""
    url = (
        f"https://geocoding-api.open-meteo.com/v1/search"
        f"?name={urllib.parse.quote(city)}&count=1&language=en&format=json"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Geocoding failed: {e}"

    results = data.get("results")
    if not results:
        return f"City not found: {city!r}"

    r = results[0]
    name = r.get("name", city)
    country = r.get("country", "")
    display = f"{name}, {country}" if country else name
    return r["latitude"], r["longitude"], display


def _fetch_weather(lat: float, lon: float) -> dict | str:
    """Fetch current weather JSON for a coordinate pair."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
        f"weather_code,wind_speed_10m,wind_direction_10m"
        f"&timezone=auto"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Weather API failed: {e}"


def _fetch_forecast(lat: float, lon: float, days: int) -> dict | str:
    """Fetch daily forecast JSON for a coordinate pair."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,weather_code,"
        f"precipitation_sum,wind_speed_10m_max"
        f"&timezone=auto&forecast_days={days}"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Forecast API failed: {e}"


def _format_current_weather(data: dict, display: str, unit: str = "c") -> str:
    """Format the current-weather response into a human-readable string."""
    current = data.get("current", {})
    temp = current.get("temperature_2m", "?")
    feels = current.get("apparent_temperature", "?")
    humidity = current.get("relative_humidity_2m", "?")
    wind = current.get("wind_speed_10m", "?")
    wind_dir = current.get("wind_direction_10m", "?")
    code = current.get("weather_code", -1)
    condition = _WMO_CODES.get(code, "Unknown")
    tz = data.get("timezone", "")

    unit = unit.lower().strip()
    if unit not in {"c", "f"}:
        return f"Invalid unit: {unit!r}. Use 'c' or 'f'."

    temp_unit = "°C"
    wind_unit = "km/h"
    if unit == "f":
        temp = _c_to_f(temp)
        feels = _c_to_f(feels)
        wind = _kmh_to_mph(wind)
        temp_unit = "°F"
        wind_unit = "mph"

    return (
        f"{display} ({tz}):\n"
        f"  Temperature: {_format_number(temp)}{temp_unit} "
        f"(feels like {_format_number(feels)}{temp_unit})\n"
        f"  Conditions: {condition}\n"
        f"  Humidity: {humidity}%\n"
        f"  Wind: {_format_number(wind)} {wind_unit} (direction: {wind_dir}°)"
    )


def _format_forecast(data: dict, display: str, days: int) -> str:
    """Format the forecast response into a human-readable string."""
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    highs = daily.get("temperature_2m_max", [])
    lows = daily.get("temperature_2m_min", [])
    codes = daily.get("weather_code", [])
    precip = daily.get("precipitation_sum", [])
    wind = daily.get("wind_speed_10m_max", [])

    lines = [f"{display} — {days}-day forecast:"]
    for i, date in enumerate(dates):
        condition = _WMO_CODES.get(codes[i] if i < len(codes) else -1, "Unknown")
        hi = highs[i] if i < len(highs) else "?"
        lo = lows[i] if i < len(lows) else "?"
        rain = precip[i] if i < len(precip) else 0
        w = wind[i] if i < len(wind) else "?"
        lines.append(f"  {date}: {lo}°C - {hi}°C, {condition}, precip: {rain}mm, wind: {w} km/h")

    return "\n".join(lines)


def _c_to_f(value: float | str) -> float | str:
    """Convert Celsius to Fahrenheit if the value is numeric."""
    if isinstance(value, (int, float)):
        return value * 9 / 5 + 32
    return value


def _kmh_to_mph(value: float | str) -> float | str:
    """Convert km/h to mph if the value is numeric."""
    if isinstance(value, (int, float)):
        return value * 0.621371
    return value


def _format_number(value: float | str) -> str:
    """Format numeric values without unnecessary trailing zeroes."""
    if isinstance(value, (int, float)):
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return str(value)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city using Open-Meteo (free, no API key).

    Returns temperature, conditions, humidity, wind speed, and "feels like" temperature.

    Args:
        city: City name, e.g. "Tokyo", "London", "New York".
    """
    geo = _geocode(city)
    if isinstance(geo, str):
        return geo
    lat, lon, display = geo
    data = _fetch_weather(lat, lon)
    if isinstance(data, str):
        return data
    return _format_current_weather(data, display)


@tool
def get_forecast(city: str, days: int = 3) -> str:
    """Get a multi-day weather forecast for a city using Open-Meteo (free, no API key).

    Args:
        city: City name, e.g. "Tokyo", "London", "New York".
        days: Number of forecast days (1-7). Defaults to 3.
    """
    days = max(1, min(days, 7))
    geo = _geocode(city)
    if isinstance(geo, str):
        return geo
    lat, lon, display = geo
    data = _fetch_forecast(lat, lon, days)
    if isinstance(data, str):
        return data
    return _format_forecast(data, display, days)


@tool
def get_weather_by_coords(lat: float, lon: float) -> str:
    """Get the current weather for a latitude/longitude pair using Open-Meteo.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
    """
    data = _fetch_weather(lat, lon)
    if isinstance(data, str):
        return data
    return _format_current_weather(data, f"{lat}, {lon}")


@tool
def get_forecast_by_coords(lat: float, lon: float, days: int = 3) -> str:
    """Get a multi-day weather forecast for a latitude/longitude pair.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        days: Number of forecast days (1-7). Defaults to 3.
    """
    days = max(1, min(days, 7))
    data = _fetch_forecast(lat, lon, days)
    if isinstance(data, str):
        return data
    return _format_forecast(data, f"{lat}, {lon}", days)


@tool
def weather_units(city: str, unit: str = "c") -> str:
    """Get current weather for a city with converted output units.

    Args:
        city: City name, e.g. "Tokyo", "London", "New York".
        unit: Output unit: "c" or "f". Defaults to Celsius.
    """
    geo = _geocode(city)
    if isinstance(geo, str):
        return geo
    lat, lon, display = geo
    data = _fetch_weather(lat, lon)
    if isinstance(data, str):
        return data
    return _format_current_weather(data, display, unit=unit)
