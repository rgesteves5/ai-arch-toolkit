"""Geography tools — geocoding, IP lookup, country info (free, no API key)."""

from __future__ import annotations

import ipaddress
import math
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_GEOCODING = Api(base="https://geocoding-api.open-meteo.com/v1", name="Open-Meteo", query_safe=",")
_FORECAST = Api(base="https://api.open-meteo.com/v1", name="Open-Meteo", query_safe=",")
# One request per second, a clock the osm_* tools share:
# https://operations.osmfoundation.org/policies/nominatim/
_NOMINATIM = Api(base="https://nominatim.openstreetmap.org", name="Nominatim", min_interval_s=1.1)
# The free endpoint is HTTPS and allows commercial use: https://ipwhois.io/documentation
_IPWHOIS = Api(base="https://ipwho.is", name="ipwho.is", segment_safe=":")
_COUNTRIES = Api(base="https://restcountries.com/v3.1", name="REST Countries", query_safe=",")
_COUNTRY_FIELDS = (
    "name,capital,population,area,region,subregion,languages,currencies,timezones,flags,borders"
)


@tool(capability="network")
def geocode(city: str) -> str:
    """Get the coordinates and country for a city using Open-Meteo geocoding.

    Args:
        city: City name, e.g. "Tokyo", "London", "São Paulo".
    """
    params = {"name": city, "count": "3", "language": "en", "format": "json"}
    try:
        lines = _GEOCODING.get_json("search", params=params, parse=_geocode_lines)
    except HttpError as e:
        return f"Geocoding failed: {e}"
    if not lines:
        return f"No results for: {city!r}"
    return f"Geocoding results for {city!r}:\n" + "\n".join(lines)


@tool(capability="network")
def reverse_geocode(lat: float, lon: float) -> str:
    """Look up a place name from latitude and longitude.

    Uses OpenStreetMap Nominatim reverse geocoding (free, no API key).

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
    """
    error = _validate_coords(lat, lon)
    if error:
        return error
    params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": "10", "addressdetails": "1"}
    try:
        return _NOMINATIM.get_json(
            "reverse", params=params, parse=lambda data: _place_text(data, lat, lon)
        )
    except HttpError as e:
        return f"Reverse geocoding failed: {e}"


@tool(capability="network")
def timezone_lookup(lat: float, lon: float) -> str:
    """Look up the timezone for a coordinate pair.

    Uses Open-Meteo forecast metadata (free, no API key).

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
    """
    error = _validate_coords(lat, lon)
    if error:
        return error
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m",
        "forecast_days": "1",
        "timezone": "auto",
    }
    try:
        return _FORECAST.get_json(
            "forecast", params=params, parse=lambda data: _timezone_text(data, lat, lon)
        )
    except HttpError as e:
        return f"Timezone lookup failed: {e}"


@tool(capability="compute")
def distance_between(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    unit: str = "km",
) -> str:
    """Calculate the great-circle distance between two coordinate pairs.

    Args:
        lat1: Starting latitude in decimal degrees.
        lon1: Starting longitude in decimal degrees.
        lat2: Ending latitude in decimal degrees.
        lon2: Ending longitude in decimal degrees.
        unit: Output unit: "km" or "mi". Defaults to kilometers.
    """
    start_error = _validate_coords(lat1, lon1)
    if start_error:
        return start_error.replace("Coordinates", "Start coordinates")
    end_error = _validate_coords(lat2, lon2)
    if end_error:
        return end_error.replace("Coordinates", "End coordinates")

    unit = unit.lower().strip()
    if unit not in {"km", "mi"}:
        return f"Invalid unit: {unit!r}. Use 'km' or 'mi'."

    radius = 6371.0088 if unit == "km" else 3958.7613
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return f"{lat1}, {lon1} → {lat2}, {lon2} = {distance:.2f} {unit}"


@tool(capability="network")
def ip_lookup(ip: str = "") -> str:
    """Look up geographic location and ISP info for an IP address.

    Uses ipwho.is (free, no API key, 1000 requests/day per client IP).

    Args:
        ip: Explicit IPv4 or IPv6 address to look up.
    """
    try:
        target = str(ipaddress.ip_address(ip))
    except ValueError:
        return "IP lookup failed: provide a valid IPv4 or IPv6 address"
    try:
        return _IPWHOIS.get_json(target, parse=_ip_text)
    except HttpError as e:
        return f"IP lookup failed: {e}"


@tool(capability="network")
def country_info(name: str) -> str:
    """Get information about a country (capital, population, languages, etc.).

    Uses restcountries.com (free, no API key).

    Args:
        name: Country name, e.g. "Japan", "France", "Brazil".
    """
    try:
        return _COUNTRIES.get_json_list(
            "name",
            name,
            params={"fields": _COUNTRY_FIELDS},
            parse=lambda data: _country_text(data, name),
        )
    except HttpError as e:
        if e.status == 404:
            return f"Country not found: {name!r}"
        return f"Country info failed: {e}"


def _geocode_lines(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for r in data.get("results") or []:
        name = r.get("name", "")
        country = r.get("country", "")
        admin = r.get("admin1", "")
        loc = f"{name}, {admin}, {country}" if admin else f"{name}, {country}"
        line = f"  {loc}: {r.get('latitude', '?')}°N, {r.get('longitude', '?')}°E"
        if r.get("population"):
            line += f", pop: {r['population']:,}"
        if r.get("timezone"):
            line += f", tz: {r['timezone']}"
        lines.append(line)
    return lines


def _place_text(data: dict[str, Any], lat: float, lon: float) -> str:
    display = data.get("display_name")
    if not display:
        return f"No reverse geocoding result for coordinates: {lat}, {lon}"
    address = data.get("address", {})
    country = address.get("country", "?")
    state = (
        address.get("state")
        or address.get("region")
        or address.get("county")
        or address.get("state_district")
        or "?"
    )
    city = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("hamlet")
        or "?"
    )
    return (
        f"Coordinates: {lat}, {lon}\n"
        f"Location: {display}\n"
        f"City: {city}\n"
        f"Region: {state}\n"
        f"Country: {country}"
    )


def _timezone_text(data: dict[str, Any], lat: float, lon: float) -> str:
    timezone = data.get("timezone")
    if not timezone:
        return f"No timezone found for coordinates: {lat}, {lon}"
    offset = _format_utc_offset(data.get("utc_offset_seconds"))
    return f"Coordinates: {lat}, {lon}\nTimezone: {timezone}\nUTC offset: {offset}"


def _ip_text(data: dict[str, Any]) -> str:
    if data.get("success") is not True:
        return f"IP lookup failed: {data.get('message', 'unknown error')}"
    connection = data.get("connection") or {}
    timezone = data.get("timezone") or {}
    return (
        f"IP: {data.get('ip', '?')}\n"
        f"Location: {data.get('city', '?')}, {data.get('region', '?')}, "
        f"{data.get('country', '?')}\n"
        f"Coordinates: {data.get('latitude', '?')}°N, {data.get('longitude', '?')}°E\n"
        f"Timezone: {timezone.get('id', '?')}\n"
        f"ISP: {connection.get('isp', '?')}\n"
        f"Organization: {connection.get('org', '?')}"
    )


def _country_text(data: list[Any], name: str) -> str:
    if not data:
        return f"No data for: {name!r}"
    c = data[0]
    official = c.get("name", {}).get("official", name)
    common = c.get("name", {}).get("common", name)
    capitals = c.get("capital", [])
    languages = c.get("languages", {})
    timezones = c.get("timezones", [])
    lang_str = ", ".join(languages.values()) if languages else "?"
    currencies = [
        f"{info.get('name', code)} ({code}{', ' + info['symbol'] if info.get('symbol') else ''})"
        for code, info in c.get("currencies", {}).items()
    ]
    subregion = c.get("subregion", "")
    return (
        f"{common} ({official}):\n"
        f"  Capital: {', '.join(capitals) if capitals else '?'}\n"
        f"  Population: {c.get('population', 0):,}\n"
        f"  Area: {c.get('area', 0):,.0f} km²\n"
        f"  Region: {c.get('region', '?')}" + (f" / {subregion}" if subregion else "") + "\n"
        f"  Languages: {lang_str}\n"
        f"  Currencies: {', '.join(currencies) if currencies else '?'}\n"
        f"  Timezones: {', '.join(timezones[:5]) if timezones else '?'}"
    )


def _validate_coords(lat: float, lon: float) -> str | None:
    """Validate a latitude/longitude pair."""
    if not -90 <= lat <= 90:
        return f"Coordinates out of range: latitude must be between -90 and 90, got {lat}."
    if not -180 <= lon <= 180:
        return f"Coordinates out of range: longitude must be between -180 and 180, got {lon}."
    return None


def _format_utc_offset(offset_seconds: int | None) -> str:
    """Format a UTC offset in seconds as UTC+HH:MM."""
    if offset_seconds is None:
        return "?"
    sign = "+" if offset_seconds >= 0 else "-"
    total_minutes = abs(offset_seconds) // 60
    hours, minutes = divmod(total_minutes, 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"
