"""Geography tools — geocoding, IP lookup, country info (free, no API key)."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

from ai_arch_toolkit.core import tool

_TIMEOUT = 10
_USER_AGENT = "ai-arch-toolkit/1.0"


@tool
def geocode(city: str) -> str:
    """Get the coordinates and country for a city using Open-Meteo geocoding.

    Args:
        city: City name, e.g. "Tokyo", "London", "São Paulo".
    """
    url = (
        f"https://geocoding-api.open-meteo.com/v1/search"
        f"?name={urllib.request.quote(city)}&count=3&language=en&format=json"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Geocoding failed: {e}"

    results = data.get("results")
    if not results:
        return f"No results for: {city!r}"

    lines: list[str] = []
    for r in results:
        name = r.get("name", "")
        country = r.get("country", "")
        admin = r.get("admin1", "")
        lat = r.get("latitude", "?")
        lon = r.get("longitude", "?")
        pop = r.get("population")
        tz = r.get("timezone", "")
        loc = f"{name}, {admin}, {country}" if admin else f"{name}, {country}"
        line = f"  {loc}: {lat}°N, {lon}°E"
        if pop:
            line += f", pop: {pop:,}"
        if tz:
            line += f", tz: {tz}"
        lines.append(line)

    return f"Geocoding results for {city!r}:\n" + "\n".join(lines)


@tool
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

    url = (
        "https://nominatim.openstreetmap.org/reverse"
        f"?format=jsonv2&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Reverse geocoding failed: {e}"

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


@tool
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

    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m&forecast_days=1&timezone=auto"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Timezone lookup failed: {e}"

    timezone = data.get("timezone")
    if not timezone:
        return f"No timezone found for coordinates: {lat}, {lon}"

    offset = _format_utc_offset(data.get("utc_offset_seconds"))
    return (
        f"Coordinates: {lat}, {lon}\n"
        f"Timezone: {timezone}\n"
        f"UTC offset: {offset}"
    )


@tool
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


@tool
def ip_lookup(ip: str = "") -> str:
    """Look up geographic location and ISP info for an IP address.

    Uses ip-api.com (free, no API key, max 45 requests/minute).

    Args:
        ip: IP address to look up. Leave empty for your own public IP.
    """
    target = ip or ""
    url = f"http://ip-api.com/json/{target}?fields=status,message,query,country,regionName,city,zip,lat,lon,timezone,isp,org,as"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"IP lookup failed: {e}"

    if data.get("status") != "success":
        return f"IP lookup failed: {data.get('message', 'unknown error')}"

    return (
        f"IP: {data.get('query', '?')}\n"
        f"Location: {data.get('city', '?')}, {data.get('regionName', '?')}, "
        f"{data.get('country', '?')}\n"
        f"Coordinates: {data.get('lat', '?')}°N, {data.get('lon', '?')}°E\n"
        f"Timezone: {data.get('timezone', '?')}\n"
        f"ISP: {data.get('isp', '?')}\n"
        f"Organization: {data.get('org', '?')}"
    )


@tool
def country_info(name: str) -> str:
    """Get information about a country (capital, population, languages, etc.).

    Uses restcountries.com (free, no API key).

    Args:
        name: Country name, e.g. "Japan", "France", "Brazil".
    """
    url = (
        f"https://restcountries.com/v3.1/name/{urllib.request.quote(name)}"
        f"?fields=name,capital,population,area,region,subregion,languages,"
        f"currencies,timezones,flags,borders"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"Country not found: {name!r}"
        return f"API error: {e.code}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Country info failed: {e}"

    if not isinstance(data, list) or not data:
        return f"No data for: {name!r}"

    c = data[0]
    official = c.get("name", {}).get("official", name)
    common = c.get("name", {}).get("common", name)
    capitals = c.get("capital", [])
    pop = c.get("population", 0)
    area = c.get("area", 0)
    region = c.get("region", "?")
    subregion = c.get("subregion", "")
    languages = c.get("languages", {})
    currencies = c.get("currencies", {})
    timezones = c.get("timezones", [])

    lang_str = ", ".join(languages.values()) if languages else "?"
    curr_list = []
    for code, info in currencies.items():
        symbol = info.get("symbol", "")
        curr_name = info.get("name", code)
        curr_list.append(f"{curr_name} ({code}{', ' + symbol if symbol else ''})")
    curr_str = ", ".join(curr_list) if curr_list else "?"
    tz_str = ", ".join(timezones[:5]) if timezones else "?"

    return (
        f"{common} ({official}):\n"
        f"  Capital: {', '.join(capitals) if capitals else '?'}\n"
        f"  Population: {pop:,}\n"
        f"  Area: {area:,.0f} km²\n"
        f"  Region: {region}" + (f" / {subregion}" if subregion else "") + "\n"
        f"  Languages: {lang_str}\n"
        f"  Currencies: {curr_str}\n"
        f"  Timezones: {tz_str}"
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
