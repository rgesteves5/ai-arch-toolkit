"""Geography tools — geocoding, IP lookup, country info (free, no API key)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ai_arch_toolkit.core import tool

_TIMEOUT = 10


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
