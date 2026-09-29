"""Geography tools — geocoding, IP lookup, country info (free, no API key)."""

from __future__ import annotations

import ipaddress
import math
import re
from decimal import Decimal
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
# Country facts come from Wikidata, free and without a key: the search API finds the candidates,
# one SPARQL query reads those that hold an ISO 3166-1 code. REST Countries took v1-v4 down and
# its v5 needs a key: https://restcountries.com/docs/countries/legacy-api-deprecation
_WIKIDATA = Api(base="https://www.wikidata.org/w/api.php", name="Wikidata", timeout_s=15)
_WIKIDATA_SPARQL = Api(
    base="https://query.wikidata.org/sparql", name="Wikidata Query Service", timeout_s=15
)
_COUNTRY_RE = re.compile(r"^[\w .,'\u2019()&-]{1,80}$")
_QID_RE = re.compile(r"^Q\d+$")
# One row per fact and value, so a fact with many values adds rows instead of multiplying them.
# Best-rank statements only; the area is normalised to square metres.
_COUNTRY_FACTS = """
  ?c wdt:P297 [] .
  FILTER NOT EXISTS { ?c wdt:P576 [] }
  { BIND("name" AS ?prop) ?c rdfs:label ?value . FILTER(LANG(?value) = "en") }
  UNION { BIND("official" AS ?prop) ?c wdt:P1448 ?value . FILTER(LANG(?value) = "en") }
  UNION { BIND("iso2" AS ?prop) ?c wdt:P297 ?value }
  UNION { BIND("iso3" AS ?prop) ?c wdt:P298 ?value }
  UNION { BIND("capital" AS ?prop) ?c wdt:P36 ?value }
  UNION { BIND("population" AS ?prop) ?c p:P1082 ?s . ?s a wikibase:BestRank ;
          ps:P1082 ?value . OPTIONAL { ?s pq:P585 ?extra } }
  UNION { BIND("area" AS ?prop) ?c p:P2046 ?a . ?a a wikibase:BestRank ;
          psn:P2046/wikibase:quantityAmount ?value }
  UNION { BIND("continent" AS ?prop) ?c wdt:P30 ?value }
  UNION { BIND("language" AS ?prop) ?c wdt:P37 ?value }
  UNION { BIND("currency" AS ?prop) ?c wdt:P38 ?value . OPTIONAL { ?value wdt:P498 ?extra } }
  UNION { BIND("timezone" AS ?prop) ?c wdt:P421 ?value }
  OPTIONAL { ?value rdfs:label ?label . FILTER(LANG(?label) = "en") }
"""
_UTC_OFFSET_RE = re.compile(r"^UTC([+\u2212-])?(\d{1,2}):(\d{2})")

type _Facts = dict[str, list[tuple[str, str]]]


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

    Uses Wikidata (free, no API key).

    Args:
        name: Country name or ISO 3166-1 code, e.g. "Japan", "France", "BR".
    """
    name = name.strip()
    if not _COUNTRY_RE.fullmatch(name):
        return "Country info failed: invalid name."
    search = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "uselang": "en",
        "type": "item",
        "limit": "10",
        "format": "json",
    }
    try:
        qids = _WIKIDATA.get_json(params=search, parse=_candidate_qids)
        text = (
            _WIKIDATA_SPARQL.get_json(
                params={"query": _country_query(qids), "format": "json"},
                parse=lambda data: _country_text(data, qids),
            )
            if qids
            else ""
        )
    except HttpError as e:
        return f"Country info failed: {e}"
    return text or f"Country not found: {name!r}"


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


def _candidate_qids(data: dict[str, Any]) -> list[str]:
    ids = [item.get("id") for item in data.get("search") or [] if isinstance(item, dict)]
    return [qid for qid in ids if isinstance(qid, str) and _QID_RE.fullmatch(qid)]


def _country_query(qids: list[str]) -> str:
    items = " ".join(f"wd:{qid}" for qid in qids)
    return (
        f"SELECT ?c ?prop ?value ?label ?extra WHERE {{ VALUES ?c {{ {items} }}{_COUNTRY_FACTS}}}"
    )


def _country_text(data: dict[str, Any], qids: list[str]) -> str:
    """The best match in full, then the other countries the search found; ``""`` for none."""
    countries: dict[str, _Facts] = {}
    for row in data["results"]["bindings"]:
        qid = row["c"]["value"].rsplit("/", 1)[-1]
        value = row["value"]
        # An item shows by its English label and a literal by its value; an unlabelled item
        # keeps only its extra (a currency's ISO code), or nothing.
        text = row["label"]["value"] if "label" in row else ""
        if value["type"] != "uri":
            text = value["value"]
        extra = row["extra"]["value"] if "extra" in row else ""
        countries.setdefault(qid, {}).setdefault(row["prop"]["value"], []).append((text, extra))
    found = [qid for qid in qids if qid in countries]
    if not found:
        return ""
    lines = _country_lines(found[0], countries[found[0]])
    others = [
        f"{_first(countries[qid], 'name') or qid} ({_first(countries[qid], 'iso2')})"
        for qid in found[1:]
    ]
    if others:
        lines.append(f"Other matches: {', '.join(others)}")
    return "\n".join(lines)


def _country_lines(qid: str, facts: _Facts) -> list[str]:
    name = _first(facts, "name") or qid
    official = _first(facts, "official")
    codes = ", ".join(code for code in (_first(facts, "iso2"), _first(facts, "iso3")) if code)
    currencies = sorted(
        {
            f"{label} ({code})" if label and code else label or code
            for label, code in facts.get("currency", [])
        }
        - {""}
    )
    zones = _labels(facts, "timezone")
    offsets = [zone for zone in zones if _UTC_OFFSET_RE.match(zone)]
    return [
        f"{name} ({official}):" if official and official != name else f"{name}:",
        f"  ISO 3166-1: {codes or '?'}",
        f"  Capital: {', '.join(_labels(facts, 'capital')) or '?'}",
        f"  Population: {_population(facts)}",
        f"  Area: {_area(facts)}",
        f"  Continent: {', '.join(_labels(facts, 'continent')) or '?'}",
        f"  Official languages: {', '.join(_labels(facts, 'language')) or '?'}",
        f"  Currencies: {', '.join(currencies) or '?'}",
        f"  Timezones: {', '.join(sorted(offsets, key=_utc_minutes) or zones) or '?'}",
        f"  Wikidata: https://www.wikidata.org/wiki/{qid}",
    ]


def _first(facts: _Facts, prop: str) -> str:
    return next((text for text, _ in facts.get(prop, []) if text), "")


def _labels(facts: _Facts, prop: str) -> list[str]:
    return sorted({text for text, _ in facts.get(prop, []) if text})


def _population(facts: _Facts) -> str:
    """The latest best-rank figure, with its year when Wikidata dates it."""
    figures = facts.get("population", [])
    if not figures:
        return "?"
    value, when = max(figures, key=lambda figure: figure[1])
    year = f" ({when[:4]})" if when[:4].isdigit() else ""
    return f"{int(Decimal(value)):,}{year}"


def _area(facts: _Facts) -> str:
    areas = [Decimal(value) for value, _ in facts.get("area", [])]
    return f"{max(areas) / 1_000_000:,.0f} km²" if areas else "?"


def _utc_minutes(label: str) -> int:
    match = _UTC_OFFSET_RE.match(label)
    if not match:
        return 0
    sign, hours, minutes = match.groups()
    return (-1 if sign in {"-", "\u2212"} else 1) * (int(hours) * 60 + int(minutes))


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
