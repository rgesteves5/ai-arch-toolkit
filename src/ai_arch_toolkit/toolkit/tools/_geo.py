"""Geography tools: places by name, time zones, distances, IP addresses and countries (free, no
key). Reverse geocoding is ``osm_reverse_geocode``'s (``_osm``)."""

from __future__ import annotations

import ipaddress
import math
import re
from decimal import Decimal
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._first_results import asked, first_results_window
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._mediawiki import mediawiki_error
from ai_arch_toolkit.toolkit.tools._open_meteo import (
    FORECAST,
    GEOCODING,
    GEOCODING_DEPTH,
    Place,
    geocoding_params,
    places,
)
from ai_arch_toolkit.toolkit.tools._values import plain


def _ipwhois_error(reply: Reply) -> ToolFailure | str | None:
    """The error an ipwho.is answer reports; ``None`` for a result.

    ipwho.is says ``"success": false`` and why, in ``message``: with HTTP 200 for an address it
    cannot locate ("Invalid IP address", "Reserved range"), and with a 4xx for the rest ("Rate
    limit exceeded" with a 429) (https://ipwhois.io/documentation, "Errors"). An address it
    cannot locate is the caller's to change, a ``validation_error``.
    """
    body = reply.body
    if not isinstance(body, dict) or body.get("success") is not False:
        return None
    message = _string(body.get("message")) or "no reason given"
    if reply.status == 200 and message in _UNLOCATABLE:
        return ToolFailure(
            "validation_error",
            f"ipwho.is cannot locate this address ({message}): a private, reserved or malformed "
            "address has no public location; give a public IPv4 or IPv6 address",
        )
    return message


_UNLOCATABLE = frozenset({"Invalid IP address", "Reserved range"})
# The free endpoint is HTTPS, 1000 requests a day per client IP: https://ipwhois.io/documentation
_IPWHOIS = Api(
    base="https://ipwho.is", name="ipwho.is", segment_safe=":", error_reader=_ipwhois_error
)
# Country facts come from Wikidata, free and without a key: the search API finds the candidates,
# one SPARQL query reads those that hold an ISO 3166-1 code. REST Countries took v1-v4 down and
# its v5 needs a key: https://restcountries.com/docs/countries/legacy-api-deprecation
_WIKIDATA = Api(
    base="https://www.wikidata.org/w/api.php",
    name="Wikidata",
    timeout_s=15,
    error_reader=mediawiki_error,
)
_WIKIDATA_SPARQL = Api(
    base="https://query.wikidata.org/sparql", name="Wikidata Query Service", timeout_s=15
)
_NAME_CHARS = 200
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
def geocode(
    city: str,
    max_results: Annotated[int, Range(1, GEOCODING_DEPTH)] = 10,
    offset: Annotated[int, Range(0, GEOCODING_DEPTH - 1)] = 0,
) -> ToolResult:
    """Find places by name with Open-Meteo's geocoding, in its order: their coordinates, region,
    country, time zone and population.

    Args:
        city: The place's name, e.g. "Tokyo", "London" or "São Paulo".
        max_results: How many places to list.
        offset: How many places to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when ``city`` is empty or too long.
    """
    name = " ".join(city.split())
    if not name or len(name) > _NAME_CHARS:
        raise ToolFailure(
            "validation_error",
            f"invalid city {city[:100]!r}; give a place name of 1 to {_NAME_CHARS} characters, "
            "e.g. 'Lisbon'",
        )
    params = geocoding_params(name, asked(offset, max_results, GEOCODING_DEPTH))
    return GEOCODING.get_json(
        "search",
        params=params,
        parse=lambda data: _geocode_answer(places(data), name, offset, max_results),
    )


@tool(capability="network")
def timezone_lookup(lat: float, lon: float) -> str:
    """Look up the time zone of a point and its UTC offset now, from Open-Meteo.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Raises:
        ToolFailure: validation_error when the coordinates are out of range.
    """
    _validate_coords(lat, lon)
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m",
        "forecast_days": "1",
        "timezone": "auto",
    }
    return FORECAST.get_json(
        "forecast", params=params, parse=lambda data: _timezone_text(data, lat, lon)
    )


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

    Raises:
        ToolFailure: validation_error when a coordinate is out of range or ``unit`` is unknown.
    """
    _validate_coords(lat1, lon1, "start ")
    _validate_coords(lat2, lon2, "end ")

    unit = unit.lower().strip()
    if unit not in {"km", "mi"}:
        raise ToolFailure("validation_error", f"invalid unit {unit!r}; use 'km' or 'mi'.")

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
    """Look up the approximate location, time zone and network of a public IP address, from
    ipwho.is (1000 requests a day per client IP).

    Args:
        ip: An IPv4 or IPv6 address.

    Raises:
        ToolFailure: validation_error when ``ip`` is not an IP address, or ipwho.is cannot locate
            it (a private or reserved range).
    """
    try:
        target = str(ipaddress.ip_address(ip))
    except ValueError as e:
        msg = f"invalid IP address {ip!r}; pass an IPv4 or IPv6 address, e.g. '8.8.8.8'."
        raise ToolFailure("validation_error", msg) from e
    return _IPWHOIS.get_json(target, parse=lambda data: _ip_text(data, target))


@tool(capability="network")
def country_info(name: str) -> str:
    """Get information about a country (capital, population, languages, etc.).

    Uses Wikidata (free, no API key).

    Args:
        name: Country name or ISO 3166-1 code, e.g. "Japan", "France", "BR".

    Raises:
        ToolFailure: validation_error when ``name`` is empty or has unsupported characters;
            not_found when Wikidata has no country by that name or code.
    """
    name = name.strip()
    if not _COUNTRY_RE.fullmatch(name):
        msg = (
            f"invalid name {name!r}; pass a country name or ISO 3166-1 code of up to 80 "
            "letters, e.g. 'Japan' or 'BR'."
        )
        raise ToolFailure("validation_error", msg)
    search = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "uselang": "en",
        "type": "item",
        "limit": "10",
        "format": "json",
    }
    qids = _WIKIDATA.get_json(params=search, parse=_candidate_qids)
    text = (
        _WIKIDATA_SPARQL.get_json(
            params={"query": _country_query(qids), "format": "json"},
            parse=lambda data: _country_text(data, qids),
        )
        if qids
        else ""
    )
    if not text:
        msg = (
            f"no country named {name!r} on Wikidata; try its English name or its ISO 3166-1 "
            "code, e.g. 'JP'."
        )
        raise ToolFailure("not_found", msg)
    return text


def _geocode_answer(found: list[Place], name: str, offset: int, limit: int) -> ToolResult:
    if not found:
        return ToolResult.success(
            f"No places named {name!r} in Open-Meteo's geocoding; check the spelling, or search "
            "OpenStreetMap with osm_search_place."
        )
    lines = [f"{number}. {_place_line(place)}" for number, place in enumerate(found, start=1)]
    window = first_results_window(
        lines,
        offset=offset,
        limit=limit,
        depth=GEOCODING_DEPTH,
        narrow="add the region or country to the name, or search with osm_search_place",
    )
    return window.result(heading=f"Places named {name!r} (Open-Meteo geocoding):")


def _place_line(place: Place) -> str:
    parts = [place.label(), place.position()]
    if place.timezone:
        parts.append(f"time zone {place.timezone}")
    if place.population is not None:
        parts.append(f"population {place.population}")
    if place.elevation is not None:
        parts.append(f"elevation {plain(place.elevation)} m")
    return " | ".join(parts)


def _timezone_text(data: dict[str, Any], lat: float, lon: float) -> str:
    timezone = _string(data.get("timezone"))
    if not timezone:
        raise ToolFailure("upstream", "Open-Meteo answered without a time zone; try again later.")
    abbreviation = _string(data.get("timezone_abbreviation"))
    named = f"{timezone} ({abbreviation})" if abbreviation else timezone
    offset = _format_utc_offset(data.get("utc_offset_seconds"))
    return (
        f"Coordinates: latitude {plain(lat)}, longitude {plain(lon)}\n"
        f"Time zone: {named}\n"
        f"UTC offset now: {offset}"
    )


def _ip_text(data: dict[str, Any], target: str) -> str:
    if data.get("success") is not True:
        raise ToolFailure(
            "upstream", f"ipwho.is answered without a result for {target}; try again later."
        )
    connection = data.get("connection") or {}
    timezone = data.get("timezone") or {}
    code = _string(data.get("country_code"))
    country = _string(data.get("country")) or "?"
    zone = _string(timezone.get("id")) or "?"
    utc_offset = _string(timezone.get("utc"))
    return (
        f"IP: {_string(data.get('ip')) or target}\n"
        f"Location: {_string(data.get('city')) or '?'}, {_string(data.get('region')) or '?'}, "
        f"{country}{f' ({code})' if code else ''}\n"
        f"Coordinates (approximate): latitude {plain(data.get('latitude')) or '?'}, "
        f"longitude {plain(data.get('longitude')) or '?'}\n"
        f"Time zone: {zone}{f' (UTC{utc_offset})' if utc_offset else ''}\n"
        f"ISP: {_string(connection.get('isp')) or '?'}\n"
        f"Organization: {_string(connection.get('org')) or '?'}"
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


def _validate_coords(lat: float, lon: float, which: str = "") -> None:
    """Validate a latitude/longitude pair; ``which`` names it ("start ") in the message."""
    if not -90 <= lat <= 90:
        msg = f"{which}latitude out of range: it must be between -90 and 90, got {lat}."
        raise ToolFailure("validation_error", msg)
    if not -180 <= lon <= 180:
        msg = f"{which}longitude out of range: it must be between -180 and 180, got {lon}."
        raise ToolFailure("validation_error", msg)


def _format_utc_offset(offset_seconds: int | None) -> str:
    """Format a UTC offset in seconds as UTC+HH:MM."""
    if offset_seconds is None:
        return "?"
    sign = "+" if offset_seconds >= 0 else "-"
    total_minutes = abs(offset_seconds) // 60
    hours, minutes = divmod(total_minutes, 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
