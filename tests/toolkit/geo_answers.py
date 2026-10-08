"""Answers of the geo, weather and natural-event sources, shaped as their documentation shows them,
for the tools' tests and the contract cases (T08b).

- Open-Meteo: https://open-meteo.com/en/docs, /en/docs/geocoding-api, /en/docs/air-quality-api
- Nominatim: https://nominatim.org/release-docs/latest/api/Search/ and /Output/
- Overpass: https://dev.overpass-api.de/overpass-doc/en/preface/commons.html
- NASA EONET: https://eonet.gsfc.nasa.gov/docs/v3
- USGS: https://earthquake.usgs.gov/fdsnws/event/1/ and the FDSN error text,
  https://www.fdsn.org/webservices/FDSN-WS-Specification-Commonalities-1.2.pdf
"""

from __future__ import annotations

from typing import Any

# --- Open-Meteo ----------------------------------------------------------------------------------

# The documented error examples (the forecast and air-quality pages, and the geocoding page).
OPEN_METEO_REASON = "Cannot initialize WeatherVariable from invalid String value tempeture_2m"
GEOCODING_REASON = "Parameter count must be between 1 and 100."
# An answer with no place: empty fields are left out, ``results`` among them.
NO_PLACES: dict[str, Any] = {"generationtime_ms": 0.6}


def geocoding(count: int, name: str = "Springfield") -> dict[str, Any]:
    """A geocoding answer with ``count`` places named ``name``, most populated first."""
    return {
        "results": [
            {
                "id": 4250542 + index,
                "name": name,
                "latitude": 39.80172 + index,
                "longitude": -89.64371 - index,
                "elevation": 182.0,
                "feature_code": "PPLA",
                "country_code": "US",
                "country": "United States",
                "admin1": f"State {index + 1}",
                "timezone": "America/Chicago",
                "population": 116250 - index,
            }
            for index in range(count)
        ],
        "generationtime_ms": 0.9,
    }


def air_quality_hours(count: int, *, start: int = 1781280000) -> dict[str, Any]:
    """An hourly air-quality answer of ``count`` hours from ``start`` (Unix time, UTC)."""
    return {
        "latitude": 38.75,
        "longitude": -9.15,
        "timezone": "Europe/Lisbon",
        "utc_offset_seconds": 3600,
        "hourly_units": {"time": "unixtime", "pm10": "μg/m³", "european_aqi": "EAQI"},
        "hourly": {
            "time": [start + 3600 * hour for hour in range(count)],
            "pm10": [10.5 + hour for hour in range(count)],
            "european_aqi": [20 + hour for hour in range(count)],
        },
    }


# --- Nominatim -----------------------------------------------------------------------------------

NOMINATIM_LAYER_MESSAGE = (
    "Parameter 'layer' must be a comma-separated list of: address, poi, railway, natural, manmade"
)


def nominatim_place(index: int = 0, name: str = "Springfield") -> dict[str, Any]:
    """A jsonv2 place with its address details."""
    return {
        "place_id": 300000 + index,
        "licence": "Data © OpenStreetMap contributors, ODbL 1.0. http://osm.org/copyright",
        "osm_type": "relation",
        "osm_id": 126 + index,
        "lat": f"{39.7990175 + index}",
        "lon": f"{-89.6439575 - index}",
        "category": "boundary",
        "type": "administrative",
        "place_rank": 16,
        "importance": 0.61,
        "addresstype": "city",
        "name": name,
        "display_name": f"{name}, Sangamon County, Illinois, United States {index + 1}",
        "address": {
            "city": name,
            "county": "Sangamon County",
            "state": "Illinois",
            "ISO3166-2-lvl4": "US-IL",
            "country": "United States",
            "country_code": "us",
        },
        "boundingbox": ["39.6", "39.9", "-89.8", "-89.5"],
    }


def nominatim_places(count: int) -> list[dict[str, Any]]:
    return [nominatim_place(index) for index in range(count)]


# --- Overpass ------------------------------------------------------------------------------------

OVERPASS_TIMEOUT = 'runtime error: Query timed out in "query" at line 1 after 26 seconds.'


def overpass_page(*errors: str) -> bytes:
    """An Overpass error page, as the server sends it, with one paragraph per error."""
    paragraphs = "".join(
        f'<p><strong style="color:#FF0000">Error</strong>: {error} </p>\n' for error in errors
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html>\n<html><head>'
        "<title>OSM3S Response</title></head>\n<body>\n"
        "<p>The data included in this document is from www.openstreetmap.org.</p>\n"
        f"{paragraphs}</body>\n</html>\n"
    ).encode()


def overpass_elements(count: int) -> dict[str, Any]:
    """An Overpass JSON answer with ``count`` cafés: nodes, then a way with its center."""
    nodes = [
        {
            "type": "node",
            "id": 1000 + index,
            "lat": 38.7 + index / 1000,
            "lon": -9.1,
            "tags": {"name": f"Café {index + 1}", "amenity": "cafe"},
        }
        for index in range(max(count - 1, 0))
    ]
    way = {
        "type": "way",
        "id": 77,
        "center": {"lat": 38.71, "lon": -9.14},
        "tags": {"name": "Café Way", "amenity": "cafe", "opening_hours": "Mo-Fr 08:00-18:00"},
    }
    return {"version": 0.6, "generator": "Overpass API", "elements": [*nodes, way][:count]}


# --- NASA EONET ----------------------------------------------------------------------------------

EONET_ERROR_PAGE = b"<!DOCTYPE html><title>Server Error</title>"


def eonet_event(points: int = 2, *, number: int = 1) -> dict[str, Any]:
    """An event with a track of ``points`` dated positions (GeoJSON: longitude first)."""
    return {
        "id": f"EONET_{number}",
        "title": f"Tropical Storm {number}",
        "description": "A storm over the Atlantic.",
        "link": f"https://eonet.gsfc.nasa.gov/api/v3/events/EONET_{number}",
        "closed": None,
        "categories": [{"id": "severeStorms", "title": "Severe Storms"}],
        "sources": [
            {"id": "JTWC", "url": "https://www.metoc.navy.mil/jtwc/products/al012026.tcw"},
            {"id": "NOAA_NHC", "url": "https://www.nhc.noaa.gov/"},
        ],
        "geometry": [
            {
                "magnitudeValue": 35.0 + point,
                "magnitudeUnit": "kts",
                "date": f"2026-09-{point + 1:02d}T00:00:00Z",
                "type": "Point",
                "coordinates": [-40.0 - point, 15.0 + point],
            }
            for point in range(points)
        ],
    }


def eonet_events(count: int) -> dict[str, Any]:
    return {
        "title": "EONET Events",
        "link": "https://eonet.gsfc.nasa.gov/api/v3/events",
        "events": [eonet_event(number=number) for number in range(1, count + 1)],
    }


# --- USGS ----------------------------------------------------------------------------------------

USGS_BAD_START = 'Bad starttime value "2024-13-01".'


def fdsn_error(code: int, description: str, detail: str) -> bytes:
    """An FDSN error text (Commonalities 1.2, "Errors messages")."""
    return (
        f"Error {code}: {description}\n\n{detail}\n\n"
        "Usage details are available from https://earthquake.usgs.gov/fdsnws/event/1\n\n"
        "Request:\n/fdsnws/event/1/query?starttime=2024-13-01\n\n"
        "Request Submitted:\n2026-10-08T12:00:00+00:00\n\n"
        "Service version:\n1.15.2\n"
    ).encode()


def usgs_feature(number: int = 1, *, time_ms: int = 1710000000000) -> dict[str, Any]:
    """A GeoJSON event: ``time`` in milliseconds since the epoch, depth in km."""
    return {
        "type": "Feature",
        "id": f"us{number}",
        "properties": {
            "mag": 5.0 + number / 10,
            "place": "Portugal",
            "time": time_ms,
            "updated": time_ms + 60_000,
            "url": f"https://earthquake.usgs.gov/earthquakes/eventpage/us{number}",
            "felt": 12,
            "alert": "green",
            "status": "reviewed",
            "tsunami": 0,
            "sig": 400,
            "magType": "mww",
            "type": "earthquake",
            "title": f"M {5.0 + number / 10} - Portugal",
        },
        "geometry": {"type": "Point", "coordinates": [-9.1, 38.7, 10.0]},
    }


def usgs_features(*numbers: int) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "metadata": {"status": 200, "count": len(numbers)},
        "features": [usgs_feature(number) for number in numbers],
    }
