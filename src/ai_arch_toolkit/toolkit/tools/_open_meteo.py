"""The Open-Meteo APIs that the weather, geography and air-quality tools share: one error reader,
the geocoding search and the place it finds (free, no key; https://open-meteo.com/en/docs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._values import plain


def open_meteo_error(reply: Reply) -> ToolFailure | str | None:
    """The error an Open-Meteo answer reports; ``None`` for a result.

    Every Open-Meteo API refuses a request with HTTP 400 and ``{"error": true, "reason": "..."}``
    (https://open-meteo.com/en/docs, https://open-meteo.com/en/docs/geocoding-api and
    https://open-meteo.com/en/docs/air-quality-api, "Errors"): an argument the API does not take,
    a ``validation_error`` in its words. Any other answer that carries a reason is the source
    failing, in its words, typed by its status.
    """
    body = reply.body
    reason = _string(body.get("reason")) if isinstance(body, dict) else ""
    if not reason:
        return None
    if reply.status == 400:
        return ToolFailure(
            "validation_error",
            f"Open-Meteo rejected the request: {reason.rstrip('.')}; correct the argument it "
            "names and call again",
        )
    return reason


GEOCODING = Api(
    base="https://geocoding-api.open-meteo.com/v1",
    name="Open-Meteo geocoding",
    timeout_s=15,
    error_reader=open_meteo_error,
)
FORECAST = Api(
    base="https://api.open-meteo.com/v1",
    name="Open-Meteo",
    timeout_s=15,
    query_safe=",",
    error_reader=open_meteo_error,
)
AIR_QUALITY = Api(
    base="https://air-quality-api.open-meteo.com/v1/air-quality",
    name="Open-Meteo air quality",
    timeout_s=15,
    query_safe=",",
    error_reader=open_meteo_error,
)
# The geocoding search returns up to 100 places for a name, and has no offset
# (https://open-meteo.com/en/docs/geocoding-api, "count").
GEOCODING_DEPTH = 100


@dataclass(frozen=True, slots=True, kw_only=True)
class Place:
    """A place the geocoding search found."""

    name: str
    region: str
    country: str
    country_code: str
    latitude: float
    longitude: float
    timezone: str
    population: int | None
    elevation: float | None

    def label(self) -> str:
        """The place's name, region and country, e.g. "Lisbon, Lisbon, Portugal (PT)"."""
        country = f"{self.country} ({self.country_code})" if self.country_code else self.country
        parts = [self.name, self.region if self.region != self.name else "", country]
        return ", ".join(part for part in parts if part) or "(unnamed)"

    def position(self) -> str:
        """The place's coordinates, labelled."""
        return f"latitude {plain(self.latitude)}, longitude {plain(self.longitude)}"


def geocoding_params(name: str, count: int) -> dict[str, str]:
    """The query of a geocoding search for ``name``: its first ``count`` places, in English."""
    return {"name": name, "count": str(count), "language": "en", "format": "json"}


def places(data: dict[str, Any]) -> list[Place]:
    """The places of a geocoding answer, in its order. An answer without places has no
    ``results`` (empty fields are left out of the answer)."""
    return [_place(item) for item in data.get("results") or []]


def measured(value: object, unit: object) -> str:
    """A value with the unit the answer names for it (``current_units``, ``hourly_units``,
    ``daily_units``); Open-Meteo sends ``null`` for a value it does not have."""
    if value is None:
        return "not reported"
    unit_text = _string(unit)
    return f"{plain(value)} {unit_text}" if unit_text else plain(value)


def _place(item: dict[str, Any]) -> Place:
    population = item.get("population")
    elevation = item.get("elevation")
    return Place(
        name=_string(item.get("name")),
        region=_string(item.get("admin1")),
        country=_string(item.get("country")),
        country_code=_string(item.get("country_code")),
        latitude=float(item["latitude"]),
        longitude=float(item["longitude"]),
        timezone=_string(item.get("timezone")),
        population=population if isinstance(population, int) else None,
        elevation=float(elevation) if isinstance(elevation, int | float) else None,
    )


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
