"""WHO Global Health Observatory tools — public health indicators."""

from __future__ import annotations

import re
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_API = Api(
    base="https://ghoapi.azureedge.net/api",
    name="WHO GHO",
    timeout_s=20,
    query_safe="'() ,",
    status_messages={404: "no matching records found."},
)
_MAX_LIMIT = 100
_CODE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_YEAR_RE = re.compile(r"^\d{4}$")


@tool(capability="network")
def who_indicators(query: str = "", max_results: int = 25, skip: int = 0) -> str:
    """Search WHO Global Health Observatory indicators.

    Args:
        query: Optional text filter across indicator code and name.
        max_results: Number of indicators to return (1-100). Defaults to 25.
        skip: Number of matching indicators to skip. Defaults to 0.
    """
    if query and not _valid_text(query):
        return "WHO GHO indicators failed: invalid query."
    if skip < 0:
        return "WHO GHO indicators failed: skip must be greater than or equal to 0."
    params = {"$top": str(_bounded(max_results)), "$skip": str(skip)}
    if query.strip():
        escaped = query.strip().replace("'", "''")
        params["$filter"] = (
            f"contains(tolower(IndicatorName),'{escaped.lower()}') "
            f"or contains(tolower(IndicatorCode),'{escaped.lower()}')"
        )
    try:
        return _API.get_json(
            "Indicator", params=params, parse=lambda data: _indicators_text(data, skip)
        )
    except HttpError as e:
        return f"WHO GHO indicators failed: {e}"


@tool(capability="network")
def who_indicator(indicator_code: str) -> str:
    """Get WHO GHO indicator metadata by code.

    Args:
        indicator_code: WHO GHO indicator code, e.g. "WHOSIS_000001".
    """
    code = indicator_code.strip()
    if not _CODE_RE.fullmatch(code):
        return f"WHO GHO indicator failed: invalid indicator_code: {indicator_code!r}"
    escaped_code = code.replace("'", "''")
    params = {"$filter": f"IndicatorCode eq '{escaped_code}'", "$top": "1"}
    try:
        return _API.get_json(
            "Indicator", params=params, parse=lambda data: _indicator_text(data, code)
        )
    except HttpError as e:
        return f"WHO GHO indicator failed: {e}"


@tool(capability="network")
def who_series(
    indicator_code: str,
    country: str = "",
    from_year: str = "",
    to_year: str = "",
    dim1: str = "",
    max_results: int = 25,
    skip: int = 0,
) -> str:
    """Fetch WHO GHO observations for an indicator.

    Args:
        indicator_code: WHO GHO indicator code, e.g. "WHOSIS_000001".
        country: Optional ISO3 country code filter, e.g. "PRT".
        from_year: Optional lower year bound as YYYY.
        to_year: Optional upper year bound as YYYY.
        dim1: Optional first-dimension code filter, e.g. sex or age code.
        max_results: Number of observations to return (1-100). Defaults to 25.
        skip: Number of observations to skip. Defaults to 0.
    """
    code = indicator_code.strip()
    if not _CODE_RE.fullmatch(code):
        return f"WHO GHO series failed: invalid indicator_code: {indicator_code!r}"
    problem = _series_problem(country, from_year, to_year, dim1, skip)
    if problem:
        return f"WHO GHO series failed: {problem}"

    params = {"$top": str(_bounded(max_results)), "$skip": str(skip)}
    series_filter = _series_filter(country, from_year, to_year, dim1)
    if series_filter:
        params["$filter"] = series_filter
    try:
        return _API.get_json(
            code, params=params, parse=lambda data: _series_text(data, code, skip)
        )
    except HttpError as e:
        return f"WHO GHO series failed: {e}"


def _series_problem(country: str, from_year: str, to_year: str, dim1: str, skip: int) -> str:
    if country and not re.fullmatch(r"^[A-Za-z]{3}$", country.strip()):
        return "invalid country. Use ISO3."
    if from_year and not _YEAR_RE.fullmatch(from_year.strip()):
        return "invalid from_year."
    if to_year and not _YEAR_RE.fullmatch(to_year.strip()):
        return "invalid to_year."
    if from_year and to_year and int(from_year) > int(to_year):
        return "from_year must be before or equal to to_year."
    if dim1 and not _CODE_RE.fullmatch(dim1.strip()):
        return "invalid dim1."
    if skip < 0:
        return "skip must be greater than or equal to 0."
    return ""


def _series_filter(country: str, from_year: str, to_year: str, dim1: str) -> str:
    filters = []
    if country.strip():
        filters.append(f"SpatialDim eq '{country.strip().upper()}'")
    if from_year.strip():
        filters.append(f"TimeDim ge {int(from_year)}")
    if to_year.strip():
        filters.append(f"TimeDim le {int(to_year)}")
    if dim1.strip():
        filters.append(f"Dim1 eq '{dim1.strip()}'")
    return " and ".join(filters)


def _indicators_text(data: dict[str, Any], skip: int) -> str:
    items = _values(data)
    if not items:
        return "No WHO GHO indicators found."
    lines = [f"WHO GHO indicators (returned {len(items)}, skip {skip}):"]
    for index, item in enumerate(items, start=1):
        lines.append(
            f"{index}. {_string(item.get('IndicatorCode'))} — {_string(item.get('IndicatorName'))}"
        )
    return "\n".join(lines)


def _indicator_text(data: dict[str, Any], code: str) -> str:
    items = _values(data)
    if not items:
        return f"WHO GHO indicator not found: {code}"
    item = items[0]
    return "\n".join(
        [
            f"WHO GHO indicator {code}:",
            _string(item.get("IndicatorName")) or "(no name)",
            f"   language: {_string(item.get('Language')) or '?'}",
        ]
    )


def _series_text(data: dict[str, Any], code: str, skip: int) -> str:
    items = _values(data)
    if not items:
        return f"No WHO GHO observations found for {code}."
    lines = [f"WHO GHO series {code} (returned {len(items)}, skip {skip}):"]
    for index, item in enumerate(items, start=1):
        country_name = _string(item.get("SpatialDim"))
        year = _string(item.get("TimeDim"))
        value = _string(item.get("Value")) or _string(item.get("NumericValue"))
        dims = []
        for key in ("ParentLocation", "Dim1", "Dim2", "Dim3"):
            if _string(item.get(key)):
                dims.append(f"{key}: {_string(item.get(key))}")
        suffix = f" | {'; '.join(dims)}" if dims else ""
        lines.append(f"{index}. {country_name} {year}: {value}{suffix}")
    return "\n".join(lines)


def _values(data: dict[str, Any]) -> list[dict[str, Any]]:
    value = data.get("value")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
