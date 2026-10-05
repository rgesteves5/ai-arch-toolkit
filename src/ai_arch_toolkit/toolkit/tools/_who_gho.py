"""WHO Global Health Observatory tools — public health indicators."""

from __future__ import annotations

import re
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api

_API = Api(
    base="https://ghoapi.azureedge.net/api",
    name="WHO GHO",
    timeout_s=20,
    query_safe="'() ,",
)
_MAX_LIMIT = 100
_CODE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_YEAR_RE = re.compile(r"^\d{4}$")
_TEXT_HINT = "use 1-180 letters, digits, spaces and ,.'()/%:+-"


@tool(capability="network")
def who_indicators(query: str = "", max_results: int = 25, skip: int = 0) -> str:
    """Search WHO Global Health Observatory indicators.

    Args:
        query: Optional text filter across indicator code and name.
        max_results: Number of indicators to return (1-100). Defaults to 25.
        skip: Number of matching indicators to skip. Defaults to 0.

    Raises:
        ToolFailure: validation_error when the query or ``skip`` is invalid.
    """
    if query and not _valid_text(query):
        raise ToolFailure("validation_error", f"invalid query {query!r}; {_TEXT_HINT}")
    _check_skip(skip)
    params = {"$top": str(_bounded(max_results)), "$skip": str(skip)}
    if query.strip():
        escaped = query.strip().replace("'", "''")
        params["$filter"] = (
            f"contains(tolower(IndicatorName),'{escaped.lower()}') "
            f"or contains(tolower(IndicatorCode),'{escaped.lower()}')"
        )
    return _API.get_json(
        "Indicator", params=params, parse=lambda data: _indicators_text(data, skip)
    )


@tool(capability="network")
def who_indicator(indicator_code: str) -> str:
    """Get WHO GHO indicator metadata by code.

    Args:
        indicator_code: WHO GHO indicator code, e.g. "WHOSIS_000001".

    Raises:
        ToolFailure: validation_error when the code is malformed; not_found when WHO GHO has no
            indicator with that code.
    """
    code = _indicator_code(indicator_code)
    escaped_code = code.replace("'", "''")
    params = {"$filter": f"IndicatorCode eq '{escaped_code}'", "$top": "1"}
    return _API.get_json(
        "Indicator", params=params, parse=lambda data: _indicator_text(data, code)
    )


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

    Raises:
        ToolFailure: validation_error when the code, the country, a year, ``dim1`` or ``skip`` is
            invalid; not_found when WHO GHO has no indicator with that code.
    """
    code = _indicator_code(indicator_code)
    _check_series(country, from_year, to_year, dim1, skip)

    params = {"$top": str(_bounded(max_results)), "$skip": str(skip)}
    series_filter = _series_filter(country, from_year, to_year, dim1)
    if series_filter:
        params["$filter"] = series_filter
    # Each indicator is an OData entity set of its own, /api/{code}: a 404 is an unknown code.
    return _API.get_json(
        code,
        params=params,
        parse=lambda data: _series_text(data, code, skip),
        missing=_no_indicator(code),
    )


def _indicator_code(indicator_code: str) -> str:
    """The stripped code; a malformed one raises ``ToolFailure`` (validation_error)."""
    code = indicator_code.strip()
    if not _CODE_RE.fullmatch(code):
        raise ToolFailure(
            "validation_error",
            f"invalid indicator_code {indicator_code!r}; a WHO GHO code looks like "
            "WHOSIS_000001 (find one with who_indicators)",
        )
    return code


def _no_indicator(code: str) -> str:
    return f"WHO GHO has no indicator {code}; search for one with who_indicators"


def _check_skip(skip: int) -> None:
    if skip < 0:
        raise ToolFailure("validation_error", f"invalid skip {skip}; use 0 or more")


def _check_series(country: str, from_year: str, to_year: str, dim1: str, skip: int) -> None:
    """Raises ``ToolFailure`` (validation_error) for the first invalid series filter."""
    problem = ""
    if country and not re.fullmatch(r"^[A-Za-z]{3}$", country.strip()):
        problem = f"invalid country {country!r}; use an ISO3 code such as PRT"
    elif from_year and not _YEAR_RE.fullmatch(from_year.strip()):
        problem = f"invalid from_year {from_year!r}; use YYYY"
    elif to_year and not _YEAR_RE.fullmatch(to_year.strip()):
        problem = f"invalid to_year {to_year!r}; use YYYY"
    elif from_year and to_year and int(from_year) > int(to_year):
        problem = f"from_year {from_year} is after to_year {to_year}; swap them"
    elif dim1 and not _CODE_RE.fullmatch(dim1.strip()):
        problem = f"invalid dim1 {dim1!r}; use a dimension code such as SEX_MLE"
    if problem:
        raise ToolFailure("validation_error", problem)
    _check_skip(skip)


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
        raise ToolFailure("not_found", _no_indicator(code))
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
