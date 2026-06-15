"""WHO Global Health Observatory tools — public health indicators."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://ghoapi.azureedge.net/api"
_TIMEOUT = 20
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_LIMIT = 100
_CODE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_YEAR_RE = re.compile(r"^\d{4}$")


@tool
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
        data = _fetch_json("/Indicator", params)
        items = _values(data)
    except urllib.error.HTTPError as e:
        return _http_error("WHO GHO indicators failed", e)
    except urllib.error.URLError as e:
        return f"WHO GHO indicators failed: URL error: {e.reason}"
    except TimeoutError:
        return "WHO GHO indicators failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"WHO GHO indicators failed: could not parse API response: {e}"

    if not items:
        return "No WHO GHO indicators found."
    lines = [f"WHO GHO indicators (returned {len(items)}, skip {skip}):"]
    for index, item in enumerate(items, start=1):
        lines.append(
            f"{index}. {_string(item.get('IndicatorCode'))} — {_string(item.get('IndicatorName'))}"
        )
    return "\n".join(lines)


@tool
def who_indicator(indicator_code: str) -> str:
    """Get WHO GHO indicator metadata by code.

    Args:
        indicator_code: WHO GHO indicator code, e.g. "WHOSIS_000001".
    """
    code = indicator_code.strip()
    if not _CODE_RE.fullmatch(code):
        return f"WHO GHO indicator failed: invalid indicator_code: {indicator_code!r}"
    escaped_code = code.replace("'", "''")
    try:
        data = _fetch_json(
            "/Indicator",
            {"$filter": f"IndicatorCode eq '{escaped_code}'", "$top": "1"},
        )
        items = _values(data)
    except urllib.error.HTTPError as e:
        return _http_error("WHO GHO indicator failed", e)
    except urllib.error.URLError as e:
        return f"WHO GHO indicator failed: URL error: {e.reason}"
    except TimeoutError:
        return "WHO GHO indicator failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"WHO GHO indicator failed: could not parse API response: {e}"

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


@tool
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
    if country and not re.fullmatch(r"^[A-Za-z]{3}$", country.strip()):
        return "WHO GHO series failed: invalid country. Use ISO3."
    if from_year and not _YEAR_RE.fullmatch(from_year.strip()):
        return "WHO GHO series failed: invalid from_year."
    if to_year and not _YEAR_RE.fullmatch(to_year.strip()):
        return "WHO GHO series failed: invalid to_year."
    if from_year and to_year and int(from_year) > int(to_year):
        return "WHO GHO series failed: from_year must be before or equal to to_year."
    if dim1 and not _CODE_RE.fullmatch(dim1.strip()):
        return "WHO GHO series failed: invalid dim1."
    if skip < 0:
        return "WHO GHO series failed: skip must be greater than or equal to 0."

    filters = []
    if country.strip():
        filters.append(f"SpatialDim eq '{country.strip().upper()}'")
    if from_year.strip():
        filters.append(f"TimeDim ge {int(from_year)}")
    if to_year.strip():
        filters.append(f"TimeDim le {int(to_year)}")
    if dim1.strip():
        filters.append(f"Dim1 eq '{dim1.strip()}'")
    params = {"$top": str(_bounded(max_results)), "$skip": str(skip)}
    if filters:
        params["$filter"] = " and ".join(filters)
    try:
        data = _fetch_json(f"/{urllib.parse.quote(code)}", params)
        items = _values(data)
    except urllib.error.HTTPError as e:
        return _http_error("WHO GHO series failed", e)
    except urllib.error.URLError as e:
        return f"WHO GHO series failed: URL error: {e.reason}"
    except TimeoutError:
        return "WHO GHO series failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"WHO GHO series failed: could not parse API response: {e}"

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


def _fetch_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{_BASE_URL}{path}"
    if params:
        query = urllib.parse.urlencode(params, safe="'() ,")
        url = f"{url}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _values(data: dict[str, Any]) -> list[dict[str, Any]]:
    value = data.get("value")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 404:
        return f"{prefix}: no matching records found."
    if error.code == 429:
        return f"{prefix}: rate limited by WHO GHO (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
