"""WHO Global Health Observatory tools: public health indicators and their observations.

The GHO OData API (https://www.who.int/data/gho/info/gho-odata-api) pages with OData's ``$top``
and ``$skip`` and gives no count, so a list asks for one row more than it shows: that row, or
an ``@odata.nextLink``, says there is more, and the window's footer gives the next ``skip``
(D39). Errors are OData's error object, ``{"error": {"code", "message"}}``
(https://docs.oasis-open.org/odata/odata-json-format/v4.01/odata-json-format-v4.01.html#sec_ErrorResponse).
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._numbers import plain_number
from ai_arch_toolkit.toolkit.tools._window import list_window


def _odata_error(reply: Reply) -> ToolFailure | str | None:
    """The error an OData answer explains, in its words; ``None`` for a result.

    A 400 is a request the service refuses (a code or a filter it does not know): the caller's
    to fix, a ``validation_error``.
    """
    error = reply.body.get("error") if isinstance(reply.body, dict) else None
    message = " ".join(str(error.get("message") or "").split()) if isinstance(error, dict) else ""
    if not message:
        return None
    if reply.status == 400:
        return ToolFailure(
            "validation_error",
            f"{message.rstrip('.')}; check the code and the filters (who_indicators lists the "
            "codes)",
        )
    return message


_API = Api(
    base="https://ghoapi.azureedge.net/api",
    name="WHO GHO",
    timeout_s=20,
    query_safe="'() ,",
    error_reader=_odata_error,
)
_CODE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_YEAR_RE = re.compile(r"^\d{4}$")
_TEXT_HINT = "use 1-180 letters, digits, spaces and ,.'()/%:+-"
# The dimensions a row may carry besides place and time, each with its type.
_DIMENSIONS = (("Dim1Type", "Dim1"), ("Dim2Type", "Dim2"), ("Dim3Type", "Dim3"))


@tool(capability="network")
def who_indicators(
    query: str = "",
    max_results: Annotated[int, Range(1, 100)] = 25,
    skip: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Search WHO Global Health Observatory indicators by code or name.

    Args:
        query: Text to look for in the indicator's code or name; empty lists them all.
        max_results: How many indicators to show.
        skip: How many matching indicators to skip; the footer gives the next skip.

    Raises:
        ToolFailure: validation_error when the query is invalid.
    """
    if query and not _valid_text(query):
        raise ToolFailure("validation_error", f"invalid query {query!r}; {_TEXT_HINT}")
    params = {"$top": str(max_results + 1), "$skip": str(skip)}
    if query.strip():
        escaped = query.strip().replace("'", "''").lower()
        params["$filter"] = (
            f"contains(tolower(IndicatorName),'{escaped}') "
            f"or contains(tolower(IndicatorCode),'{escaped}')"
        )
    return _API.get_json(
        "Indicator",
        params=params,
        parse=lambda data: _indicators_answer(data, query.strip(), max_results, skip),
    )


@tool(capability="network")
def who_indicator(indicator_code: str) -> ToolResult:
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
        "Indicator", params=params, parse=lambda data: _indicator_answer(data, code)
    )


@tool(capability="network")
def who_series(
    indicator_code: str,
    country: str = "",
    from_year: str = "",
    to_year: str = "",
    dim1: str = "",
    max_results: Annotated[int, Range(1, 100)] = 25,
    skip: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Fetch WHO GHO observations for an indicator: place, year, value (with its uncertainty
    interval) and the other dimensions of each row.

    Args:
        indicator_code: WHO GHO indicator code, e.g. "WHOSIS_000001".
        country: ISO3 country code to keep, e.g. "PRT".
        from_year: First year, as YYYY.
        to_year: Last year, as YYYY.
        dim1: First-dimension code to keep, e.g. a sex or age code such as "SEX_MLE".
        max_results: How many observations to show.
        skip: How many observations to skip; the footer gives the next skip.

    Raises:
        ToolFailure: validation_error when the code, the country, a year or ``dim1`` is
            invalid; not_found when WHO GHO has no indicator with that code.
    """
    code = _indicator_code(indicator_code)
    _check_series(country, from_year, to_year, dim1)

    params = {"$top": str(max_results + 1), "$skip": str(skip)}
    series_filter = _series_filter(country, from_year, to_year, dim1)
    if series_filter:
        params["$filter"] = series_filter
    asked = _asked(country, from_year, to_year, dim1)
    # Each indicator is an OData entity set of its own, /api/{code}: a 404 is an unknown code.
    return _API.get_json(
        code,
        params=params,
        parse=lambda data: _series_answer(data, code, asked, max_results, skip),
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


def _check_series(country: str, from_year: str, to_year: str, dim1: str) -> None:
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


def _asked(country: str, from_year: str, to_year: str, dim1: str) -> str:
    """The series filters in words: ``country=PRT, from 2019``; empty without any."""
    parts = []
    if country.strip():
        parts.append(f"country={country.strip().upper()}")
    if from_year.strip():
        parts.append(f"from {from_year.strip()}")
    if to_year.strip():
        parts.append(f"to {to_year.strip()}")
    if dim1.strip():
        parts.append(f"dim1={dim1.strip()}")
    return ", ".join(parts)


def _window_of(
    data: dict[str, Any], rows: list[dict[str, Any]], max_results: int, skip: int
) -> tuple[list[dict[str, Any]], dict[str, int] | None]:
    """The rows to show and the next call: one row more than asked, or a next link, says there
    is more."""
    more = len(rows) > max_results or bool(data.get("@odata.nextLink"))
    shown = rows[:max_results]
    return shown, ({"skip": skip + len(shown)} if more and shown else None)


def _indicators_answer(
    data: dict[str, Any], query: str, max_results: int, skip: int
) -> ToolResult:
    rows = _values(data)
    if not rows and skip == 0:
        found = f"match {query!r}" if query else "are listed"
        return ToolResult.success(f"No WHO GHO indicators {found}.")
    shown, next_call = _window_of(data, rows, max_results, skip)
    lines = [
        f"{number}. {_string(item.get('IndicatorCode'))}: {_string(item.get('IndicatorName'))}"
        for number, item in enumerate(shown, start=skip + 1)
    ]
    matching = f" that match {query!r}" if query else ""
    window = list_window(lines, first=skip + 1, next_call=next_call)
    return window.result(heading=f"WHO GHO indicators{matching} (who_series reads one):")


def _indicator_answer(data: dict[str, Any], code: str) -> ToolResult:
    items = _values(data)
    if not items:
        raise ToolFailure("not_found", _no_indicator(code))
    item = items[0]
    lines = [
        f"WHO GHO indicator {code}:",
        _string(item.get("IndicatorName")) or "(no name)",
        f"   language: {_string(item.get('Language')) or '?'}",
    ]
    return ToolResult.success("\n".join(lines))


def _series_answer(
    data: dict[str, Any], code: str, asked: str, max_results: int, skip: int
) -> ToolResult:
    rows = _values(data)
    if not rows and skip == 0:
        matching = f" match {asked}" if asked else ""
        return ToolResult.success(f"No WHO GHO observations of {code}{matching}.")
    shown, next_call = _window_of(data, rows, max_results, skip)
    lines = [_row_text(number, row) for number, row in enumerate(shown, start=skip + 1)]
    window = list_window(lines, first=skip + 1, next_call=next_call)
    where = f" ({asked})" if asked else ""
    return window.result(heading=f"WHO GHO observations of {code}{where}:")


def _row_text(number: int, row: dict[str, Any]) -> str:
    """``5. PRT (country, region Europe EUR), 2019: 81.3 (80.9 to 81.7) | sex: SEX_BTSX``:
    each code with its dimension's type, and the labels the row brings."""
    place = _string(row.get("SpatialDim")) or "(no place)"
    about = [_string(row.get("SpatialDimType")).lower()]
    region = " ".join(
        part
        for part in (_string(row.get("ParentLocation")), _string(row.get("ParentLocationCode")))
        if part
    )
    if region:
        about.append(f"region {region}")
    described = ", ".join(part for part in about if part)
    head = f"{number}. {place}" + (f" ({described})" if described else "")
    time = _string(row.get("TimeDim")) or "(no time)"
    others = [
        f"{_string(row.get(kind)).lower() or column.lower()}: {_string(row.get(column))}"
        for kind, column in _DIMENSIONS
        if _string(row.get(column))
    ]
    return " | ".join([f"{head}, {time}: {_value(row)}", *others])


def _value(row: dict[str, Any]) -> str:
    """The numeric value, every digit, with the uncertainty interval when the row gives one; the
    displayed value when there is no number."""
    number = row.get("NumericValue")
    if not isinstance(number, int | float) or isinstance(number, bool):
        return _string(row.get("Value")) or "no value"
    low, high = row.get("Low"), row.get("High")
    if isinstance(low, int | float) and isinstance(high, int | float):
        return f"{plain_number(number)} ({plain_number(low)} to {plain_number(high)})"
    return plain_number(number)


def _values(data: dict[str, Any]) -> list[dict[str, Any]]:
    value = data.get("value")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
