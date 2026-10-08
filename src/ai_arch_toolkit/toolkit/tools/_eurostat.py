"""Eurostat tools: find a dataset, read its dimensions and codes, and read its observations.

The statistics API answers JSON-stat 2.0 (https://json-stat.org/format/), with each dimension's
codes and their labels; the tools show both, and every list reads on through the window (D39).
The API's guide
(https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-detailed-guidelines/api-statistics)
sets the rules the tools follow: several codes of one dimension are one parameter each
(``geo=PT&geo=ES``), one time parameter per query, and its error table: 404 for a dataset it
does not disseminate, 400 with error 100 for a query whose result is empty, 400 with 140 or 150
for a query it refuses, and 413 (``ASYNCHRONOUS_RESPONSE``) for one it will only serve later.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._values import decimal_text, plain
from ai_arch_toolkit.toolkit.tools._window import Window, page_window


def _eurostat_error(reply: Reply, query: str = "for this query") -> ToolFailure | str | None:
    """The error a Eurostat answer explains: ``{"error": [{"status", "id", "label"}]}`` (or one
    object), or the guide's ``{"warning": {"status": 413, "label"}}``.

    By the guide's error table: a 400 with error 100 is a query with no data (``not_found``,
    naming ``query``); any other 400 a query the API refuses (``validation_error``); a 413 a
    request it will only serve later, worth a retry. A 404 never gets here: each dataset call
    declares it ``missing``. Any other error is the labels; any other warning is no error.
    """
    body = reply.body if isinstance(reply.body, dict) else {}
    later = [
        item
        for item in _error_items(body.get("warning"))
        if _int(item.get("status")) == _ASYNCHRONOUS
    ]
    items = _error_items(body.get("error")) or later
    full = "; ".join(label for item in items if (label := _string(item.get("label"))))
    said = full.rstrip(".")
    if not said:
        return None
    statuses = {reply.status, *(_int(item.get("status")) for item in items)}
    if _ASYNCHRONOUS in statuses:
        return ToolFailure(
            "upstream",
            f"{said}; try again in a few minutes, or narrow the request with filters",
            retryable=True,
        )
    if reply.status == 400 and any(_int(item.get("id")) == _NO_RESULTS for item in items):
        return ToolFailure(
            "not_found",
            f"Eurostat has no data {query} ({said}); other codes or periods may have some: "
            "eurostat_dataset lists the codes",
        )
    if reply.status == 400:
        return ToolFailure(
            "validation_error",
            f"{said}; list the codes with eurostat_dataset(dataset_id, dimension=...)",
        )
    return full


# The status of a request Eurostat treats asynchronously: the answer is ready later.
_ASYNCHRONOUS = 413
# The error of a query whose result is empty, with a 400 (the guide's error table).
_NO_RESULTS = 100
# Stubs are each dataset's ID and title, about 1.5 MB; the full catalogue is 20 MB, nine tenths of
# it annotations. eurostat_dataset reads one dataset's details.
_DATAFLOWS = Api(
    base="https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/all/latest",
    name="Eurostat",
    timeout_s=30,
    params={"format": "JSON", "lang": "en", "detail": "allstubs"},
    error_reader=_eurostat_error,
)
_DATA = Api(
    base="https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data",
    name="Eurostat",
    timeout_s=30,
    params={"format": "JSON", "lang": "en"},
    error_reader=_eurostat_error,
)
_DATASET_LINES = 60
_DATASET_RE = re.compile(r"^[A-Za-z0-9_]{2,60}$")
_CODE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
# The parameters that set the time of a query: only one goes in a query (the guide).
_TIME_KEYS = frozenset(
    {"time", "time_period", "sincetimeperiod", "untiltimeperiod", "lasttimeperiod"}
)
# The parameters the tool sets on every data request (``_DATA``): no filter replaces them.
_FIXED_KEYS = frozenset({"format", "lang"})


@dataclass(frozen=True, slots=True, kw_only=True)
class _Dimension:
    """A dimension of a JSON-stat answer: its codes in index order, with their labels."""

    id: str
    label: str
    codes: tuple[str, ...]
    labels: Mapping[str, str]

    def named(self, code: str) -> str:
        """``Portugal (PT)``; a code whose label is itself (a year) stays as it is."""
        label = self.labels.get(code, "")
        return f"{label} ({code})" if label and label != code else code


@tool(capability="network")
def eurostat_dataset_search(
    query: str,
    max_results: Annotated[int, Range(1, 50)] = 10,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Search Eurostat datasets by ID or title; eurostat_dataset reads one's dimensions and
    codes.

    Args:
        query: Words of the dataset's title, or its ID.
        max_results: How many datasets to show.
        offset: How many matching datasets to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when ``query`` is empty or has unsupported characters.
    """
    if not _TEXT_RE.fullmatch(query.strip()):
        msg = (
            f"invalid query {query!r}; pass 1-180 characters of words, digits and basic "
            "punctuation, e.g. 'population'."
        )
        raise ToolFailure("validation_error", msg)
    return _DATAFLOWS.get_json(
        parse=lambda data: _search_answer(data, query.strip(), offset, max_results)
    )


@tool(capability="network")
def eurostat_dataset(
    dataset_id: str, dimension: str = "", offset: Annotated[int, Range(0)] = 0
) -> ToolResult:
    """Read a Eurostat dataset: its title, update time, period and description, and each
    dimension's codes with their labels (the codes eurostat_series takes as filters).

    Args:
        dataset_id: Eurostat dataset ID, e.g. "TPS00001" (eurostat_dataset_search finds them).
        dimension: One dimension's ID, e.g. "geo", to list only its codes; empty lists them all.
        offset: How many lines of codes to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when ``dataset_id`` is malformed or the dataset has no
            such dimension; not_found when Eurostat does not disseminate that dataset.
    """
    dataset = _dataset_id(dataset_id)
    return _DATA.get_json(
        dataset,
        params={"lastTimePeriod": "1"},
        parse=lambda data: _dataset_answer(data, dataset, dimension.strip(), offset),
        missing=_missing(dataset),
    )


@tool(capability="network")
def eurostat_series(
    dataset_id: str,
    filters: str = "",
    last_time_periods: Annotated[int, Range(1, 100)] = 5,
    max_points: Annotated[int, Range(1, 100)] = 30,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Read Eurostat observations of a dataset, one row per observation, each code with its
    label. Several codes of a dimension compare them, e.g. geo=PT+ES+FR.

    Args:
        dataset_id: Eurostat dataset ID, e.g. "TPS00001".
        filters: Comma-separated dimension filters, codes joined by '+', e.g.
            "geo=PT+ES,unit=NR"; a time filter (time=2020, sinceTimePeriod=2015) replaces
            ``last_time_periods``. eurostat_dataset lists the dimensions and codes.
        last_time_periods: How many of the latest periods to read, without a time filter.
        max_points: How many observations to show.
        offset: How many observations to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when ``dataset_id`` or ``filters`` is malformed, or
            Eurostat refuses a code; not_found when Eurostat does not disseminate that dataset
            or has no data for the query.
    """
    dataset = _dataset_id(dataset_id)
    parsed = _parse_filters(filters)
    params: dict[str, list[str] | str] = dict(parsed)
    filtered = ", ".join(f"{key}={'+'.join(codes)}" for key, codes in parsed.items())
    asked = f" for {filtered}" if filtered else ""
    if not _TIME_KEYS & {key.lower() for key in parsed}:
        params["lastTimePeriod"] = str(last_time_periods)
        asked += f" in the last {last_time_periods} periods"
    api = replace(_DATA, error_reader=lambda reply: _eurostat_error(reply, f"of {dataset}{asked}"))
    return api.get_json(
        dataset,
        params=params,
        parse=lambda data: _series_answer(data, dataset, asked, max_points, offset),
        missing=_missing(dataset, filtered),
    )


def _dataset_id(dataset_id: str) -> str:
    dataset = dataset_id.strip().upper()
    if not _DATASET_RE.fullmatch(dataset):
        msg = (
            f"invalid dataset_id {dataset_id!r}; a Eurostat dataset ID looks like TPS00001 "
            "(eurostat_dataset_search finds them)."
        )
        raise ToolFailure("validation_error", msg)
    return dataset


def _missing(dataset: str, filtered: str = "") -> str:
    """What a 404 of a dataset call means: "the requested resource is not available", which the
    guide reads as the dataset. With filters, the answer does not say which it was."""
    missing = (
        f"Eurostat has no dataset {dataset} to disseminate; find its ID with "
        "eurostat_dataset_search"
    )
    if not filtered:
        return missing
    return (
        f"{missing} (a 404 does not say whether the filters {filtered} were at fault: "
        "eurostat_dataset lists the codes)"
    )


def _parse_filters(filters: str) -> dict[str, list[str]]:
    """``geo=PT+ES,unit=NR`` as ``{"geo": ["PT", "ES"], "unit": ["NR"]}``."""
    out: dict[str, list[str]] = {}
    for raw in filters.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ToolFailure("validation_error", f"invalid filter {item!r}; use key=value.")
        key, value = (part.strip() for part in item.split("=", 1))
        if not _CODE_RE.fullmatch(key):
            msg = f"invalid filter dimension {key!r}; eurostat_dataset lists the dimensions."
            raise ToolFailure("validation_error", msg)
        if key.lower() in _FIXED_KEYS:
            msg = (
                f"invalid filter {key!r}: the tool sets format and lang itself; filter by a "
                "dimension (eurostat_dataset lists them)."
            )
            raise ToolFailure("validation_error", msg)
        # Geo codes are upper case (PT, EU27_2020), as eurostat_compare sent them.
        codes = [
            code.strip().upper() if key.lower() == "geo" else code.strip()
            for code in value.split("+")
        ]
        if not value or not all(_CODE_RE.fullmatch(code) for code in codes):
            msg = (
                f"invalid filter value for {key!r}; use codes joined by '+', e.g. {key}=PT+ES "
                "(eurostat_dataset lists the codes)."
            )
            raise ToolFailure("validation_error", msg)
        out.setdefault(key, []).extend(codes)
    return out


# --- Answers -----------------------------------------------------------------------------------


def _search_answer(data: dict[str, Any], query: str, offset: int, max_results: int) -> ToolResult:
    terms = query.lower().split()
    items = data.get("link", {}).get("item", [])
    lines = [
        f"{_string(item.get('extension', {}).get('id'))}: {_string(item.get('label'))}"
        for item in (items if isinstance(items, list) else [])
        if isinstance(item, dict) and _matches(item, terms)
    ]
    if not lines:
        return ToolResult.success(f"No Eurostat datasets match {query!r}.")
    window = page_window(
        [f"{number}. {line}" for number, line in enumerate(lines, start=1)],
        offset=offset,
        limit=max_results,
    )
    return window.result(
        heading=f"Eurostat datasets that match {query!r} (eurostat_dataset reads one):"
    )


def _matches(item: dict[str, Any], terms: list[str]) -> bool:
    extension = item.get("extension")
    code = _string(extension.get("id")) if isinstance(extension, dict) else ""
    text = f"{_string(item.get('label'))} {code}".lower()
    return all(term in text for term in terms)


def _dataset_answer(data: dict[str, Any], dataset: str, dimension: str, offset: int) -> ToolResult:
    dimensions = _dimensions(data)
    if dimension:
        dimensions = [dim for dim in dimensions if dim.id.lower() == dimension.lower()]
        if not dimensions:
            have = _and([dim.id for dim in _dimensions(data)])
            raise ToolFailure(
                "validation_error", f"{dataset} has no dimension {dimension!r}; it has {have}"
            )
    period = _period(data)
    owned = [(dim, line) for dim in dimensions for line in _dimension_lines(dim, period)]
    lines = [line for _, line in owned]
    title = f"Eurostat dataset {dataset}: {_string(data.get('label')) or '(no title)'}"
    if offset or dimension:
        # A window that starts among a dimension's codes names it: its line is in an earlier one.
        within = offset < len(owned) and owned[offset][1].startswith(" ")
        dim = owned[offset][0] if within else None
        heading = (
            f"{title}, codes ({dim.id}: {dim.label}, continued):" if dim else f"{title}, codes:"
        )
    else:
        heading = "\n".join(
            [
                title,
                *_details(data, period),
                "Dimensions and codes (eurostat_series takes them as filters, e.g. geo=PT+ES):",
            ]
        )
    window = page_window(lines, offset=offset, limit=_DATASET_LINES)
    return _naming(window, dimension).result(heading=heading)


def _naming(window: Window, dimension: str) -> Window:
    """A window of one dimension's codes names the dimension in its next call."""
    if not dimension or window.next_call is None:
        return window
    return replace(window, next_call={**window.next_call, "dimension": dimension})


def _details(data: dict[str, Any], period: str) -> list[str]:
    updated = _utc(_string(data.get("updated"))) or "?"
    lines = [f"   updated: {updated} | source: {_string(data.get('source')) or '?'}"]
    count = _annotations(data).get("OBS_COUNT")
    facts = [f"observations: {count}"] if count else []
    if period:
        facts.append(f"period: {period}")
    if facts:
        lines.append("   " + " | ".join(facts))
    extension = data.get("extension")
    description = extension.get("description") if isinstance(extension, dict) else None
    if text := _string(re.sub(r"<[^>]+>", " ", _string(description))):
        lines.append(f"   description: {text}")
    return lines


def _dimension_lines(dim: _Dimension, period: str) -> list[str]:
    count = f"{len(dim.codes)} code{'' if len(dim.codes) == 1 else 's'}"
    if dim.id.lower() == "time":  # the request asks for the latest period only
        span = f"; the data cover {period}" if period else ""
        head = f"{dim.id}: {dim.label} (the latest period here{span})"
    else:
        head = f"{dim.id}: {dim.label} ({count})"
    codes = [
        f"  {code}: {dim.labels[code]}" if dim.labels.get(code, code) != code else f"  {code}"
        for code in dim.codes
    ]
    return [head, *codes]


def _series_answer(
    data: dict[str, Any], dataset: str, asked: str, max_points: int, offset: int
) -> ToolResult:
    dimensions = _dimensions(data)
    observations = _observations(data, dimensions)
    if not observations:
        return ToolResult.success(f"Eurostat has no observations of {dataset}{asked}.")
    varying = [dim for dim in dimensions if len(dim.codes) > 1]
    flags = _flag_labels(data)
    rows = [
        f"{number}. {_row(coordinates, value, _flag_text(flag, flags), dimensions, varying)}"
        for number, (coordinates, value, flag) in enumerate(observations, start=1)
    ]
    heading = [f"Eurostat {dataset}: {_string(data.get('label')) or '(no title)'}"]
    if fixed := [dim for dim in dimensions if len(dim.codes) == 1]:
        heading.append(
            "fixed: "
            + "; ".join(f"{dim.label} ({dim.id}) = {dim.named(dim.codes[0])}" for dim in fixed)
        )
    if others := [dim for dim in varying if dim.id.lower() not in ("geo", "time")]:
        verb, pronoun = ("vary", "them") if len(others) > 1 else ("varies", "it")
        example = f"{others[0].id}={others[0].codes[0]}"
        heading.append(
            f"{_and([dim.id for dim in others])} {verb} too: each row names its series; filter "
            f"{pronoun} (e.g. {example}) to compare one series"
        )
    heading.append(
        "rows: "
        + " | ".join([*(dim.id for dim in varying if dim.id != "time"), _point_name(dimensions)])
    )
    window = page_window(rows, offset=offset, limit=max_points)
    return window.result(heading="\n".join(heading))


def _point_name(dimensions: list[_Dimension]) -> str:
    """How a row ends: ``time: value``, or ``value`` for a dataset without time."""
    return "time: value" if any(dim.id == "time" for dim in dimensions) else "value"


def _row(
    coordinates: tuple[int, ...],
    value: object,
    flag: str,
    dimensions: list[_Dimension],
    varying: list[_Dimension],
) -> str:
    """``Number (NR) | Portugal (PT) | 2024: 10639726 (flag p)``: the observation's codes of the
    dimensions that vary, with their labels, then its time and its value."""
    codes = {dim.id: dim.codes[index] for dim, index in zip(dimensions, coordinates, strict=True)}
    named = [dim.named(codes[dim.id]) for dim in varying if dim.id != "time"]
    time = codes.get("time", "")
    if isinstance(value, str):
        shown = decimal_text(value)
    else:
        shown = plain(value) if isinstance(value, int | float) else "no value"
    point = f"{time}: {shown}" if time else shown
    return " | ".join([*named, point]) + (f" (flag {flag})" if flag else "")


# --- JSON-stat ---------------------------------------------------------------------------------


def _dimensions(data: dict[str, Any]) -> list[_Dimension]:
    """The answer's dimensions in ``id`` order, each with its codes in index order."""
    ids, dims = data.get("id", []), data.get("dimension", {})
    if not isinstance(ids, list) or not isinstance(dims, dict):
        return []
    return [_dimension(str(dim_id), dims.get(dim_id)) for dim_id in ids]


def _dimension(dim_id: str, dim: object) -> _Dimension:
    raw: dict[str, Any] = dim if isinstance(dim, dict) else {}
    category = raw.get("category")
    category = category if isinstance(category, dict) else {}
    index, named = category.get("index"), category.get("label")
    labels = (
        {str(code): _string(label) for code, label in named.items()}
        if isinstance(named, dict)
        else {}
    )
    if isinstance(index, list):  # JSON-stat allows the codes as an array, in order
        codes = [str(code) for code in index]
    elif isinstance(index, dict):
        codes = [str(code) for code, _ in sorted(index.items(), key=lambda item: int(item[1]))]
    else:
        codes = list(labels)
    return _Dimension(
        id=dim_id, label=_string(raw.get("label")) or dim_id, codes=tuple(codes), labels=labels
    )


def _observations(
    data: dict[str, Any], dimensions: list[_Dimension]
) -> list[tuple[tuple[int, ...], object, str]]:
    """Each observation's coordinates (one index per dimension), value and flag, in the
    answer's order: series by series, time fastest (row-major)."""
    raw, status = data.get("value"), data.get("status")
    if isinstance(raw, list):  # JSON-stat allows every cell in a list, null where none
        raw = {str(position): value for position, value in enumerate(raw) if value is not None}
    if not isinstance(raw, dict) or not dimensions:
        return []
    values = {str(key): value for key, value in raw.items() if str(key).isdigit()}
    sizes = [len(dim.codes) for dim in dimensions]
    return [
        (coordinates, values[key], _flag(status, position))
        for position, key in sorted((int(key), key) for key in values)
        if (coordinates := _coordinates(position, sizes)) is not None
    ]


def _coordinates(position: int, sizes: list[int]) -> tuple[int, ...] | None:
    """The indices of a flat position (row-major: the last dimension fastest); ``None`` past
    the cube."""
    coordinates = []
    for size in reversed(sizes):
        if size < 1:
            return None
        coordinates.append(position % size)
        position //= size
    return tuple(reversed(coordinates)) if position == 0 else None


def _flag_labels(data: dict[str, Any]) -> dict[str, str]:
    """What each flag means, as Eurostat's answer names them (``extension.status.label``:
    ``p`` is provisional)."""
    extension = data.get("extension")
    status = extension.get("status") if isinstance(extension, dict) else None
    labels = status.get("label") if isinstance(status, dict) else None
    if not isinstance(labels, dict):
        return {}
    return {str(code): text for code, label in labels.items() if (text := _string(label))}


def _flag_text(flag: str, labels: Mapping[str, str]) -> str:
    """``p: provisional``; flags given together (``ep``) each with its label; a flag the answer
    does not name stays a code."""
    if not flag:
        return ""
    if flag in labels:
        return f"{flag}: {labels[flag]}"
    if all(letter in labels for letter in flag):
        return f"{flag}: {', '.join(labels[letter] for letter in flag)}"
    return flag


def _flag(status: object, position: int) -> str:
    """An observation's status code: JSON-stat gives one for all, a list, or an object."""
    if isinstance(status, str):
        return status
    if isinstance(status, list):
        return _string(status[position]) if position < len(status) else ""
    if isinstance(status, dict):
        return _string(status.get(str(position)))
    return ""


def _annotations(data: dict[str, Any]) -> dict[str, str]:
    extension = data.get("extension")
    items = extension.get("annotation") if isinstance(extension, dict) else None
    out: dict[str, str] = {}
    for item in items if isinstance(items, list) else ():
        if isinstance(item, dict) and (key := _string(item.get("type"))):
            text = _string(item.get("title")) or _string(item.get("text"))
            if text:
                out[key] = text
    return out


def _period(data: dict[str, Any]) -> str:
    notes = _annotations(data)
    oldest, latest = notes.get("OBS_PERIOD_OVERALL_OLDEST"), notes.get("OBS_PERIOD_OVERALL_LATEST")
    return f"{oldest or '?'} to {latest or '?'}" if oldest or latest else ""


def _error_items(value: object) -> list[dict[str, Any]]:
    items = [value] if isinstance(value, dict) else value
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _utc(stamp: str) -> str:
    """``2026-04-30T23:00:00+0200`` as ``2026-04-30T21:00:00Z``; other text as it is."""
    try:
        when = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return stamp
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _and(items: Iterable[str]) -> str:
    names = list(items)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _int(value: object) -> int | None:
    try:
        return int(str(value))
    except ValueError:
        return None


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
