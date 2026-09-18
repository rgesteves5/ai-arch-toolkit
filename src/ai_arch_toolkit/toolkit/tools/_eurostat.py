"""Eurostat tools — public EU statistics dataset discovery and series lookup."""

from __future__ import annotations

import re
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_DATAFLOWS = Api(
    base="https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/all/latest",
    name="Eurostat",
    timeout_s=30,
    params={"format": "JSON", "lang": "en"},
    status_messages={404: "no matching records found."},
)
_DATA = Api(
    base="https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data",
    name="Eurostat",
    timeout_s=30,
    params={"format": "JSON", "lang": "en"},
    status_messages={404: "no matching records found."},
)
_MAX_LIMIT = 50
_DATASET_RE = re.compile(r"^[A-Za-z0-9_]{2,60}$")
_CODE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)


@tool(capability="network")
def eurostat_dataset_search(query: str, max_results: int = 10, offset: int = 0) -> str:
    """Search Eurostat datasets/dataflows.

    Args:
        query: Dataset ID or title search text.
        max_results: Number of datasets to return (1-50). Defaults to 10.
        offset: Zero-based offset in matching local results. Defaults to 0.
    """
    if not _valid_text(query):
        return "Eurostat dataset search failed: invalid query."
    if offset < 0:
        return "Eurostat dataset search failed: offset must be greater than or equal to 0."
    try:
        return _DATAFLOWS.get_json(
            parse=lambda data: _dataset_search_text(data, query, offset, max_results)
        )
    except HttpError as e:
        return f"Eurostat dataset search failed: {e}"


@tool(capability="network")
def eurostat_dataset(dataset_id: str) -> str:
    """Get Eurostat dataset metadata using a small last-period query.

    Args:
        dataset_id: Eurostat dataset/dataflow ID, e.g. "TPS00001".
    """
    dataset = dataset_id.strip().upper()
    if not _DATASET_RE.fullmatch(dataset):
        return f"Eurostat dataset lookup failed: invalid dataset_id: {dataset_id!r}"
    try:
        return _DATA.get_json(
            dataset,
            params={"lastTimePeriod": "1"},
            parse=lambda data: _dataset_text(data, dataset),
        )
    except HttpError as e:
        return f"Eurostat dataset lookup failed: {e}"


@tool(capability="network")
def eurostat_dimensions(dataset_id: str, max_values: int = 20) -> str:
    """List Eurostat dataset dimensions and sample category codes.

    Args:
        dataset_id: Eurostat dataset/dataflow ID, e.g. "TPS00001".
        max_values: Number of category values to show per dimension (1-50). Defaults to 20.
    """
    dataset = dataset_id.strip().upper()
    if not _DATASET_RE.fullmatch(dataset):
        return f"Eurostat dimensions failed: invalid dataset_id: {dataset_id!r}"
    try:
        return _DATA.get_json(
            dataset,
            params={"lastTimePeriod": "1"},
            parse=lambda data: _dimensions_text(data, dataset, max_values),
        )
    except HttpError as e:
        return f"Eurostat dimensions failed: {e}"


@tool(capability="network")
def eurostat_series(
    dataset_id: str,
    filters: str = "",
    last_time_periods: int = 5,
    max_points: int = 25,
) -> str:
    """Get Eurostat observations for a dataset using generic dimension filters.

    Args:
        dataset_id: Eurostat dataset/dataflow ID, e.g. "TPS00001".
        filters: Comma-separated dimension filters, e.g. "geo=PT,unit=NR".
        last_time_periods: Number of latest time periods to request when no time filter is given.
        max_points: Number of observations to return (1-50). Defaults to 25.
    """
    dataset = dataset_id.strip().upper()
    if not _DATASET_RE.fullmatch(dataset):
        return f"Eurostat series failed: invalid dataset_id: {dataset_id!r}"
    parsed = _parse_filters(filters)
    if isinstance(parsed, str):
        return f"Eurostat series failed: {parsed}"
    params = _with_last_periods(parsed, last_time_periods)
    try:
        return _DATA.get_json(
            dataset, params=params, parse=lambda data: _series_text(data, dataset, max_points)
        )
    except HttpError as e:
        return f"Eurostat series failed: {e}"


@tool(capability="network")
def eurostat_compare(
    dataset_id: str,
    geo_codes: str,
    filters: str = "",
    last_time_periods: int = 1,
) -> str:
    """Compare a Eurostat dataset across multiple geo codes.

    Args:
        dataset_id: Eurostat dataset/dataflow ID, e.g. "TPS00001".
        geo_codes: Comma-separated geo codes, e.g. "PT,ES,FR".
        filters: Additional comma-separated dimension filters except geo.
        last_time_periods: Number of latest time periods to request. Defaults to 1.
    """
    geos = [geo.strip().upper() for geo in geo_codes.split(",") if geo.strip()]
    if not geos or len(geos) > 10:
        return "Eurostat compare failed: provide 1-10 comma-separated geo_codes."
    if any(not _CODE_RE.fullmatch(geo) for geo in geos):
        return "Eurostat compare failed: invalid geo code."
    parsed = _parse_filters(filters)
    if isinstance(parsed, str):
        return f"Eurostat compare failed: {parsed}"
    if "geo" in {key.lower() for key in parsed}:
        return "Eurostat compare failed: provide geo filters via geo_codes, not filters."
    dataset = dataset_id.strip().upper()
    if not _DATASET_RE.fullmatch(dataset):
        return f"Eurostat compare failed: invalid dataset_id: {dataset_id!r}"

    try:
        rows = _compare_rows(dataset, geos, parsed, last_time_periods)
    except HttpError as e:
        return f"Eurostat compare failed: {e}"
    if not rows:
        return f"No Eurostat comparison observations found for {dataset}."
    lines = [f"Eurostat comparison {dataset}:"]
    lines.extend(f"{index}. {row}" for index, row in enumerate(rows, start=1))
    return "\n".join(lines)


def _with_last_periods(params: dict[str, str], last_time_periods: int) -> dict[str, str]:
    if not any(key.lower() == "time" for key in params):
        params["lastTimePeriod"] = str(max(1, min(last_time_periods, 20)))
    return params


def _compare_rows(
    dataset: str, geos: list[str], filters: dict[str, str], last_time_periods: int
) -> list[str]:
    rows: list[str] = []
    for geo in geos:
        params = _with_last_periods({**filters, "geo": geo}, last_time_periods)
        rows.extend(
            _DATA.get_json(
                dataset,
                params=params,
                parse=lambda data: [
                    _point_text(point) for point in _observations(data)[:last_time_periods]
                ],
            )
        )
    return rows


def _dataset_search_text(data: dict[str, Any], query: str, offset: int, max_results: int) -> str:
    terms = query.lower().split()
    matches = [item for item in _dataflow_items(data) if _matches_dataflow(item, terms)]
    page = matches[offset : offset + _bounded(max_results)]
    if not page:
        return "No Eurostat datasets found."
    lines = [
        (
            f"Eurostat datasets for {query!r} "
            f"(returned {len(page)}, total matches {len(matches)}, offset {offset}):"
        )
    ]
    for index, item in enumerate(page, start=1):
        lines.extend(_format_dataflow(item, index=index))
    return "\n".join(lines)


def _dataset_text(data: dict[str, Any], dataset: str) -> str:
    lines = [f"Eurostat dataset {dataset}:", _string(data.get("label")) or "(no label)"]
    updated = _string(data.get("updated")) or "?"
    source = _string(data.get("source")) or "?"
    lines.append(f"   updated: {updated} | source: {source}")
    description = _strip_html(_nested(data, "extension", "description"))
    if description:
        lines.append(f"   description: {_trim(description, 500)}")
    annotations = _annotations(data)
    if annotations:
        lines.append("   " + " | ".join(annotations))
    dims = _dimension_summaries(data, max_values=5)
    if dims:
        lines.append("   dimensions: " + "; ".join(dims))
    return "\n".join(lines)


def _dimensions_text(data: dict[str, Any], dataset: str, max_values: int) -> str:
    dims = data.get("dimension", {})
    ids = data.get("id", [])
    if not isinstance(dims, dict) or not isinstance(ids, list):
        return f"No Eurostat dimensions found for {dataset}."
    lines = [f"Eurostat dimensions for {dataset}:"]
    for dim_id in ids:
        dim = dims.get(dim_id, {})
        if not isinstance(dim, dict):
            continue
        labels = dim.get("category", {}).get("label", {})
        values = []
        if isinstance(labels, dict):
            for code, label in list(labels.items())[: _bounded(max_values)]:
                values.append(f"{code}={_string(label)}")
        lines.append(f"{dim_id} — {_string(dim.get('label')) or '?'}")
        if values:
            lines.append(f"   values: {'; '.join(values)}")
    return "\n".join(lines)


def _series_text(data: dict[str, Any], dataset: str, max_points: int) -> str:
    points = _observations(data)
    if not points:
        return f"No Eurostat observations found for {dataset}."
    returned = min(len(points), _bounded(max_points))
    lines = [f"Eurostat series {dataset} (returned {returned} of {len(points)}):"]
    for index, point in enumerate(points[: _bounded(max_points)], start=1):
        lines.append(f"{index}. {_point_text(point)}")
    return "\n".join(lines)


def _point_text(point: dict[str, Any]) -> str:
    dims = ", ".join(f"{key}={value}" for key, value in point["dimensions"].items())
    return f"{point['value']} | {dims}"


def _dataflow_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = data.get("link", {}).get("item", [])
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _matches_dataflow(item: dict[str, Any], terms: list[str]) -> bool:
    extension = item.get("extension", {})
    text = " ".join(
        [
            _string(item.get("label")),
            _string(extension.get("id")) if isinstance(extension, dict) else "",
            _strip_html(_string(extension.get("description")))
            if isinstance(extension, dict)
            else "",
        ]
    ).lower()
    return all(term in text for term in terms)


def _format_dataflow(item: dict[str, Any], *, index: int) -> list[str]:
    extension = item.get("extension", {})
    if not isinstance(extension, dict):
        extension = {}
    dataset_id = _string(extension.get("id"))
    label = _string(item.get("label"))
    lines = [f"{index}. {dataset_id} — {label}"]
    annotations = _annotation_map(extension.get("annotation"))
    obs = annotations.get("OBS_COUNT")
    latest = annotations.get("OBS_PERIOD_OVERALL_LATEST")
    oldest = annotations.get("OBS_PERIOD_OVERALL_OLDEST")
    if obs or latest or oldest:
        lines.append(f"   observations: {obs or '?'} | period: {oldest or '?'}-{latest or '?'}")
    return lines


def _annotations(data: dict[str, Any]) -> list[str]:
    values = _annotation_map(data.get("extension", {}).get("annotation"))
    out = []
    if values.get("OBS_COUNT"):
        out.append(f"observations: {values['OBS_COUNT']}")
    if values.get("OBS_PERIOD_OVERALL_OLDEST") or values.get("OBS_PERIOD_OVERALL_LATEST"):
        oldest = values.get("OBS_PERIOD_OVERALL_OLDEST", "?")
        latest = values.get("OBS_PERIOD_OVERALL_LATEST", "?")
        out.append(f"period: {oldest}-{latest}")
    return out


def _annotation_map(value: Any) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    out = {}
    for item in value:
        if isinstance(item, dict):
            key = _string(item.get("type"))
            text = (
                _string(item.get("title"))
                or _string(item.get("text"))
                or _string(item.get("date"))
            )
            if key and text:
                out[key] = text
    return out


def _dimension_summaries(data: dict[str, Any], *, max_values: int) -> list[str]:
    dims = data.get("dimension", {})
    ids = data.get("id", [])
    if not isinstance(dims, dict) or not isinstance(ids, list):
        return []
    out = []
    for dim_id in ids:
        dim = dims.get(dim_id, {})
        if not isinstance(dim, dict):
            continue
        labels = dim.get("category", {}).get("label", {})
        count = len(labels) if isinstance(labels, dict) else 0
        sample = ", ".join(list(labels)[:max_values]) if isinstance(labels, dict) else ""
        out.append(f"{dim_id} ({count}: {sample})")
    return out


def _observations(data: dict[str, Any]) -> list[dict[str, Any]]:
    value_map = data.get("value", {})
    if not isinstance(value_map, dict):
        return []
    ids = data.get("id", [])
    sizes = data.get("size", [])
    dims = data.get("dimension", {})
    if not isinstance(ids, list) or not isinstance(sizes, list) or not isinstance(dims, dict):
        return []
    index_to_code = [_dimension_index(dims.get(dim_id, {})) for dim_id in ids]
    points = []
    for flat_index, value in value_map.items():
        if not str(flat_index).isdigit():
            continue
        coordinates = _decode_index(int(flat_index), [int(size) for size in sizes])
        dimensions = {
            str(dim_id): index_to_code[position].get(coord, str(coord))
            for position, (dim_id, coord) in enumerate(zip(ids, coordinates, strict=False))
        }
        points.append({"value": value, "dimensions": dimensions})
    return points


def _dimension_index(dim: dict[str, Any]) -> dict[int, str]:
    index = dim.get("category", {}).get("index", {}) if isinstance(dim, dict) else {}
    if not isinstance(index, dict):
        return {}
    return {int(position): code for code, position in index.items() if isinstance(position, int)}


def _decode_index(flat_index: int, sizes: list[int]) -> list[int]:
    coords = []
    for size in reversed(sizes):
        coords.append(flat_index % size)
        flat_index //= size
    return list(reversed(coords))


def _parse_filters(filters: str) -> dict[str, str] | str:
    out: dict[str, str] = {}
    if not filters.strip():
        return out
    for raw in filters.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            return f"invalid filter {item!r}; use key=value."
        key, value = [part.strip() for part in item.split("=", 1)]
        if not _CODE_RE.fullmatch(key):
            return f"invalid filter dimension {key!r}."
        if not value or any(not _CODE_RE.fullmatch(part.strip()) for part in value.split("+")):
            return f"invalid filter value for {key!r}."
        out[key] = value
    return out


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value)


def _trim(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def _nested(data: dict[str, Any], *keys: str) -> str:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return _string(current)


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
