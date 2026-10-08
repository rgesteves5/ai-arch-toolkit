"""Data processing tools — JSON extraction and CSV reading (T09).

A CSV file is read a row at a time: a page of rows is kept, and every row is counted, so the
total is the whole file's and the footer reads on (D39).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._filesystem import _is_regular_file, path_failure
from ai_arch_toolkit.toolkit.tools._window import list_window

_DEFAULT_MAX_ROWS = 100
_MAX_ROWS = 10_000
# The keys of an object a not_found names.
_KEYS_NAMED = 20


@tool(capability="compute")
def json_extract(json_string: str, path: str) -> str:
    """Extract a value from a JSON string using dot-notation path.

    Supports array indexing with brackets: "results[0].name", "data.items[2].value".

    Args:
        json_string: A valid JSON string.
        path: Dot-notation path, e.g. "user.address.city" or "items[0].name".

    Raises:
        ToolFailure: validation_error when ``json_string`` is not valid JSON or a segment of
            ``path`` indexes a value that cannot take it; not_found when ``path`` names a key or
            an index the JSON does not have.
    """
    try:
        data = json.loads(json_string)
    except (ValueError, RecursionError) as e:
        raise ToolFailure(
            "validation_error", f"invalid JSON: {e}; pass a complete JSON document"
        ) from e

    current = data
    for segment in _parse_path(path):
        try:
            current = current[segment]
        except (KeyError, IndexError) as e:
            raise ToolFailure(
                "not_found",
                f"the JSON has no {segment!r} at that point of path {path!r}; {_holds(current)}",
            ) from e
        except TypeError as e:
            raise ToolFailure(
                "validation_error",
                f"cannot index a JSON {_kind(current)} with {segment!r} in path {path!r}; "
                "use a key for an object and a number in brackets for a list",
            ) from e

    if isinstance(current, (dict, list)):
        return json.dumps(current, indent=2, ensure_ascii=False)
    return str(current)


def _holds(value: object) -> str:
    """What an object, a list or a string has, for a path that asked it for something it
    lacks."""
    if isinstance(value, dict) and value:
        keys = ", ".join(repr(key) for key in list(value)[:_KEYS_NAMED])
        more = f" and {len(value) - _KEYS_NAMED} more" if len(value) > _KEYS_NAMED else ""
        return f"the object there has the keys {keys}{more}"
    size = len(value) if isinstance(value, dict | list | str) else 0
    if not size:
        return f"the {_kind(value)} there is empty"
    unit = "characters" if isinstance(value, str) else "items"
    return f"the {_kind(value)} there has {size} {unit} (indexes 0 to {size - 1})"


def _kind(value: object) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "list"
    if isinstance(value, str):
        return "string"
    return "number, boolean or null"


def _parse_path(path: str) -> list[str | int]:
    """Parse "foo.bar[0].baz" into ['foo', 'bar', 0, 'baz']."""
    parts: list[str | int] = []
    for segment in path.split("."):
        if not segment:
            continue
        if "[" in segment:
            key, rest = segment.split("[", 1)
            if key:
                parts.append(key)
            for bracket in rest.split("["):
                idx_str = bracket.rstrip("]")
                try:
                    parts.append(int(idx_str))
                except ValueError:
                    parts.append(idx_str)
        else:
            parts.append(segment)
    return parts


@tool(
    capability="filesystem",
    risk_level="high",
    requires_approval=True,
    approval_reason="Reading CSV files can expose secrets or private data.",
)
def csv_read(
    path: str,
    offset: Annotated[int, Range(0)] = 0,
    max_rows: Annotated[int, Range(1, _MAX_ROWS)] = _DEFAULT_MAX_ROWS,
) -> ToolResult:
    """Read a CSV file as a table: its header, then a page of its rows.

    Args:
        path: Path to the CSV file.
        offset: How many data rows to skip; the footer gives the next offset.
        max_rows: How many data rows to return.

    Raises:
        ToolFailure: not_found when ``path`` does not exist; validation_error when it is not a
            regular file, a malformed path or not readable CSV; upstream when the filesystem
            refuses it.
    """
    p = Path(path).expanduser()
    try:
        if not _is_regular_file(p):
            raise ToolFailure(
                "validation_error", f"{path!r} is not a file; pass a CSV file's path"
            )
        with p.open(encoding="utf-8", errors="replace", newline="") as handle:
            table = _page(csv.reader(handle), offset, max_rows)
    except (OSError, ValueError) as e:
        raise path_failure(e, "read", path) from e
    return _answer(path, table, offset)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Table:
    """A CSV file's header, a page of its data rows, and how many data rows it has."""

    header: list[str] | None
    rows: list[list[str]]
    total: int


def _page(rows: Iterator[list[str]], offset: int, max_rows: int) -> _Table:
    """The header and the data rows from ``offset`` (at most ``max_rows``); every row is read,
    one at a time, and counted.

    Raises:
        ToolFailure: validation_error when a row is not readable CSV.
    """
    header: list[str] | None = None
    page: list[list[str]] = []
    total = 0
    try:
        header = next(rows, None)
        for row in rows:
            if offset <= total < offset + max_rows:
                page.append(row)
            total += 1
    except csv.Error as e:  # a field over the limit, a NUL byte
        number = total + (1 if header is None else 2)
        raise ToolFailure(
            "validation_error",
            f"the file is not readable CSV at row {number}: {e}; read it with read_file",
        ) from e
    return _Table(header=header, rows=page, total=total)


def _answer(path: str, table: _Table, offset: int) -> ToolResult:
    """The header and the page as a table, with the window's footer when rows are left out."""
    if table.header is None:
        return ToolResult.success("Empty CSV file.")
    shown = [table.header, *table.rows]
    widths = [0] * max(len(row) for row in shown)
    for row in shown:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    def line(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[j]) for j, cell in enumerate(row))

    end = offset + len(table.rows)
    window = list_window(
        [line(row) for row in table.rows],
        first=offset + 1,
        total=table.total,
        next_call={"offset": end} if end < table.total else None,
    )
    separator = "-+-".join("-" * width for width in widths)
    heading = f"{path} ({table.total} rows):\n{line(table.header)}\n{separator}"
    return window.result(heading=heading)
