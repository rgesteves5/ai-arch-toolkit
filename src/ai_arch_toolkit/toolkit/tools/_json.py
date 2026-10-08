"""Data processing tools — JSON extraction and CSV reading (T09).

A CSV file is read a row at a time: a page of rows is kept, within a number of rows and of
characters, and the rows after it are counted up to a bound, so the total is the whole file's
unless the file is huge, and the footer reads on (D39).
"""

from __future__ import annotations

import csv
import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TextIO

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._filesystem import _is_regular_file, path_failure
from ai_arch_toolkit.toolkit.tools._window import list_window

_DEFAULT_MAX_ROWS = 100
_MAX_ROWS = 10_000
# The characters a page of rows holds at most (one row at least, whatever its length), as a
# window of read_file does.
_PAGE_CHARS = 100_000
# A cell is padded to its column's width up to here: a longer cell shows whole, unaligned.
_PAD_CHARS = 40
# The rows after a page are counted up to here: past it, the heading says "at least", so a page
# never parses a huge file to its end (the parser reads about 80 MB/s, so 0.6 s at most).
_COUNT_CHARS = 50_000_000
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
            table = _page(_Lines(handle), offset, max_rows)
    except (OSError, ValueError) as e:
        raise path_failure(e, "read", path) from e
    return _answer(path, table, offset)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Table:
    """A CSV file's header, a page of its data rows, and how many data rows it has: all of them
    when ``whole``, else at least that many."""

    header: list[str] | None
    rows: list[list[str]]
    total: int
    whole: bool


class _Lines:
    """The lines of a file, counting their characters as the CSV reader takes them."""

    __slots__ = ("_handle", "chars")

    def __init__(self, handle: TextIO) -> None:
        self._handle = handle
        self.chars = 0

    def __iter__(self) -> _Lines:
        return self

    def __next__(self) -> str:
        line = next(self._handle)
        self.chars += len(line)
        return line


def _page(lines: _Lines, offset: int, max_rows: int) -> _Table:
    """The header and the data rows from ``offset``: at most ``max_rows``, within
    ``_PAGE_CHARS`` characters; then the rows after them are counted, up to ``_COUNT_CHARS``
    characters.

    Raises:
        ToolFailure: validation_error when a row is not readable CSV.
    """
    reader = csv.reader(lines)
    header: list[str] | None = None
    seen = 0  # data rows read
    try:
        header = next(reader, None)
        seen = sum(1 for _ in itertools.islice(reader, offset))
        page, more = _rows(reader, max_rows)
        seen += len(page) + more
        mark, whole = lines.chars, True
        for _ in reader:
            seen += 1
            if lines.chars - mark > _COUNT_CHARS:
                whole = False
                break
    except csv.Error as e:  # a field over the limit, a NUL byte
        number = seen + (1 if header is None else 2)
        raise ToolFailure(
            "validation_error",
            f"the file is not readable CSV at row {number}: {e}; read it with read_file",
        ) from e
    return _Table(header=header, rows=_fitted(header or [], page), total=seen, whole=whole)


def _rows(reader: Iterator[list[str]], max_rows: int) -> tuple[list[list[str]], int]:
    """Up to ``max_rows`` rows whose cells, unpadded, fit ``_PAGE_CHARS`` (the first whatever its
    length), and 1 when one more row was read that did not fit, else 0."""
    page: list[list[str]] = []
    used = 0
    for row in reader:
        size = sum(map(len, row)) + 3 * max(len(row) - 1, 0) + 1  # its line, unpadded
        if page and used + size > _PAGE_CHARS:
            return page, 1
        page.append(row)
        used += size
        if len(page) == max_rows:
            break
    return page, 0


def _fitted(header: list[str], rows: list[list[str]]) -> list[list[str]]:
    """The first of ``rows`` whose lines, padded, fit ``_PAGE_CHARS`` (one row at least)."""
    widths = _widths([header, *rows])
    used = 0
    for count, row in enumerate(rows):
        used += len(_line(row, widths)) + 1
        if count and used > _PAGE_CHARS:
            return rows[:count]
    return rows


def _widths(rows: list[list[str]]) -> list[int]:
    """Each column's width: its widest cell's, up to ``_PAD_CHARS``."""
    widths = [0] * max(map(len, rows), default=0)
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], min(len(cell), _PAD_CHARS))
    return widths


def _line(row: list[str], widths: list[int]) -> str:
    return " | ".join(cell.ljust(widths[j]) for j, cell in enumerate(row))


def _answer(path: str, table: _Table, offset: int) -> ToolResult:
    """The header and the page as a table, with the window's footer when rows are left out."""
    if table.header is None:
        return ToolResult.success("Empty CSV file.")
    widths = _widths([table.header, *table.rows])
    end = offset + len(table.rows)
    more = not table.whole or end < table.total
    window = list_window(
        [_line(row, widths) for row in table.rows],
        first=offset + 1,
        total=table.total if table.whole else None,
        next_call={"offset": end} if more else None,
    )
    separator = "-+-".join("-" * width for width in widths)
    size = f"{table.total} rows"
    if not table.whole:
        size = f"at least {size}, counted up to {_COUNT_CHARS} characters past this page"
    heading = f"{path} ({size}):\n{_line(table.header, widths)}\n{separator}"
    return window.result(heading=heading)
