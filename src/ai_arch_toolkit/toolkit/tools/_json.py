"""Data processing tools — JSON extraction and CSV reading."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._filesystem import (
    _error_text,
    _is_regular_file,
    clamp,
    read_prefix,
)

_DEFAULT_MAX_ROWS = 100
_MAX_ROWS = 10_000
_MAX_CSV_CHARS = 1_000_000


@tool(capability="compute")
def json_extract(json_string: str, path: str) -> str:
    """Extract a value from a JSON string using dot-notation path.

    Supports array indexing with brackets: "results[0].name", "data.items[2].value".

    Args:
        json_string: A valid JSON string.
        path: Dot-notation path, e.g. "user.address.city" or "items[0].name".
    """
    try:
        data = json.loads(json_string)
    except (ValueError, RecursionError) as e:
        return f"Invalid JSON: {e}"

    current = data
    for segment in _parse_path(path):
        try:
            current = current[segment]
        except (KeyError, IndexError, TypeError) as e:
            return f"Path error at {segment!r}: {e}"

    if isinstance(current, (dict, list)):
        return json.dumps(current, indent=2, ensure_ascii=False)
    return str(current)


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
def csv_read(path: str, max_rows: int = _DEFAULT_MAX_ROWS) -> str:
    """Read a CSV file and return it as a formatted table.

    Args:
        path: Path to the CSV file.
        max_rows: Maximum number of data rows to return (1-10000). Defaults to 100.
    """
    p = Path(path).expanduser()
    try:
        if not _is_regular_file(p):
            return f"Not a file: {path}"
        text, _ = read_prefix(p, _MAX_CSV_CHARS)
    except FileNotFoundError:
        return f"File not found: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except (OSError, ValueError) as e:
        return f"Cannot read {path!r}: {_error_text(e)}"
    return _table(text, clamp(max_rows, 1, _MAX_ROWS))


def _table(text: str, max_rows: int) -> str:
    rows: list[list[str]] = []
    for i, row in enumerate(csv.reader(StringIO(text))):
        rows.append(row)
        if i >= max_rows:  # header + max_rows data rows
            break
    if not rows:
        return "Empty CSV file."
    widths = [0] * max(len(r) for r in rows)
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    def _fmt_row(row: list[str]) -> str:
        cells = [cell.ljust(widths[j]) if j < len(widths) else cell for j, cell in enumerate(row)]
        return " | ".join(cells)

    lines = [_fmt_row(rows[0]), "-+-".join("-" * w for w in widths)]
    lines.extend(_fmt_row(row) for row in rows[1:])
    total_rows = text.count("\n")
    result = "\n".join(lines)
    if total_rows > max_rows + 1:
        result += f"\n\n[Showing {max_rows} of {total_rows} rows]"
    return result
