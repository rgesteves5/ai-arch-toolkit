"""Table parsing tool for BBEH benchmark tasks.

Parses column-major or row-major flat text data into structured tables.
Designed for the BBEH buggy_tables task where data is given as:
  [col_name, val1, val2, ...], [col_name, val1, val2, ...], ...
"""

from __future__ import annotations

import re

from ai_arch_toolkit.core import tool


def _parse_column_major(text: str) -> dict[str, list[str]]:
    """Parse column-major bracket-delimited data.

    Input format: [col, v1, v2, ...], [col, v1, v2, ...], ...
    Returns dict mapping column name to list of values.
    """
    # Find all bracket groups
    groups = re.findall(r"\[([^\]]*)\]", text)
    table: dict[str, list[str]] = {}
    for group in groups:
        # Split by comma, strip whitespace
        parts = [p.strip() for p in group.split(",")]
        if not parts:
            continue
        col_name = parts[0]
        values = parts[1:]
        table[col_name] = values
    return table


def _to_rows(table: dict[str, list[str]]) -> list[dict[str, str]]:
    """Convert column-major dict to list of row dicts."""
    if not table:
        return []
    cols = list(table.keys())
    n_rows = max(len(v) for v in table.values()) if table else 0
    rows = []
    for i in range(n_rows):
        row = {}
        for col in cols:
            vals = table[col]
            row[col] = vals[i] if i < len(vals) else ""
        rows.append(row)
    return rows


def _format_table(rows: list[dict[str, str]], max_rows: int = 50) -> str:
    """Format rows as a readable text table."""
    if not rows:
        return "Empty table"
    cols = list(rows[0].keys())
    # Compute column widths
    widths = {c: len(c) for c in cols}
    display_rows = rows[:max_rows]
    for row in display_rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))
    # Cap column width
    for c in cols:
        widths[c] = min(widths[c], 20)

    # Header
    header = " | ".join(c.ljust(widths[c])[: widths[c]] for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    lines = [header, sep]
    for row in display_rows:
        line = " | ".join(str(row.get(c, "")).ljust(widths[c])[: widths[c]] for c in cols)
        lines.append(line)
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    lines.append(f"\n{len(rows)} rows x {len(cols)} columns")
    return "\n".join(lines)


@tool
def table_parse(data: str, output_format: str = "table") -> str:
    """Parse tabular data from text into a structured format.

    Handles column-major bracket format: [col, v1, v2, ...], [col, v1, v2, ...]
    Also handles simple CSV-like formats.

    Args:
        data: The raw table data text to parse.
        output_format: "table" for readable table, "python" for Python dict repr,
            "rows" for list-of-dicts repr, "columns" for column names only.
    """
    try:
        # Try column-major bracket format first
        if "[" in data and "]" in data:
            table = _parse_column_major(data)
            if table:
                rows = _to_rows(table)
                if output_format == "python":
                    return repr(table)
                if output_format == "rows":
                    return repr(rows[:50])
                if output_format == "columns":
                    return f"Columns: {list(table.keys())}\nRows: {len(rows)}"
                return _format_table(rows)

        # Try CSV-like (lines with commas/tabs)
        lines = [line.strip() for line in data.strip().split("\n") if line.strip()]
        if len(lines) >= 2:
            # Detect separator
            sep = "\t" if "\t" in lines[0] else ","
            header = [h.strip() for h in lines[0].split(sep)]
            rows: list[dict[str, str]] = []
            for line in lines[1:]:
                values = [v.strip() for v in line.split(sep)]
                row = dict(zip(header, values, strict=False))
                rows.append(row)
            if output_format == "python":
                table_dict: dict[str, list[str]] = {h: [] for h in header}
                for row in rows:
                    for h in header:
                        table_dict[h].append(row.get(h, ""))
                return repr(table_dict)
            if output_format == "rows":
                return repr(rows[:50])
            if output_format == "columns":
                return f"Columns: {header}\nRows: {len(rows)}"
            return _format_table(rows)

        return "Error: Could not parse table format. Expected column-major brackets or CSV."
    except Exception as e:
        return f"Error: {e}"
