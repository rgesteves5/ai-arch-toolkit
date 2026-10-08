"""A table as lines a model reads: one per row, its cells joined by `` | `` (D40).

A cell that spans rows repeats on each of them, so every row reads whole, and one that spans
columns leaves the others empty. The HTML a wiki renders (``_wiki_html``) and the narrative of a
drug label (``_spl``, CDA's ``rowspan`` and ``colspan``) are laid out the same way, within the same
budgets: hostile markup cannot multiply the text it reads as.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

# A span larger than this is the markup's mistake, not a table's shape; past this many laid-out
# cells a table stops repeating its spans (each row keeps its own cells).
_MAX_ROWSPAN = 500
_MAX_COLSPAN = 50
_MAX_TABLE_CELLS = 100_000


@dataclass(slots=True)
class Cell:
    """A table cell: its values, and the rows and columns it spans."""

    parts: list[str] = field(default_factory=list)
    rowspan: int = 1
    colspan: int = 1

    def text(self) -> str:
        return "; ".join(part for part in self.parts if part)


def cell(rowspan: str | None, colspan: str | None, *parts: str) -> Cell:
    """A cell with the spans its ``rowspan`` and ``colspan`` attributes give, within the caps (a
    value that is no positive integer spans one)."""
    return Cell(
        parts=list(parts),
        rowspan=_span(rowspan, _MAX_ROWSPAN),
        colspan=_span(colspan, _MAX_COLSPAN),
    )


def table_lines(rows: Sequence[Sequence[Cell]], caption: str = "") -> list[str]:
    """``Table: caption``, then one line per row with every span laid out; a row's empty cells at
    its end are left out, and a row with nothing in it."""
    lines = []
    for row in grid(rows):
        while row and not row[-1]:
            row.pop()
        if row:
            lines.append(" | ".join(row))
    return ([f"Table: {caption}"] if caption else []) + lines


def grid(rows: Sequence[Sequence[Cell]]) -> list[list[str]]:
    """The rows with every span laid out: a row-spanning cell repeats below, a column-spanning
    one leaves its other columns empty, until the table has laid out ``_MAX_TABLE_CELLS``."""
    laid_out: list[list[str]] = []
    below: dict[int, tuple[str, int]] = {}  # column -> (text, rows still to fill)
    laid = 0
    for row in rows:
        out: list[str] = []
        far = max(below, default=-1)  # the last column a span from above fills in this row
        cells = iter(row)
        current = next(cells, None)
        while current is not None or len(out) <= far:
            column = len(out)
            if column in below:
                text, left = below.pop(column)
                out.append(text)
                if left > 1:
                    below[column] = (text, left - 1)
            elif current is None:  # only spans from above remain, further right
                out.append("")
            else:
                spanning = laid < _MAX_TABLE_CELLS
                text = current.text()
                for offset in range(current.colspan if spanning else 1):
                    out.append(text if offset == 0 else "")
                    if spanning and current.rowspan > 1:
                        below[column + offset] = (text if offset == 0 else "", current.rowspan - 1)
                current = next(cells, None)
            laid += 1
        if laid >= _MAX_TABLE_CELLS:
            below.clear()
        laid_out.append(out)
    return laid_out


def _span(value: str | None, ceiling: int) -> int:
    try:
        number = int(value or "")
    except ValueError:
        number = 1
    return min(max(number, 1), ceiling)
