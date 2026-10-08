"""A page of the first results of a source that has no offset of its own (D39).

Some sources return their first N results and no way to skip any: Open-Meteo's geocoding gives up
to 100 places, Nominatim's search up to 40, EONET as many events as ``limit`` asks. A tool pages
them by asking, from the start, for one more result than the page ends on (:func:`asked`), and
showing ``limit`` of them from ``offset``. The one more says whether others follow; the total is
known when the source returned fewer than asked. Past ``depth`` the source gives nothing more, and
the page says how to narrow the query instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ai_arch_toolkit.toolkit.tools._window import Window, list_window


def asked(offset: int, limit: int, depth: int) -> int:
    """How many results to ask the source for: the page from the start, and one more, to know
    whether others follow (at most ``depth``, the most the source returns)."""
    return min(offset + limit + 1, depth)


def first_results_window(
    found: Sequence[str], *, offset: int, limit: int, depth: int, narrow: str
) -> Window:
    """The page of ``found`` (the source's first :func:`asked` results, one line each) that
    starts at ``offset``.

    Args:
        found: The results the source returned, from the first.
        offset: How many to skip.
        limit: How many to show.
        depth: The most results the source returns for one query.
        narrow: What to change in the query to reach results past ``depth``.
    """
    shown = list(found[offset : offset + limit])
    more = len(found) > offset + limit
    whole = len(found) < asked(offset, limit, depth)
    window = list_window(
        shown,
        first=offset + 1,
        total=len(found) if whole else None,
        next_call={"offset": offset + len(shown)} if more else None,
    )
    if more or whole or not shown:
        return window
    note = f"(the source returns no more than {depth} results; {narrow})"
    return replace(window, body=f"{window.body}\n{note}")
