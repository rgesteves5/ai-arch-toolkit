"""What the literature and identifier tools share (T06): a record read through the window, a list
of names that says how many more there are and where to read them all, and a DOI as the sources
take it.

A record tool (``arxiv_paper``, ``crossref_work``, ``nvd_cve`` …) writes the whole record, every
field and every item of its lists, and returns it through the window (D39): its footer says what
was shown and the offset that reads on. A search shows the first names of a long list and says
how many more there are, and which record tool lists them all.
"""

from __future__ import annotations

from collections.abc import Sequence

from ai_arch_toolkit.core import ToolResult
from ai_arch_toolkit.toolkit.tools._window import text_window

# A record's window, as the wiki family's (T05): enough for a whole record, bounded by the model.
MAX_CHARS = 20_000
DEFAULT_CHARS = 6_000
# How many names of a list a search result shows before it says how many more there are.
NAMES_SHOWN = 8

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)


def record(text: str, *, heading: str, offset: int, max_chars: int) -> ToolResult:
    """``text`` from ``offset``, at most ``max_chars`` characters, under ``heading``: the window's
    footer says what was shown and the offset that reads on."""
    return text_window(text, offset=offset, limit=max_chars).result(heading=heading)


def names(items: Sequence[str], *, whole: str) -> str:
    """The first names of ``items``; past them, how many more there are and the call (``whole``)
    that lists them all, when the record has an ID to call it with (else ``whole`` is empty)."""
    if len(items) <= NAMES_SHOWN:
        return ", ".join(items)
    shown = ", ".join(items[:NAMES_SHOWN])
    more = f"+{len(items) - NAMES_SHOWN} more"
    return f"{shown} ({more}; {whole} lists all)" if whole else f"{shown} ({more})"


def call(tool: str, identifier: str) -> str:
    """The call that reads a record by its ID, ``tool('ID')``; empty without an ID."""
    return f"{tool}({identifier!r})" if identifier else ""


def doi_of(value: str) -> str:
    """``value`` as a bare DOI (``10.1000/xyz``), from a DOI, ``doi:…`` or a doi.org URL; empty
    when it is not one.

    A DOI is a prefix (the directory indicator ``10``, a full stop and a registrant code), a
    forward slash and a suffix (the DOI Handbook, "Numbering": https://doi.org/10.1000/182).
    """
    doi = value.strip()
    lower = doi.lower()
    for prefix in _DOI_PREFIXES:
        if lower.startswith(prefix):
            doi = doi[len(prefix) :].strip()
            break
    prefix, slash, suffix = doi.partition("/")
    if not (prefix.startswith("10.") and len(prefix) > 3 and slash and suffix):
        return ""
    return "" if any(char.isspace() for char in doi) else doi
