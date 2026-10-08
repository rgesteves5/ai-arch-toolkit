"""Internet Archive tools: the advanced search, with its total, and whole item records read window
by window (T09; D39).

Two APIs on archive.org, each with its error reader: the advanced search
(https://archive.org/advancedsearch.php), paged by ``rows`` and ``page`` down to the 10,000th
result (https://archive.org/help/aboutsearch.htm), and the metadata read API
(https://archive.org/developers/md-read.html).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._window import find_window, list_window, text_window

# The search serves sorted pages only until the 10,000th result
# (https://archive.org/help/aboutsearch.htm).
_SEARCH_DEPTH = 10_000
_MAX_RESULTS = 20
_MAX_CHARS = 20_000
_DEFAULT_CHARS = 6_000
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
# What a search result shows of each list; the item shows them whole.
_SHOWN = {"creator": 3, "collections": 3, "subjects": 5}
# Extended error codes of the metadata API (``extended_err=1``): deleted, and the ones that pass.
_DELETED = 104
_PASSING = frozenset({101, 102, 105})  # creation pending, unavailable, primary node not found


def _said(reply: Reply) -> str:
    """The ``error`` text an answer carries in place of the result; empty when it has none."""
    error = reply.body.get("error") if isinstance(reply.body, dict) else None
    return " ".join(str(error).split()) if error else ""


def _search_error(reply: Reply) -> str | None:
    """The advanced search's error, in its words, with what to check.

    It answers a query it cannot run with an ``error`` text instead of the result, with HTTP 200
    (seen 2026-09-29); a server error says only what it said (its status makes it retryable).
    """
    said = _said(reply)
    if not said:
        return None
    if reply.status >= 500:
        return said
    return (
        f"the Internet Archive's search could not run the query: {said}; check its syntax "
        "(field:value, AND, OR, quotes, parentheses)"
    )


def _metadata_error(reply: Reply) -> ToolFailure | str | None:
    """The metadata API's error, typed by its extended code when it gives one.

    It answers an error with a human-readable ``error``, and with ``extended_err=1`` an
    ``errcode`` beside it (https://archive.org/developers/md-read.html): 104 is a deleted item;
    101, 102 and 105 an item that cannot be read for now; any other is the source's words.
    """
    said = _said(reply)
    if not said:
        return None
    code = reply.body.get("errcode") if isinstance(reply.body, dict) else None
    if code == _DELETED:
        return ToolFailure(
            "not_found",
            f"the Internet Archive deleted the item ({said}); find another with "
            "internet_archive_search",
        )
    if code in _PASSING:
        return ToolFailure(
            "upstream", f"the item cannot be read now ({said}); try again later", retryable=True
        )
    return said


_SEARCH = Api(
    base="https://archive.org/advancedsearch.php",
    name="Internet Archive",
    timeout_s=15,
    error_reader=_search_error,
)
_METADATA = Api(
    base="https://archive.org/metadata",
    name="Internet Archive",
    timeout_s=15,
    error_reader=_metadata_error,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Item:
    """An item's metadata, from a search result or the metadata API."""

    identifier: str
    title: str
    creator: tuple[str, ...]
    date: str
    mediatype: str
    collection: tuple[str, ...]
    subjects: tuple[str, ...]
    downloads: int | None
    item_size: int | None
    description: str = ""
    files: tuple[str, ...] = ()
    files_count: int | None = None


@tool(capability="network")
def internet_archive_search(
    query: str,
    max_results: Annotated[int, Range(1, _MAX_RESULTS)] = 5,
    page: Annotated[int, Range(1, _SEARCH_DEPTH)] = 1,
    mediatype: str = "",
    collection: str = "",
) -> ToolResult:
    """Search Internet Archive items using the public advanced search API.

    Args:
        query: Search text, in the archive's syntax: words, "a phrase", or field:value
            (title:, creator:, subject:, date:[1900 TO 1950]).
        max_results: How many items a page has.
        page: Which page of results, from 1; the footer gives the next.
        mediatype: Optional mediatype filter, e.g. texts, audio, movies, software.
        collection: Optional collection filter, by its identifier.

    Raises:
        ToolFailure: validation_error when ``query`` is empty or the page starts past the
            10,000th result, the deepest the search pages; upstream when the archive cannot run
            the query.
    """
    query = query.strip()
    if not query:
        raise ToolFailure("validation_error", "query cannot be empty; pass the text to search")
    if (first := (page - 1) * max_results + 1) > _SEARCH_DEPTH:
        raise ToolFailure(
            "validation_error",
            f"page {page} starts at result {first}, past the first {_SEARCH_DEPTH} the Internet "
            "Archive's search pages; narrow the query (mediatype, collection, more words)",
        )

    filters = {"mediatype": mediatype.strip(), "collection": collection.strip()}
    search_query = " AND ".join(
        [f"({query})", *(f"{name}:{value}" for name, value in filters.items() if value)]
    )
    shown = ", ".join(f"{name}: {value}" for name, value in filters.items() if value)
    wanted = f"{query!r}" + (f" ({shown})" if shown else "")
    params = {
        "q": search_query,
        "fl[]": [
            "identifier",
            "title",
            "creator",
            "date",
            "mediatype",
            "collection",
            "subject",
            "downloads",
            "item_size",
        ],
        "rows": str(max_results),
        "page": str(page),
        "output": "json",
    }
    return _SEARCH.get_json(
        params=params, parse=lambda data: _search_answer(data, wanted, page, max_results)
    )


@tool(capability="network")
def internet_archive_item(
    identifier: str,
    find: str = "",
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, _MAX_CHARS)] = _DEFAULT_CHARS,
) -> ToolResult:
    """Fetch an Internet Archive item's metadata and its files, whole.

    Args:
        identifier: The item's identifier, as internet_archive_search returns it.
        find: A term to look for (a file name, a format): the answer is the passages around each
            match, not the whole record.
        offset: Where to start, in characters of the record (with ``find``, where to search on
            from); the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when ``identifier`` is not an identifier; not_found when
            the Internet Archive has no such item, or deleted it; upstream when it cannot read it.
    """
    normalized = identifier.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ToolFailure(
            "validation_error",
            f"invalid identifier {identifier!r}; an identifier has only letters, digits and "
            "_.- (internet_archive_search returns them)",
        )

    missing = (
        f"no Internet Archive item {normalized!r}; find its identifier with "
        "internet_archive_search"
    )
    # An unknown identifier is an empty answer with HTTP 200 (``_item_of``); a 404 says the same.
    item = _METADATA.get_json(
        normalized, params={"extended_err": "1"}, parse=_item_of, missing=missing
    )
    if item is None:
        raise ToolFailure("not_found", missing)

    text, term = _item_text(item), find.strip()
    if term:
        window = find_window(text, term, offset=offset, limit=max_chars)
        heading = f"Internet Archive item {normalized}, passages that mention {term!r}:"
    else:
        window = text_window(text, offset=offset, limit=max_chars)
        heading = f"Internet Archive item {normalized}:"
    return window.result(heading=heading)


# --- The search --------------------------------------------------------------------------------


def _search_answer(data: dict[str, Any], wanted: str, page: int, rows: int) -> ToolResult:
    found = data.get("response", {})
    docs = [doc for doc in found.get("docs", []) if isinstance(doc, dict)]
    items = [item for item in map(_search_item, docs) if item is not None]
    total = found.get("numFound") if isinstance(found.get("numFound"), int) else None
    start = found.get("start") if isinstance(found.get("start"), int) else (page - 1) * rows
    if not items and page == 1:
        return ToolResult.success(f"No Internet Archive items match {wanted}.")
    lines = [_search_lines(start + number, item) for number, item in enumerate(items, start=1)]
    last = start + len(docs)
    more = last < total if total is not None else len(docs) == rows
    deep = more and page * rows >= _SEARCH_DEPTH  # the next page starts past the depth
    window = list_window(
        lines,
        first=start + 1,
        total=total,
        next_call={"page": page + 1} if more and not deep else None,
    )
    note = (
        f" (the search pages only its first {_SEARCH_DEPTH} results: narrow the query for the "
        "rest)"
        if deep
        else ""
    )
    return window.result(heading=f"Internet Archive items that match {wanted}{note}:")


def _search_item(data: dict[str, Any]) -> _Item | None:
    identifier = _string(data.get("identifier"))
    if not identifier:
        return None
    return _Item(
        identifier=identifier,
        title=_string(data.get("title")) or "(untitled)",
        creator=_string_tuple(data.get("creator")),
        date=_date(_string(data.get("date"))),
        mediatype=_string(data.get("mediatype")),
        collection=_string_tuple(data.get("collection")),
        subjects=_string_tuple(data.get("subject")),
        downloads=_int_or_none(data.get("downloads")),
        item_size=_int_or_none(data.get("item_size")),
    )


def _search_lines(number: int, item: _Item) -> str:
    """A result: its title, then its facts, then the start of its lists, each saying how many
    more the item has."""
    facts = [f"identifier: {item.identifier}"]
    facts += [f"{name}: {value}" for name, value in _facts(item)]
    lists = [
        f"{name}: {_some(values, _SHOWN[name])}"
        for name, values in (
            ("creator", item.creator),
            ("collections", item.collection),
            ("subjects", item.subjects),
        )
        if values
    ]
    lines = [f"{number}. {item.title}", "   " + " | ".join(facts)]
    if lists:
        lines.append("   " + " | ".join(lists))
    return "\n".join(lines)


def _facts(item: _Item) -> list[tuple[str, str]]:
    facts = [("mediatype", item.mediatype), ("date", item.date)]
    if item.downloads is not None:
        facts.append(("downloads", str(item.downloads)))
    if item.item_size is not None:
        facts.append(("size", f"{item.item_size} bytes"))
    return [(name, value) for name, value in facts if value]


def _some(values: tuple[str, ...], shown: int) -> str:
    more = len(values) - shown
    return ", ".join(values[:shown]) + (f" (+{more} more)" if more > 0 else "")


# --- The item ----------------------------------------------------------------------------------


def _item_of(data: dict[str, Any]) -> _Item | None:
    """The item of a metadata answer; ``None`` for the empty answer of an unknown identifier."""
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return None
    identifier = _string(metadata.get("identifier") or data.get("item"))
    if not identifier:
        return None
    files = tuple(text for text in map(_file_line, data.get("files", []) or []) if text)
    return _Item(
        identifier=identifier,
        title=_string(metadata.get("title")) or "(untitled)",
        creator=_string_tuple(metadata.get("creator")),
        date=_date(_string(metadata.get("date"))),
        mediatype=_string(metadata.get("mediatype")),
        collection=_string_tuple(metadata.get("collection")),
        subjects=_string_tuple(metadata.get("subject")),
        downloads=None,
        item_size=_int_or_none(data.get("item_size")),
        description=_string(metadata.get("description")),
        files=files,
        files_count=_int_or_none(data.get("files_count")),
    )


def _file_line(entry: object) -> str:
    if not isinstance(entry, dict):
        return ""
    name, fmt, size = (_string(entry.get(key)) for key in ("name", "format", "size"))
    text = name + (f" ({fmt})" if fmt else "")
    return text + (f", {size} bytes" if size else "") if text else ""


def _item_text(item: _Item) -> str:
    """The whole record: the facts first, then the description and the lists, nothing cut."""
    facts = [f"{name.capitalize()}: {value}" for name, value in _facts(item)]
    count = item.files_count if item.files_count is not None else len(item.files)
    lines = [
        f"Title: {item.title}",
        " | ".join([*facts, f"Files: {count}"]),
        f"Page: https://archive.org/details/{item.identifier}",
    ]
    for name, values in (
        ("Creator", item.creator),
        ("Collections", item.collection),
        ("Subjects", item.subjects),
    ):
        if values:
            lines.append(f"{name}: {', '.join(values)}")
    if item.description:
        lines.append(f"Description: {item.description}")
    if item.files:
        where = f"https://archive.org/download/{item.identifier}/<name>"
        lines.append(f"Files (download one at {where}):")
        lines += [f"- {file}" for file in item.files]
    return "\n".join(lines)


# --- Values ------------------------------------------------------------------------------------


def _date(text: str) -> str:
    """An ISO 8601 date: the search gives midnight UTC timestamps (``1888-01-01T00:00:00Z``)."""
    return text.removesuffix("T00:00:00Z")


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(_string(item) for item in value if _string(item))
    text = _string(value)
    return (text,) if text else ()


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_string(item) for item in value if _string(item))
    if isinstance(value, dict):
        for key in ("value", "text", "description"):
            text = _string(value.get(key))
            if text:
                return text
        return ""
    return " ".join(str(value).split())


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    try:
        if value is not None and str(value).strip():
            return int(value)
    except ValueError:
        return None
    return None
