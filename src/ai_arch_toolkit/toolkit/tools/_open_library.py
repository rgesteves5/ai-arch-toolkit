"""Open Library tools: the book search, with its total, and whole work and edition records, with
their authors' names, read window by window (T09; D39).

The search pages by ``limit`` and ``offset`` and counts its results in ``numFound``
(https://openlibrary.org/dev/docs/api/search). A work or an edition names its authors by key only
(``/authors/OL…A``); one request to the author search, ``key:(… OR …)``, names them all
(https://openlibrary.org/dev/docs/api/authors). A caller that sends no email may make one request a
second (https://openlibrary.org/developers/api).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api
from ai_arch_toolkit.toolkit.tools._window import list_window, text_window

_API = Api(base="https://openlibrary.org", name="Open Library", min_interval_s=1.0)
_MAX_RESULTS = 20
_MAX_CHARS = 20_000
_DEFAULT_CHARS = 6_000
_WORK_ID_RE = re.compile(r"^OL\d+W$", re.IGNORECASE)
_RECORD_KEY_RE = re.compile(r"^/(works|books)/(OL\d+[WM])$")
_AUTHOR_KEY_RE = re.compile(r"^/authors/(OL\d+A)$")
_MAX_REDIRECTS = 3
# The authors one request names; a record with more lists the rest by key.
_AUTHOR_NAMES = 50
# The fields a search result shows (https://openlibrary.org/dev/docs/api/search, ``fields``).
_SEARCH_FIELDS = (
    "key,title,author_name,author_key,first_publish_year,edition_count,language,publisher,isbn,"
    "ebook_access"
)
# What a search result shows of each list; ``(+N more)`` says the rest.
_SHOWN = 3
# How Open Library writes dates, and the ISO 8601 form of each.
_DATE_FORMATS = (
    ("%Y-%m-%d", "%Y-%m-%d"),
    ("%B %d, %Y", "%Y-%m-%d"),
    ("%b %d, %Y", "%Y-%m-%d"),
    ("%d %B %Y", "%Y-%m-%d"),
    ("%B %Y", "%Y-%m"),
    ("%b %Y", "%Y-%m"),
    ("%Y", "%Y"),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Book:
    """A search result, a work or an edition, as the tools show it.

    Attributes:
        authors: Each author as shown: ``Name (OL…A)``.
        author_keys: A record's authors by key (``OL…A``), until their names are read.
        works: The works an edition belongs to, by ID (``OL…W``).
    """

    key: str
    title: str
    authors: tuple[str, ...] = ()
    author_keys: tuple[str, ...] = ()
    first_published: str = ""
    edition_count: int | None = None
    publishers: tuple[str, ...] = ()
    published: str = ""
    languages: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()
    isbn_13: tuple[str, ...] = ()
    isbn_10: tuple[str, ...] = ()
    cover_id: int | None = None
    ebook_access: str = ""
    pages: int | None = None
    works: tuple[str, ...] = ()
    description: str = ""
    links: tuple[str, ...] = ()


@tool(capability="network")
def open_library_search(
    query: str,
    max_results: Annotated[int, Range(1, _MAX_RESULTS)] = 5,
    start: Annotated[int, Range(0)] = 0,
    title: str = "",
    author: str = "",
    subject: str = "",
    isbn: str = "",
) -> ToolResult:
    """Search books and works using the public Open Library API.

    Args:
        query: General search text, in Open Library's search syntax: words, or field:value
            (``author_key:OL26320A`` lists an author's books, by the key the results show).
        max_results: How many books to return.
        start: How many results to skip; the footer gives the next start.
        title: Optional title-specific search.
        author: Optional author-specific search.
        subject: Optional subject-specific search.
        isbn: Optional ISBN-specific search.

    Raises:
        ToolFailure: validation_error when no search field is given.
    """
    fields = {"q": query, "title": title, "author": author, "subject": subject, "isbn": isbn}
    filters = {key: value.strip() for key, value in fields.items() if value.strip()}
    if not filters:
        raise ToolFailure(
            "validation_error", "nothing to search; provide query, title, author, subject, or isbn"
        )

    params = {
        "limit": str(max_results),
        "offset": str(start),
        "fields": _SEARCH_FIELDS,
        **filters,
    }
    wanted = ", ".join(
        f"{'query' if key == 'q' else key} {value!r}" for key, value in filters.items()
    )
    return _API.get_json(
        "search.json",
        params=params,
        parse=lambda data: _search_answer(data, wanted, start, max_results),
    )


@tool(capability="network")
def open_library_work(
    work_id: str,
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, _MAX_CHARS)] = _DEFAULT_CHARS,
) -> ToolResult:
    """Fetch an Open Library work, whole: its authors by name, description, subjects and links.

    Args:
        work_id: Open Library work ID or URL, e.g. "OL27448W" or "/works/OL27448W".
        offset: Where to start, in characters of the record; the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when ``work_id`` is not a work ID; not_found when Open
            Library has no such work, or deleted it.
    """
    normalized = _normalize_work_id(work_id)
    if not normalized:
        raise ToolFailure(
            "validation_error",
            f"invalid work_id {work_id!r}; a work ID looks like OL27448W "
            "(open_library_search returns them)",
        )

    book = _record(
        "works",
        normalized,
        _parse_work,
        missing=f"Open Library has no work {normalized}; find works with open_library_search",
    )

    heading = normalized
    if book.key.startswith("/works/") and book.key != f"/works/{normalized}":
        heading += f" (merged into {book.key.removeprefix('/works/')})"
    window = text_window(_work_text(_named(book)), offset=offset, limit=max_chars)
    return window.result(heading=f"Open Library work {heading}:")


@tool(capability="network")
def open_library_isbn(
    isbn: str,
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, _MAX_CHARS)] = _DEFAULT_CHARS,
) -> ToolResult:
    """Fetch the Open Library edition with an ISBN, whole: its authors by name, publishers,
    date, ISBNs and work.

    Args:
        isbn: ISBN-10 or ISBN-13 string.
        offset: Where to start, in characters of the record; the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when ``isbn`` is not an ISBN-10 or ISBN-13; not_found
            when Open Library has no edition with it, or deleted it.
    """
    normalized = _normalize_isbn(isbn)
    if not normalized:
        raise ToolFailure(
            "validation_error",
            f"invalid ISBN {isbn!r}; an ISBN has 10 characters (digits, the last may be X) "
            "or 13 digits",
        )

    book = _record(
        "isbn",
        normalized,
        _parse_edition,
        missing=(
            f"Open Library has no edition with ISBN {normalized}; find books with "
            "open_library_search"
        ),
    )

    window = text_window(_edition_text(_named(book)), offset=offset, limit=max_chars)
    return window.result(heading=f"Open Library ISBN {normalized}:")


# --- Records -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Merged:
    """A record merged into another: Open Library keeps it as a redirect to ``segments``."""

    segments: tuple[str, str]


def _record(
    section: str,
    identifier: str,
    parse: Callable[[dict[str, Any]], _Book | None],
    *,
    missing: str,
) -> _Book:
    """The record at ``/{section}/{identifier}.json``, followed through merges.

    Open Library keeps a merged record as a ``/type/redirect`` to the one it went into, and a
    deleted one as ``/type/delete`` (https://openlibrary.org/type/redirect), both with HTTP 200
    (seen 2026-09-30); read as books, they had no title.

    Raises:
        ToolFailure: not_found, saying ``missing``, for a 404 or a record without a title or
            key; not_found for a deleted record; upstream for a chain of too many redirects.
    """
    segments = (section, f"{identifier}.json")
    for _ in range(_MAX_REDIRECTS + 1):
        record = _API.get_json(
            *segments, parse=lambda data: _live_record(data, parse), missing=missing
        )
        if record is None:
            raise ToolFailure("not_found", missing)
        if not isinstance(record, _Merged):
            return record
        segments = record.segments
    raise ToolFailure(
        "upstream", f"more than {_MAX_REDIRECTS} redirects between Open Library records"
    )


def _live_record(
    data: dict[str, Any], parse: Callable[[dict[str, Any]], _Book | None]
) -> _Book | _Merged | None:
    kind = _string(data.get("type", {}).get("key"))
    if kind == "/type/delete":
        raise ToolFailure(
            "not_found",
            f"Open Library deleted {_string(data.get('key')) or 'the record'}; "
            "find another with open_library_search",
        )
    if kind != "/type/redirect":
        return parse(data)
    target = _RECORD_KEY_RE.fullmatch(_string(data.get("location")))
    if target is None:
        raise ToolFailure(
            "upstream", "could not parse API response: a redirect without a record to go to"
        )
    return _Merged((target.group(1), f"{target.group(2)}.json"))


def _parse_work(data: dict[str, Any]) -> _Book | None:
    title, key = _string(data.get("title")), _string(data.get("key"))
    if not title and not key:
        return None
    roles = [
        item.get("author") for item in data.get("authors", []) or [] if isinstance(item, dict)
    ]
    return _Book(
        key=key,
        title=title or "(untitled)",
        author_keys=_author_keys(roles),
        first_published=_iso_date(_string(data.get("first_publish_date"))),
        subjects=_string_tuple(data.get("subjects")),
        cover_id=_first_int(data.get("covers")),
        description=_description(data.get("description")),
        links=_links(data),
    )


def _parse_edition(data: dict[str, Any]) -> _Book | None:
    title, key = _string(data.get("title")), _string(data.get("key"))
    if not title and not key:
        return None
    works = (_string(item.get("key")) for item in _dicts(data.get("works")))
    return _Book(
        key=key,
        title=title or "(untitled)",
        author_keys=_author_keys(data.get("authors")),
        publishers=_string_tuple(data.get("publishers")),
        published=_iso_date(_string(data.get("publish_date"))),
        languages=tuple(
            _string(item.get("key")).removeprefix("/languages/")
            for item in _dicts(data.get("languages"))
            if _string(item.get("key"))
        ),
        subjects=_string_tuple(data.get("subjects")),
        isbn_13=_string_tuple(data.get("isbn_13")),
        isbn_10=_string_tuple(data.get("isbn_10")),
        cover_id=_first_int(data.get("covers")),
        pages=_int_or_none(data.get("number_of_pages")),
        works=tuple(work.removeprefix("/works/") for work in works if work.startswith("/works/")),
        description=_description(data.get("description")),
    )


def _author_keys(entries: object) -> tuple[str, ...]:
    """The ``OL…A`` keys of a record's ``{"key": "/authors/OL…A"}`` entries, in order."""
    keys = (_AUTHOR_KEY_RE.fullmatch(_string(entry.get("key"))) for entry in _dicts(entries))
    return tuple(dict.fromkeys(match.group(1) for match in keys if match is not None))


def _named(book: _Book) -> _Book:
    """``book`` with its authors' names, read in one request for the first ``_AUTHOR_NAMES``; an
    author the search does not know keeps its key, and says so."""
    if not book.author_keys:
        return book
    asked = book.author_keys[:_AUTHOR_NAMES]
    params = {
        "q": "key:(" + " OR ".join(f"/authors/{key}" for key in asked) + ")",
        "fields": "key,name",
        "limit": str(len(asked)),
    }
    names = _API.get_json("search", "authors.json", params=params, parse=_names)
    shown = tuple(
        f"{names[key]} ({key})"
        if key in names
        else key + (" (no name on Open Library)" if key in asked else "")
        for key in book.author_keys
    )
    return replace(book, authors=shown, author_keys=())


def _names(data: dict[str, Any]) -> dict[str, str]:
    """The name of each author an author search found, by bare key (``OL23919A``)."""
    found = (
        (_string(doc.get("key")).removeprefix("/authors/"), _string(doc.get("name")))
        for doc in _dicts(data.get("docs"))
    )
    return {key: name for key, name in found if key and name}


def _work_text(book: _Book) -> str:
    lines = [
        f"Title: {book.title}",
        *_labelled("Authors", book.authors),
        *_facts(("First published", book.first_published)),
        f"Page: https://openlibrary.org{book.key}" if book.key else "",
        *_facts(("Cover", _cover_url(book.cover_id)), ("Description", book.description)),
        f"Subjects ({len(book.subjects)}): {', '.join(book.subjects)}" if book.subjects else "",
        *_facts(("Links", " | ".join(book.links))),
    ]
    return "\n".join(line for line in lines if line)


def _edition_text(book: _Book) -> str:
    pages = str(book.pages) if book.pages is not None else ""
    isbns = (("ISBN-13", ", ".join(book.isbn_13)), ("ISBN-10", ", ".join(book.isbn_10)))
    works = ", ".join(book.works)
    lines = [
        f"Title: {book.title}",
        *_labelled("Authors", book.authors),
        " | ".join(_facts(("Published", book.published), ("Pages", pages))),
        *_labelled("Publishers", book.publishers),
        *_labelled("Languages", book.languages),
        " | ".join(_facts(*isbns)),
        f"Work{'s' if len(book.works) > 1 else ''}: {works} (open_library_work reads it)"
        if works
        else "",
        f"Page: https://openlibrary.org{book.key}" if book.key else "",
        *_facts(("Cover", _cover_url(book.cover_id)), ("Description", book.description)),
        f"Subjects ({len(book.subjects)}): {', '.join(book.subjects)}" if book.subjects else "",
    ]
    return "\n".join(line for line in lines if line)


def _labelled(label: str, values: tuple[str, ...]) -> list[str]:
    return [f"{label}: {', '.join(values)}"] if values else []


def _facts(*pairs: tuple[str, str]) -> list[str]:
    return [f"{label}: {value}" for label, value in pairs if value]


# --- The search --------------------------------------------------------------------------------


def _search_answer(data: dict[str, Any], wanted: str, start: int, rows: int) -> ToolResult:
    docs = list(_dicts(data.get("docs")))
    counted = (_int_or_none(data.get(name)) for name in ("numFound", "num_found"))
    total = next((value for value in counted if value is not None), None)
    reported = _int_or_none(data.get("start"))
    first = start if reported is None else reported
    if not docs and start == 0:
        return ToolResult.success(f"No Open Library books match {wanted}.")
    books = [_search_book(doc) for doc in docs]
    lines = [_search_lines(first + number, book) for number, book in enumerate(books, start=1)]
    last = first + len(docs)
    more = last < total if total is not None else len(docs) == rows
    window = list_window(
        lines, first=first + 1, total=total, next_call={"start": last} if more else None
    )
    return window.result(heading=f"Open Library books that match {wanted}:")


def _search_book(data: dict[str, Any]) -> _Book:
    names, keys = _string_tuple(data.get("author_name")), _string_tuple(data.get("author_key"))
    authors = tuple(
        f"{name} ({keys[index]})" if index < len(keys) else name
        for index, name in enumerate(names)
    )
    year = _int_or_none(data.get("first_publish_year"))
    return _Book(
        key=_string(data.get("key")),
        title=_string(data.get("title")) or "(untitled)",
        authors=authors,
        first_published=str(year) if year is not None else "",
        edition_count=_int_or_none(data.get("edition_count")),
        publishers=_string_tuple(data.get("publisher")),
        languages=_string_tuple(data.get("language")),
        isbn_13=_string_tuple(data.get("isbn")),
        ebook_access=_string(data.get("ebook_access")),
    )


def _search_lines(number: int, book: _Book) -> str:
    """A result: its title and work ID, its facts, then the start of its long lists, each
    saying how many more it has."""
    work = book.key.removeprefix("/works/")
    title = f"{number}. {book.title}" + (
        f" (work {work})" if book.key.startswith("/works/") else ""
    )
    editions = f"{book.edition_count} editions" if book.edition_count is not None else ""
    facts = [
        f"by {', '.join(book.authors)}" if book.authors else "",
        f"first published {book.first_published}" if book.first_published else "",
        editions,
        f"languages: {', '.join(book.languages)}" if book.languages else "",
        f"ebook: {book.ebook_access}" if book.ebook_access else "",
    ]
    lists = _facts(("ISBN", _some(book.isbn_13)), ("publishers", _some(book.publishers)))
    lines = [title, "   " + " | ".join(fact for fact in facts if fact)]
    if lists:
        lines.append("   " + " | ".join(lists))
    return "\n".join(line for line in lines if line.strip())


def _some(values: tuple[str, ...]) -> str:
    more = len(values) - _SHOWN
    return ", ".join(values[:_SHOWN]) + (f" (+{more} more)" if more > 0 else "")


# --- Values ------------------------------------------------------------------------------------


def _normalize_work_id(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith("https://openlibrary.org/works/"):
        raw = raw.removeprefix("https://openlibrary.org/works/")
    elif raw.startswith("http://openlibrary.org/works/"):
        raw = raw.removeprefix("http://openlibrary.org/works/")
    elif raw.startswith("/works/"):
        raw = raw.removeprefix("/works/")
    if raw.endswith(".json"):
        raw = raw[:-5]
    raw = raw.strip("/")
    return raw.upper() if _WORK_ID_RE.fullmatch(raw) else ""


def _normalize_isbn(value: str) -> str:
    raw = value.strip().replace("-", "").replace(" ", "")
    if len(raw) == 10 and raw[:-1].isdigit() and (raw[-1].isdigit() or raw[-1].upper() == "X"):
        return raw.upper()
    if len(raw) == 13 and raw.isdigit():
        return raw
    return ""


def _iso_date(text: str) -> str:
    """``text`` in ISO 8601 when it is a date Open Library writes ("October 1, 1988" is
    1988-10-01, "September 1970" 1970-09); anything else as written."""
    for written, iso in _DATE_FORMATS:
        try:
            return datetime.strptime(text, written).strftime(iso)
        except ValueError:
            continue
    return text


def _description(value: Any) -> str:
    if isinstance(value, dict):
        return _string(value.get("value"))
    return _string(value)


def _links(data: dict[str, Any]) -> tuple[str, ...]:
    links: list[str] = []
    for item in _dicts(data.get("links")):
        title, url = _string(item.get("title")), _string(item.get("url"))
        if url:
            links.append(f"{title}: {url}" if title else url)
    return tuple(links)


def _cover_url(cover_id: int | None) -> str:
    if cover_id is None or cover_id < 0:
        return ""
    return f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"


def _dicts(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(_string(item) for item in value if _string(item))


def _first_int(value: Any) -> int | None:
    if isinstance(value, list):
        for item in value:
            parsed = _int_or_none(item)
            if parsed is not None:
                return parsed
    return _int_or_none(value)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
