"""Open Library tools — public book, work, and ISBN lookup."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://openlibrary.org"
_TIMEOUT = 10
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_RESULTS_LIMIT = 20
_DESCRIPTION_MAX_CHARS = 1000
_WORK_ID_RE = re.compile(r"^OL\d+W$", re.IGNORECASE)


@dataclass(frozen=True, slots=True, kw_only=True)
class _OpenLibraryBook:
    """Normalized metadata for an Open Library search result or edition."""

    key: str
    title: str
    authors: tuple[str, ...]
    first_publish_year: int | None
    edition_count: int | None
    publishers: tuple[str, ...]
    publish_date: str
    languages: tuple[str, ...]
    subjects: tuple[str, ...]
    isbn: tuple[str, ...]
    cover_id: int | None
    ebook_access: str
    has_fulltext: bool | None
    pages: int | None
    works: tuple[str, ...]
    description: str
    links: tuple[str, ...]


@tool
def open_library_search(
    query: str,
    max_results: int = 5,
    start: int = 0,
    title: str = "",
    author: str = "",
    subject: str = "",
    isbn: str = "",
) -> str:
    """Search books and works using the public Open Library API.

    Args:
        query: General search text.
        max_results: Number of books to return (1-20). Defaults to 5.
        start: Zero-based result offset for pagination. Defaults to 0.
        title: Optional title-specific search.
        author: Optional author-specific search.
        subject: Optional subject-specific search.
        isbn: Optional ISBN-specific search.
    """
    if start < 0:
        return "Open Library search failed: start must be greater than or equal to 0."
    if not any(value.strip() for value in (query, title, author, subject, isbn)):
        return "Open Library search failed: provide query, title, author, subject, or isbn."

    max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))
    params = {
        "limit": str(max_results),
        "offset": str(start),
    }
    if query.strip():
        params["q"] = query.strip()
    if title.strip():
        params["title"] = title.strip()
    if author.strip():
        params["author"] = author.strip()
    if subject.strip():
        params["subject"] = subject.strip()
    if isbn.strip():
        params["isbn"] = isbn.strip()

    try:
        data = _fetch_json("/search.json", params)
        docs = data.get("docs", [])
        books = [_parse_search_doc(item) for item in docs if isinstance(item, dict)]
        books = [book for book in books if book is not None]
    except urllib.error.HTTPError as e:
        return f"Open Library search failed: HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"Open Library search failed: URL error: {e.reason}"
    except TimeoutError:
        return "Open Library search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Open Library search failed: could not parse API response: {e}"

    if not books:
        return "No Open Library results found."

    return "Open Library results:\n" + _format_books(books, include_description=False)


@tool
def open_library_work(work_id: str) -> str:
    """Fetch Open Library metadata for a work.

    Args:
        work_id: Open Library work ID or URL, e.g. "OL27448W" or "/works/OL27448W".
    """
    normalized = _normalize_work_id(work_id)
    if not normalized:
        return f"Open Library work lookup failed: invalid work_id: {work_id!r}"

    try:
        data = _fetch_json(f"/works/{normalized}.json", {})
        book = _parse_work(data)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"Open Library work not found: {normalized}"
        return f"Open Library work lookup failed: HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"Open Library work lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "Open Library work lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Open Library work lookup failed: could not parse API response: {e}"

    if book is None:
        return f"Open Library work not found: {normalized}"

    return f"Open Library work {normalized}:\n" + _format_books(
        [book],
        include_index=False,
        include_description=True,
    )


@tool
def open_library_isbn(isbn: str) -> str:
    """Fetch Open Library edition metadata for an ISBN.

    Args:
        isbn: ISBN-10 or ISBN-13 string.
    """
    normalized = _normalize_isbn(isbn)
    if not normalized:
        return f"Open Library ISBN lookup failed: invalid ISBN: {isbn!r}"

    try:
        data = _fetch_json(f"/isbn/{normalized}.json", {})
        book = _parse_isbn(data)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"Open Library ISBN not found: {normalized}"
        return f"Open Library ISBN lookup failed: HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"Open Library ISBN lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "Open Library ISBN lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Open Library ISBN lookup failed: could not parse API response: {e}"

    if book is None:
        return f"Open Library ISBN not found: {normalized}"

    return f"Open Library ISBN {normalized}:\n" + _format_books(
        [book],
        include_index=False,
        include_description=True,
    )


def _fetch_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{_BASE_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _parse_search_doc(data: dict[str, Any]) -> _OpenLibraryBook | None:
    title = _string(data.get("title"))
    key = _string(data.get("key"))
    if not title and not key:
        return None

    return _OpenLibraryBook(
        key=key,
        title=title or "(untitled)",
        authors=_string_tuple(data.get("author_name")),
        first_publish_year=_int_or_none(data.get("first_publish_year")),
        edition_count=_int_or_none(data.get("edition_count")),
        publishers=_string_tuple(data.get("publisher")),
        publish_date="",
        languages=_string_tuple(data.get("language")),
        subjects=_string_tuple(data.get("subject")),
        isbn=_string_tuple(data.get("isbn")),
        cover_id=_int_or_none(data.get("cover_i")),
        ebook_access=_string(data.get("ebook_access")),
        has_fulltext=_bool_or_none(data.get("has_fulltext")),
        pages=None,
        works=(key,) if key.startswith("/works/") else (),
        description="",
        links=(),
    )


def _parse_work(data: dict[str, Any]) -> _OpenLibraryBook | None:
    title = _string(data.get("title"))
    key = _string(data.get("key"))
    if not title and not key:
        return None

    authors: list[str] = []
    for item in data.get("authors", []) or []:
        if not isinstance(item, dict):
            continue
        author = item.get("author")
        if isinstance(author, dict):
            author_key = _string(author.get("key"))
            if author_key:
                authors.append(author_key)

    return _OpenLibraryBook(
        key=key,
        title=title or "(untitled)",
        authors=tuple(authors),
        first_publish_year=None,
        edition_count=None,
        publishers=(),
        publish_date=_string(data.get("first_publish_date")),
        languages=(),
        subjects=_string_tuple(data.get("subjects")),
        isbn=(),
        cover_id=_first_int(data.get("covers")),
        ebook_access="",
        has_fulltext=None,
        pages=None,
        works=(key,) if key else (),
        description=_description(data.get("description")),
        links=_links(data),
    )


def _parse_isbn(data: dict[str, Any]) -> _OpenLibraryBook | None:
    title = _string(data.get("title"))
    key = _string(data.get("key"))
    if not title and not key:
        return None

    authors = []
    for item in data.get("authors", []) or []:
        if isinstance(item, dict):
            author_key = _string(item.get("key"))
            if author_key:
                authors.append(author_key)

    works = []
    for item in data.get("works", []) or []:
        if isinstance(item, dict):
            work_key = _string(item.get("key"))
            if work_key:
                works.append(work_key)

    return _OpenLibraryBook(
        key=key,
        title=title or "(untitled)",
        authors=tuple(authors),
        first_publish_year=None,
        edition_count=None,
        publishers=_string_tuple(data.get("publishers")),
        publish_date=_string(data.get("publish_date")),
        languages=_language_keys(data.get("languages")),
        subjects=_string_tuple(data.get("subjects")),
        isbn=tuple(
            dict.fromkeys(
                list(_string_tuple(data.get("isbn_13"))) + list(_string_tuple(data.get("isbn_10")))
            )
        ),
        cover_id=_first_int(data.get("covers")),
        ebook_access="",
        has_fulltext=None,
        pages=_int_or_none(data.get("number_of_pages")),
        works=tuple(works),
        description=_description(data.get("description")),
        links=(),
    )


def _format_books(
    books: list[_OpenLibraryBook],
    *,
    include_index: bool = True,
    include_description: bool = False,
) -> str:
    blocks: list[str] = []
    for index, book in enumerate(books, start=1):
        title = f"{index}. {book.title}" if include_index else book.title
        lines = [title]

        meta: list[str] = []
        if book.key:
            meta.append(f"key: {book.key}")
        if book.first_publish_year is not None:
            meta.append(f"first published: {book.first_publish_year}")
        if book.publish_date:
            meta.append(f"published: {book.publish_date}")
        if book.edition_count is not None:
            meta.append(f"editions: {book.edition_count}")
        if book.pages is not None:
            meta.append(f"pages: {book.pages}")
        if meta:
            lines.append("   " + " | ".join(meta))

        if book.authors:
            lines.append("   Authors: " + ", ".join(book.authors[:8]))
        if book.publishers:
            lines.append("   Publishers: " + ", ".join(book.publishers[:5]))
        if book.isbn:
            lines.append("   ISBN: " + ", ".join(book.isbn[:8]))
        if book.languages:
            lines.append("   Languages: " + ", ".join(book.languages[:12]))
        if book.subjects:
            lines.append("   Subjects: " + ", ".join(book.subjects[:12]))
        if book.works:
            lines.append("   Works: " + ", ".join(book.works[:5]))
        if book.ebook_access:
            lines.append(f"   Ebook access: {book.ebook_access}")
        if book.has_fulltext is not None:
            lines.append(f"   Has fulltext: {book.has_fulltext}")
        if include_description and book.description:
            lines.append(f"   Description: {_truncate(book.description, _DESCRIPTION_MAX_CHARS)}")
        cover_url = _cover_url(book.cover_id)
        if cover_url:
            lines.append(f"   Cover: {cover_url}")
        if book.links:
            lines.append("   Links: " + " | ".join(book.links[:5]))
        if book.key:
            lines.append(f"   URL: https://openlibrary.org{book.key}")

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


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


def _description(value: Any) -> str:
    if isinstance(value, dict):
        return _string(value.get("value"))
    return _string(value)


def _links(data: dict[str, Any]) -> tuple[str, ...]:
    links: list[str] = []
    for item in data.get("links", []) or []:
        if not isinstance(item, dict):
            continue
        title = _string(item.get("title"))
        url = _string(item.get("url"))
        if title and url:
            links.append(f"{title}: {url}")
        elif url:
            links.append(url)
    return tuple(links)


def _language_keys(value: Any) -> tuple[str, ...]:
    languages: list[str] = []
    for item in value or []:
        if isinstance(item, dict):
            key = _string(item.get("key"))
            if key:
                languages.append(key.removeprefix("/languages/"))
    return tuple(languages)


def _cover_url(cover_id: int | None) -> str:
    if cover_id is None or cover_id < 0:
        return ""
    return f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"


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
    if isinstance(value, int):
        return value
    return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"
