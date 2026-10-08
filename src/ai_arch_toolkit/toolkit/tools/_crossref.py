"""Crossref tools: search works, and read one work's record by DOI (T06).

A list says how many works match (``total-results``) and pages by ``offset``, which reaches the
10,000th result (https://github.com/CrossRef/rest-api-doc, "Offsets for /works are limited to
10K"). A work's record gives every field and every item of its lists (authors, licenses, links,
references) through the window (D39).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._records import (
    DEFAULT_CHARS,
    MAX_CHARS,
    call,
    doi_of,
    names,
    record,
)
from ai_arch_toolkit.toolkit.tools._window import list_window


def _crossref_error(reply: Reply) -> ToolFailure | str | None:
    """The reason Crossref gives for a request it refuses; ``None`` for any other answer.

    Crossref answers a parameter it does not take with HTTP 400 and ``{"status": "failed",
    "message-type": "validation-failure", "message": [{"type", "value", "message"}]}``
    (CrossRef/cayenne, src/cayenne/api/v1/validate.clj): the caller's argument to fix, a
    ``validation_error``. An unknown DOI is a 404 with "Resource not found." in plain text, which
    ``crossref_work`` declares.
    """
    body = reply.body
    if not isinstance(body, dict) or body.get("status") != "failed":
        return None
    failures = body.get("message")
    items = failures if isinstance(failures, list) else [failures]
    said = "; ".join(
        text
        for item in items
        if (text := _string(item.get("message") if isinstance(item, dict) else item))
    )
    if not said:
        return None
    if reply.status == 400:
        msg = f"Crossref refused the request: {said}; correct that parameter"
        return ToolFailure("validation_error", msg)
    return said


_API = Api(base="https://api.crossref.org/works", name="Crossref", error_reader=_crossref_error)
# The deepest offset a /works list takes (cayenne's query.clj: max-offset 10000).
_DEPTH = 10_000
_VALID_TYPE_FILTER = re.compile(r"^[a-z0-9-]+$")
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True, kw_only=True)
class _CrossrefWork:
    """A Crossref work, as its deposit gives it."""

    doi: str
    title: str
    authors: tuple[str, ...]
    authors_in_full: tuple[str, ...]
    published: str
    container_title: str
    publisher: str
    work_type: str
    url: str
    abstract: str
    referenced_by_count: int | None
    reference_count: int | None
    references: tuple[str, ...]
    licenses: tuple[str, ...]
    links: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Page:
    works: list[_CrossrefWork]
    total: int | None


@tool(capability="network")
def crossref_search(
    query: str,
    max_results: Annotated[int, Range(1, 20)] = 5,
    start: Annotated[int, Range(0, _DEPTH)] = 0,
    from_date: str = "",
    to_date: str = "",
    type_filter: str = "",
) -> ToolResult:
    """Search Crossref works by title, topic, author, or a citation's words, numbered, with
    the total.

    Args:
        query: Search text, such as a title, topic, author or citation fragment.
        max_results: How many works to list.
        start: How many results to skip; the footer gives the next start.
        from_date: The earliest publication date, YYYY-MM-DD.
        to_date: The latest publication date, YYYY-MM-DD.
        type_filter: A Crossref type, e.g. "journal-article" or "proceedings-article".

    Raises:
        ToolFailure: validation_error when an argument is invalid (here or for Crossref).
    """
    query = query.strip()
    if not query:
        msg = "query cannot be empty; pass a title, topic, author or citation fragment."
        raise ToolFailure("validation_error", msg)
    params = {"query": query, "rows": str(max_results), "offset": str(start)}
    if filter_value := _build_filter(from_date, to_date, type_filter):
        params["filter"] = filter_value
    page = _API.get_json(params=params, parse=_works)
    return _search_answer(page, query, start)


@tool(capability="network")
def crossref_work(
    doi: str,
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, MAX_CHARS)] = DEFAULT_CHARS,
) -> ToolResult:
    """Read a work's Crossref record by DOI: its whole abstract, every author with the
    affiliations, the licenses, the full-text links and the references.

    Args:
        doi: A DOI or DOI URL, e.g. "10.1038/nature14539" or "https://doi.org/...".
        offset: Where to start, in characters of the record; the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when the DOI is malformed; not_found when Crossref has no
            work with it.
    """
    normalized = doi_of(doi)
    if not normalized:
        msg = f"invalid DOI {doi!r}; a DOI looks like 10.1000/xyz."
        raise ToolFailure("validation_error", msg)
    missing = (
        f"no Crossref work with DOI {normalized}; search with crossref_search, or look the DOI "
        "up with datacite_doi."
    )
    work = _API.get_json(normalized, parse=_work, missing=missing)
    heading = f"Crossref work {normalized}:"
    return record(_record_text(work), heading=heading, offset=offset, max_chars=max_chars)


def _search_answer(page: _Page, query: str, start: int) -> ToolResult:
    if not page.works and start == 0:
        return ToolResult.success(f"No Crossref works match {query!r}.")
    blocks = [_result_block(start + n, work) for n, work in enumerate(page.works, start=1)]
    end = start + len(page.works)
    more = page.total is not None and end < page.total
    next_call = {"start": end} if page.works and more and end <= _DEPTH else None
    window = list_window(blocks, first=start + 1, total=page.total, next_call=next_call)
    return window.result(heading=f"Crossref works that match {query!r}:")


def _build_filter(from_date: str, to_date: str, type_filter: str) -> str:
    filters: list[str] = []
    from_date = from_date.strip()
    to_date = to_date.strip()
    type_filter = type_filter.strip()

    parsed_start: date | None = None
    parsed_end: date | None = None
    if from_date:
        parsed_start = _parse_date(from_date)
        if parsed_start is None:
            msg = f"invalid from_date {from_date!r}; use YYYY-MM-DD."
            raise ToolFailure("validation_error", msg)
        filters.append(f"from-pub-date:{from_date}")
    if to_date:
        parsed_end = _parse_date(to_date)
        if parsed_end is None:
            msg = f"invalid to_date {to_date!r}; use YYYY-MM-DD."
            raise ToolFailure("validation_error", msg)
        filters.append(f"until-pub-date:{to_date}")
    if parsed_start and parsed_end and parsed_start > parsed_end:
        msg = f"from_date {from_date} must be before or equal to to_date {to_date}."
        raise ToolFailure("validation_error", msg)
    if type_filter:
        if not _VALID_TYPE_FILTER.fullmatch(type_filter):
            msg = (
                f"invalid type_filter {type_filter!r}; use a Crossref type such as "
                "journal-article or proceedings-article."
            )
            raise ToolFailure("validation_error", msg)
        filters.append(f"type:{type_filter}")
    return ",".join(filters)


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _works(data: dict[str, Any]) -> _Page:
    message = data.get("message", {})
    items = message.get("items", [])
    total = message.get("total-results")
    works = [_parse_work(item) for item in items if isinstance(item, dict)]
    return _Page(works, total if isinstance(total, int) else None)


def _work(data: dict[str, Any]) -> _CrossrefWork:
    message = data.get("message")
    if not isinstance(message, dict):
        raise TypeError(f"expected a work in message, got {type(message).__name__}")
    return _parse_work(message)


def _parse_work(item: dict[str, Any]) -> _CrossrefWork:
    authors = [author for author in item.get("author") or [] if isinstance(author, dict)]
    return _CrossrefWork(
        doi=_string(item.get("DOI")),
        title=_title(item),
        authors=tuple(name for author in authors if (name := _author_name(author))),
        authors_in_full=tuple(name for author in authors if (name := _author_in_full(author))),
        published=_published_date(item),
        container_title=_first_string(item.get("container-title")),
        publisher=_string(item.get("publisher")),
        work_type=_string(item.get("type")),
        url=_string(item.get("URL")),
        abstract=_clean_text(str(item.get("abstract", "") or "")),
        referenced_by_count=_int_or_none(item.get("is-referenced-by-count")),
        reference_count=_int_or_none(item.get("reference-count")),
        references=_references(item),
        licenses=_licenses(item),
        links=_links(item),
    )


def _title(item: dict[str, Any]) -> str:
    title = _first_string(item.get("title"))
    subtitle = _first_string(item.get("subtitle"))
    if title and subtitle:
        return f"{title}: {subtitle}"
    return title or "(untitled)"


def _author_name(author: dict[str, Any]) -> str:
    name = _string(author.get("name"))
    if name:
        return name
    return " ".join(
        part for part in (_string(author.get("given")), _string(author.get("family"))) if part
    )


def _author_in_full(author: dict[str, Any]) -> str:
    """An author with the affiliations and the ORCID iD the deposit gives."""
    name = _author_name(author)
    if not name:
        return ""
    places = [
        text
        for place in author.get("affiliation") or []
        if isinstance(place, dict) and (text := _string(place.get("name")))
    ]
    orcid = (
        _string(author.get("ORCID"))
        .removeprefix("https://orcid.org/")
        .removeprefix("http://orcid.org/")
    )
    details = [*places, f"ORCID {orcid}"] if orcid else places
    return f"{name} ({'; '.join(details)})" if details else name


def _published_date(item: dict[str, Any]) -> str:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        value = item.get(key)
        if isinstance(value, dict):
            date_parts = value.get("date-parts")
            if isinstance(date_parts, list) and date_parts:
                return _format_date_parts(date_parts[0])
    return ""


def _format_date_parts(parts: Any) -> str:
    """Crossref's ``date-parts`` ([year, month, day], the last two optional) in ISO 8601."""
    if not isinstance(parts, list) or not parts:
        return ""
    values = [str(part).zfill(2) for part in parts[:3] if part is not None]
    if values:
        values[0] = values[0].lstrip("0") or "0"
    return "-".join(values)


def _date_of(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    parts = value.get("date-parts")
    return _format_date_parts(parts[0]) if isinstance(parts, list) and parts else ""


def _references(item: dict[str, Any]) -> tuple[str, ...]:
    references: list[str] = []
    for ref in item.get("reference") or []:
        if not isinstance(ref, dict):
            continue
        parts = [
            _string(ref.get("author")),
            _string(ref.get("article-title")),
            _string(ref.get("journal-title")),
            _string(ref.get("year")),
        ]
        text = "; ".join(part for part in parts if part)
        if not text:
            text = _string(ref.get("unstructured"))
        if doi := _string(ref.get("DOI")):
            text = f"{text}; DOI: {doi}" if text else f"DOI: {doi}"
        if text:
            references.append(_clean_text(text))
    return tuple(references)


def _licenses(item: dict[str, Any]) -> tuple[str, ...]:
    """Each license, with the version it covers and when it starts."""
    licenses: list[str] = []
    for license_item in item.get("license") or []:
        if not isinstance(license_item, dict) or not (url := _string(license_item.get("URL"))):
            continue
        details = [_string(license_item.get("content-version"))]
        if start := _date_of(license_item.get("start")):
            details.append(f"from {start}")
        said = ", ".join(detail for detail in details if detail)
        licenses.append(f"{url} ({said})" if said else url)
    return tuple(dict.fromkeys(licenses))


def _links(item: dict[str, Any]) -> tuple[str, ...]:
    """Each full-text link, with its content type and intended application."""
    links: list[str] = []
    for link in item.get("link") or []:
        if not isinstance(link, dict) or not (url := _string(link.get("URL"))):
            continue
        details = (_string(link.get("content-type")), _string(link.get("intended-application")))
        said = ", ".join(detail for detail in details if detail and detail != "unspecified")
        links.append(f"{url} ({said})" if said else url)
    return tuple(dict.fromkeys(links))


def _meta(work: _CrossrefWork) -> str:
    meta = [f"DOI: {work.doi}"] if work.doi else []
    if work.work_type:
        meta.append(f"type: {work.work_type}")
    if work.published:
        meta.append(f"published: {work.published}")
    return " | ".join(meta)


def _venue(work: _CrossrefWork) -> str:
    venue = [f"Venue: {work.container_title}"] if work.container_title else []
    if work.publisher:
        venue.append(f"Publisher: {work.publisher}")
    return " | ".join(venue)


def _cited(work: _CrossrefWork) -> str:
    if work.referenced_by_count is None:
        return ""
    return f"Cited by: {work.referenced_by_count} works in Crossref"


def _result_block(number: int, work: _CrossrefWork) -> str:
    lines = [_meta(work)]
    if work.authors:
        lines.append(f"Authors: {names(work.authors, whole=call('crossref_work', work.doi))}")
    lines += [_venue(work), _cited(work)]
    return "\n".join([f"{number}. {work.title}", *(f"   {line}" for line in lines if line)])


def _record_text(work: _CrossrefWork) -> str:
    """The whole record: the abstract before the lists, which can run long."""
    counts = [_cited(work)]
    if work.reference_count is not None:
        counts.append(f"References: {work.reference_count} deposited")
    lines = [work.title, _meta(work), _venue(work), " | ".join(c for c in counts if c)]
    if work.url:
        lines.append(f"URL: {work.url}")
    if work.abstract:
        lines.append(f"Abstract: {work.abstract}")
    if work.authors_in_full:
        lines.append(f"Authors ({len(work.authors_in_full)}): {', '.join(work.authors_in_full)}")
    if work.licenses:
        lines.append(f"License: {' | '.join(work.licenses)}")
    if work.links:
        lines.append(f"Links: {' | '.join(work.links)}")
    lines += _reference_lines(work)
    return "\n".join(line for line in lines if line) + "\n"


def _reference_lines(work: _CrossrefWork) -> list[str]:
    if work.references:
        return [f"References ({len(work.references)}):", *(f"- {r}" for r in work.references)]
    if work.reference_count:
        return [f"References: {work.reference_count} deposited, none listed in Crossref's record"]
    return []


def _first_string(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return _clean_text(item)
    if isinstance(value, str):
        return _clean_text(value)
    return ""


def _clean_text(text: str) -> str:
    """Text without its JATS or HTML markup."""
    return " ".join(html.unescape(_TAG_RE.sub(" ", text)).split())


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
