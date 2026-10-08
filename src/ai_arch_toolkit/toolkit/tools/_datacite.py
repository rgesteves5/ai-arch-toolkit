"""DataCite tools: search DOI records, and read one DOI's record (T06).

A list says how many records match (``meta.total``) and pages by ``page[number]`` through the
first 10,000 records (https://support.datacite.org/docs/pagination). A DOI's record gives every
field and every item of its lists (titles, creators, descriptions, subjects, rights, related
identifiers) through the window (D39).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


def _datacite_error(reply: Reply) -> ToolFailure | str | None:
    """The reason DataCite gives for a request it refuses; ``None`` without one.

    The REST API speaks JSON:API, whose errors are a list of objects with a ``status``, a
    ``title`` and an optional ``detail`` (https://jsonapi.org/format/#error-objects), with the
    titles DataCite's error page lists (https://support.datacite.org/docs/api-error-codes). A 400
    is a request DataCite could not run, most often a query it cannot read: the caller's to fix.
    """
    errors = reply.body.get("errors") if isinstance(reply.body, dict) else None
    if reply.status < 400 or not isinstance(errors, list):
        return None
    said = "; ".join(
        ": ".join(part for part in (_string(e.get("title")), _string(e.get("detail"))) if part)
        for e in errors
        if isinstance(e, dict)
    )
    if not said.strip("; "):
        return None
    if reply.status == 400:
        msg = (
            f"DataCite refused the request: {said}; check the query syntax "
            "(https://support.datacite.org/docs/api-queries) or the resource_type"
        )
        return ToolFailure("validation_error", msg)
    return said


_API = Api(
    base="https://api.datacite.org/dois",
    name="DataCite",
    timeout_s=15,
    error_reader=_datacite_error,
)
# Paging by number reaches the first 10,000 records (https://support.datacite.org/docs/pagination).
_DEPTH = 10_000
_RESOURCE_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z _-]{0,59}$")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclass(frozen=True, slots=True, kw_only=True)
class _DataCiteDoi:
    """A DataCite DOI's metadata, as the record gives it."""

    doi: str
    title: str
    other_titles: tuple[str, ...]
    creators: tuple[str, ...]
    creators_in_full: tuple[str, ...]
    publisher: str
    publication_year: int | None
    resource_type: str
    version: str
    descriptions: tuple[str, ...]
    subjects: tuple[str, ...]
    url: str
    rights: tuple[str, ...]
    related_identifiers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Page:
    """A page of records, with DataCite's count of all the records that match."""

    dois: list[_DataCiteDoi]
    total: int | None = None


@tool(capability="network")
def datacite_search(
    query: str,
    resource_type: str = "",
    max_results: Annotated[int, Range(1, 20)] = 5,
    page: Annotated[int, Range(1, _DEPTH)] = 1,
) -> ToolResult:
    """Search DataCite DOI records (datasets, software, texts and other research outputs),
    numbered, with the total.

    Args:
        query: Metadata search text, e.g. words of a title or a creator's name.
        resource_type: A resourceTypeGeneral to keep, e.g. "Dataset", "Software" or
            "JournalArticle".
        max_results: How many records a page lists.
        page: Which page of ``max_results`` records to show; the footer gives the next page.

    Raises:
        ToolFailure: validation_error when an argument is invalid, the page lies past the first
            10000 records, or DataCite refuses the query.
    """
    query = query.strip()
    if not query:
        msg = "query cannot be empty; pass metadata search text such as a title or creator."
        raise ToolFailure("validation_error", msg)
    if (page - 1) * max_results >= _DEPTH:
        msg = (
            f"page {page} of {max_results} lies past the first {_DEPTH} records, the most "
            "DataCite pages through; narrow the query or set a resource_type"
        )
        raise ToolFailure("validation_error", msg)
    params = {"query": query, "page[size]": str(max_results), "page[number]": str(page)}
    if resource_type.strip():
        params["resource-type-id"] = _resource_type_id(resource_type)
    found = _API.get_json(params=params, parse=_records)
    return _search_answer(found, query, page, max_results)


@tool(capability="network")
def datacite_doi(
    doi: str,
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, MAX_CHARS)] = DEFAULT_CHARS,
) -> ToolResult:
    """Read a DOI's DataCite record: every title, creator, description, subject, right and
    related identifier.

    Args:
        doi: A DOI or DOI URL.
        offset: Where to start, in characters of the record; the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when the DOI is malformed; not_found when DataCite has no
            record of it.
    """
    normalized = doi_of(doi)
    if not normalized:
        msg = f"invalid DOI {doi!r}; a DOI looks like 10.1000/xyz."
        raise ToolFailure("validation_error", msg)
    missing = (
        f"no DataCite record of DOI {normalized}; search with datacite_search, "
        "or look the DOI up with crossref_work."
    )
    found = _API.get_json(normalized, parse=_record, missing=missing)
    heading = f"DataCite DOI {normalized}:"
    return record(_record_text(found), heading=heading, offset=offset, max_chars=max_chars)


def _search_answer(found: _Page, query: str, page: int, max_results: int) -> ToolResult:
    first = (page - 1) * max_results
    if not found.dois and page == 1:
        return ToolResult.success(f"No DataCite DOIs match {query!r}.")
    blocks = [_result_block(first + n, doi) for n, doi in enumerate(found.dois, start=1)]
    more = found.total is not None and first + len(found.dois) < found.total
    # A page is numbered in pages of max_results: the next call keeps the size.
    onward = {"page": page + 1, "max_results": max_results}
    next_call = onward if found.dois and more and page * max_results < _DEPTH else None
    window = list_window(blocks, first=first + 1, total=found.total, next_call=next_call)
    return window.result(heading=f"DataCite DOIs that match {query!r}:")


def _resource_type_id(value: str) -> str:
    """A resourceTypeGeneral as the ``resource-type-id`` filter takes it, in kebab case:
    ``JournalArticle`` or ``Journal Article`` is ``journal-article``
    (https://support.datacite.org/docs/api-queries).

    Raises:
        ToolFailure: validation_error unless it is a resource type's name.
    """
    text = value.strip()
    if not _RESOURCE_TYPE_RE.fullmatch(text):
        msg = (
            f"invalid resource_type {value[:80]!r}; give a DataCite resourceTypeGeneral, e.g. "
            "'Dataset', 'Software' or 'JournalArticle'"
        )
        raise ToolFailure("validation_error", msg)
    words = _CAMEL_RE.sub(" ", text).replace("_", " ").replace("-", " ")
    return "-".join(words.lower().split())


def _records(data: dict[str, Any]) -> _Page:
    items = data.get("data", [])
    dois = [doi for item in items if isinstance(item, dict) if (doi := _parse_doi(item))]
    meta = data.get("meta")
    return _Page(dois, _int_or_none(meta.get("total")) if isinstance(meta, dict) else None)


def _record(data: dict[str, Any]) -> _DataCiteDoi:
    item = data.get("data")
    found = _parse_doi(item) if isinstance(item, dict) else None
    if found is None:
        raise TypeError("expected a DOI record in data")
    return found


def _parse_doi(data: dict[str, Any]) -> _DataCiteDoi | None:
    attrs = data.get("attributes")
    if not isinstance(attrs, dict):
        return None
    doi = _string(attrs.get("doi") or data.get("id"))
    if not doi:
        return None
    titles = _titles(attrs.get("titles"))
    creators = [item for item in attrs.get("creators") or [] if isinstance(item, dict)]
    return _DataCiteDoi(
        doi=doi,
        title=titles[0] if titles else "(untitled)",
        other_titles=tuple(titles[1:]),
        creators=tuple(name for item in creators if (name := _string(item.get("name")))),
        creators_in_full=tuple(name for item in creators if (name := _creator(item))),
        publisher=_publisher(attrs.get("publisher")),
        publication_year=_int_or_none(attrs.get("publicationYear")),
        resource_type=_resource_type(attrs),
        version=_string(attrs.get("version")),
        descriptions=_descriptions(attrs.get("descriptions")),
        subjects=_texts(attrs.get("subjects"), "subject"),
        url=_string(attrs.get("url")),
        rights=_rights(attrs.get("rightsList")),
        related_identifiers=_related_identifiers(attrs.get("relatedIdentifiers")),
    )


def _titles(value: Any) -> list[str]:
    """The main title first, then the others with their type."""
    titles: list[str] = []
    for item in value or []:
        if isinstance(item, dict) and (title := _string(item.get("title"))):
            kind = _string(item.get("titleType"))
            titles.append(f"{title} ({kind})" if kind and titles else title)
    return titles


def _creator(item: dict[str, Any]) -> str:
    """A creator with the affiliations (text, or objects with a ``name``)."""
    name = _string(item.get("name"))
    places = [
        text
        for place in item.get("affiliation") or []
        if (text := _string(place.get("name") if isinstance(place, dict) else place))
    ]
    return f"{name} ({'; '.join(places)})" if name and places else name


def _publisher(value: Any) -> str:
    """The publisher: text, or an object with a ``name`` (``publisher=true``)."""
    return _string(value.get("name") if isinstance(value, dict) else value)


def _resource_type(attrs: dict[str, Any]) -> str:
    """The resourceTypeGeneral, with the free-text resourceType when it says more."""
    types = attrs.get("types")
    if not isinstance(types, dict):
        return ""
    general, specific = (
        _string(types.get("resourceTypeGeneral")),
        _string(types.get("resourceType")),
    )
    if general and specific and specific != general:
        return f"{general} ({specific})"
    return general or specific


def _descriptions(value: Any) -> tuple[str, ...]:
    """Each description, labelled with its type (Abstract, Methods …)."""
    descriptions: list[str] = []
    for item in value or []:
        if isinstance(item, dict) and (text := _string(item.get("description"))):
            kind = _string(item.get("descriptionType"))
            descriptions.append(
                f"Description ({kind}): {text}" if kind else f"Description: {text}"
            )
    return tuple(descriptions)


def _texts(value: Any, key: str) -> tuple[str, ...]:
    return tuple(
        text for item in value or [] if isinstance(item, dict) if (text := _string(item.get(key)))
    )


def _rights(value: Any) -> tuple[str, ...]:
    rights: list[str] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        text, uri = _string(item.get("rights")), _string(item.get("rightsUri"))
        if text or uri:
            rights.append(f"{text} ({uri})" if text and uri else text or uri)
    return tuple(rights)


def _related_identifiers(value: Any) -> tuple[str, ...]:
    """Each related identifier: the relation, the identifier and its type."""
    related: list[str] = []
    for item in value or []:
        if not isinstance(item, dict) or not (
            identifier := _string(item.get("relatedIdentifier"))
        ):
            continue
        relation, kind = (
            _string(item.get("relationType")),
            _string(item.get("relatedIdentifierType")),
        )
        text = f"{relation}: {identifier}" if relation else identifier
        related.append(f"{text} ({kind})" if kind else text)
    return tuple(related)


def _meta(doi: _DataCiteDoi) -> str:
    meta = [f"DOI: {doi.doi}"]
    if doi.resource_type:
        meta.append(f"type: {doi.resource_type}")
    if doi.publication_year is not None:
        meta.append(f"year: {doi.publication_year}")
    return " | ".join(meta)


def _result_block(number: int, doi: _DataCiteDoi) -> str:
    lines = [_meta(doi)]
    if doi.creators:
        lines.append(f"Creators: {names(doi.creators, whole=call('datacite_doi', doi.doi))}")
    if doi.publisher:
        lines.append(f"Publisher: {doi.publisher}")
    return "\n".join([f"{number}. {doi.title}", *(f"   {line}" for line in lines)])


def _record_text(doi: _DataCiteDoi) -> str:
    """The whole record: the descriptions before the lists, which can run long."""
    meta = _meta(doi) + (f" | version: {doi.version}" if doi.version else "")
    lines = [doi.title, meta]
    if doi.other_titles:
        lines.append(f"Also titled: {'; '.join(doi.other_titles)}")
    if doi.publisher:
        lines.append(f"Publisher: {doi.publisher}")
    if doi.url:
        lines.append(f"URL: {doi.url}")
    lines.append(f"DataCite: https://commons.datacite.org/doi.org/{doi.doi}")
    lines += doi.descriptions
    if doi.creators_in_full:
        lines.append(f"Creators ({len(doi.creators_in_full)}): {', '.join(doi.creators_in_full)}")
    if doi.subjects:
        lines.append(f"Subjects: {', '.join(doi.subjects)}")
    if doi.rights:
        lines.append(f"Rights: {' | '.join(doi.rights)}")
    if doi.related_identifiers:
        related = doi.related_identifiers
        lines += [f"Related identifiers ({len(related)}):", *(f"- {item}" for item in related)]
    return "\n".join(lines) + "\n"


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = _string(value)
    return int(text) if text.isdigit() else None
