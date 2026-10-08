"""Europe PMC tools: search articles, read one article's record, and list the articles that
cite one (T06).

A search says how many articles match (``hitCount``) and pages by cursor (``cursorMark``, then
the ``nextCursorMark`` each page gives); a citation list says how many cite the record and pages
by ``page`` and ``pageSize`` (https://europepmc.org/RestfulWebService). An article's record
gives its whole abstract and every author, MeSH heading and full-text link, through the window
(D39).
"""

from __future__ import annotations

import html
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
    names,
    record,
)
from ai_arch_toolkit.toolkit.tools._window import list_window


def _epmc_error(reply: Reply) -> str | None:
    """The error Europe PMC reports in place of the result; ``None`` for a result.

    Europe PMC answers a request it cannot run with ``errCode`` and ``errMsg`` and no result
    list, with HTTP 200 (``{"errCode": 404, "errMsg": "Invalid page size provided. ..."}``, as
    recorded in K-Dense-AI/scientific-agent-skills, paper-lookup/references/europepmc.md; the
    service's page, https://europepmc.org/RestfulWebService, does not load without a browser).
    The code does not say whose fault it is, so the error stays in Europe PMC's words.
    """
    body = reply.body
    message = _string(body.get("errMsg")) if isinstance(body, dict) else ""
    if not message:
        return None
    code = _string(body.get("errCode")) if isinstance(body, dict) else ""
    said = f"Europe PMC error {code}: {message}" if code else f"Europe PMC error: {message}"
    return f"{said}; check the query and the identifiers, or try again later"


_API = Api(
    base="https://www.ebi.ac.uk/europepmc/webservices/rest",
    name="Europe PMC",
    timeout_s=15,
    error_reader=_epmc_error,
)
_RESULT_TYPES = {"lite", "core", "idlist"}
_SOURCE_RE = re.compile(r"^[A-Z]{3}$", re.IGNORECASE)
_SOURCED_ID_RE = re.compile(r"^([A-Za-z]{3})/(\S+)$")
_FLAGS = (
    ("open access", "isOpenAccess"),
    ("in Europe PMC", "inEPMC"),
    ("in PMC", "inPMC"),
    ("PDF", "hasPDF"),
    ("references", "hasReferences"),
)
_YES_NO = {"Y": "yes", "N": "no"}


@dataclass(frozen=True, slots=True, kw_only=True)
class _Article:
    """A Europe PMC article, as a result or a citation gives it."""

    id: str
    source: str
    pmid: str
    pmcid: str
    doi: str
    title: str
    authors: tuple[str, ...]
    authors_in_full: tuple[str, ...]
    journal: str
    year: str
    published: str
    publication_types: tuple[str, ...]
    abstract: str
    flags: str
    cited_by_count: int | None
    mesh: tuple[str, ...]
    keywords: tuple[str, ...]
    full_text: tuple[str, ...]

    @property
    def key(self) -> str:
        """The record as ``SOURCE/ID``, which ``europe_pmc_article`` takes."""
        return f"{self.source}/{self.id}"


@dataclass(frozen=True, slots=True, kw_only=True)
class _Found:
    """A page of articles: how many there are in all, and the cursor of the next page."""

    articles: list[_Article]
    total: int | None
    next_cursor: str = ""


@tool(capability="network")
def europe_pmc_search(
    query: str,
    max_results: Annotated[int, Range(1, 20)] = 5,
    cursor_mark: str = "*",
    offset: Annotated[int, Range(0)] = 0,
    result_type: str = "lite",
) -> ToolResult:
    """Search Europe PMC articles (PubMed, PMC, preprints, patents …), numbered, with the
    total.

    Args:
        query: Search text or Europe PMC query syntax, e.g. 'AUTH:"Hinton G" AND deep learning'.
        max_results: How many articles to list.
        cursor_mark: Where the page starts: "*" for the first; the footer gives the next.
        offset: How many results come before the cursor's page; the footer gives it with the
            cursor.
        result_type: How much Europe PMC sends per article: lite, core or idlist (IDs only);
            the list shows the same fields from lite and core, and europe_pmc_article gives an
            article's whole record.

    Raises:
        ToolFailure: validation_error when ``query`` is empty or ``result_type`` is unknown;
            upstream when Europe PMC reports an error.
    """
    query = query.strip()
    if not query:
        raise ToolFailure("validation_error", "query cannot be empty; pass search text.")
    result_type = result_type.strip().lower() or "lite"
    if result_type not in _RESULT_TYPES:
        msg = f"result_type must be one of lite, core, idlist, got {result_type!r}."
        raise ToolFailure("validation_error", msg)
    cursor = cursor_mark.strip() or "*"
    params = {
        "query": query,
        "format": "json",
        "pageSize": str(max_results),
        "cursorMark": cursor,
        "resultType": result_type,
    }
    found = _API.get_json("search", params=params, parse=_search_results)
    return _search_answer(found, query, cursor, offset)


@tool(capability="network")
def europe_pmc_article(
    identifier: str,
    source: str = "",
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, MAX_CHARS)] = DEFAULT_CHARS,
) -> ToolResult:
    """Read an article's Europe PMC record: its whole abstract, every author with the
    affiliations, the MeSH headings, the keywords and every full-text link.

    Args:
        identifier: A PMID, a PMCID, a DOI, or a record as SOURCE/ID, e.g. "MED/26017442".
        source: The Europe PMC source of a bare ID, e.g. MED, PMC, PPR, AGR, CBA or PAT.
        offset: Where to start, in characters of the record; the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when ``identifier`` is empty or ``source`` is malformed;
            not_found when Europe PMC has no article with that identifier.
    """
    wanted = identifier.strip()
    query = _article_query(wanted, source)
    params = {"query": query, "format": "json", "pageSize": "1", "resultType": "core"}
    found = _API.get_json("search", params=params, parse=_search_results)
    if not found.articles:
        msg = f"no Europe PMC article with identifier {wanted!r}; search with europe_pmc_search."
        raise ToolFailure("not_found", msg)
    article = found.articles[0]
    heading = f"Europe PMC article {article.key}"
    if found.total is not None and found.total > 1:
        heading += (
            f" (the first of {found.total} records with ID {wanted!r}; give source= for another)"
        )
    text = _record_text(article)
    return record(text, heading=f"{heading}:", offset=offset, max_chars=max_chars)


@tool(capability="network")
def europe_pmc_citations(
    source: str,
    identifier: str,
    max_results: Annotated[int, Range(1, 25)] = 10,
    page: Annotated[int, Range(1)] = 1,
) -> ToolResult:
    """List the articles in Europe PMC that cite a record, numbered, with the total.

    Args:
        source: The record's Europe PMC source, e.g. MED or PMC.
        identifier: The record's ID in that source, e.g. a PMID for MED.
        max_results: How many citing articles a page lists.
        page: Which page of ``max_results`` articles to show; the footer gives the next page.

    Raises:
        ToolFailure: validation_error when ``source`` is malformed or ``identifier`` is empty;
            not_found when Europe PMC has no such record.
    """
    normalized_source = source.strip().upper()
    identifier = identifier.strip()
    if not _SOURCE_RE.fullmatch(normalized_source):
        raise ToolFailure("validation_error", _invalid_source(source))
    if not identifier:
        msg = "identifier cannot be empty; pass the record's ID in that source, e.g. a PMID."
        raise ToolFailure("validation_error", msg)
    key = f"{normalized_source}/{identifier}"
    found = _API.get_json(
        normalized_source,
        identifier,
        "citations",
        params={"format": "json", "page": str(page), "pageSize": str(max_results)},
        parse=_citations,
    )
    if not found.articles and page == 1:
        _known(normalized_source, identifier)
        return ToolResult.success(f"No articles in Europe PMC cite {key}.")
    first = (page - 1) * max_results
    blocks = [_result_block(first + n, a) for n, a in enumerate(found.articles, start=1)]
    more = found.total is not None and first + len(found.articles) < found.total
    # A page is numbered in pages of max_results: the next call keeps the size.
    onward = {"page": page + 1, "max_results": max_results}
    next_call = onward if found.articles and more else None
    window = list_window(blocks, first=first + 1, total=found.total, next_call=next_call)
    return window.result(heading=f"Articles in Europe PMC that cite {key}:")


def _search_answer(found: _Found, query: str, cursor: str, offset: int) -> ToolResult:
    if not found.articles and cursor == "*":
        return ToolResult.success(f"No Europe PMC articles match {query!r}.")
    blocks = [_result_block(offset + n, a) for n, a in enumerate(found.articles, start=1)]
    end = offset + len(found.articles)
    more = found.total is not None and end < found.total
    moved = found.next_cursor not in ("", cursor)
    next_call = (
        {"cursor_mark": found.next_cursor, "offset": end}
        if found.articles and more and moved
        else None
    )
    window = list_window(blocks, first=offset + 1, total=found.total, next_call=next_call)
    return window.result(heading=f"Europe PMC articles that match {query!r}:")


def _known(source: str, identifier: str) -> None:
    """Whether Europe PMC has the record: its citation list answers an unknown one as a record
    nobody cites.

    Raises:
        ToolFailure: not_found when it has no such record.
    """
    params = {
        "query": f"SRC:{source} AND EXT_ID:{identifier}",
        "format": "json",
        "pageSize": "1",
        "resultType": "idlist",
    }
    found = _API.get_json("search", params=params, parse=_search_results)
    if found.total == 0 or (found.total is None and not found.articles):
        msg = (
            f"Europe PMC has no record {source}/{identifier}; find the article's source and ID "
            "with europe_pmc_search"
        )
        raise ToolFailure("not_found", msg)


def _article_query(identifier: str, source: str) -> str:
    source = source.strip().upper()
    if not identifier:
        msg = "identifier cannot be empty; pass a PMID, PMCID, DOI, or Europe PMC ID."
        raise ToolFailure("validation_error", msg)
    if source and not _SOURCE_RE.fullmatch(source):
        raise ToolFailure("validation_error", _invalid_source(source))
    if not identifier.startswith("10.") and (sourced := _SOURCED_ID_RE.fullmatch(identifier)):
        source, identifier = sourced[1].upper(), sourced[2]
    if identifier.upper().startswith("PMC") and identifier[3:].isdigit():
        query = f"PMCID:{identifier}"
    elif identifier.lower().startswith("10."):
        query = f'DOI:"{identifier}"'
    else:
        query = f"EXT_ID:{identifier}"
    if source:
        query = f"SRC:{source} AND {query}"
    return query


def _invalid_source(source: str) -> str:
    return f"invalid source {source!r}; a Europe PMC source is three letters, e.g. MED or PMC."


def _search_results(data: dict[str, Any]) -> _Found:
    results = data.get("resultList", {}).get("result", [])
    return _Found(
        articles=_articles(results),
        total=_int_or_none(data.get("hitCount")),
        next_cursor=_string(data.get("nextCursorMark")),
    )


def _citations(data: dict[str, Any]) -> _Found:
    results = data.get("citationList", {}).get("citation", [])
    return _Found(articles=_articles(results), total=_int_or_none(data.get("hitCount")))


def _articles(results: Any) -> list[_Article]:
    if not isinstance(results, list):
        return []
    return [a for item in results if isinstance(item, dict) if (a := _parse_article(item))]


def _parse_article(data: dict[str, Any]) -> _Article | None:
    article_id = _string(data.get("id"))
    if not article_id:
        return None
    authors = tuple(
        name.strip().rstrip(".")
        for name in _string(data.get("authorString")).split(", ")
        if name.strip().rstrip(".")
    )
    return _Article(
        id=article_id,
        source=_string(data.get("source")),
        pmid=_string(data.get("pmid")),
        pmcid=_string(data.get("pmcid")),
        doi=_string(data.get("doi")),
        title=_clean_text(data.get("title")) or "(untitled)",
        authors=authors,
        authors_in_full=_authors_in_full(data.get("authorList")) or authors,
        journal=_string(data.get("journalTitle") or data.get("journalAbbreviation")),
        year=_string(data.get("pubYear")),
        published=_string(data.get("firstPublicationDate") or data.get("firstIndexDate")),
        publication_types=_listed(data.get("pubTypeList"), "pubType")
        or tuple(t for t in [_string(data.get("pubType") or data.get("citationType"))] if t),
        abstract=_clean_text(data.get("abstractText")),
        flags=" | ".join(
            f"{label}: {_YES_NO.get(value, value)}"
            for label, key in _FLAGS
            if (value := _string(data.get(key)))
        ),
        cited_by_count=_int_or_none(data.get("citedByCount")),
        mesh=_mesh(data.get("meshHeadingList")),
        keywords=_listed(data.get("keywordList"), "keyword"),
        full_text=_full_text(data.get("fullTextUrlList")),
    )


def _authors_in_full(value: Any) -> tuple[str, ...]:
    """Each author of a ``core`` result, with the affiliations."""
    authors = value.get("author") if isinstance(value, dict) else None
    found: list[str] = []
    for author in authors if isinstance(authors, list) else []:
        if not isinstance(author, dict) or not (name := _string(author.get("fullName"))):
            continue
        details = author.get("authorAffiliationDetailsList")
        places = details.get("authorAffiliation") if isinstance(details, dict) else None
        named = [
            text
            for place in (places if isinstance(places, list) else [])
            if isinstance(place, dict) and (text := _string(place.get("affiliation")))
        ]
        found.append(f"{name} ({'; '.join(named)})" if named else name)
    return tuple(found)


def _listed(value: Any, key: str) -> tuple[str, ...]:
    items = value.get(key) if isinstance(value, dict) else None
    return (
        tuple(text for item in items or [] if (text := _string(item)))
        if isinstance(items, list)
        else ()
    )


def _mesh(value: Any) -> tuple[str, ...]:
    """Each MeSH heading, marked when it is a major topic, with its qualifiers."""
    headings = value.get("meshHeading") if isinstance(value, dict) else None
    found: list[str] = []
    for heading in headings if isinstance(headings, list) else []:
        if not isinstance(heading, dict) or not (name := _string(heading.get("descriptorName"))):
            continue
        qualifiers = heading.get("meshQualifierList")
        listed = qualifiers.get("meshQualifier") if isinstance(qualifiers, dict) else None
        details = ["major"] if _string(heading.get("majorTopic_YN")) == "Y" else []
        details += [
            text
            for q in (listed if isinstance(listed, list) else [])
            if isinstance(q, dict) and (text := _string(q.get("qualifierName")))
        ]
        found.append(f"{name} ({'; '.join(details)})" if details else name)
    return tuple(found)


def _full_text(value: Any) -> tuple[str, ...]:
    """Each full-text link, with its availability and form."""
    items = value.get("fullTextUrl") if isinstance(value, dict) else None
    links: list[str] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not (url := _string(item.get("url"))):
            continue
        details = [_string(item.get("availability")), _string(item.get("documentStyle"))]
        said = ", ".join(detail for detail in details if detail)
        links.append(f"{url} ({said})" if said else url)
    return tuple(links)


def _ids(article: _Article) -> str:
    meta = [f"id: {article.key}"]
    labelled = (("PMID", article.pmid), ("PMCID", article.pmcid), ("DOI", article.doi))
    meta += [f"{label}: {value}" for label, value in labelled if value]
    if article.year:
        meta.append(f"year: {article.year}")
    if article.cited_by_count is not None:
        meta.append(f"cited by: {article.cited_by_count}")
    return " | ".join(meta)


def _venue(article: _Article) -> str:
    venue = [f"Journal: {article.journal}"] if article.journal else []
    if article.publication_types:
        venue.append(f"Type: {'; '.join(article.publication_types)}")
    return " | ".join(venue)


def _result_block(number: int, article: _Article) -> str:
    lines = [_ids(article), article.flags]
    if article.authors:
        whole = call("europe_pmc_article", article.key)
        lines.append(f"Authors: {names(article.authors, whole=whole)}")
    lines.append(_venue(article))
    return "\n".join([f"{number}. {article.title}", *(f"   {line}" for line in lines if line)])


def _record_text(article: _Article) -> str:
    """The whole record: the abstract before the lists, which can run long."""
    published = f"published: {article.published}" if article.published else ""
    dated = " | ".join(part for part in (published, article.flags) if part)
    lines = [article.title, _ids(article), dated, _venue(article)]
    if article.abstract:
        lines.append(f"Abstract: {article.abstract}")
    if article.authors_in_full:
        authors = article.authors_in_full
        lines.append(f"Authors ({len(authors)}): {', '.join(authors)}")
    if article.mesh:
        lines.append(f"MeSH: {', '.join(article.mesh)}")
    if article.keywords:
        lines.append(f"Keywords: {', '.join(article.keywords)}")
    if article.full_text:
        lines.append(f"Full text: {' | '.join(article.full_text)}")
    lines.append(f"Europe PMC: https://europepmc.org/article/{article.key}")
    return "\n".join(line for line in lines if line) + "\n"


def _clean_text(value: Any) -> str:
    text = html.unescape(_string(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


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
