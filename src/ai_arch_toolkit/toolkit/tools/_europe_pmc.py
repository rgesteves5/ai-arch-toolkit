"""Europe PMC tools — public biomedical/life-sciences search and citation lookup."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api

_API = Api(
    base="https://www.ebi.ac.uk/europepmc/webservices/rest", name="Europe PMC", timeout_s=15
)
_MAX_RESULTS_LIMIT = 20
_ABSTRACT_MAX_CHARS = 1200
_RESULT_TYPES = {"lite", "core", "idlist"}
_SOURCE_RE = re.compile(r"^[A-Z]{3}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True, kw_only=True)
class _EuropePmcArticle:
    """Normalized Europe PMC article metadata."""

    id: str
    source: str
    pmid: str
    pmcid: str
    doi: str
    title: str
    authors: str
    journal: str
    year: str
    published: str
    publication_type: str
    abstract: str
    is_open_access: str
    in_epmc: str
    in_pmc: str
    has_pdf: str
    has_references: str
    cited_by_count: int | None
    full_text_urls: tuple[str, ...]


@tool(capability="network")
def europe_pmc_search(
    query: str,
    max_results: int = 5,
    cursor_mark: str = "*",
    result_type: str = "lite",
) -> str:
    """Search Europe PMC using the public REST API.

    Args:
        query: Europe PMC query text or native query syntax.
        max_results: Number of records to return (1-20). Defaults to 5.
        cursor_mark: Cursor mark for pagination. Defaults to "*".
        result_type: Result detail: lite, core, or idlist. Defaults to lite.

    Raises:
        ToolFailure: validation_error when ``query`` is empty or ``result_type`` is unknown.
    """
    query = query.strip()
    if not query:
        raise ToolFailure("validation_error", "query cannot be empty; pass search text.")
    result_type = result_type.strip().lower() or "lite"
    if result_type not in _RESULT_TYPES:
        msg = f"result_type must be one of lite, core, idlist, got {result_type!r}."
        raise ToolFailure("validation_error", msg)

    params = {
        "query": query,
        "format": "json",
        "pageSize": str(_bounded(max_results)),
        "cursorMark": cursor_mark.strip() or "*",
        "resultType": result_type,
    }
    return _API.get_json("search", params=params, parse=lambda data: _search_text(data, query))


@tool(capability="network")
def europe_pmc_article(identifier: str, source: str = "") -> str:
    """Fetch article metadata from Europe PMC by PMID, PMCID, DOI, or source ID.

    Args:
        identifier: PMID, PMCID, DOI, or Europe PMC external ID.
        source: Optional Europe PMC source, e.g. MED, PMC, AGR, CBA, PAT.

    Raises:
        ToolFailure: validation_error when ``identifier`` is empty or ``source`` is malformed;
            not_found when Europe PMC has no article with that identifier.
    """
    query = _article_query(identifier, source)
    params = {"query": query, "format": "json", "pageSize": "1", "resultType": "core"}
    return _API.get_json(
        "search", params=params, parse=lambda data: _article_text(data, identifier.strip())
    )


@tool(capability="network")
def europe_pmc_citations(source: str, identifier: str, max_results: int = 10) -> str:
    """Fetch articles that cite a Europe PMC record.

    Args:
        source: Europe PMC source, e.g. MED or PMC.
        identifier: Source-specific article ID, e.g. a PMID for MED.
        max_results: Number of citing articles to return (1-20). Defaults to 10.

    Raises:
        ToolFailure: validation_error when ``source`` is malformed or ``identifier`` is empty.
    """
    normalized_source = source.strip().upper()
    identifier = identifier.strip()
    if not _SOURCE_RE.fullmatch(normalized_source):
        raise ToolFailure("validation_error", _invalid_source(source))
    if not identifier:
        msg = "identifier cannot be empty; pass the record's ID in that source, e.g. a PMID."
        raise ToolFailure("validation_error", msg)

    record = f"{normalized_source}/{identifier}"
    return _API.get_json(
        normalized_source,
        identifier,
        "citations",
        params={"format": "json", "pageSize": str(_bounded(max_results))},
        parse=lambda data: _citations_text(data, record),
    )


def _search_text(data: dict[str, Any], query: str) -> str:
    articles = _articles_from_search(data)
    if not articles:
        return f"No Europe PMC results for: {query!r}"
    return _search_header(query, data) + "\n" + _format_articles(articles, include_abstract=False)


def _article_text(data: dict[str, Any], identifier: str) -> str:
    articles = _articles_from_search(data)
    if not articles:
        msg = (
            f"no Europe PMC article with identifier {identifier!r}; search with europe_pmc_search."
        )
        raise ToolFailure("not_found", msg)
    article = articles[0]
    return f"Europe PMC article {article.source}/{article.id}:\n" + _format_articles(
        [article],
        include_index=False,
        include_abstract=True,
    )


def _citations_text(data: dict[str, Any], record: str) -> str:
    citations = _citations_from_data(data)
    if not citations:
        return f"No Europe PMC citations found for {record}."
    total = _string(data.get("hitCount")) or str(len(citations))
    return (
        f"Europe PMC citations for {record} "
        f"(returned {len(citations)}, total {total}):\n"
        + _format_articles(citations, include_abstract=False)
    )


def _articles_from_search(data: dict[str, Any]) -> list[_EuropePmcArticle]:
    results = data.get("resultList", {}).get("result", [])
    if not isinstance(results, list):
        return []
    return [
        article for item in results if isinstance(item, dict) if (article := _parse_article(item))
    ]


def _citations_from_data(data: dict[str, Any]) -> list[_EuropePmcArticle]:
    results = data.get("citationList", {}).get("citation", [])
    if not isinstance(results, list):
        return []
    return [
        article for item in results if isinstance(item, dict) if (article := _parse_article(item))
    ]


def _parse_article(data: dict[str, Any]) -> _EuropePmcArticle | None:
    article_id = _string(data.get("id"))
    if not article_id:
        return None
    return _EuropePmcArticle(
        id=article_id,
        source=_string(data.get("source")),
        pmid=_string(data.get("pmid")),
        pmcid=_string(data.get("pmcid")),
        doi=_string(data.get("doi")),
        title=_clean_text(data.get("title")) or "(untitled)",
        authors=_string(data.get("authorString")),
        journal=_string(data.get("journalTitle") or data.get("journalAbbreviation")),
        year=_string(data.get("pubYear")),
        published=_string(data.get("firstPublicationDate") or data.get("firstIndexDate")),
        publication_type=_string(data.get("pubType") or data.get("citationType")),
        abstract=_clean_text(data.get("abstractText")),
        is_open_access=_string(data.get("isOpenAccess")),
        in_epmc=_string(data.get("inEPMC")),
        in_pmc=_string(data.get("inPMC")),
        has_pdf=_string(data.get("hasPDF")),
        has_references=_string(data.get("hasReferences")),
        cited_by_count=_int_or_none(data.get("citedByCount")),
        full_text_urls=_full_text_urls(data.get("fullTextUrlList")),
    )


def _format_articles(
    articles: list[_EuropePmcArticle],
    *,
    include_index: bool = True,
    include_abstract: bool = False,
) -> str:
    blocks: list[str] = []
    for index, article in enumerate(articles, start=1):
        title = f"{index}. {article.title}" if include_index else article.title
        lines = [title]
        meta = []
        if article.source or article.id:
            meta.append(f"id: {article.source}/{article.id}")
        if article.pmid:
            meta.append(f"PMID: {article.pmid}")
        if article.pmcid:
            meta.append(f"PMCID: {article.pmcid}")
        if article.doi:
            meta.append(f"DOI: {article.doi}")
        if article.year:
            meta.append(f"year: {article.year}")
        if article.cited_by_count is not None:
            meta.append(f"cited by: {article.cited_by_count}")
        if meta:
            lines.append("   " + " | ".join(meta))
        flags = []
        for label, value in (
            ("open access", article.is_open_access),
            ("in EPMC", article.in_epmc),
            ("in PMC", article.in_pmc),
            ("PDF", article.has_pdf),
            ("references", article.has_references),
        ):
            if value:
                flags.append(f"{label}: {value}")
        if flags:
            lines.append("   " + " | ".join(flags))
        if article.authors:
            lines.append(f"   Authors: {article.authors}")
        if article.journal:
            lines.append(f"   Journal: {article.journal}")
        if article.publication_type:
            lines.append(f"   Type: {article.publication_type}")
        if include_abstract and article.abstract:
            lines.append(f"   Abstract: {_truncate(article.abstract, _ABSTRACT_MAX_CHARS)}")
        if article.full_text_urls:
            lines.append("   Full text: " + " | ".join(article.full_text_urls[:5]))
        lines.append(f"   Europe PMC: https://europepmc.org/article/{article.source}/{article.id}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _search_header(query: str, data: dict[str, Any]) -> str:
    hit_count = _string(data.get("hitCount")) or "?"
    cursor = _string(data.get("nextCursorMark"))
    suffix = f" | nextCursorMark: {cursor}" if cursor else ""
    return f"Europe PMC results for {query!r} (total {hit_count}){suffix}:"


def _article_query(identifier: str, source: str) -> str:
    identifier = identifier.strip()
    source = source.strip().upper()
    if not identifier:
        msg = "identifier cannot be empty; pass a PMID, PMCID, DOI, or Europe PMC ID."
        raise ToolFailure("validation_error", msg)
    if source and not _SOURCE_RE.fullmatch(source):
        raise ToolFailure("validation_error", _invalid_source(source))
    if identifier.upper().startswith("PMC"):
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


def _full_text_urls(value: Any) -> tuple[str, ...]:
    urls: list[str] = []
    items = value.get("fullTextUrl") if isinstance(value, dict) else []
    for item in items or []:
        if isinstance(item, dict):
            url = _string(item.get("url"))
            if url:
                urls.append(url)
    return tuple(urls)


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_RESULTS_LIMIT))


def _clean_text(value: Any) -> str:
    text = html.unescape(_string(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def _string(value: Any) -> str:
    if value is None:
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


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"
