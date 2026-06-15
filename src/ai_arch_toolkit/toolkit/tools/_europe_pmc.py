"""Europe PMC tools — public biomedical/life-sciences search and citation lookup."""

from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_SEARCH_URL = f"{_BASE_URL}/search"
_TIMEOUT = 15
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
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


@tool
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
    """
    query = query.strip()
    if not query:
        return "Europe PMC search failed: query cannot be empty."
    result_type = result_type.strip().lower() or "lite"
    if result_type not in _RESULT_TYPES:
        return "Europe PMC search failed: result_type must be one of lite, core, idlist."

    try:
        data = _fetch_json(
            _SEARCH_URL,
            {
                "query": query,
                "format": "json",
                "pageSize": str(_bounded(max_results)),
                "cursorMark": cursor_mark.strip() or "*",
                "resultType": result_type,
            },
        )
        articles = _articles_from_search(data)
    except urllib.error.HTTPError as e:
        return _http_error("Europe PMC search failed", e)
    except urllib.error.URLError as e:
        return f"Europe PMC search failed: URL error: {e.reason}"
    except TimeoutError:
        return "Europe PMC search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Europe PMC search failed: could not parse API response: {e}"

    if not articles:
        return f"No Europe PMC results for: {query!r}"
    return _search_header(query, data) + "\n" + _format_articles(articles, include_abstract=False)


@tool
def europe_pmc_article(identifier: str, source: str = "") -> str:
    """Fetch article metadata from Europe PMC by PMID, PMCID, DOI, or source ID.

    Args:
        identifier: PMID, PMCID, DOI, or Europe PMC external ID.
        source: Optional Europe PMC source, e.g. MED, PMC, AGR, CBA, PAT.
    """
    query_or_error = _article_query(identifier, source)
    if query_or_error.startswith("Europe PMC article lookup failed:"):
        return query_or_error

    try:
        data = _fetch_json(
            _SEARCH_URL,
            {
                "query": query_or_error,
                "format": "json",
                "pageSize": "1",
                "resultType": "core",
            },
        )
        articles = _articles_from_search(data)
    except urllib.error.HTTPError as e:
        return _http_error("Europe PMC article lookup failed", e)
    except urllib.error.URLError as e:
        return f"Europe PMC article lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "Europe PMC article lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Europe PMC article lookup failed: could not parse API response: {e}"

    if not articles:
        return f"Europe PMC article not found: {identifier.strip()}"
    article = articles[0]
    return f"Europe PMC article {article.source}/{article.id}:\n" + _format_articles(
        [article],
        include_index=False,
        include_abstract=True,
    )


@tool
def europe_pmc_citations(source: str, identifier: str, max_results: int = 10) -> str:
    """Fetch articles that cite a Europe PMC record.

    Args:
        source: Europe PMC source, e.g. MED or PMC.
        identifier: Source-specific article ID, e.g. a PMID for MED.
        max_results: Number of citing articles to return (1-20). Defaults to 10.
    """
    normalized_source = source.strip().upper()
    identifier = identifier.strip()
    if not _SOURCE_RE.fullmatch(normalized_source):
        return f"Europe PMC citations failed: invalid source: {source!r}"
    if not identifier:
        return "Europe PMC citations failed: identifier cannot be empty."

    try:
        data = _fetch_json(
            f"{_BASE_URL}/{urllib.parse.quote(normalized_source)}/{urllib.parse.quote(identifier)}/citations",
            {"format": "json", "pageSize": str(_bounded(max_results))},
        )
        citations = _citations_from_data(data)
    except urllib.error.HTTPError as e:
        return _http_error("Europe PMC citations failed", e)
    except urllib.error.URLError as e:
        return f"Europe PMC citations failed: URL error: {e.reason}"
    except TimeoutError:
        return "Europe PMC citations failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Europe PMC citations failed: could not parse API response: {e}"

    if not citations:
        return f"No Europe PMC citations found for {normalized_source}/{identifier}."
    total = _string(data.get("hitCount")) or str(len(citations))
    return (
        f"Europe PMC citations for {normalized_source}/{identifier} "
        f"(returned {len(citations)}, total {total}):\n"
        + _format_articles(citations, include_abstract=False)
    )


def _fetch_json(url: str, params: dict[str, str]) -> dict[str, Any]:
    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(request_url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


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
        return "Europe PMC article lookup failed: identifier cannot be empty."
    if source and not _SOURCE_RE.fullmatch(source):
        return f"Europe PMC article lookup failed: invalid source: {source!r}"
    if identifier.upper().startswith("PMC"):
        query = f"PMCID:{identifier}"
    elif identifier.lower().startswith("10."):
        query = f'DOI:"{identifier}"'
    else:
        query = f"EXT_ID:{identifier}"
    if source:
        query = f"SRC:{source} AND {query}"
    return query


def _full_text_urls(value: Any) -> tuple[str, ...]:
    urls: list[str] = []
    items = value.get("fullTextUrl") if isinstance(value, dict) else []
    for item in items or []:
        if isinstance(item, dict):
            url = _string(item.get("url"))
            if url:
                urls.append(url)
    return tuple(urls)


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 429:
        return f"{prefix}: rate limited by Europe PMC (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


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
