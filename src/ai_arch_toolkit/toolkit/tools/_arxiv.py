"""arXiv tools — public paper search and metadata lookup."""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date

from ai_arch_toolkit.core import tool

_API_URL = "https://export.arxiv.org/api/query"
_TIMEOUT = 10
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_RESULTS_LIMIT = 20
_SUMMARY_MAX_CHARS = 700
_VALID_CATEGORIES = re.compile(r"^[A-Za-z0-9.-]+$")
_ADVANCED_QUERY_TOKENS = (
    "all:",
    "ti:",
    "au:",
    "abs:",
    "co:",
    "jr:",
    "cat:",
    "rn:",
    "id:",
    "submittedDate:",
    "AND",
    "OR",
    "ANDNOT",
)
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


@dataclass(frozen=True, slots=True, kw_only=True)
class _ArxivPaper:
    """Normalized metadata for an arXiv entry."""

    paper_id: str
    title: str
    authors: tuple[str, ...]
    summary: str
    published: str
    updated: str
    primary_category: str
    categories: tuple[str, ...]
    abs_url: str
    pdf_url: str
    doi: str | None
    journal_ref: str | None
    comment: str | None


@tool
def arxiv_search(
    query: str,
    max_results: int = 5,
    start: int = 0,
    category: str = "",
    sort_by: str = "relevance",
    sort_order: str = "descending",
    from_date: str = "",
    to_date: str = "",
) -> str:
    """Search arXiv papers using the public arXiv API.

    Args:
        query: Search text or arXiv API query syntax, e.g. "LLM agents" or "ti:agent".
        max_results: Number of papers to return (1-20). Defaults to 5.
        start: Zero-based result offset for pagination. Defaults to 0.
        category: Optional arXiv category filter, e.g. "cs.AI" or "stat.ML".
        sort_by: Sort field: relevance, lastUpdatedDate, or submittedDate.
        sort_order: Sort order: ascending or descending.
        from_date: Optional submitted date lower bound as YYYY-MM-DD.
        to_date: Optional submitted date upper bound as YYYY-MM-DD.
    """
    query = query.strip()
    if not query:
        return "arXiv search failed: query cannot be empty."

    max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))
    if start < 0:
        return "arXiv search failed: start must be greater than or equal to 0."
    sort_by = sort_by.strip() or "relevance"
    sort_order = sort_order.strip() or "descending"

    validation_error = _validate_search_options(category, sort_by, sort_order)
    if validation_error:
        return validation_error

    search_query = _build_search_query(query, category, from_date, to_date)
    if search_query.startswith("arXiv search failed:"):
        return search_query

    try:
        xml_text = _fetch_arxiv(
            {
                "search_query": search_query,
                "start": str(start),
                "max_results": str(max_results),
                "sortBy": sort_by,
                "sortOrder": sort_order,
            }
        )
        papers = _parse_atom(xml_text)
    except urllib.error.HTTPError as e:
        return f"arXiv search failed: HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"arXiv search failed: URL error: {e.reason}"
    except TimeoutError:
        return "arXiv search failed: request timed out."
    except ET.ParseError as e:
        return f"arXiv search failed: could not parse API response: {e}"

    if not papers:
        return f"No arXiv results for: {query!r}"

    return f"arXiv results for {query!r}:\n" + _format_papers(papers)


@tool
def arxiv_paper(arxiv_id: str) -> str:
    """Fetch metadata for a specific arXiv paper by ID.

    Args:
        arxiv_id: arXiv identifier, e.g. "1706.03762", "1706.03762v1", or an arXiv URL.
    """
    paper_id = _normalize_arxiv_id(arxiv_id)
    if not paper_id:
        return f"arXiv paper lookup failed: invalid arXiv ID: {arxiv_id!r}"

    try:
        xml_text = _fetch_arxiv({"id_list": paper_id, "start": "0", "max_results": "1"})
        papers = _parse_atom(xml_text)
    except urllib.error.HTTPError as e:
        return f"arXiv paper lookup failed: HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"arXiv paper lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "arXiv paper lookup failed: request timed out."
    except ET.ParseError as e:
        return f"arXiv paper lookup failed: could not parse API response: {e}"

    if not papers:
        return f"arXiv paper not found: {paper_id}"

    return f"arXiv paper {paper_id}:\n" + _format_papers(papers, include_index=False)


def _validate_search_options(category: str, sort_by: str, sort_order: str) -> str:
    if category and not _VALID_CATEGORIES.fullmatch(category.strip()):
        return f"arXiv search failed: invalid category: {category!r}"
    if sort_by not in {"relevance", "lastUpdatedDate", "submittedDate"}:
        return (
            "arXiv search failed: sort_by must be one of "
            "relevance, lastUpdatedDate, submittedDate."
        )
    if sort_order not in {"ascending", "descending"}:
        return "arXiv search failed: sort_order must be ascending or descending."
    return ""


def _build_search_query(
    query: str,
    category: str = "",
    from_date: str = "",
    to_date: str = "",
) -> str:
    parts: list[str] = []
    if category:
        parts.append(f"cat:{category.strip()}")

    if _looks_advanced_query(query):
        parts.append(f"({query})")
    else:
        parts.append(f'all:"{_escape_arxiv_phrase(query)}"')

    date_filter = _build_submitted_date_filter(from_date, to_date)
    if date_filter.startswith("arXiv search failed:"):
        return date_filter
    if date_filter:
        parts.append(date_filter)

    return " AND ".join(parts)


def _build_submitted_date_filter(from_date: str, to_date: str) -> str:
    from_date = from_date.strip()
    to_date = to_date.strip()
    if not from_date and not to_date:
        return ""

    parsed_start: date | None = None
    parsed_end: date | None = None
    start = "197001010000"
    end = "999912312359"
    if from_date:
        parsed_start = _parse_date(from_date)
        if parsed_start is None:
            return f"arXiv search failed: invalid from_date {from_date!r}. Use YYYY-MM-DD."
        start = f"{parsed_start:%Y%m%d}0000"
    if to_date:
        parsed_end = _parse_date(to_date)
        if parsed_end is None:
            return f"arXiv search failed: invalid to_date {to_date!r}. Use YYYY-MM-DD."
        end = f"{parsed_end:%Y%m%d}2359"
    if parsed_start and parsed_end and parsed_start > parsed_end:
        return "arXiv search failed: from_date must be before or equal to to_date."
    return f"submittedDate:[{start} TO {end}]"


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _looks_advanced_query(query: str) -> bool:
    return any(token in query for token in _ADVANCED_QUERY_TOKENS)


def _escape_arxiv_phrase(query: str) -> str:
    return query.replace('"', '\\"')


def _fetch_arxiv(params: dict[str, str]) -> str:
    url = f"{_API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_atom(xml_text: str) -> list[_ArxivPaper]:
    root = ET.fromstring(xml_text)
    papers: list[_ArxivPaper] = []
    for entry in root.findall(f"{_ATOM}entry"):
        entry_id = _text(entry, "id")
        paper_id = _id_from_abs_url(entry_id)
        abs_url = _normalize_abs_url(entry_id, paper_id)
        pdf_url = _pdf_url(entry, paper_id)
        primary_category = _primary_category(entry)
        categories = tuple(
            category.attrib.get("term", "")
            for category in entry.findall(f"{_ATOM}category")
            if category.attrib.get("term")
        )

        papers.append(
            _ArxivPaper(
                paper_id=paper_id,
                title=_normalize_text(_text(entry, "title")),
                authors=tuple(
                    _normalize_text(_text(author, "name"))
                    for author in entry.findall(f"{_ATOM}author")
                    if _text(author, "name")
                ),
                summary=_normalize_text(_text(entry, "summary")),
                published=_date_only(_text(entry, "published")),
                updated=_date_only(_text(entry, "updated")),
                primary_category=primary_category,
                categories=categories,
                abs_url=abs_url,
                pdf_url=pdf_url,
                doi=_optional_text(entry, f"{_ARXIV}doi"),
                journal_ref=_optional_text(entry, f"{_ARXIV}journal_ref"),
                comment=_optional_text(entry, f"{_ARXIV}comment"),
            )
        )
    return papers


def _text(element: ET.Element, tag: str) -> str:
    return element.findtext(f"{_ATOM}{tag}", default="")


def _optional_text(element: ET.Element, tag: str) -> str | None:
    text = element.findtext(tag, default="")
    text = _normalize_text(text)
    return text or None


def _primary_category(entry: ET.Element) -> str:
    category = entry.find(f"{_ARXIV}primary_category")
    if category is not None:
        return category.attrib.get("term", "")
    categories = entry.findall(f"{_ATOM}category")
    if categories:
        return categories[0].attrib.get("term", "")
    return ""


def _pdf_url(entry: ET.Element, paper_id: str) -> str:
    for link in entry.findall(f"{_ATOM}link"):
        href = link.attrib.get("href", "")
        if not href:
            continue
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            return href.replace("http://", "https://")
    if paper_id:
        return f"https://arxiv.org/pdf/{paper_id}"
    return ""


def _normalize_abs_url(entry_id: str, paper_id: str) -> str:
    if entry_id:
        return entry_id.replace("http://", "https://")
    if paper_id:
        return f"https://arxiv.org/abs/{paper_id}"
    return ""


def _id_from_abs_url(url: str) -> str:
    if "/abs/" in url:
        return url.rsplit("/abs/", 1)[1].strip()
    return url.strip()


def _normalize_arxiv_id(arxiv_id: str) -> str:
    value = arxiv_id.strip()
    if not value:
        return ""
    value = value.removeprefix("arXiv:").removeprefix("arxiv:")
    if "/abs/" in value:
        value = value.rsplit("/abs/", 1)[1]
    if "/pdf/" in value:
        value = value.rsplit("/pdf/", 1)[1]
    value = value.removesuffix(".pdf").strip("/")
    if not re.fullmatch(r"[A-Za-z0-9./-]+(?:v\d+)?", value):
        return ""
    return value


def _format_papers(papers: list[_ArxivPaper], include_index: bool = True) -> str:
    blocks: list[str] = []
    for index, paper in enumerate(papers, start=1):
        title = f"{index}. {paper.title}" if include_index else paper.title
        lines = [title]
        meta = [f"arXiv: {paper.paper_id}"]
        if paper.primary_category:
            meta.append(paper.primary_category)
        if paper.published:
            meta.append(f"published: {paper.published}")
        if paper.updated and paper.updated != paper.published:
            meta.append(f"updated: {paper.updated}")
        lines.append("   " + " | ".join(meta))

        if paper.authors:
            lines.append(f"   Authors: {_format_authors(paper.authors)}")
        if paper.summary:
            lines.append(f"   Summary: {_truncate(paper.summary, _SUMMARY_MAX_CHARS)}")
        if paper.comment:
            lines.append(f"   Comment: {paper.comment}")
        if paper.journal_ref:
            lines.append(f"   Journal: {paper.journal_ref}")
        if paper.doi:
            lines.append(f"   DOI: {paper.doi}")

        links = [link for link in (paper.abs_url, paper.pdf_url) if link]
        if links:
            lines.append("   Links: " + " | ".join(links))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_authors(authors: tuple[str, ...]) -> str:
    if len(authors) <= 8:
        return ", ".join(authors)
    return ", ".join(authors[:8]) + f", ... (+{len(authors) - 8} more)"


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _date_only(value: str) -> str:
    if "T" in value:
        return value.split("T", 1)[0]
    return value


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"
