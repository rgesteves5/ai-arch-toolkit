"""arXiv tools: search papers, and read one paper's record (T06).

The API answers an Atom feed whose OpenSearch elements say how many papers match
(``totalResults``) and where the page starts (``startIndex``); a search pages by ``start``
through the first 30,000 results (https://info.arxiv.org/help/api/user-manual.html, "Paging").
A search shows each paper's whole summary; a paper's record gives every field and every author,
through the window (D39).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from typing import Annotated

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


def _feed_error(reply: Reply) -> ToolFailure | str | None:
    """The error an arXiv feed reports; ``None`` for a feed of papers.

    The API answers a request it cannot run with a feed whose one entry is the error
    (https://info.arxiv.org/help/api/user-manual.html#34-errors), with HTTP 400 (seen
    2026-09-30): a request it could not read, a ``validation_error``. An error feed with any
    other status is the source failing, in its words.
    """
    if not isinstance(reply.body, str):
        return None
    try:
        entries = ET.fromstring(reply.body).findall(f"{_ATOM}entry")
    except ET.ParseError:
        return None
    error = next((error for entry in entries if (error := _entry_error(entry))), None)
    if error is None:
        return None
    return _rejected(error) if reply.status == 400 else error


def _rejected(error: str) -> ToolFailure:
    """The failure of a request arXiv refused, with its reason.

    Every error the user manual lists is about the request (a malformed ID, a bad offset or
    count, a query it cannot read), so the caller's arguments are what to fix.
    """
    msg = (
        f"arXiv rejected the request: {error}; check the ID or the query syntax "
        "(https://info.arxiv.org/help/api/user-manual.html#query_details)."
    )
    return ToolFailure("validation_error", msg)


# arXiv asks for "a 3 second delay" between calls (https://info.arxiv.org/help/api/user-manual.html).
_API = Api(
    base="https://export.arxiv.org/api/query",
    name="arXiv",
    min_interval_s=3.0,
    error_reader=_feed_error,
)
# A search reaches the first 30,000 results (the user manual, "Paging").
_DEPTH = 30_000
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
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
_ERROR_ID_RE = re.compile(r"^https?://arxiv\.org/api/errors(?:#.*)?$")


@dataclass(frozen=True, slots=True)
class _Author:
    name: str
    affiliations: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _ArxivPaper:
    """An arXiv entry, as the feed gives it."""

    paper_id: str
    title: str
    authors: tuple[_Author, ...]
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


@dataclass(frozen=True, slots=True, kw_only=True)
class _Feed:
    """A feed's papers, how many entries it held (papers or not), and how many papers match in
    all (``None`` when it does not say)."""

    papers: list[_ArxivPaper]
    entries: int
    total: int | None


@tool(capability="network")
def arxiv_search(
    query: str,
    max_results: Annotated[int, Range(1, 20)] = 5,
    start: Annotated[int, Range(0, _DEPTH - 1)] = 0,
    category: str = "",
    sort_by: str = "relevance",
    sort_order: str = "descending",
    from_date: str = "",
    to_date: str = "",
) -> ToolResult:
    """Search arXiv papers: each with its whole summary, numbered, with the total.

    Args:
        query: Search text or arXiv API query syntax, e.g. "LLM agents" or "ti:agent".
        max_results: How many papers to list.
        start: How many results to skip; the footer gives the next start.
        category: An arXiv category to keep, e.g. "cs.AI" or "stat.ML".
        sort_by: relevance, lastUpdatedDate or submittedDate.
        sort_order: ascending or descending.
        from_date: The earliest submission date, YYYY-MM-DD.
        to_date: The latest submission date, YYYY-MM-DD.

    Raises:
        ToolFailure: validation_error when an argument is invalid or arXiv rejects the query;
            upstream when arXiv fails.
    """
    query = query.strip()
    if not query:
        msg = "query cannot be empty; pass search text such as 'LLM agents' or 'ti:agent'."
        raise ToolFailure("validation_error", msg)
    sort_by = sort_by.strip() or "relevance"
    sort_order = sort_order.strip() or "descending"
    _validate_search_options(category, sort_by, sort_order)
    # The last page within arXiv's reach asks for what is left of it.
    count = max_results if start + max_results <= _DEPTH else _DEPTH - start
    params = {
        "search_query": _build_search_query(query, category, from_date, to_date),
        "start": str(start),
        "max_results": str(count),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    found = _API.get_text(params=params, parse=_parse_atom)
    return _search_answer(found, query, start, max_results)


@tool(capability="network")
def arxiv_paper(
    arxiv_id: str,
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, MAX_CHARS)] = DEFAULT_CHARS,
) -> ToolResult:
    """Read an arXiv paper's record: its whole summary, every author with the affiliations,
    the categories, the journal reference, the DOI and the links.

    Args:
        arxiv_id: An arXiv ID, e.g. "1706.03762" or "1706.03762v1", or an arXiv URL.
        offset: Where to start, in characters of the record; the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when the ID is malformed (here or by arXiv); not_found
            when arXiv has no paper with it; upstream when arXiv fails.
    """
    paper_id = _normalize_arxiv_id(arxiv_id)
    if not paper_id:
        msg = f"invalid arXiv ID {arxiv_id!r}; an arXiv ID looks like 1706.03762 or 1706.03762v1."
        raise ToolFailure("validation_error", msg)
    params = {"id_list": paper_id, "start": "0", "max_results": "1"}
    papers = _API.get_text(params=params, parse=_parse_atom).papers
    if not papers:
        msg = f"no arXiv paper with ID {paper_id}; search with arxiv_search."
        raise ToolFailure("not_found", msg)
    text = _record_text(papers[0])
    return record(text, heading=f"arXiv paper {paper_id}:", offset=offset, max_chars=max_chars)


def _search_answer(found: _Feed, query: str, start: int, max_results: int) -> ToolResult:
    if not found.papers and start == 0:
        return ToolResult.success(f"No arXiv papers match {query!r}.")
    if not found.entries and found.total is not None and start < found.total:
        msg = (
            f"arXiv sent an empty page at start={start} though it counts {found.total} results; "
            "send the same call again (its API sends empty pages at times)"
        )
        raise ToolFailure("upstream", msg, retryable=True)
    blocks = [_result_block(start + n, paper) for n, paper in enumerate(found.papers, start=1)]
    end = start + found.entries  # an entry that held no paper still took its place
    # A feed without its total may have more after a full page.
    more = found.entries >= max_results if found.total is None else end < found.total
    next_call = {"start": end} if found.entries and more and end < _DEPTH else None
    window = list_window(blocks, first=start + 1, total=found.total, next_call=next_call)
    return window.result(heading=f"arXiv papers that match {query!r}:")


def _validate_search_options(category: str, sort_by: str, sort_order: str) -> None:
    if category and not _VALID_CATEGORIES.fullmatch(category.strip()):
        msg = f"invalid category {category!r}; an arXiv category looks like cs.AI or stat.ML."
        raise ToolFailure("validation_error", msg)
    if sort_by not in {"relevance", "lastUpdatedDate", "submittedDate"}:
        msg = f"invalid sort_by {sort_by!r}; use relevance, lastUpdatedDate or submittedDate."
        raise ToolFailure("validation_error", msg)
    if sort_order not in {"ascending", "descending"}:
        msg = f"invalid sort_order {sort_order!r}; use ascending or descending."
        raise ToolFailure("validation_error", msg)


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
    if date_filter:
        parts.append(date_filter)

    return " AND ".join(parts)


def _build_submitted_date_filter(from_date: str, to_date: str) -> str:
    """The ``submittedDate:[… TO …]`` filter, in GMT to the minute (the user manual)."""
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
            msg = f"invalid from_date {from_date!r}; use YYYY-MM-DD."
            raise ToolFailure("validation_error", msg)
        start = f"{parsed_start:%Y%m%d}0000"
    if to_date:
        parsed_end = _parse_date(to_date)
        if parsed_end is None:
            msg = f"invalid to_date {to_date!r}; use YYYY-MM-DD."
            raise ToolFailure("validation_error", msg)
        end = f"{parsed_end:%Y%m%d}2359"
    if parsed_start and parsed_end and parsed_start > parsed_end:
        msg = f"from_date {from_date} must be before or equal to to_date {to_date}."
        raise ToolFailure("validation_error", msg)
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


def _parse_atom(xml_text: str) -> _Feed:
    root = ET.fromstring(xml_text)
    papers: list[_ArxivPaper] = []
    entries = root.findall(f"{_ATOM}entry")
    for entry in entries:
        if error := _entry_error(entry):
            raise _rejected(error)
        if paper := _paper(entry):
            papers.append(paper)
    total = (root.findtext(f"{_OPENSEARCH}totalResults") or "").strip()
    return _Feed(
        papers=papers, entries=len(entries), total=int(total) if total.isdigit() else None
    )


def _paper(entry: ET.Element) -> _ArxivPaper | None:
    """The paper an entry holds; ``None`` for an entry with no title and no summary."""
    title = _normalize_text(_text(entry, "title"))
    summary = _normalize_text(_text(entry, "summary"))
    if not title and not summary:
        return None
    entry_id = _text(entry, "id")
    paper_id = _id_from_abs_url(entry_id)
    return _ArxivPaper(
        paper_id=paper_id,
        title=title or "(untitled)",
        authors=tuple(_authors(entry)),
        summary=summary,
        published=_date_only(_text(entry, "published")),
        updated=_date_only(_text(entry, "updated")),
        primary_category=_primary_category(entry),
        categories=tuple(
            term
            for category in entry.findall(f"{_ATOM}category")
            if (term := category.attrib.get("term", ""))
        ),
        abs_url=_normalize_abs_url(entry_id, paper_id),
        pdf_url=_pdf_url(entry, paper_id),
        doi=_optional_text(entry, f"{_ARXIV}doi"),
        journal_ref=_optional_text(entry, f"{_ARXIV}journal_ref"),
        comment=_optional_text(entry, f"{_ARXIV}comment"),
    )


def _authors(entry: ET.Element) -> list[_Author]:
    authors: list[_Author] = []
    for author in entry.findall(f"{_ATOM}author"):
        name = _normalize_text(_text(author, "name"))
        if name:
            places = author.findall(f"{_ARXIV}affiliation")
            affiliations = tuple(_normalize_text(place.text or "") for place in places)
            authors.append(_Author(name, tuple(a for a in affiliations if a)))
    return authors


def _entry_error(entry: ET.Element) -> str | None:
    """The error an entry reports, in its summary; ``None`` for a paper."""
    if not _ERROR_ID_RE.fullmatch(_text(entry, "id").strip()):
        return None
    return _normalize_text(_text(entry, "summary")) or "unknown error"


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


def _meta(paper: _ArxivPaper) -> str:
    meta = [f"arXiv: {paper.paper_id}"]
    if paper.primary_category:
        meta.append(paper.primary_category)
    if paper.published:
        meta.append(f"published: {paper.published}")
    if paper.updated and paper.updated != paper.published:
        meta.append(f"updated: {paper.updated}")
    return " | ".join(meta)


def _notes(paper: _ArxivPaper) -> list[str]:
    """The comment, the journal reference and the DOI, one line each, when given."""
    labelled = (("Comment", paper.comment), ("Journal", paper.journal_ref), ("DOI", paper.doi))
    return [f"{label}: {value}" for label, value in labelled if value]


def _links(paper: _ArxivPaper) -> str:
    return " | ".join(link for link in (paper.abs_url, paper.pdf_url) if link)


def _result_block(number: int, paper: _ArxivPaper) -> str:
    """A search result: the paper, its whole summary, and the first authors."""
    lines = [f"{number}. {paper.title}", _meta(paper)]
    if paper.authors:
        whole = call("arxiv_paper", paper.paper_id)
        lines.append(f"Authors: {names([a.name for a in paper.authors], whole=whole)}")
    if paper.summary:
        lines.append(f"Summary: {paper.summary}")
    lines.extend(_notes(paper))
    if links := _links(paper):
        lines.append(f"Links: {links}")
    return "\n".join([lines[0], *(f"   {line}" for line in lines[1:])])


def _record_text(paper: _ArxivPaper) -> str:
    """The whole record: the summary before the author list, which can run to thousands."""
    lines = [paper.title, _meta(paper)]
    if paper.categories:
        primary = paper.primary_category
        tagged = [f"{c} (primary)" if c == primary else c for c in paper.categories]
        lines.append(f"Categories: {', '.join(tagged)}")
    if paper.summary:
        lines.append(f"Summary: {paper.summary}")
    if paper.authors:
        people = [
            f"{a.name} ({'; '.join(a.affiliations)})" if a.affiliations else a.name
            for a in paper.authors
        ]
        lines.append(f"Authors ({len(people)}): {', '.join(people)}")
    lines.extend(_notes(paper))
    if links := _links(paper):
        lines.append(f"Links: {links}")
    return "\n".join(lines) + "\n"


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _date_only(value: str) -> str:
    if "T" in value:
        return value.split("T", 1)[0]
    return value
