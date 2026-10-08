"""Semantic Scholar tools: search papers, read one paper's record, and list the papers that
cite one (T06).

The relevance search takes plain text, "no special query syntax", says how many papers match
(``total``) and gives the next ``offset`` (``next``) while there is one, through the first 1,000
results; a paper's citations page the same way, through the first 10,000, without a total
(https://api.semanticscholar.org/api-docs/graph; the 1,000 since 2023-10-31,
https://github.com/allenai/s2-folks/blob/main/API_RELEASE_NOTES.md). A paper's record gives
its whole abstract and every author, through the window (D39).
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


def _s2_error(reply: Reply) -> ToolFailure | None:
    """A parameter Semantic Scholar refuses; ``None`` for any other answer (the door reads its
    ``error`` text).

    A 400's ``error`` is "Unrecognized or unsupported fields: […]", "Unacceptable query params:
    […]" or a message of its own (the API's Error400 schema): the second names an argument the
    caller gave, a ``validation_error``; the first is the tool's fields, the source's to say.
    """
    error = reply.body.get("error") if isinstance(reply.body, dict) else None
    said = " ".join(str(error).split()) if error else ""
    if reply.status != 400 or not said.startswith("Unacceptable query params"):
        return None
    msg = f"Semantic Scholar refused the request: {said}; correct that parameter"
    return ToolFailure("validation_error", msg)


# Without a key, every caller shares one limit, spent on 2026-10-04 (a 429 at the first request);
# a free key gives its holder 1 request per second, in the x-api-key header
# (https://www.semanticscholar.org/product/api/tutorial). The key is optional (D52).
_API = Api(
    base="https://api.semanticscholar.org/graph/v1",
    name="Semantic Scholar",
    segment_safe=":",
    min_interval_s=1.0,
    key_env="SEMANTIC_SCHOLAR_API_KEY",
    key_header="x-api-key",
    key_url="https://www.semanticscholar.org/product/api#api-key-form",
    error_reader=_s2_error,
)
# The relevance search reaches offset + limit = 999 (under 1,000); citations, 9,999.
_SEARCH_DEPTH = 999
_CITATION_DEPTH = 9_999
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", re.IGNORECASE)
_YEAR_RE = re.compile(r"^(?:\d{4}|\d{4}-|-\d{4}|\d{4}-\d{4})$")
# The prefixes the paper endpoints take, by the lower-case name a caller may write.
_PREFIXES = {
    "doi": "DOI",
    "arxiv": "ARXIV",
    "pmid": "PMID",
    "pmcid": "PMCID",
    "corpusid": "CorpusId",
    "url": "URL",
    "mag": "MAG",
    "acl": "ACL",
}
_NUMERIC = frozenset({"PMID", "CorpusId"})
# externalIds keys, and the prefix the paper endpoints take for each.
_ID_PREFIXES = {
    "DOI": "DOI",
    "ArXiv": "ARXIV",
    "PubMed": "PMID",
    "PubMedCentral": "PMCID",
    "CorpusId": "CorpusId",
    "MAG": "MAG",
    "ACL": "ACL",
}
_PAPER_FIELDS = ",".join(
    [
        "paperId",
        "corpusId",
        "title",
        "abstract",
        "year",
        "venue",
        "publicationVenue",
        "publicationTypes",
        "publicationDate",
        "url",
        "externalIds",
        "authors",
        "citationCount",
        "referenceCount",
        "influentialCitationCount",
        "openAccessPdf",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
    ]
)
_CITATION_FIELDS = ",".join(
    [
        "contexts",
        "intents",
        "isInfluential",
        "citingPaper.paperId",
        "citingPaper.corpusId",
        "citingPaper.title",
        "citingPaper.year",
        "citingPaper.venue",
        "citingPaper.publicationDate",
        "citingPaper.url",
        "citingPaper.externalIds",
        "citingPaper.authors",
        "citingPaper.citationCount",
        "citingPaper.referenceCount",
        "citingPaper.openAccessPdf",
    ]
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Paper:
    """A Semantic Scholar paper, as the graph gives it."""

    paper_id: str
    title: str
    authors: tuple[str, ...]
    abstract: str
    year: int | None
    venue: str
    publication_date: str
    url: str
    external_ids: tuple[str, ...]
    citation_count: int | None
    reference_count: int | None
    influential_citation_count: int | None
    open_access_pdf: str
    fields_of_study: tuple[str, ...]
    publication_types: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Citation:
    """A citing paper, with the sentences that cite and why."""

    paper: _Paper
    contexts: tuple[str, ...]
    intents: tuple[str, ...]
    is_influential: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class _Batch[T]:
    """A batch: its items, the total when the API gives one, and the next offset."""

    items: list[T]
    total: int | None = None
    next_offset: int | None = None


@tool(capability="network")
def semantic_scholar_search(
    query: str,
    max_results: Annotated[int, Range(1, 20)] = 5,
    start: Annotated[int, Range(0, _SEARCH_DEPTH - 1)] = 0,
    year: str = "",
    venue: str = "",
) -> ToolResult:
    """Search Semantic Scholar papers by the words of their titles and abstracts, numbered, with
    the total.

    A keyword search: look a paper up by DOI, arXiv ID or PMID with semantic_scholar_paper.

    Args:
        query: Plain search text, such as a title, a topic or an author.
        max_results: How many papers to list.
        start: How many results to skip; the footer gives the next start.
        year: A publication year or range: "2019", "2016-2020", "2010-" or "-2015".
        venue: A venue to keep, e.g. "NeurIPS" or "Nature".

    Raises:
        ToolFailure: validation_error when the query is empty or an identifier, or ``year`` is
            not a year or a range of years.
    """
    query = query.strip()
    if not query:
        raise ToolFailure(
            "validation_error", "query cannot be empty; give a title, topic or author."
        )
    if doi_of(query) or _ARXIV_ID_RE.fullmatch(query) or _ARXIV_URL_RE.search(query):
        msg = (
            f"{query!r} is an identifier, and this search matches words: look the paper up with "
            f"semantic_scholar_paper({query!r})"
        )
        raise ToolFailure("validation_error", msg)
    if year.strip() and not _YEAR_RE.fullmatch(year.strip()):
        msg = f"invalid year {year!r}; give 2019, 2016-2020, 2010- or -2015"
        raise ToolFailure("validation_error", msg)
    # The last page within the search's reach asks for what is left of it.
    limit = max_results if start + max_results <= _SEARCH_DEPTH else _SEARCH_DEPTH - start
    params = {"query": query, "limit": str(limit), "offset": str(start), "fields": _PAPER_FIELDS}
    if year.strip():
        params["year"] = year.strip()
    if venue.strip():
        params["venue"] = venue.strip()
    found = _API.get_json("paper", "search", params=params, parse=_papers)
    if not found.items and start == 0:
        return ToolResult.success(f"No Semantic Scholar papers match {query!r}.")
    blocks = [_paper_block(start + n, p) for n, p in enumerate(found.items, start=1)]
    window = list_window(
        blocks,
        first=start + 1,
        total=found.total,
        next_call=_onward(found, _SEARCH_DEPTH),
    )
    return window.result(heading=f"Semantic Scholar papers that match {query!r}:")


@tool(capability="network")
def semantic_scholar_paper(
    paper_id: str,
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, MAX_CHARS)] = DEFAULT_CHARS,
) -> ToolResult:
    """Read a paper's Semantic Scholar record: its whole abstract, every author, the counts,
    the identifiers, the fields of study and the open PDF.

    Args:
        paper_id: A Semantic Scholar paper ID, a DOI or DOI URL, an arXiv ID or URL, a PMID, or a
            prefixed ID such as "CorpusId:13756489" or "PMCID:PMC4567".
        offset: Where to start, in characters of the record; the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when ``paper_id`` is empty or malformed; not_found when
            Semantic Scholar has no paper with it.
    """
    normalized = _paper_id(paper_id)
    paper = _API.get_json(
        "paper",
        normalized,
        params={"fields": _PAPER_FIELDS},
        parse=_one_paper,
        missing=_missing(normalized),
    )
    heading = f"Semantic Scholar paper {normalized}:"
    return record(_record_text(paper), heading=heading, offset=offset, max_chars=max_chars)


@tool(capability="network")
def semantic_scholar_citations(
    paper_id: str,
    max_results: Annotated[int, Range(1, 20)] = 10,
    start: Annotated[int, Range(0, _CITATION_DEPTH - 1)] = 0,
) -> ToolResult:
    """List the papers that cite a paper, numbered, each with every sentence that cites it.

    Args:
        paper_id: A Semantic Scholar paper ID, a DOI or DOI URL, an arXiv ID or URL, a PMID, or a
            prefixed ID such as "CorpusId:13756489".
        max_results: How many citing papers to list.
        start: How many citing papers to skip; the footer gives the next start.

    Raises:
        ToolFailure: validation_error when ``paper_id`` is empty or malformed; not_found when
            Semantic Scholar has no paper with that ID.
    """
    normalized = _paper_id(paper_id)
    limit = max_results if start + max_results <= _CITATION_DEPTH else _CITATION_DEPTH - start
    params = {"limit": str(limit), "offset": str(start), "fields": _CITATION_FIELDS}
    found = _API.get_json(
        "paper",
        normalized,
        "citations",
        params=params,
        parse=_citations,
        missing=_missing(normalized),
    )
    if not found.items and start == 0:
        return ToolResult.success(f"No papers in Semantic Scholar cite {normalized}.")
    blocks = [_citation_block(start + n, c) for n, c in enumerate(found.items, start=1)]
    next_call = _onward(found, _CITATION_DEPTH)
    # Past the API's reach, the paper's count says how many cannot be read here.
    beyond = next_call is None and found.next_offset is not None
    total = _citation_count(normalized) if beyond else None
    window = list_window(blocks, first=start + 1, total=total, next_call=next_call)
    return window.result(heading=f"Papers in Semantic Scholar that cite {normalized}:")


def _citation_count(paper_id: str) -> int | None:
    """How many papers cite ``paper_id``, as its record counts them."""
    return _API.get_json(
        "paper",
        paper_id,
        params={"fields": "citationCount"},
        parse=lambda data: _int_or_none(data.get("citationCount")),
        missing=_missing(paper_id),
    )


def _onward[T](found: _Batch[T], depth: int) -> dict[str, object] | None:
    """The next call while the API gives a next offset within its reach."""
    if not found.items or found.next_offset is None or found.next_offset >= depth:
        return None
    return {"start": found.next_offset}


def _papers(data: dict[str, Any]) -> _Batch[_Paper]:
    items = [paper for item in data.get("data", []) if (paper := _parse_paper(item))]
    total, onward = _int_or_none(data.get("total")), _int_or_none(data.get("next"))
    return _Batch(items=items, total=total, next_offset=onward)


def _one_paper(data: dict[str, Any]) -> _Paper:
    paper = _parse_paper(data)
    if paper is None:
        raise TypeError("expected a paper with a paperId or a title")
    return paper


def _citations(data: dict[str, Any]) -> _Batch[_Citation]:
    items = [citation for item in data.get("data", []) if (citation := _parse_citation(item))]
    return _Batch(items=items, next_offset=_int_or_none(data.get("next")))


def _paper_id(value: str) -> str:
    """The ID Semantic Scholar takes for ``value``, or a validation_error."""
    normalized = _normalize_paper_id(value)
    if not normalized:
        raise ToolFailure(
            "validation_error",
            f"invalid paper_id {value!r}; give a Semantic Scholar paper ID, a DOI, an arXiv ID, "
            "a PMID or a prefixed ID such as 'CorpusId:123'; find one with "
            "semantic_scholar_search.",
        )
    return normalized


def _missing(paper_id: str) -> str:
    """The not_found message for a paper ID Semantic Scholar does not know (its 404)."""
    return f"no Semantic Scholar paper with ID {paper_id}; search with semantic_scholar_search."


def _normalize_paper_id(value: str) -> str:
    """``value`` as the paper endpoints take it: a paperId as it is, any other ID with its
    prefix (``DOI:``, ``ARXIV:``, ``PMID:``, ``CorpusId:`` …); empty when it is malformed."""
    paper_id = value.strip()
    if not paper_id:
        return ""
    if doi := doi_of(paper_id):
        return f"DOI:{doi}"
    if arxiv := _ARXIV_URL_RE.search(paper_id):
        return f"ARXIV:{arxiv[1].strip().removesuffix('.pdf')}"
    prefix, colon, rest = paper_id.partition(":")
    canonical = _PREFIXES.get(prefix.strip().lower()) if colon else None
    if canonical is not None:
        rest = rest.strip().removesuffix(".pdf") if canonical == "ARXIV" else rest.strip()
        valid = bool(rest) and (canonical not in _NUMERIC or rest.isdigit())
        return f"{canonical}:{rest}" if valid else ""
    if paper_id.lower().startswith(("http://", "https://")):
        return f"URL:{paper_id}"
    if _ARXIV_ID_RE.fullmatch(paper_id):
        return f"ARXIV:{paper_id}"
    return f"PMID:{paper_id}" if paper_id.isdigit() else paper_id


def _parse_paper(data: Any) -> _Paper | None:
    if not isinstance(data, dict):
        return None
    paper_id = _string(data.get("paperId"))
    title = _string(data.get("title"))
    if not paper_id and not title:
        return None
    return _Paper(
        paper_id=paper_id,
        title=title or "(untitled)",
        authors=tuple(
            name
            for author in data.get("authors") or []
            if isinstance(author, dict) and (name := _string(author.get("name")))
        ),
        abstract=_string(data.get("abstract")),
        year=_int_or_none(data.get("year")),
        venue=_venue(data),
        publication_date=_string(data.get("publicationDate")),
        url=_string(data.get("url")),
        external_ids=_external_ids(data.get("externalIds")),
        citation_count=_int_or_none(data.get("citationCount")),
        reference_count=_int_or_none(data.get("referenceCount")),
        influential_citation_count=_int_or_none(data.get("influentialCitationCount")),
        open_access_pdf=_open_access_pdf(data),
        fields_of_study=_fields_of_study(data),
        publication_types=_string_tuple(data.get("publicationTypes")),
    )


def _parse_citation(data: Any) -> _Citation | None:
    paper = _parse_paper(data.get("citingPaper")) if isinstance(data, dict) else None
    if paper is None:
        return None
    return _Citation(
        paper=paper,
        contexts=_string_tuple(data.get("contexts")),
        intents=_string_tuple(data.get("intents")),
        is_influential=bool(data.get("isInfluential")),
    )


def _venue(data: dict[str, Any]) -> str:
    publication_venue = data.get("publicationVenue")
    if isinstance(publication_venue, dict) and (name := _string(publication_venue.get("name"))):
        return name
    return _string(data.get("venue"))


def _external_ids(value: Any) -> tuple[str, ...]:
    """The paper's identifiers that the paper endpoints take, written as they take them
    (``DOI:10.…``, ``ARXIV:…``); the others (DBLP keys and the like) no tool takes."""
    if not isinstance(value, dict):
        return ()
    found = {
        prefix: text
        for key, raw in value.items()
        if (prefix := _ID_PREFIXES.get(str(key))) and (text := _string(raw))
    }
    return tuple(
        f"{prefix}:{found[prefix]}" for prefix in _ID_PREFIXES.values() if prefix in found
    )


def _open_access_pdf(data: dict[str, Any]) -> str:
    pdf = data.get("openAccessPdf")
    return _string(pdf.get("url")) if isinstance(pdf, dict) else ""


def _fields_of_study(data: dict[str, Any]) -> tuple[str, ...]:
    fields = list(_string_tuple(data.get("fieldsOfStudy")))
    fields += [
        category
        for item in data.get("s2FieldsOfStudy") or []
        if isinstance(item, dict) and (category := _string(item.get("category")))
    ]
    return tuple(dict.fromkeys(fields))


def _meta(paper: _Paper) -> str:
    meta = [f"paperId: {paper.paper_id}"] if paper.paper_id else []
    if paper.year is not None:
        meta.append(f"year: {paper.year}")
    if paper.publication_date:
        meta.append(f"published: {paper.publication_date}")
    return " | ".join(meta)


def _counts(paper: _Paper) -> str:
    labelled = (
        ("Citations", paper.citation_count),
        ("Influential citations", paper.influential_citation_count),
        ("References", paper.reference_count),
    )
    return " | ".join(f"{label}: {count}" for label, count in labelled if count is not None)


def _paper_lines(paper: _Paper) -> list[str]:
    """A paper in a list: what tells it apart, and the first authors."""
    lines = [_meta(paper)]
    if paper.authors:
        whole = call("semantic_scholar_paper", paper.paper_id)
        lines.append(f"Authors: {names(paper.authors, whole=whole)}")
    lines += [f"Venue: {paper.venue}" if paper.venue else "", _counts(paper)]
    if paper.external_ids:
        lines.append(f"IDs: {' | '.join(paper.external_ids)}")
    if paper.open_access_pdf:
        lines.append(f"Open PDF: {paper.open_access_pdf}")
    return [line for line in lines if line]


def _paper_block(number: int, paper: _Paper) -> str:
    return "\n".join([f"{number}. {paper.title}", *(f"   {line}" for line in _paper_lines(paper))])


def _citation_block(number: int, citation: _Citation) -> str:
    lines = _paper_lines(citation.paper)
    why = ["influential"] if citation.is_influential else []
    if citation.intents:
        why.append(f"intents: {', '.join(citation.intents)}")
    if why:
        lines.append(f"Citation: {' | '.join(why)}")
    lines += [f'Context: "{context}"' for context in citation.contexts]
    return "\n".join([f"{number}. {citation.paper.title}", *(f"   {line}" for line in lines)])


def _record_text(paper: _Paper) -> str:
    """The whole record: the abstract before the author list, which can run long."""
    lines = [paper.title, _meta(paper), f"Venue: {paper.venue}" if paper.venue else ""]
    lines.append(_counts(paper))
    if paper.external_ids:
        lines.append(f"IDs: {' | '.join(paper.external_ids)}")
    kinds = []
    if paper.publication_types:
        kinds.append(f"Publication types: {', '.join(paper.publication_types)}")
    if paper.fields_of_study:
        kinds.append(f"Fields: {', '.join(paper.fields_of_study)}")
    lines.append(" | ".join(kinds))
    if paper.abstract:
        lines.append(f"Abstract: {paper.abstract}")
    if paper.authors:
        lines.append(f"Authors ({len(paper.authors)}): {', '.join(paper.authors)}")
    if paper.open_access_pdf:
        lines.append(f"Open PDF: {paper.open_access_pdf}")
    if paper.url:
        lines.append(f"URL: {paper.url}")
    return "\n".join(line for line in lines if line) + "\n"


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(text for item in value if (text := _string(item)))


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
