"""Semantic Scholar tools — public academic graph search and citation lookup."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_TIMEOUT = 10
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_RESULTS_LIMIT = 20
_ABSTRACT_MAX_CHARS = 900
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", re.IGNORECASE)
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
        "citingPaper.abstract",
        "citingPaper.year",
        "citingPaper.venue",
        "citingPaper.publicationDate",
        "citingPaper.url",
        "citingPaper.externalIds",
        "citingPaper.authors",
        "citingPaper.citationCount",
        "citingPaper.referenceCount",
        "citingPaper.openAccessPdf",
        "citingPaper.fieldsOfStudy",
    ]
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _SemanticScholarPaper:
    """Normalized metadata for a Semantic Scholar paper."""

    paper_id: str
    corpus_id: str
    title: str
    authors: tuple[str, ...]
    abstract: str
    year: int | None
    venue: str
    publication_date: str
    url: str
    external_ids: tuple[tuple[str, str], ...]
    citation_count: int | None
    reference_count: int | None
    influential_citation_count: int | None
    open_access_pdf: str
    fields_of_study: tuple[str, ...]
    publication_types: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _SemanticScholarCitation:
    """Normalized citation edge from Semantic Scholar."""

    paper: _SemanticScholarPaper
    contexts: tuple[str, ...]
    intents: tuple[str, ...]
    is_influential: bool


@tool
def semantic_scholar_search(
    query: str,
    max_results: int = 5,
    start: int = 0,
    year: str = "",
    venue: str = "",
) -> str:
    """Search Semantic Scholar papers using the public Academic Graph API.

    Args:
        query: Search text, such as a title, topic, author, DOI, or citation fragment.
        max_results: Number of papers to return (1-20). Defaults to 5.
        start: Zero-based result offset for pagination. Defaults to 0.
        year: Optional year filter accepted by Semantic Scholar, e.g. "2024" or "2020-2024".
        venue: Optional venue filter, e.g. "NeurIPS" or "Nature".
    """
    query = query.strip()
    if not query:
        return "Semantic Scholar search failed: query cannot be empty."
    if start < 0:
        return "Semantic Scholar search failed: start must be greater than or equal to 0."

    max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))
    params = {
        "query": query,
        "limit": str(max_results),
        "offset": str(start),
        "fields": _PAPER_FIELDS,
    }
    if year.strip():
        params["year"] = year.strip()
    if venue.strip():
        params["venue"] = venue.strip()

    try:
        data = _fetch_json("/paper/search", params)
        items = data.get("data", [])
        papers = [_parse_paper(item) for item in items if isinstance(item, dict)]
        papers = [paper for paper in papers if paper is not None]
    except urllib.error.HTTPError as e:
        return _http_error("Semantic Scholar search failed", e)
    except urllib.error.URLError as e:
        return f"Semantic Scholar search failed: URL error: {e.reason}"
    except TimeoutError:
        return "Semantic Scholar search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Semantic Scholar search failed: could not parse API response: {e}"

    if not papers:
        return f"No Semantic Scholar results for: {query!r}"

    return f"Semantic Scholar results for {query!r}:\n" + _format_papers(
        papers,
        include_abstract=False,
    )


@tool
def semantic_scholar_paper(paper_id: str) -> str:
    """Fetch detailed Semantic Scholar metadata for a paper.

    Args:
        paper_id: Semantic Scholar paper ID, DOI/DOI URL, arXiv ID/URL, PMID, or prefixed ID.
    """
    normalized = _normalize_paper_id(paper_id)
    if not normalized:
        return f"Semantic Scholar paper lookup failed: invalid paper_id: {paper_id!r}"

    try:
        data = _fetch_json(
            f"/paper/{urllib.parse.quote(normalized, safe=':')}", {"fields": _PAPER_FIELDS}
        )
        paper = _parse_paper(data)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"Semantic Scholar paper not found: {normalized}"
        return _http_error("Semantic Scholar paper lookup failed", e)
    except urllib.error.URLError as e:
        return f"Semantic Scholar paper lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "Semantic Scholar paper lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Semantic Scholar paper lookup failed: could not parse API response: {e}"

    if paper is None:
        return f"Semantic Scholar paper not found: {normalized}"

    return f"Semantic Scholar paper {normalized}:\n" + _format_papers(
        [paper],
        include_index=False,
        include_abstract=True,
        include_details=True,
    )


@tool
def semantic_scholar_citations(
    paper_id: str,
    max_results: int = 10,
    start: int = 0,
) -> str:
    """Fetch papers that cite a Semantic Scholar paper.

    Args:
        paper_id: Semantic Scholar paper ID, DOI/DOI URL, arXiv ID/URL, PMID, or prefixed ID.
        max_results: Number of citing papers to return (1-20). Defaults to 10.
        start: Zero-based result offset for pagination. Defaults to 0.
    """
    normalized = _normalize_paper_id(paper_id)
    if not normalized:
        return f"Semantic Scholar citations lookup failed: invalid paper_id: {paper_id!r}"
    if start < 0:
        return (
            "Semantic Scholar citations lookup failed: start must be greater than or equal to 0."
        )

    max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))
    params = {
        "limit": str(max_results),
        "offset": str(start),
        "fields": _CITATION_FIELDS,
    }

    try:
        path = f"/paper/{urllib.parse.quote(normalized, safe=':')}/citations"
        data = _fetch_json(path, params)
        items = data.get("data", [])
        citations = [_parse_citation(item) for item in items if isinstance(item, dict)]
        citations = [citation for citation in citations if citation is not None]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"Semantic Scholar paper not found: {normalized}"
        return _http_error("Semantic Scholar citations lookup failed", e)
    except urllib.error.URLError as e:
        return f"Semantic Scholar citations lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "Semantic Scholar citations lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Semantic Scholar citations lookup failed: could not parse API response: {e}"

    if not citations:
        return f"No Semantic Scholar citations found for: {normalized}"

    return f"Semantic Scholar citations for {normalized}:\n" + _format_citations(citations)


def _fetch_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _normalize_paper_id(value: str) -> str:
    paper_id = value.strip()
    if not paper_id:
        return ""

    lower = paper_id.lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if lower.startswith(prefix):
            return f"DOI:{paper_id[len(prefix) :].strip()}"

    arxiv_match = _ARXIV_URL_RE.search(paper_id)
    if arxiv_match:
        return f"ARXIV:{_strip_arxiv_pdf_suffix(arxiv_match.group(1).strip())}"

    if lower.startswith("doi:"):
        return "DOI:" + paper_id[4:].strip()
    if lower.startswith("arxiv:"):
        return "ARXIV:" + _strip_arxiv_pdf_suffix(paper_id[6:].strip())
    if lower.startswith("pmid:"):
        pmid = paper_id[5:].strip()
        return f"PMID:{pmid}" if pmid.isdigit() else ""
    if lower.startswith("pmcid:"):
        pmcid = paper_id[6:].strip()
        return f"PMCID:{pmcid}" if pmcid else ""
    if lower.startswith("corpusid:"):
        corpus_id = paper_id[9:].strip()
        return f"CorpusId:{corpus_id}" if corpus_id.isdigit() else ""
    if lower.startswith("url:"):
        url = paper_id[4:].strip()
        return f"URL:{url}" if url else ""
    if lower.startswith(("mag:", "acl:", "pubmedcentral:")):
        prefix, raw_id = paper_id.split(":", 1)
        raw_id = raw_id.strip()
        return f"{prefix.upper()}:{raw_id}" if raw_id else ""
    if lower.startswith(("http://", "https://")):
        return f"URL:{paper_id}"
    if paper_id.lower().startswith("10.") and "/" in paper_id and " " not in paper_id:
        return f"DOI:{paper_id}"
    if _ARXIV_ID_RE.fullmatch(paper_id):
        return f"ARXIV:{paper_id}"
    if paper_id.isdigit():
        return f"PMID:{paper_id}"

    return paper_id


def _strip_arxiv_pdf_suffix(value: str) -> str:
    if value.lower().endswith(".pdf"):
        return value[:-4]
    return value


def _parse_paper(data: dict[str, Any]) -> _SemanticScholarPaper | None:
    paper_id = str(data.get("paperId", "")).strip()
    title = str(data.get("title", "")).strip()
    if not paper_id and not title:
        return None

    return _SemanticScholarPaper(
        paper_id=paper_id,
        corpus_id=_string_or_empty(data.get("corpusId")),
        title=title or "(untitled)",
        authors=_authors(data),
        abstract=str(data.get("abstract", "") or "").strip(),
        year=_int_or_none(data.get("year")),
        venue=_venue(data),
        publication_date=str(data.get("publicationDate", "") or "").strip(),
        url=str(data.get("url", "") or "").strip(),
        external_ids=_external_ids(data),
        citation_count=_int_or_none(data.get("citationCount")),
        reference_count=_int_or_none(data.get("referenceCount")),
        influential_citation_count=_int_or_none(data.get("influentialCitationCount")),
        open_access_pdf=_open_access_pdf(data),
        fields_of_study=_fields_of_study(data),
        publication_types=_string_tuple(data.get("publicationTypes")),
    )


def _parse_citation(data: dict[str, Any]) -> _SemanticScholarCitation | None:
    paper_data = data.get("citingPaper")
    if not isinstance(paper_data, dict):
        return None
    paper = _parse_paper(paper_data)
    if paper is None:
        return None
    return _SemanticScholarCitation(
        paper=paper,
        contexts=_string_tuple(data.get("contexts")),
        intents=_string_tuple(data.get("intents")),
        is_influential=bool(data.get("isInfluential")),
    )


def _authors(data: dict[str, Any]) -> tuple[str, ...]:
    authors: list[str] = []
    for author in data.get("authors", []):
        if isinstance(author, dict):
            name = str(author.get("name", "") or "").strip()
            if name:
                authors.append(name)
    return tuple(authors)


def _venue(data: dict[str, Any]) -> str:
    publication_venue = data.get("publicationVenue")
    if isinstance(publication_venue, dict):
        name = str(publication_venue.get("name", "") or "").strip()
        if name:
            return name
    return str(data.get("venue", "") or "").strip()


def _external_ids(data: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    external_ids = data.get("externalIds")
    if not isinstance(external_ids, dict):
        return ()
    pairs: list[tuple[str, str]] = []
    for key, value in external_ids.items():
        if value is None:
            continue
        if isinstance(value, list):
            text = ", ".join(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value).strip()
        if text:
            pairs.append((str(key).strip(), text))
    return tuple(pairs)


def _open_access_pdf(data: dict[str, Any]) -> str:
    pdf = data.get("openAccessPdf")
    if not isinstance(pdf, dict):
        return ""
    return str(pdf.get("url", "") or "").strip()


def _fields_of_study(data: dict[str, Any]) -> tuple[str, ...]:
    fields = list(_string_tuple(data.get("fieldsOfStudy")))
    for item in data.get("s2FieldsOfStudy", []) or []:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "") or "").strip()
        if category:
            fields.append(category)
    return tuple(dict.fromkeys(fields))


def _format_papers(
    papers: list[_SemanticScholarPaper],
    *,
    include_index: bool = True,
    include_abstract: bool = False,
    include_details: bool = False,
) -> str:
    blocks: list[str] = []
    for index, paper in enumerate(papers, start=1):
        title = f"{index}. {paper.title}" if include_index else paper.title
        lines = [title]

        meta: list[str] = []
        if paper.paper_id:
            meta.append(f"paperId: {paper.paper_id}")
        if paper.year is not None:
            meta.append(f"year: {paper.year}")
        if paper.publication_date:
            meta.append(f"published: {paper.publication_date}")
        if meta:
            lines.append("   " + " | ".join(meta))

        if paper.authors:
            lines.append(f"   Authors: {_format_authors(paper.authors)}")
        if paper.venue:
            lines.append(f"   Venue: {paper.venue}")
        counts = _format_counts(paper)
        if counts:
            lines.append(f"   {counts}")
        if paper.external_ids:
            lines.append("   External IDs: " + _format_external_ids(paper.external_ids))
        if include_abstract and paper.abstract:
            lines.append(f"   Abstract: {_truncate(paper.abstract, _ABSTRACT_MAX_CHARS)}")
        if include_details and paper.fields_of_study:
            lines.append("   Fields: " + ", ".join(paper.fields_of_study[:8]))
        if include_details and paper.publication_types:
            lines.append("   Publication types: " + ", ".join(paper.publication_types[:8]))
        if paper.open_access_pdf:
            lines.append(f"   Open PDF: {paper.open_access_pdf}")
        if paper.url:
            lines.append(f"   URL: {paper.url}")

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_citations(citations: list[_SemanticScholarCitation]) -> str:
    blocks: list[str] = []
    for index, citation in enumerate(citations, start=1):
        lines = _format_papers([citation.paper], include_index=False).splitlines()
        lines[0] = f"{index}. {lines[0]}"
        citation_meta: list[str] = []
        if citation.is_influential:
            citation_meta.append("influential")
        if citation.intents:
            citation_meta.append("intents: " + ", ".join(citation.intents[:5]))
        if citation_meta:
            lines.append("   Citation: " + " | ".join(citation_meta))
        if citation.contexts:
            lines.append(f"   Context: {_truncate(citation.contexts[0], 260)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_counts(paper: _SemanticScholarPaper) -> str:
    counts: list[str] = []
    if paper.citation_count is not None:
        counts.append(f"Citations: {paper.citation_count}")
    if paper.influential_citation_count is not None:
        counts.append(f"Influential citations: {paper.influential_citation_count}")
    if paper.reference_count is not None:
        counts.append(f"References: {paper.reference_count}")
    return " | ".join(counts)


def _format_external_ids(external_ids: tuple[tuple[str, str], ...]) -> str:
    preferred = {"DOI", "ArXiv", "PubMed", "PMID", "PMCID", "ACL", "DBLP", "CorpusId"}
    ordered = [item for item in external_ids if item[0] in preferred]
    ordered.extend(item for item in external_ids if item[0] not in preferred)
    return " | ".join(f"{key}: {value}" for key, value in ordered[:8])


def _format_authors(authors: tuple[str, ...]) -> str:
    if len(authors) <= 8:
        return ", ".join(authors)
    return ", ".join(authors[:8]) + f", ... (+{len(authors) - 8} more)"


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 429:
        return f"{prefix}: rate limited by Semantic Scholar (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"
