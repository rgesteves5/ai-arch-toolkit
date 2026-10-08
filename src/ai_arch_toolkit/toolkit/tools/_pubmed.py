"""PubMed tools: search articles, and read one article's record by PMID (T06).

ESearch says how many articles match (``count``) and pages by ``retstart`` through the first
10,000 (https://www.nlm.nih.gov/pubs/techbull/so22/so22_updated_pubmed_e_utilities.html:
``retstart + retmax <= 10,000``); EFetch gives each article's XML
(https://www.ncbi.nlm.nih.gov/books/NBK25499/). An article's record gives its whole abstract
and every author, MeSH heading and keyword, through the window (D39).
"""

from __future__ import annotations

import html
import xml.etree.ElementTree as ET
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
    names,
    record,
)
from ai_arch_toolkit.toolkit.tools._window import list_window


def _esearch_error(reply: Reply) -> str | None:
    """The error an ESearch answer reports in place of a result; ``None`` for a result.

    ESearch answers ``ERROR`` instead of the count and ids
    (https://eutils.ncbi.nlm.nih.gov/eutils/dtd/20060628/esearch.dtd), with HTTP 200 (seen
    2026-09-29). A query that matches nothing is a result, with notes under ``warninglist``.
    The error does not say whose fault it is (a malformed term or a backend that failed), so it
    stays in ESearch's words, ``upstream``. Any other error status keeps the door's reading.
    """
    data = reply.body
    result = data.get("esearchresult") if isinstance(data, dict) else None
    error = result.get("ERROR") if isinstance(result, dict) else None
    return _clean_text(str(error)) if error else None


# NCBI asks clients without an API key for at most three requests a second
# (https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/).
_EUTILS = Api(
    base="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    name="NCBI E-utilities",
    min_interval_s=0.34,
    params={"tool": "ai_arch_toolkit"},
    error_reader=_esearch_error,
)
# ESearch reaches the first 10,000 records of a query.
_DEPTH = 10_000
_SORT_VALUES = {
    "relevance": "relevance",
    "pub_date": "pub date",
    "first_author": "first author",
    "journal": "journal",
}
_MONTHS = {
    name: f"{number:02d}"
    for number, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}


@dataclass(frozen=True, slots=True, kw_only=True)
class _PubmedArticle:
    """A PubMed article, as EFetch gives it."""

    pmid: str
    pmcid: str
    doi: str
    title: str
    authors: tuple[str, ...]
    authors_in_full: tuple[str, ...]
    journal: str
    published: str
    abstract: tuple[str, ...]
    mesh_terms: tuple[str, ...]
    publication_types: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Search:
    """ESearch's page: the PMIDs, how many match, and the phrases it did not find."""

    ids: list[str]
    count: int | None
    not_found: tuple[str, ...]


@tool(capability="network")
def pubmed_search(
    query: str,
    max_results: Annotated[int, Range(1, 20)] = 5,
    start: Annotated[int, Range(0, _DEPTH - 1)] = 0,
    from_date: str = "",
    to_date: str = "",
    sort: str = "relevance",
) -> ToolResult:
    """Search PubMed articles, numbered, with the total.

    Args:
        query: PubMed search text or native PubMed query syntax.
        max_results: How many articles to list.
        start: How many results to skip; the footer gives the next start.
        from_date: The earliest publication date, YYYY-MM-DD.
        to_date: The latest publication date, YYYY-MM-DD.
        sort: relevance, pub_date, first_author or journal.

    Raises:
        ToolFailure: validation_error when the query is empty, ``sort`` is unknown or a date is
            not YYYY-MM-DD or out of order; upstream when ESearch reports an error for the query.
    """
    query = query.strip()
    if not query:
        raise ToolFailure("validation_error", "query cannot be empty; give PubMed search text.")
    sort = sort.strip() or "relevance"
    if sort not in _SORT_VALUES:
        raise ToolFailure(
            "validation_error",
            f"unknown sort {sort!r}; use one of relevance, pub_date, first_author, journal.",
        )
    # The last page within ESearch's reach asks for what is left of it.
    count = max_results if start + max_results <= _DEPTH else _DEPTH - start
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retstart": str(start),
        "retmax": str(count),
        "sort": _SORT_VALUES[sort],
        **_build_date_params(from_date, to_date),
    }
    found = _EUTILS.get_json("esearch.fcgi", params=params, parse=_search)
    if not found.ids and start == 0:
        missed = f" PubMed did not find: {', '.join(found.not_found)}." if found.not_found else ""
        return ToolResult.success(f"No PubMed articles match {query!r}.{missed}")
    articles = {article.pmid: article for article in _articles(found.ids)} if found.ids else {}
    blocks = [
        _result_block(start + n, pmid, articles.get(pmid)) for n, pmid in enumerate(found.ids, 1)
    ]
    end = start + len(found.ids)
    more = found.count is not None and end < found.count
    next_call = {"start": end} if found.ids and more and end < _DEPTH else None
    window = list_window(blocks, first=start + 1, total=found.count, next_call=next_call)
    return window.result(heading=f"PubMed articles that match {query!r}:")


@tool(capability="network")
def pubmed_article(
    pmid: str,
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, MAX_CHARS)] = DEFAULT_CHARS,
) -> ToolResult:
    """Read a PubMed article's record by PMID: its whole abstract, every author with the
    affiliations, the MeSH headings, the keywords and the publication types.

    Args:
        pmid: A PubMed identifier, e.g. "26017442".
        offset: Where to start, in characters of the record; the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when ``pmid`` is not all digits; not_found when PubMed
            has no article with it; upstream when EFetch reports an error.
    """
    normalized = pmid.strip()
    if not normalized.isdigit():
        raise ToolFailure(
            "validation_error",
            f"invalid PMID {pmid!r}; a PMID is digits only, e.g. '26017442'.",
        )
    articles = _articles([normalized])
    if not articles:
        raise ToolFailure(
            "not_found", f"no PubMed article with PMID {normalized}; search with pubmed_search."
        )
    heading = f"PubMed article {normalized}:"
    return record(_record_text(articles[0]), heading=heading, offset=offset, max_chars=max_chars)


def _build_date_params(from_date: str, to_date: str) -> dict[str, str]:
    from_date = from_date.strip()
    to_date = to_date.strip()
    if not from_date and not to_date:
        return {}

    parsed_start: date | None = None
    parsed_end: date | None = None
    params = {"datetype": "pdat"}
    if from_date:
        parsed_start = _parse_date(from_date)
        if parsed_start is None:
            raise ToolFailure(
                "validation_error", f"invalid from_date {from_date!r}; use YYYY-MM-DD."
            )
        params["mindate"] = f"{parsed_start:%Y/%m/%d}"
    if to_date:
        parsed_end = _parse_date(to_date)
        if parsed_end is None:
            raise ToolFailure("validation_error", f"invalid to_date {to_date!r}; use YYYY-MM-DD.")
        params["maxdate"] = f"{parsed_end:%Y/%m/%d}"
    if parsed_start and parsed_end and parsed_start > parsed_end:
        raise ToolFailure(
            "validation_error",
            f"from_date {from_date} is after to_date {to_date}; swap them or widen the range.",
        )
    return params


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _search(data: dict[str, Any]) -> _Search:
    result = data.get("esearchresult", {})
    ids = [str(pmid).strip() for pmid in result.get("idlist", []) if str(pmid).strip()]
    count = str(result.get("count", "")).strip()
    notes = result.get("warninglist") if isinstance(result.get("warninglist"), dict) else {}
    errors = result.get("errorlist") if isinstance(result.get("errorlist"), dict) else {}
    missed = [
        *(notes.get("quotedphrasesnotfound") or []),
        *(errors.get("phrasesnotfound") or []),
    ]
    not_found = tuple(text for item in missed if (text := _clean_text(str(item))))
    return _Search(ids=ids, count=int(count) if count.isdigit() else None, not_found=not_found)


def _articles(pmids: list[str]) -> list[_PubmedArticle]:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    return _EUTILS.get_text("efetch.fcgi", params=params, parse=_parse_pubmed_xml)


def _parse_pubmed_xml(xml_text: str) -> list[_PubmedArticle]:
    """The articles of an EFetch answer; one that reports an error is that error.

    Raises:
        ToolFailure: upstream when EFetch answers an ``eFetchResult`` with an ``ERROR``.
    """
    root = ET.fromstring(xml_text)
    if root.tag == "eFetchResult":
        said = _clean_text(root.findtext("ERROR") or "") or "no articles and no reason"
        msg = f"PubMed EFetch error: {said}; try again later, or check the PMID with pubmed_search"
        raise ToolFailure("upstream", msg, retryable=True)
    return [
        article for node in root.findall(".//PubmedArticle") if (article := _parse_article(node))
    ]


def _parse_article(node: ET.Element) -> _PubmedArticle | None:
    pmid = _text(node.find("./MedlineCitation/PMID"))
    article = node.find("./MedlineCitation/Article")
    if not pmid or article is None:
        return None
    people = article.findall("./AuthorList/Author")
    return _PubmedArticle(
        pmid=pmid,
        pmcid=_article_id(node, "pmc"),
        doi=_article_id(node, "doi"),
        title=_clean_text(_element_text(article.find("./ArticleTitle"))) or "(untitled)",
        authors=tuple(name for author in people if (name := _author_name(author))),
        authors_in_full=tuple(name for author in people if (name := _author_in_full(author))),
        journal=_journal(article),
        published=_published_date(article),
        abstract=_abstract(article),
        mesh_terms=_mesh_terms(node),
        publication_types=_texts(article.findall("./PublicationTypeList/PublicationType")),
        keywords=_texts(node.findall("./MedlineCitation/KeywordList/Keyword")),
    )


def _author_name(author: ET.Element) -> str:
    collective = _text(author.find("./CollectiveName"))
    if collective:
        return collective
    given = _text(author.find("./ForeName")) or _text(author.find("./Initials"))
    return " ".join(part for part in (given, _text(author.find("./LastName"))) if part)


def _author_in_full(author: ET.Element) -> str:
    """An author with the affiliations."""
    name = _author_name(author)
    places = _texts(author.findall("./AffiliationInfo/Affiliation"))
    return f"{name} ({'; '.join(places)})" if name and places else name


def _journal(article: ET.Element) -> str:
    return (
        _text(article.find("./Journal/Title"))
        or _text(article.find("./Journal/ISOAbbreviation"))
        or _text(article.find("./Journal/MedlineTA"))
    )


def _published_date(article: ET.Element) -> str:
    article_date = article.find("./ArticleDate")
    if article_date is not None and (formatted := _date_from_node(article_date)):
        return formatted
    pub_date = article.find("./Journal/JournalIssue/PubDate")
    return _date_from_node(pub_date) if pub_date is not None else ""


def _date_from_node(node: ET.Element) -> str:
    """A PubMed date in ISO 8601 (year, month and day as far as given); a ``MedlineDate``
    ("2015 Spring") as written."""
    year = _text(node.find("./Year"))
    if not year:
        return _text(node.find("./MedlineDate"))
    month = _normalize_month(_text(node.find("./Month")))
    day = _text(node.find("./Day")).zfill(2)
    parts = [year, month] if month else [year]
    if month and day != "00":
        parts.append(day)
    return "-".join(parts)


def _normalize_month(value: str) -> str:
    if value.isdigit():
        return f"{int(value):02d}" if 1 <= int(value) <= 12 else ""
    return _MONTHS.get(value[:3].lower(), "")


def _abstract(article: ET.Element) -> tuple[str, ...]:
    """The abstract's parts, each with its label (BACKGROUND, METHODS …) when it has one."""
    parts: list[str] = []
    for abstract_text in article.findall("./Abstract/AbstractText"):
        text = _clean_text(_element_text(abstract_text))
        if text:
            label = abstract_text.attrib.get("Label", "").strip()
            parts.append(f"{label}: {text}" if label else text)
    return tuple(parts)


def _mesh_terms(node: ET.Element) -> tuple[str, ...]:
    """Each MeSH heading, marked when it is a major topic, with its qualifiers."""
    terms: list[str] = []
    for heading in node.findall("./MedlineCitation/MeshHeadingList/MeshHeading"):
        descriptor = heading.find("./DescriptorName")
        name = _text(descriptor)
        if not name or descriptor is None:
            continue
        details = ["major"] if descriptor.attrib.get("MajorTopicYN") == "Y" else []
        details += _texts(heading.findall("./QualifierName"))
        terms.append(f"{name} ({'; '.join(details)})" if details else name)
    return tuple(terms)


def _article_id(node: ET.Element, id_type: str) -> str:
    for article_id in node.findall("./PubmedData/ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType") == id_type:
            return _text(article_id)
    return ""


def _ids(article: _PubmedArticle) -> str:
    labelled = (("PMID", article.pmid), ("PMCID", article.pmcid), ("DOI", article.doi))
    meta = [f"{label}: {value}" for label, value in labelled if value]
    if article.published:
        meta.append(f"published: {article.published}")
    return " | ".join(meta)


def _result_block(number: int, pmid: str, article: _PubmedArticle | None) -> str:
    if article is None:
        missing = f"PubMed returned no record for it; try pubmed_article({pmid!r})"
        return f"{number}. PMID {pmid}: {missing}"
    lines = [_ids(article)]
    if article.authors:
        whole = call("pubmed_article", article.pmid)
        lines.append(f"Authors: {names(article.authors, whole=whole)}")
    if article.journal:
        lines.append(f"Journal: {article.journal}")
    return "\n".join([f"{number}. {article.title}", *(f"   {line}" for line in lines)])


def _record_text(article: _PubmedArticle) -> str:
    """The whole record: the abstract before the lists, which can run long."""
    venue = [f"Journal: {article.journal}"] if article.journal else []
    if article.publication_types:
        venue.append(f"Publication types: {', '.join(article.publication_types)}")
    lines = [article.title, _ids(article), " | ".join(venue)]
    if len(article.abstract) == 1:
        lines.append(f"Abstract: {article.abstract[0]}")
    elif article.abstract:
        lines += ["Abstract:", *article.abstract]
    if article.authors_in_full:
        authors = article.authors_in_full
        lines.append(f"Authors ({len(authors)}): {', '.join(authors)}")
    if article.mesh_terms:
        lines.append(f"MeSH: {', '.join(article.mesh_terms)}")
    if article.keywords:
        lines.append(f"Keywords: {', '.join(article.keywords)}")
    lines.append(f"URL: https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/")
    return "\n".join(line for line in lines if line) + "\n"


def _texts(nodes: list[ET.Element]) -> tuple[str, ...]:
    return tuple(text for node in nodes if (text := _text(node)))


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return _clean_text(node.text)


def _element_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return _clean_text("".join(node.itertext()))


def _clean_text(text: str) -> str:
    return " ".join(html.unescape(text).split())
