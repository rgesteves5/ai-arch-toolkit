"""PubMed tools — public biomedical literature search and metadata lookup."""

from __future__ import annotations

import html
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_ESEARCH_URL = f"{_BASE_URL}/esearch.fcgi"
_EFETCH_URL = f"{_BASE_URL}/efetch.fcgi"
_TIMEOUT = 10
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_TOOL_NAME = "ai_arch_toolkit"
_MAX_RESULTS_LIMIT = 20
_ABSTRACT_MAX_CHARS = 900
_MIN_REQUEST_INTERVAL_SECONDS = 0.34
_LAST_REQUEST_AT = 0.0
_SORT_VALUES = {
    "relevance": "relevance",
    "pub_date": "pub date",
    "first_author": "first author",
    "journal": "journal",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class _PubmedArticle:
    """Normalized metadata for a PubMed article."""

    pmid: str
    doi: str
    title: str
    authors: tuple[str, ...]
    journal: str
    published: str
    abstract: str
    mesh_terms: tuple[str, ...]
    publication_types: tuple[str, ...]
    keywords: tuple[str, ...]


@tool
def pubmed_search(
    query: str,
    max_results: int = 5,
    start: int = 0,
    from_date: str = "",
    to_date: str = "",
    sort: str = "relevance",
) -> str:
    """Search PubMed using the public NCBI E-utilities API.

    Args:
        query: PubMed search text or native PubMed query syntax.
        max_results: Number of articles to return (1-20). Defaults to 5.
        start: Zero-based result offset for pagination. Defaults to 0.
        from_date: Optional publication date lower bound as YYYY-MM-DD.
        to_date: Optional publication date upper bound as YYYY-MM-DD.
        sort: Sort order: relevance, pub_date, first_author, or journal.
    """
    query = query.strip()
    if not query:
        return "PubMed search failed: query cannot be empty."
    if start < 0:
        return "PubMed search failed: start must be greater than or equal to 0."

    sort = sort.strip() or "relevance"
    if sort not in _SORT_VALUES:
        return (
            "PubMed search failed: sort must be one of relevance, pub_date, first_author, journal."
        )

    date_params = _build_date_params(from_date, to_date)
    if isinstance(date_params, str):
        return date_params

    max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retstart": str(start),
        "retmax": str(max_results),
        "sort": _SORT_VALUES[sort],
    }
    params.update(date_params)

    try:
        data = _fetch_json(_ESEARCH_URL, params)
        id_list = data.get("esearchresult", {}).get("idlist", [])
        pmids = [str(pmid).strip() for pmid in id_list if str(pmid).strip()]
        if not pmids:
            return f"No PubMed results for: {query!r}"
        articles = _fetch_articles(pmids)
    except urllib.error.HTTPError as e:
        return f"PubMed search failed: HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"PubMed search failed: URL error: {e.reason}"
    except TimeoutError:
        return "PubMed search failed: request timed out."
    except json.JSONDecodeError as e:
        return f"PubMed search failed: could not parse API response: {e}"
    except ET.ParseError as e:
        return f"PubMed search failed: could not parse article XML: {e}"

    if not articles:
        return f"No PubMed article metadata found for: {query!r}"

    return f"PubMed results for {query!r}:\n" + _format_articles(articles, include_abstract=False)


@tool
def pubmed_article(pmid: str) -> str:
    """Fetch PubMed metadata for a specific article by PMID.

    Args:
        pmid: PubMed identifier, e.g. "26017442".
    """
    normalized = pmid.strip()
    if not normalized.isdigit():
        return f"PubMed article lookup failed: invalid PMID: {pmid!r}"

    try:
        articles = _fetch_articles([normalized])
    except urllib.error.HTTPError as e:
        return f"PubMed article lookup failed: HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"PubMed article lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "PubMed article lookup failed: request timed out."
    except ET.ParseError as e:
        return f"PubMed article lookup failed: could not parse article XML: {e}"

    if not articles:
        return f"PubMed article not found: {normalized}"

    return f"PubMed article {normalized}:\n" + _format_articles(
        articles,
        include_index=False,
        include_abstract=True,
        include_terms=True,
    )


def _build_date_params(from_date: str, to_date: str) -> dict[str, str] | str:
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
            return f"PubMed search failed: invalid from_date {from_date!r}. Use YYYY-MM-DD."
        params["mindate"] = _format_ncbi_date(parsed_start)
    if to_date:
        parsed_end = _parse_date(to_date)
        if parsed_end is None:
            return f"PubMed search failed: invalid to_date {to_date!r}. Use YYYY-MM-DD."
        params["maxdate"] = _format_ncbi_date(parsed_end)
    if parsed_start and parsed_end and parsed_start > parsed_end:
        return "PubMed search failed: from_date must be before or equal to to_date."
    return params


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _format_ncbi_date(value: date) -> str:
    return f"{value:%Y/%m/%d}"


def _fetch_json(url: str, params: dict[str, str]) -> dict[str, Any]:
    text = _fetch_text(url, params)
    return json.loads(text)


def _fetch_articles(pmids: list[str]) -> list[_PubmedArticle]:
    xml_text = _fetch_text(
        _EFETCH_URL,
        {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        },
    )
    return _parse_pubmed_xml(xml_text)


def _fetch_text(url: str, params: dict[str, str]) -> str:
    params = {
        **params,
        "tool": _TOOL_NAME,
    }
    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(request_url, headers={"User-Agent": _USER_AGENT})
    _throttle()
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _throttle() -> None:
    global _LAST_REQUEST_AT

    now = time.monotonic()
    elapsed = now - _LAST_REQUEST_AT
    if elapsed < _MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    _LAST_REQUEST_AT = time.monotonic()


def _parse_pubmed_xml(xml_text: str) -> list[_PubmedArticle]:
    root = ET.fromstring(xml_text)
    articles: list[_PubmedArticle] = []
    for node in root.findall(".//PubmedArticle"):
        article = _parse_article(node)
        if article:
            articles.append(article)
    return articles


def _parse_article(node: ET.Element) -> _PubmedArticle | None:
    pmid = _text(node.find("./MedlineCitation/PMID"))
    if not pmid:
        return None

    article = node.find("./MedlineCitation/Article")
    if article is None:
        return None

    return _PubmedArticle(
        pmid=pmid,
        doi=_article_id(node, "doi"),
        title=_clean_text(_element_text(article.find("./ArticleTitle"))) or "(untitled)",
        authors=_authors(article),
        journal=_journal(article),
        published=_published_date(article),
        abstract=_abstract(article),
        mesh_terms=_mesh_terms(node),
        publication_types=_publication_types(article),
        keywords=_keywords(node),
    )


def _authors(article: ET.Element) -> tuple[str, ...]:
    authors: list[str] = []
    for author in article.findall("./AuthorList/Author"):
        collective = _text(author.find("./CollectiveName"))
        if collective:
            authors.append(collective)
            continue

        last = _text(author.find("./LastName"))
        fore = _text(author.find("./ForeName"))
        initials = _text(author.find("./Initials"))
        given = fore or initials
        name = " ".join(part for part in (given, last) if part)
        if name:
            authors.append(name)
    return tuple(authors)


def _journal(article: ET.Element) -> str:
    return (
        _text(article.find("./Journal/Title"))
        or _text(article.find("./Journal/ISOAbbreviation"))
        or _text(article.find("./Journal/MedlineTA"))
    )


def _published_date(article: ET.Element) -> str:
    article_date = article.find("./ArticleDate")
    if article_date is not None:
        formatted = _date_from_node(article_date)
        if formatted:
            return formatted

    pub_date = article.find("./Journal/JournalIssue/PubDate")
    if pub_date is not None:
        return _date_from_node(pub_date)
    return ""


def _date_from_node(node: ET.Element) -> str:
    year = _text(node.find("./Year"))
    month = _normalize_month(_text(node.find("./Month")))
    day = _text(node.find("./Day")).zfill(2)
    medline_date = _text(node.find("./MedlineDate"))

    parts = [year]
    if month:
        parts.append(month)
    if day != "00":
        parts.append(day)
    if year:
        return "-".join(parts)
    return medline_date


def _normalize_month(value: str) -> str:
    if not value:
        return ""
    if value.isdigit():
        month = int(value)
        if 1 <= month <= 12:
            return str(month).zfill(2)
        return ""
    month_names = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    return month_names.get(value[:3].lower(), "")


def _abstract(article: ET.Element) -> str:
    parts: list[str] = []
    for abstract_text in article.findall("./Abstract/AbstractText"):
        text = _clean_text(_element_text(abstract_text))
        if not text:
            continue
        label = abstract_text.attrib.get("Label", "").strip()
        if label:
            parts.append(f"{label}: {text}")
        else:
            parts.append(text)
    return " ".join(parts)


def _mesh_terms(node: ET.Element) -> tuple[str, ...]:
    terms: list[str] = []
    for heading in node.findall("./MedlineCitation/MeshHeadingList/MeshHeading"):
        descriptor = _text(heading.find("./DescriptorName"))
        qualifiers = [_text(q) for q in heading.findall("./QualifierName")]
        qualifiers = [q for q in qualifiers if q]
        if descriptor and qualifiers:
            terms.append(f"{descriptor} ({', '.join(qualifiers)})")
        elif descriptor:
            terms.append(descriptor)
    return tuple(terms)


def _publication_types(article: ET.Element) -> tuple[str, ...]:
    return tuple(
        _text(publication_type)
        for publication_type in article.findall("./PublicationTypeList/PublicationType")
        if _text(publication_type)
    )


def _keywords(node: ET.Element) -> tuple[str, ...]:
    return tuple(
        _text(keyword)
        for keyword in node.findall("./MedlineCitation/KeywordList/Keyword")
        if _text(keyword)
    )


def _article_id(node: ET.Element, id_type: str) -> str:
    for article_id in node.findall("./PubmedData/ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType") == id_type:
            return _text(article_id)
    return ""


def _format_articles(
    articles: list[_PubmedArticle],
    *,
    include_index: bool = True,
    include_abstract: bool = False,
    include_terms: bool = False,
) -> str:
    blocks: list[str] = []
    for index, article in enumerate(articles, start=1):
        title = f"{index}. {article.title}" if include_index else article.title
        lines = [title]

        meta: list[str] = [f"PMID: {article.pmid}"]
        if article.doi:
            meta.append(f"DOI: {article.doi}")
        if article.published:
            meta.append(f"published: {article.published}")
        lines.append("   " + " | ".join(meta))

        if article.authors:
            lines.append(f"   Authors: {_format_authors(article.authors)}")
        if article.journal:
            lines.append(f"   Journal: {article.journal}")
        if include_abstract and article.abstract:
            lines.append(f"   Abstract: {_truncate(article.abstract, _ABSTRACT_MAX_CHARS)}")
        if include_terms and article.publication_types:
            lines.append("   Publication types: " + ", ".join(article.publication_types[:8]))
        if include_terms and article.mesh_terms:
            lines.append("   MeSH: " + ", ".join(article.mesh_terms[:12]))
        if include_terms and article.keywords:
            lines.append("   Keywords: " + ", ".join(article.keywords[:12]))
        lines.append(f"   URL: https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/")

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_authors(authors: tuple[str, ...]) -> str:
    if len(authors) <= 8:
        return ", ".join(authors)
    return ", ".join(authors[:8]) + f", ... (+{len(authors) - 8} more)"


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


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"
