"""Answers of the literature and identifier sources (T06), shaped as their documentation shows
them, for the tests of their tools and the contract cases.

- arXiv: an Atom feed with OpenSearch totals (https://info.arxiv.org/help/api/user-manual.html).
- Crossref: the REST API's JSON envelope, ``status``, ``message-type`` and ``message``
  (https://www.crossref.org/documentation/retrieve-metadata/rest-api/).
"""

from __future__ import annotations

from typing import Any

# --- arXiv -------------------------------------------------------------------------------------

_ARXIV_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom" '
    'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">\n'
)


def arxiv_entry(
    paper_id: str = "1706.03762v7",
    *,
    title: str = "Attention Is All You Need",
    summary: str = "The dominant sequence transduction models are based on complex networks.",
    authors: tuple[str, ...] = ("Ashish Vaswani", "Noam Shazeer"),
) -> str:
    """One paper as the API's feed gives it (https://info.arxiv.org/help/api/user-manual.html)."""
    people = "".join(
        f"<author><name>{name}</name>"
        + ("<arxiv:affiliation>Google Brain</arxiv:affiliation>" if index == 0 else "")
        + "</author>"
        for index, name in enumerate(authors)
    )
    return f"""
  <entry>
    <id>http://arxiv.org/abs/{paper_id}</id>
    <updated>2023-08-02T00:00:00Z</updated>
    <published>2017-06-12T17:57:34Z</published>
    <title>{title}</title>
    <summary>
      {summary}
    </summary>
    {people}
    <arxiv:comment>15 pages, 5 figures</arxiv:comment>
    <arxiv:journal_ref>NeurIPS 2017</arxiv:journal_ref>
    <arxiv:doi>10.48550/arXiv.{paper_id}</arxiv:doi>
    <link href="http://arxiv.org/abs/{paper_id}" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/{paper_id}" rel="related"
      type="application/pdf"/>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
  </entry>"""


def arxiv_feed(*entries: str, total: int | None = None, start: int = 0) -> str:
    """A feed with the OpenSearch totals the API sends."""
    counts = ""
    if total is not None:
        counts = (
            f"<opensearch:totalResults>{total}</opensearch:totalResults>"
            f"<opensearch:startIndex>{start}</opensearch:startIndex>"
            f"<opensearch:itemsPerPage>{len(entries)}</opensearch:itemsPerPage>"
        )
    return _ARXIV_HEAD + "<title>arXiv Query</title>" + counts + "".join(entries) + "\n</feed>\n"


# What the API answered live, with HTTP 400, to a query it could not read (2026-09-30).
ARXIV_ERROR_FEED = b"""\
<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" \
xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <id>https://arxiv.org/</id>
  <title>arXiv Search Results</title>
  <updated>2026-09-29T23:41:08Z</updated>
  <opensearch:itemsPerPage>1</opensearch:itemsPerPage>
  <opensearch:totalResults>1</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <entry>
    <id>https://arxiv.org/api/errors</id>
    <title>Error</title>
    <updated>2026-09-29T23:41:08Z</updated>
    <link href="https://arxiv.org/api/errors" rel="alternate" type="text/html"/>
    <summary>Invalid query string: '( ( )'</summary>
    <author>
      <name>arXiv api core</name>
    </author>
  </entry>
</feed>
"""


def arxiv_long() -> str:
    """An entry whose record runs past one window: a large collaboration."""
    return arxiv_entry(authors=tuple(f"Collaborator number {n}" for n in range(1, 200)))


# --- Crossref ----------------------------------------------------------------------------------


def crossref_item(
    doi: str = "10.5555/example",
    *,
    title: str = "Attention Is All You Need",
    authors: int = 2,
    references: int = 1,
    abstract: str = "<jats:p>The dominant <i>sequence</i> transduction model.</jats:p>",
) -> dict[str, Any]:
    """A work as Crossref deposits it."""
    people: list[dict[str, Any]] = [
        {
            "given": "Ashish",
            "family": "Vaswani",
            "ORCID": "https://orcid.org/0000-0002-1825-0097",
            "affiliation": [{"name": "Google Brain"}],
        },
        {"name": "Noam Shazeer"},
    ]
    people += [{"given": "Given", "family": f"Family {n}"} for n in range(3, authors + 1)]
    return {
        "DOI": doi,
        "title": [title],
        "subtitle": ["Transformer paper"],
        "author": people[:authors],
        "issued": {"date-parts": [[2017, 6, 12]]},
        "container-title": ["Advances in Neural Information Processing Systems"],
        "publisher": "NeurIPS",
        "type": "proceedings-article",
        "URL": f"https://doi.org/{doi}",
        "abstract": abstract,
        "is-referenced-by-count": 1234,
        "reference-count": references,
        "license": [
            {
                "URL": "https://creativecommons.org/licenses/by/4.0/",
                "content-version": "vor",
                "start": {"date-parts": [[2017, 6, 12]]},
            }
        ],
        "link": [
            {
                "URL": "https://content.example/full.pdf",
                "content-type": "application/pdf",
                "intended-application": "text-mining",
            }
        ],
        "reference": [
            {
                "key": f"ref{n}",
                "author": "Smith",
                "article-title": f"Related Work {n}",
                "journal-title": "Journal of Tests",
                "year": "2016",
                "DOI": f"10.5555/ref{n}",
            }
            for n in range(1, references + 1)
        ],
    }


def crossref_list(*items: dict[str, Any], total: int, offset: int = 0) -> dict[str, Any]:
    """A page of ``/works`` results."""
    return {
        "status": "ok",
        "message-type": "work-list",
        "message-version": "1.0.0",
        "message": {
            "facets": {},
            "total-results": total,
            "items": list(items),
            "items-per-page": len(items),
            "query": {"start-index": offset, "search-terms": "agents"},
        },
    }


def crossref_work(item: dict[str, Any]) -> dict[str, Any]:
    """One work, from ``/works/{doi}``."""
    return {"status": "ok", "message-type": "work", "message-version": "1.0.0", "message": item}


# A filter Crossref refuses, with HTTP 400, as its validator words it (CrossRef/cayenne,
# src/cayenne/api/v1/validate.clj).
CROSSREF_REFUSED: dict[str, Any] = {
    "status": "failed",
    "message-type": "validation-failure",
    "message": [
        {
            "type": "type-not-valid",
            "value": "journal",
            "message": "Type specified as journal but must be one of: book-section, monograph",
        }
    ],
}


# --- DataCite ----------------------------------------------------------------------------------


def datacite_item(
    doi: str = "10.5061/dryad.test",
    *,
    title: str = "Example dataset",
    creators: int = 2,
    resource_type: str = "Dataset",
    description: str = "Dataset description.",
) -> dict[str, Any]:
    """A DOI's record as the REST API gives it (https://support.datacite.org/docs/api-get-doi)."""
    people: list[dict[str, Any]] = [
        {"name": "Smith, Jane", "nameType": "Personal", "affiliation": ["University of Tests"]},
        {"name": "Data Team", "nameType": "Organizational", "affiliation": []},
    ]
    people += [{"name": f"Creator {n}", "affiliation": []} for n in range(3, creators + 1)]
    return {
        "id": doi,
        "type": "dois",
        "attributes": {
            "doi": doi,
            "titles": [{"title": title}, {"title": "Exemplo", "titleType": "TranslatedTitle"}],
            "creators": people[:creators],
            "publisher": "Dryad",
            "publicationYear": 2024,
            "types": {"resourceTypeGeneral": resource_type, "resourceType": "Survey data"},
            "descriptions": [
                {"description": description, "descriptionType": "Abstract"},
                {"description": "How it was measured.", "descriptionType": "Methods"},
            ],
            "subjects": [{"subject": f"Subject {n}"} for n in range(1, 13)],
            "url": "https://datadryad.org/example",
            "version": "2",
            "rightsList": [
                {
                    "rights": "Creative Commons Zero v1.0 Universal",
                    "rightsUri": "https://creativecommons.org/publicdomain/zero/1.0/legalcode",
                }
            ],
            "relatedIdentifiers": [
                {
                    "relationType": "IsSupplementTo",
                    "relatedIdentifier": f"10.5555/article{n}",
                    "relatedIdentifierType": "DOI",
                }
                for n in range(1, 8)
            ],
        },
    }


def datacite_list(
    *items: dict[str, Any], total: int, page: int = 1, total_pages: int | None = None
) -> dict[str, Any]:
    """A page of ``/dois`` results, with the ``meta`` totals
    (https://support.datacite.org/docs/pagination)."""
    pages = total_pages if total_pages is not None else max(1, -(-total // max(len(items), 1)))
    return {
        "data": list(items),
        "meta": {"total": total, "totalPages": pages, "page": page},
        "links": {"self": "https://api.datacite.org/dois?query=example"},
    }


def datacite_record(item: dict[str, Any]) -> dict[str, Any]:
    return {"data": item}


# DataCite's errors are JSON:API error objects (https://jsonapi.org/format/#error-objects), with
# the titles its error page lists (https://support.datacite.org/docs/api-error-codes).
DATACITE_RATE_LIMITED: dict[str, Any] = {
    "errors": [{"status": "429", "title": "Your request has been rate limited"}]
}
DATACITE_REFUSED: dict[str, Any] = {"errors": [{"status": "400", "title": "Bad Request"}]}


# --- Europe PMC --------------------------------------------------------------------------------
# The REST API's JSON (https://europepmc.org/RestfulWebService): a search's ``hitCount``,
# ``nextCursorMark`` and ``resultList.result``; a citation list's ``hitCount`` and
# ``citationList.citation``.


def epmc_result(
    article_id: str = "26017442",
    *,
    source: str = "MED",
    authors: int = 3,
    abstract: str = "<p>Deep learning allows computational models.</p>",
) -> dict[str, Any]:
    """An article as a ``core`` result gives it (a ``lite`` one lacks the lists)."""
    people = ["LeCun Y", "Bengio Y", "Hinton G"] + [f"Author{n} A" for n in range(4, authors + 1)]
    return {
        "id": article_id,
        "source": source,
        "pmid": article_id,
        "pmcid": "PMC123",
        "doi": "10.1038/nature14539",
        "title": "Deep <i>learning</i>",
        "authorString": ", ".join(people[:authors]) + ".",
        "authorList": {
            "author": [
                {
                    "fullName": name,
                    "authorAffiliationDetailsList": {
                        "authorAffiliation": [{"affiliation": "NYU"}] if index == 0 else []
                    },
                }
                for index, name in enumerate(people[:authors])
            ]
        },
        "journalTitle": "Nature",
        "pubYear": "2015",
        "firstPublicationDate": "2015-05-28",
        "pubType": "journal article",
        "pubTypeList": {"pubType": ["Review", "Journal Article"]},
        "abstractText": abstract,
        "isOpenAccess": "Y",
        "inEPMC": "Y",
        "inPMC": "N",
        "hasPDF": "Y",
        "hasReferences": "Y",
        "citedByCount": 123,
        "meshHeadingList": {
            "meshHeading": [
                {
                    "majorTopic_YN": "Y",
                    "descriptorName": "Neural Networks, Computer",
                    "meshQualifierList": {"meshQualifier": [{"qualifierName": "trends"}]},
                },
                {"majorTopic_YN": "N", "descriptorName": "Humans"},
            ]
        },
        "keywordList": {"keyword": ["deep learning", "representation"]},
        "fullTextUrlList": {
            "fullTextUrl": [
                {
                    "availability": "Subscription required",
                    "documentStyle": "doi",
                    "site": "DOI",
                    "url": "https://doi.org/10.1038/nature14539",
                },
                {
                    "availability": "Open access",
                    "documentStyle": "pdf",
                    "site": "Europe_PMC",
                    "url": "https://europepmc.org/articles/PMC123?pdf=render",
                },
            ]
        },
    }


def epmc_search(
    *results: dict[str, Any], hit_count: int, cursor: str = "*", next_cursor: str = "AoJ"
) -> dict[str, Any]:
    return {
        "version": "6.9",
        "hitCount": hit_count,
        "nextCursorMark": next_cursor,
        "request": {"queryString": "deep learning", "cursorMark": cursor},
        "resultList": {"result": list(results)},
    }


def epmc_citation(article_id: str = "42207361") -> dict[str, Any]:
    return {
        "id": article_id,
        "source": "MED",
        "citationType": "journal article",
        "title": "Quantum Machine Learning",
        "authorString": "Liu H, Chen J.",
        "journalAbbreviation": "Ann Biomed Eng",
        "pubYear": 2026,
        "citedByCount": 0,
    }


def epmc_citations(*citations: dict[str, Any], hit_count: int) -> dict[str, Any]:
    return {
        "version": "6.9",
        "hitCount": hit_count,
        "request": {"id": "26017442", "source": "MED"},
        "citationList": {"citation": list(citations)},
    }


# An error in place of the result, with HTTP 200 (recorded by K-Dense-AI/scientific-agent-skills,
# paper-lookup/references/europepmc.md; the official page could not be read from here).
EPMC_ERROR: dict[str, Any] = {
    "errCode": 404,
    "errMsg": "Invalid page size provided. Valid size is between 1 and 1000",
}


# --- PubMed (NCBI E-utilities) -----------------------------------------------------------------
# ESearch's JSON (``esearchresult``: ``count``, ``retstart``, ``idlist``) and EFetch's PubMed XML
# (https://www.ncbi.nlm.nih.gov/books/NBK25499/).


def esearch(ids: list[str], *, count: int, start: int = 0) -> dict[str, Any]:
    return {
        "header": {"type": "esearch", "version": "0.3"},
        "esearchresult": {
            "count": str(count),
            "retmax": str(len(ids)),
            "retstart": str(start),
            "idlist": ids,
            "querytranslation": '"deep learning"[All Fields]',
        },
    }


# A query that matches nothing (eutils.ncbi.nlm.nih.gov, 2026-09-29).
ESEARCH_NOTHING: dict[str, Any] = {
    "header": {"type": "esearch", "version": "0.3"},
    "esearchresult": {
        "count": "0",
        "idlist": [],
        "warninglist": {
            "quotedphrasesnotfound": ['"zzqqxxnotaword"'],
            "outputmessages": ["No items found."],
        },
    },
}
# An error in place of the result, with HTTP 200 (eutils.ncbi.nlm.nih.gov, 2026-09-29).
ESEARCH_ERROR: dict[str, Any] = {
    "header": {"type": "esearch", "version": "0.3"},
    "esearchresult": {
        "ERROR": "Search Backend failed: An error occurred while processing request. "
        "Details: Empty Term in the request"
    },
}
# NCBI's answer past its rate limit (https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/
# new-api-keys-for-the-e-utilities/), with HTTP 429.
NCBI_RATE_LIMITED = (
    b'{"error":"API rate limit exceeded","api-key":"1.2.3.4","count":"4","limit":"3"}'
)


def pubmed_article(
    pmid: str = "26017442",
    *,
    authors: int = 2,
    abstract: str = "Deep learning allows computational models.",
) -> str:
    """A ``PubmedArticle`` as EFetch gives it (``retmode=xml``)."""
    people = [("Yann", "LeCun", "Facebook AI Research"), ("Yoshua", "Bengio", "")]
    people += [("Given", f"Author{n}", "") for n in range(3, authors + 1)]
    author_xml = "".join(
        f"<Author><ForeName>{given}</ForeName><LastName>{last}</LastName>"
        + (
            f"<AffiliationInfo><Affiliation>{place}</Affiliation></AffiliationInfo>"
            if place
            else ""
        )
        + "</Author>"
        for given, last, place in people[:authors]
    )
    return f"""
  <PubmedArticle>
    <MedlineCitation>
      <PMID>{pmid}</PMID>
      <Article>
        <Journal>
          <Title>Nature</Title>
          <JournalIssue><PubDate><Year>2015</Year><Month>May</Month><Day>28</Day></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Deep <i>learning</i></ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">{abstract}</AbstractText>
          <AbstractText Label="CONCLUSIONS">It is useful in many domains.</AbstractText>
        </Abstract>
        <AuthorList>{author_xml}</AuthorList>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
          <PublicationType>Review</PublicationType>
        </PublicationTypeList>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName MajorTopicYN="Y">Machine Learning</DescriptorName>
          <QualifierName>methods</QualifierName>
        </MeshHeading>
        <MeshHeading>
          <DescriptorName MajorTopicYN="N">Neural Networks, Computer</DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
      <KeywordList>
        <Keyword>deep learning</Keyword>
        <Keyword>neural networks</Keyword>
      </KeywordList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">{pmid}</ArticleId>
        <ArticleId IdType="doi">10.1038/nature14539</ArticleId>
        <ArticleId IdType="pmc">PMC4567</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>"""


def pubmed_set(*articles: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<PubmedArticleSet>'
        + "".join(articles)
        + "\n</PubmedArticleSet>\n"
    )


# EFetch's error in place of the articles, with HTTP 200: an ``eFetchResult`` with an ``ERROR``,
# as NCBI answers when its history server fails.
EFETCH_ERROR = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    "<eFetchResult><ERROR>Unable to obtain query #1</ERROR></eFetchResult>\n"
)


# --- Semantic Scholar --------------------------------------------------------------------------
# The Academic Graph API (https://api.semanticscholar.org/api-docs/graph): a search batch has
# ``total``, ``offset``, ``next`` and ``data``; a citation batch, ``offset``, ``next`` and
# ``data``; ``next`` is absent on the last batch.


def s2_paper(
    paper_id: str = "649def34f8be52c8b66281af98ae884c09aef38b",
    *,
    authors: int = 2,
    abstract: str = "The dominant sequence transduction models are based on recurrence.",
) -> dict[str, Any]:
    people = [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}]
    people += [{"name": f"Author {n}"} for n in range(3, authors + 1)]
    return {
        "paperId": paper_id,
        "corpusId": 13756489,
        "title": "Attention Is All You Need",
        "abstract": abstract,
        "year": 2017,
        "venue": "NeurIPS",
        "publicationVenue": {"name": "Neural Information Processing Systems"},
        "publicationTypes": ["Conference"],
        "publicationDate": "2017-06-12",
        "url": "https://www.semanticscholar.org/paper/example",
        "externalIds": {
            "DOI": "10.48550/arXiv.1706.03762",
            "ArXiv": "1706.03762",
            "PubMed": "26017442",
            "DBLP": "conf/nips/VaswaniSPUJGKP17",
            "CorpusId": 13756489,
        },
        "authors": people[:authors],
        "citationCount": 123456,
        "referenceCount": 50,
        "influentialCitationCount": 25000,
        "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762"},
        "fieldsOfStudy": ["Computer Science"],
        "s2FieldsOfStudy": [{"category": "Machine Learning"}],
    }


def s2_search(*papers: dict[str, Any], total: int, offset: int = 0) -> dict[str, Any]:
    batch: dict[str, Any] = {"total": total, "offset": offset, "data": list(papers)}
    if offset + len(papers) < total:
        batch["next"] = offset + len(papers)
    return batch


def s2_citation(paper_id: str = "c1", *, contexts: int = 2) -> dict[str, Any]:
    return {
        "contexts": [
            f"Context {n} of the citation, following the Transformer."
            for n in range(1, contexts + 1)
        ],
        "intents": ["background", "methodology"],
        "isInfluential": True,
        "citingPaper": s2_paper(paper_id) | {"title": "A Paper That Cites It"},
    }


def s2_citations(
    *citations: dict[str, Any], offset: int = 0, more: bool = False
) -> dict[str, Any]:
    batch: dict[str, Any] = {"offset": offset, "data": list(citations)}
    if more:
        batch["next"] = offset + len(citations)
    return batch


# The two forms of a 400's ``error`` the API documents (its Error400 schema).
S2_UNACCEPTABLE = {"error": "Unacceptable query params: [year=twenty]"}
S2_UNRECOGNIZED = {"error": "Unrecognized or unsupported fields: [nope]"}


# --- ROR ---------------------------------------------------------------------------------------
# The v2 REST API (https://ror.readme.io/v2/docs/rest-api): a search gives ``number_of_results``
# and a page of 20 ``items`` (ror-community/ror-api, rorapi/settings.py: PAGE_SIZE 20, MAX_PAGE
# 500); a refused request, ``{"errors": [...]}`` (rorapi/common/queries.py).


def ror_org(
    ror_id: str = "https://ror.org/01c27hj86", *, name: str = "University of Lisbon"
) -> dict[str, Any]:
    return {
        "id": ror_id,
        "names": [
            {"value": "ULisboa", "types": ["acronym"], "lang": None},
            {"value": name, "types": ["ror_display", "label"], "lang": "en"},
            {"value": "Universidade de Lisboa", "types": ["label"], "lang": "pt"},
        ],
        "locations": [
            {
                "geonames_id": 2267057,
                "geonames_details": {
                    "name": "Lisbon",
                    "country_subdivision_name": "Lisbon",
                    "country_name": "Portugal",
                    "country_code": "PT",
                    "lat": 38.71667,
                    "lng": -9.13333,
                },
            },
            {
                "geonames_id": 2735943,
                "geonames_details": {
                    "name": "Porto",
                    "country_name": "Portugal",
                    "country_code": "PT",
                    "lat": 41.14961,
                    "lng": -8.61099,
                },
            },
        ],
        "types": ["education", "funder"],
        "status": "active",
        "established": 2013,
        "domains": ["ulisboa.pt"],
        "links": [
            {"type": "website", "value": "https://www.ulisboa.pt"},
            {"type": "wikipedia", "value": "https://en.wikipedia.org/wiki/University_of_Lisbon"},
        ],
        "external_ids": [
            {"type": "grid", "all": ["grid.9983.4"], "preferred": "grid.9983.4"},
            {"type": "isni", "all": ["0000 0001 2181 4263"], "preferred": None},
        ],
        "relationships": [
            {"type": "child", "label": f"Child {n}", "id": f"https://ror.org/0child{n:04d}"}
            for n in range(1, 13)
        ],
        "admin": {
            "created": {"date": "2018-11-14", "schema_version": "1.0"},
            "last_modified": {"date": "2024-12-11", "schema_version": "2.1"},
        },
    }


def ror_orgs(count: int, *, first: int = 0) -> list[dict[str, Any]]:
    """``count`` organizations, each with an ID and a name of its own."""
    return [
        ror_org(f"https://ror.org/0{n:08d}", name=f"Organization {n}")
        for n in range(first, first + count)
    ]


def ror_page(*orgs: dict[str, Any], total: int) -> dict[str, Any]:
    return {"number_of_results": total, "time_taken": 5, "items": list(orgs), "meta": {}}


ROR_REFUSED: dict[str, Any] = {"errors": ["filter key 'colour' is illegal"]}


# --- NVD ---------------------------------------------------------------------------------------
# The CVE API 2.0 (https://nvd.nist.gov/developers/vulnerabilities): ``resultsPerPage``,
# ``startIndex``, ``totalResults`` and the ``vulnerabilities``, each a ``cve`` with its
# ``metrics`` by CVSS version (``cvssMetricV40``, ``V31``, ``V30``, ``V2``).


def nvd_item(
    cve_id: str = "CVE-2021-44228", *, cpes: int = 3, references: int = 2
) -> dict[str, Any]:
    return {
        "cve": {
            "id": cve_id,
            "sourceIdentifier": "security@apache.org",
            "published": "2021-12-10T10:15:09.143",
            "lastModified": "2024-11-21T08:15:28.000",
            "vulnStatus": "Analyzed",
            "descriptions": [
                {
                    "lang": "en",
                    "value": "Apache Log4j2 JNDI features do not protect against LDAP.",
                },
                {"lang": "es", "value": "Las funciones JNDI de Apache Log4j2 no protegen."},
            ],
            "metrics": {
                "cvssMetricV31": [
                    {
                        "source": "nvd@nist.gov",
                        "type": "Primary",
                        "cvssData": {
                            "version": "3.1",
                            "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                            "baseScore": 10.0,
                            "baseSeverity": "CRITICAL",
                        },
                    }
                ],
                "cvssMetricV2": [
                    {
                        "source": "nvd@nist.gov",
                        "type": "Primary",
                        "cvssData": {
                            "version": "2.0",
                            "vectorString": "AV:N/AC:M/Au:N/C:C/I:C/A:C",
                            "baseScore": 9.3,
                        },
                        "baseSeverity": "HIGH",
                    }
                ],
            },
            "weaknesses": [
                {
                    "source": "nvd@nist.gov",
                    "type": "Primary",
                    "description": [{"lang": "en", "value": "CWE-917"}],
                },
                {
                    "source": "security@apache.org",
                    "type": "Secondary",
                    "description": [{"lang": "en", "value": "CWE-502"}],
                },
            ],
            "configurations": [
                {
                    "nodes": [
                        {
                            "operator": "OR",
                            "negate": False,
                            "cpeMatch": [
                                {
                                    "vulnerable": True,
                                    "criteria": f"cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:{n}",
                                    "versionStartIncluding": "2.0.1",
                                    "versionEndExcluding": "2.3.1",
                                    "matchCriteriaId": f"id-{n}",
                                }
                                for n in range(1, cpes + 1)
                            ],
                        }
                    ]
                }
            ],
            "references": [
                {
                    "url": f"https://example.test/advisory/{n}",
                    "source": "security@apache.org",
                    "tags": ["Exploit", "Third Party Advisory"],
                }
                for n in range(1, references + 1)
            ],
        }
    }


def nvd_page(*items: dict[str, Any], total: int, start: int = 0) -> dict[str, Any]:
    return {
        "resultsPerPage": len(items),
        "startIndex": start,
        "totalResults": total,
        "format": "NVD_CVE",
        "version": "2.0",
        "timestamp": "2026-10-08T10:00:00.000",
        "vulnerabilities": list(items),
    }


# NVD says why it refuses a request in the response header ``message``
# (https://nvd.nist.gov/developers/start-here: "examine the response header for a field named
# message"); these words stand in for its own.
NVD_REFUSED_REASON = "The pubStartDate and pubEndDate range must not exceed 120 days."
