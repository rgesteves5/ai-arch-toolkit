"""Answers of the health sources, shaped as their documentation gives them, for the contract's
cases (``contract_cases.py``, T07a): ClinicalTrials.gov, RxNav, DailyMed, openFDA, Open Food Facts
and EMBL-EBI OLS."""

from __future__ import annotations

from typing import Any


def studies(*ids: int, token: str = "", total: int | None = None) -> dict[str, Any]:
    """A ClinicalTrials.gov search page (https://clinicaltrials.gov/api/oas/v2)."""
    page: dict[str, Any] = {"studies": [study(n) for n in ids]}
    if token:
        page["nextPageToken"] = token
    if total is not None:
        page["totalCount"] = total
    return page


def study(n: int = 1, *, sites: int = 0) -> dict[str, Any]:
    """A ClinicalTrials.gov study record, with ``sites`` locations."""
    return {
        "protocolSection": {
            "identificationModule": {"nctId": f"NCT{n:08d}", "briefTitle": f"Study {n}"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "contactsLocationsModule": {
                "locations": [
                    {"facility": f"Hospital {site}", "city": "Lisbon", "country": "Portugal"}
                    for site in range(sites)
                ]
            },
        }
    }


def concepts(group: str, count: int) -> dict[str, Any]:
    """An RxNav answer with ``count`` concepts under ``group`` (``drugGroup``,
    ``allRelatedGroup``; https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html)."""
    properties = [
        {"rxcui": str(1000 + n), "name": f"drug {n}", "tty": "SCD"} for n in range(count)
    ]
    return {group: {"conceptGroup": [{"tty": "SCD", "conceptProperties": properties}]}}


def ndcs(count: int) -> dict[str, Any]:
    """RxNav's NDCs of a concept, in the CMS 11-digit form."""
    return {"ndcGroup": {"ndcList": {"ndc": [f"{n:011d}" for n in range(count)]}}}


def labels(count: int, *, total: int, pages: int) -> dict[str, Any]:
    """A page of DailyMed's label search (counts as text, as DailyMed sends them;
    https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm)."""
    data = [
        {"title": f"LABEL {n}", "setid": f"{n:08d}-0000-0000-0000-000000000000"}
        for n in range(count)
    ]
    return {
        "metadata": {"total_elements": str(total), "total_pages": str(pages)},
        "data": data,
    }


def spl(sections: int, *, paragraphs: int = 1) -> str:
    """An SPL document with ``sections`` top-level sections of ``paragraphs`` paragraphs each."""
    body = "".join(
        f"<component><section><code code='42229-5' displayName='SPL UNCLASSIFIED SECTION'/>"
        f"<title>{n} PART</title><text>"
        + "".join(f"<paragraph>Paragraph {p} of part {n}.</paragraph>" for p in range(paragraphs))
        + "</text></section></component>"
        for n in range(1, sections + 1)
    )
    return (
        "<document xmlns='urn:hl7-org:v3'><title>LABEL</title>"
        f"<component><structuredBody>{body}</structuredBody></component></document>"
    )


def recalls(count: int, *, total: int, skip: int = 0) -> dict[str, Any]:
    """A page of openFDA food recalls (https://open.fda.gov/apis/query-parameters/)."""
    results = [
        {"recall_number": f"F-{1000 + skip + n}-2020", "product_description": "Cookies"}
        for n in range(count)
    ]
    return {"meta": {"results": {"skip": skip, "total": total}}, "results": results}


def products(count: int, *, total: int, first: int = 0) -> dict[str, Any]:
    """A page of the Open Food Facts search (``count`` is every match;
    https://github.com/openfoodfacts/openfoodfacts-server/blob/main/docs/api/ref/api.yaml)."""
    found = [{"code": f"{first + n:013d}", "product_name": f"Food {n}"} for n in range(count)]
    return {"count": total, "page_count": count, "products": found}


def terms(count: int, *, total: int, start: int = 0) -> dict[str, Any]:
    """A page of OLS's search (Solr's ``response``)."""
    docs = [
        {"label": f"food {start + n}", "obo_id": f"FOODON:{start + n:08d}"} for n in range(count)
    ]
    return {"response": {"numFound": total, "start": start, "docs": docs}}
