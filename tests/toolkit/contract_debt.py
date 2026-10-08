"""The contract's debt: each tool that does not keep a point yet, and the points (T04b).

The contract test (``test_tool_contract.py``) fails when a tool owes a point not listed here, and
when a tool keeps a point it still lists: the list only shrinks. Whoever migrates a module (T05 to
T09) deletes its tools' points and nothing else.

Initial size (2026-10-05): 121 tools; window 96, errors 91, not_found 22, zero 42, limits 70.
"""

from __future__ import annotations


def _owes(*points: str) -> frozenset[str]:
    return frozenset(points)


# Definitions of the copied cut helpers (``_truncate``, ``_trim``) left in ``toolkit/tools``.
CUT_HELPERS = 9

DEBT: dict[str, frozenset[str]] = {
    "arxiv_paper": _owes("window", "errors", "not_found"),
    "arxiv_search": _owes("window", "errors", "zero", "limits"),
    "crossref_search": _owes("window", "errors", "zero", "limits"),
    "crossref_work": _owes("window", "errors"),
    "datacite_doi": _owes("window", "errors"),
    "datacite_search": _owes("window", "errors", "zero", "limits"),
    "europe_pmc_article": _owes("window", "errors", "not_found"),
    "europe_pmc_citations": _owes("window", "errors", "not_found", "limits"),
    "europe_pmc_search": _owes("window", "errors", "zero", "limits"),
    "eurostat_compare": _owes("window", "limits"),
    "eurostat_dataset": _owes("window"),
    "eurostat_dataset_search": _owes("window", "errors", "zero", "limits"),
    "eurostat_dimensions": _owes("window", "limits"),
    "eurostat_series": _owes("window", "limits"),
    "gdelt_news_search": _owes("window", "errors", "zero", "limits"),
    "gdelt_timeline": _owes("window", "errors", "zero"),
    "hacker_news": _owes("window", "errors", "limits"),
    "nvd_cve": _owes("window", "errors", "not_found"),
    "nvd_cve_search": _owes("window", "errors", "zero", "limits"),
    "pubmed_article": _owes("window", "errors", "not_found"),
    "pubmed_search": _owes("window", "errors", "zero", "limits"),
    "ror_organization": _owes("window", "errors"),
    "ror_search": _owes("window", "errors", "zero", "limits"),
    "semantic_scholar_citations": _owes("window", "errors", "limits"),
    "semantic_scholar_paper": _owes("window", "errors"),
    "semantic_scholar_search": _owes("window", "errors", "zero", "limits"),
    "who_indicator": _owes("errors", "not_found"),
    "who_indicators": _owes("window", "errors", "zero", "limits"),
    "who_series": _owes("window", "errors", "limits"),
    "wikidata_entity": _owes("window", "errors"),
    "wikidata_search": _owes("window", "zero", "limits"),
    "wikidata_sparql": _owes("window", "errors", "zero", "limits"),
    "world_bank_compare": _owes("window", "errors", "not_found", "limits"),
    "world_bank_countries": _owes("window", "errors", "limits"),
    "world_bank_indicator": _owes("window", "errors"),
    "world_bank_indicators": _owes("window", "errors", "zero", "limits"),
    "world_bank_series": _owes("window", "errors", "not_found", "limits"),
    "world_bank_sources": _owes("window", "errors", "limits"),
    "world_bank_topics": _owes("window", "errors", "limits"),
}
