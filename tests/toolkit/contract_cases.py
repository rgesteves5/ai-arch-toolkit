"""What the contract test (``test_tool_contract.py``) knows about each tool, and the cases that
prove it keeps the contract (T00, T04b).

A point of the contract is kept when a case declared here passes; a tool without a case for a point
that applies to it owes that point (``contract_debt.py``). Whoever migrates a module (T05 to T09)
adds its cases here and deletes its lines from the debt.

- A case's arguments are added to the tool's benign ones (``tool_catalog.benign``); its answers are
  the source's, one per request; its files are written in the working directory first.
- A window's next call carries everything the tool needs to number the following page: a source
  that pages by cursor gets the position too (``next: cursor="…", offset=20``), so the second page
  says ``[results 21-40 …]``.
- ``KINDS``, ``WHOLE`` and ``UNBOUNDED`` decide which points apply, so a change to them is reviewed
  like a change to the debt: a tool that stops cutting moves to ``WHOLE`` only when nothing it
  returns is cut any more.
- The ``youtube_*`` tools reach their source through ``youtube-transcript-api``, not ``_http``:
  their cases put a stand-in for the library at the module's one seam (``patches``,
  ``youtube_fakes.LOADER``; T09).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from tests.toolkit import health_pages as health
from tests.toolkit import wiki_pages, youtube_fakes
from tests.toolkit.wiki_pages import MISSING_PAGE
from tests.toolkit.youtube_fakes import LOADER, FakeTranscript, FakeTranscriptList

type Kind = Literal["lookup", "search", "other"]
type Body = dict[str, Any] | list[Any] | str | bytes


@dataclass(frozen=True, slots=True, kw_only=True)
class Answer:
    """One canned answer of the source."""

    body: Body = b""
    status: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class Case:
    """A call and what the source answers to it (one answer per request; the last repeats).

    Attributes:
        args: Arguments that replace the tool's benign ones.
        answers: The source's answers, in turn.
        files: Files to write in the working directory first, by relative path.
        patches: Objects to put in place first, by dotted path: the stand-in for a source the
            tool reaches without ``_http`` (a library).
    """

    args: Mapping[str, Any]
    answers: tuple[Answer, ...] = ()
    files: Mapping[str, str] = field(default_factory=dict)
    patches: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ZeroCase(Case):
    """A search that finds nothing: the text says so ("no", "0", "nothing" …), with ``says`` (the
    query) in it."""

    says: str


# --- What each tool does ------------------------------------------------------------------------

# A lookup answers one resource that may not exist (point 3: ``not_found``); a search answers a
# list that may be empty (point 3: zero results is a success that says so); other tools neither.
_LOOKUPS = """
    arxiv_paper chembl_molecule chembl_target clinical_trial_study country_info crossref_work
    csv_read dailymed_label datacite_doi earthquake_event eonet_event europe_pmc_article
    europe_pmc_citations eurostat_compare eurostat_dataset eurostat_dimensions eurostat_series
    foodon_term gbif_species gbif_species_match get_forecast get_weather internet_archive_item
    json_extract list_directory nvd_cve open_food_facts_compare
    open_food_facts_product open_library_isbn open_library_work openfda_food_recall
    pdb_chemical_component pdb_entry pdb_ligands pubmed_article read_file ror_organization
    rxnorm_concept rxnorm_ndcs rxnorm_related semantic_scholar_citations semantic_scholar_paper
    uniprot_crossrefs uniprot_entry uniprot_features uniprot_sequence who_indicator who_series
    wiki_outline wiki_read wikidata_entity wiktionary_entry world_bank_compare
    world_bank_indicator world_bank_series youtube_transcript youtube_transcript_languages
    youtube_transcript_search
    dailymed_label_text
"""
_SEARCHES = """
    arxiv_search brave_search chembl_activity_search chembl_molecule_search
    chembl_target_search clinical_trials_search crossref_search dailymed_label_search
    datacite_search earthquake_search eonet_events europe_pmc_search eurostat_dataset_search
    foodon_search gbif_occurrence_search gbif_species_search gdelt_news_search gdelt_timeline
    geocode internet_archive_search nvd_cve_search open_food_facts_search open_library_search
    openfda_food_recall_search osm_search_place overpass_pois overpass_query pdb_search
    pubmed_search regex_search ror_search rxnorm_drug_search search_files
    semantic_scholar_search tavily_search uniprot_search who_indicators wiki_search
    wikidata_search wikidata_sparql world_bank_indicators
"""
_OTHERS = """
    air_quality_current air_quality_forecast base64_decode base64_encode date_add date_diff
    date_format datetime_now distance_between earthquake_count eonet_categories
    get_forecast_by_coords get_weather_by_coords hacker_news http_get ip_lookup math_eval
    osm_reverse_geocode python_repl reverse_geocode run_command scrape_text text_stats
    timezone_convert timezone_lookup unit_convert weather_units world_bank_countries
    world_bank_sources world_bank_topics
"""
KINDS: dict[str, Kind] = {
    **dict.fromkeys(_LOOKUPS.split(), "lookup"),
    **dict.fromkeys(_SEARCHES.split(), "search"),
    **dict.fromkeys(_OTHERS.split(), "other"),
}

# The 38 tools that never cut what they return (plan, annex A), and those that stopped cutting
# when their module was migrated (tavily_search: its source serves one page, and its excerpts
# come whole, C08): point 1, the window, does not apply. Every other tool cuts, and owes a window
# until a case proves it.
_WHOLE = """
    air_quality_current base64_decode base64_encode chembl_molecule chembl_target country_info
    date_add date_diff date_format datetime_now distance_between earthquake_count
    earthquake_event eonet_categories foodon_term gbif_species gbif_species_match get_forecast
    get_forecast_by_coords get_weather get_weather_by_coords ip_lookup json_extract math_eval
    openfda_food_recall pdb_chemical_component pdb_entry pdb_ligands python_repl
    reverse_geocode rxnorm_concept tavily_search text_stats timezone_convert timezone_lookup
    unit_convert uniprot_sequence weather_units who_indicator
    open_food_facts_compare open_food_facts_product
"""
WHOLE = frozenset(_WHOLE.split())

# Network modules whose errors no source documents, and why: point 2, the source's errors, does
# not apply to their tools. Every other network tool owes it until its source's answers prove it.
SOURCELESS: dict[str, str] = {
    "_web": "any URL the caller gives: no one source documents its errors",
    "_youtube": (
        "youtube-transcript-api, not the HTTP door: the library's exceptions, not answers, say "
        "what went wrong (test_youtube.py types each one)"
    ),
}

# Integer parameters with no limit to declare, and why. Every other one needs a ``Range``.
UNBOUNDED: dict[tuple[str, str], str] = {}


# --- Point 1: the window ---------------------------------------------------------------------


def _brave_page(titles: list[str], *, more: bool) -> dict[str, Any]:
    """A page of Brave's web results, and whether Brave has more
    (https://api-dashboard.search.brave.com/api-reference/web/search/get)."""
    results = [{"title": t, "url": f"https://{t.lower()}.example/"} for t in titles]
    return {"query": {"more_results_available": more}, "web": {"results": results}}


_WIKI_LONG = wiki_pages.parse_answer("Long page", wiki_pages.long_page())
_OUTLINED = wiki_pages.parse_answer("Long page", wiki_pages.many_sections())
_ENTRY_LONG = wiki_pages.parse_answer(wiki_pages.ENTRY_TERM, wiki_pages.long_entry())

# The web, YouTube and the archives (T09).
_ROWS = "".join(f"row {n}\n" for n in range(300))
_PARAGRAPHS = "".join(f"<p>Paragraph {n}.</p>" for n in range(300))
_LONG_TRANSCRIPT = FakeTranscriptList(manual=FakeTranscript(segments=youtube_fakes.lines(300)))
_CHORUS = FakeTranscriptList(
    manual=FakeTranscript(segments=youtube_fakes.lines(10, word="chorus"))
)
_MANY_TRANSLATIONS = FakeTranscriptList(
    manual=FakeTranscript(translations=[(f"l{n}", f"Language {n}") for n in range(400)])
)


def _archive_search(start: int, *, total: int) -> dict[str, Any]:
    """A page of two advancedsearch.php results (Solr's ``response``)."""
    docs = [{"identifier": f"item{n}", "title": f"Item {n}"} for n in (start, start + 1)]
    return {"response": {"numFound": total, "start": start, "docs": docs}}


_ARCHIVE_ITEM = {
    "metadata": {"identifier": "item", "title": "Item", "mediatype": "texts"},
    "files": [{"name": f"page{n:03d}.jpg", "format": "JPEG"} for n in range(100)],
}


def _library_search(start: int, *, total: int) -> dict[str, Any]:
    """A page of two Open Library search results."""
    docs = [{"key": f"/works/OL{n}W", "title": f"Book {n}"} for n in (start + 1, start + 2)]
    return {"numFound": total, "start": start, "docs": docs}


_SUBJECTS = [f"Subject {n}" for n in range(200)]
_LIBRARY_AUTHORS = {"numFound": 1, "docs": [{"key": "OL1A", "name": "An Author"}]}
_LIBRARY_WORK = {
    "key": "/works/OL45883W",
    "title": "A Work",
    "authors": [{"author": {"key": "/authors/OL1A"}}],
    "subjects": _SUBJECTS,
}
_LIBRARY_EDITION = {
    "key": "/books/OL1M",
    "title": "An Edition",
    "authors": [{"key": "/authors/OL1A"}],
    "subjects": _SUBJECTS,
}


def _record_twice(record: dict[str, Any]) -> tuple[Answer, ...]:
    """A record and its authors' names, for a call and the one its footer names."""
    return tuple(Answer(body=body) for body in (record, _LIBRARY_AUTHORS) * 2)


WINDOW_CASES: dict[str, Case] = {
    "wiki_read": Case(
        args={"title": "Long page", "max_chars": 500}, answers=(Answer(body=_WIKI_LONG),)
    ),
    "wiki_outline": Case(args={"title": "Long page"}, answers=(Answer(body=_OUTLINED),)),
    "wiki_search": Case(
        args={"query": "physics", "max_results": 2},
        answers=(
            Answer(body=wiki_pages.search_answer(["A", "B"], total=5, next_offset=2)),
            Answer(body=wiki_pages.search_answer(["C", "D"], total=5, next_offset=4)),
        ),
    ),
    "wiktionary_entry": Case(
        args={"term": wiki_pages.ENTRY_TERM, "max_chars": 500},
        answers=(Answer(body=_ENTRY_LONG),),
    ),
    # Brave pages by ``offset``, in pages of ``count`` results.
    "brave_search": Case(
        args={"query": "physics", "max_results": 2},
        answers=(
            Answer(body=_brave_page(["Alpha", "Beta"], more=True)),
            Answer(body=_brave_page(["Gamma", "Delta"], more=False)),
        ),
    ),
    "http_get": Case(args={"max_chars": 200}, answers=(Answer(body=_ROWS),)),
    "scrape_text": Case(args={"max_chars": 200}, answers=(Answer(body=_PARAGRAPHS),)),
    "youtube_transcript": Case(
        args={"max_chars": 300}, patches={LOADER: youtube_fakes.library(_LONG_TRANSCRIPT)}
    ),
    "youtube_transcript_search": Case(
        args={"query": "chorus", "max_results": 3},
        patches={LOADER: youtube_fakes.library(_CHORUS)},
    ),
    "youtube_transcript_languages": Case(
        args={}, patches={LOADER: youtube_fakes.library(_MANY_TRANSLATIONS)}
    ),
    "internet_archive_search": Case(
        args={"query": "book", "max_results": 2},
        answers=(
            Answer(body=_archive_search(0, total=5)),
            Answer(body=_archive_search(2, total=5)),
        ),
    ),
    "internet_archive_item": Case(args={"max_chars": 500}, answers=(Answer(body=_ARCHIVE_ITEM),)),
    "open_library_search": Case(
        args={"max_results": 2},
        answers=(
            Answer(body=_library_search(0, total=5)),
            Answer(body=_library_search(2, total=5)),
        ),
    ),
    "open_library_work": Case(args={"max_chars": 500}, answers=_record_twice(_LIBRARY_WORK)),
    "open_library_isbn": Case(args={"max_chars": 500}, answers=_record_twice(_LIBRARY_EDITION)),
    # T07a, health: sources that page (by cursor, page, skip or start) send each page; the
    # tools that page a list they hold whole get it once.
    "clinical_trials_search": Case(
        args={"max_results": 2},
        answers=(
            Answer(body=health.studies(1, 2, token="T2", total=5)),
            Answer(body=health.studies(3, 4, token="T3")),
        ),
    ),
    "clinical_trial_study": Case(
        args={"max_chars": 500}, answers=(Answer(body=health.study(sites=60)),)
    ),
    "rxnorm_drug_search": Case(
        args={"max_results": 10}, answers=(Answer(body=health.concepts("drugGroup", 30)),)
    ),
    "rxnorm_related": Case(
        args={"max_results": 10}, answers=(Answer(body=health.concepts("allRelatedGroup", 30)),)
    ),
    "rxnorm_ndcs": Case(args={"max_results": 10}, answers=(Answer(body=health.ndcs(30)),)),
    "dailymed_label_search": Case(
        args={"max_results": 2},
        answers=(
            Answer(body=health.labels(2, total=5, pages=3)),
            Answer(body=health.labels(2, total=5, pages=3)),
        ),
    ),
    "dailymed_label": Case(args={}, answers=(Answer(body=health.spl(60)),)),
    "dailymed_label_text": Case(
        args={"max_chars": 500}, answers=(Answer(body=health.spl(3, paragraphs=40)),)
    ),
    "openfda_food_recall_search": Case(
        args={"max_results": 2},
        answers=(
            Answer(body=health.recalls(2, total=5)),
            Answer(body=health.recalls(2, total=5, skip=2)),
        ),
    ),
    "open_food_facts_search": Case(
        args={"max_results": 2},
        answers=(
            Answer(body=health.products(2, total=5)),
            Answer(body=health.products(2, total=5, first=2)),
        ),
    ),
    "foodon_search": Case(
        args={"max_results": 2},
        answers=(
            Answer(body=health.terms(2, total=5)),
            Answer(body=health.terms(2, total=5, start=2)),
        ),
    ),
}


# --- Point 3: a resource that does not exist ---------------------------------------------------

# The source answers an unknown resource with a 404, which the tool declares (``missing=``, T02).
_404 = (Answer(status=404, body={"message": "Not Found"}),)


def _missing_by_404(*names: str) -> dict[str, Case]:
    return {name: Case(args={}, answers=_404) for name in names}


NOT_FOUND_CASES: dict[str, Case] = {
    **_missing_by_404(
        "chembl_molecule",
        "chembl_target",
        "clinical_trial_study",
        "crossref_work",
        "dailymed_label",
        "datacite_doi",
        "earthquake_event",
        "eonet_event",
        "eurostat_dataset",
        "eurostat_dimensions",
        "eurostat_series",
        "eurostat_compare",
        "gbif_species",
        "internet_archive_item",
        "open_library_isbn",
        "open_library_work",
        "openfda_food_recall",
        "pdb_entry",
        "pdb_ligands",
        "pdb_chemical_component",
        "ror_organization",
        "rxnorm_concept",
        "rxnorm_related",
        "rxnorm_ndcs",
        "semantic_scholar_paper",
        "semantic_scholar_citations",
        "uniprot_entry",
        "uniprot_features",
        "uniprot_crossrefs",
        "uniprot_sequence",
        "wikidata_entity",
        "who_series",
        "world_bank_indicator",
    ),
    # MediaWiki answers a missing page with an error object in a 200 (en.wikibooks.org,
    # 2026-09-29; https://www.mediawiki.org/wiki/API:Errors_and_warnings).
    **{
        name: Case(args={}, answers=(Answer(body=MISSING_PAGE),))
        for name in ("wiki_outline", "wiki_read", "wiktionary_entry")
    },
    "read_file": Case(args={"path": "missing.txt"}),
    "csv_read": Case(args={"path": "missing.csv"}),
    "list_directory": Case(args={"path": "missing"}),
    # youtube-transcript-api raises VideoUnavailable for a video YouTube does not play
    # (https://github.com/jdepoix/youtube-transcript-api, _errors.py).
    **{
        name: Case(args={}, patches={LOADER: youtube_fakes.raising("VideoUnavailable")})
        for name in (
            "youtube_transcript",
            "youtube_transcript_languages",
            "youtube_transcript_search",
        )
    },
    # T07a, health: a 404 for a label or a product; OLS answers an unknown ID with no terms.
    **_missing_by_404("dailymed_label_text", "open_food_facts_product", "open_food_facts_compare"),
    "foodon_term": Case(args={}, answers=(Answer(body=health.terms(0, total=0)),)),
}


# --- Point 3: zero results ------------------------------------------------------------------

ZERO_CASES: dict[str, ZeroCase] = {
    "wiki_search": ZeroCase(
        args={"query": "zzqqxx"},
        answers=(Answer(body=wiki_pages.search_answer([], total=0)),),
        says="zzqqxx",
    ),
    "brave_search": ZeroCase(
        args={"query": "zzqqxx"},
        answers=(Answer(body=_brave_page([], more=False)),),
        says="zzqqxx",
    ),
    "tavily_search": ZeroCase(
        args={"query": "zzqqxx"},
        answers=(Answer(body={"query": "zzqqxx", "results": [], "usage": {"credits": 1}}),),
        says="zzqqxx",
    ),
    "internet_archive_search": ZeroCase(
        args={"query": "zzqqxx"},
        answers=(Answer(body={"response": {"numFound": 0, "start": 0, "docs": []}}),),
        says="zzqqxx",
    ),
    "open_library_search": ZeroCase(
        args={"query": "zzqqxx"},
        answers=(Answer(body={"numFound": 0, "start": 0, "docs": []}),),
        says="zzqqxx",
    ),
    # T07a, health: openFDA answers a search that matches nothing with a 404 NOT_FOUND
    # (https://github.com/FDA/openfda/blob/master/api/faers/api.js).
    "clinical_trials_search": ZeroCase(
        args={"query": "zzqqxx"},
        answers=(Answer(body={"studies": [], "totalCount": 0}),),
        says="zzqqxx",
    ),
    "rxnorm_drug_search": ZeroCase(
        args={"name": "zzqqxx"},
        answers=(Answer(body={"drugGroup": {"name": None}}),),
        says="zzqqxx",
    ),
    "dailymed_label_search": ZeroCase(
        args={"drug_name": "zzqqxx"},
        answers=(Answer(body=health.labels(0, total=0, pages=0)),),
        says="zzqqxx",
    ),
    "openfda_food_recall_search": ZeroCase(
        args={"query": "zzqqxx"},
        answers=(
            Answer(
                status=404,
                body={"error": {"code": "NOT_FOUND", "message": "No matches found!"}},
            ),
        ),
        says="zzqqxx",
    ),
    "open_food_facts_search": ZeroCase(
        args={"product_name": "zzqqxx"},
        answers=(Answer(body=health.products(0, total=0)),),
        says="zzqqxx",
    ),
    "foodon_search": ZeroCase(
        args={"query": "zzqqxx"}, answers=(Answer(body=health.terms(0, total=0)),), says="zzqqxx"
    ),
}
