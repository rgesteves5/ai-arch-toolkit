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
  their points wait for a seam of their own (T09).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from tests.toolkit import wiki_pages
from tests.toolkit.wiki_pages import MISSING_PAGE

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
    """

    args: Mapping[str, Any]
    answers: tuple[Answer, ...] = ()
    files: Mapping[str, str] = field(default_factory=dict)


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
    json_extract list_directory nvd_cve open_food_facts_compare open_food_facts_nutrition
    open_food_facts_product open_library_isbn open_library_work openfda_food_recall
    pdb_chemical_component pdb_entry pdb_ligands pubmed_article read_file ror_organization
    rxnorm_concept rxnorm_ndcs rxnorm_related semantic_scholar_citations semantic_scholar_paper
    uniprot_crossrefs uniprot_entry uniprot_features uniprot_sequence who_indicator who_series
    wiki_outline wiki_read wikidata_entity wiktionary_entry world_bank_compare
    world_bank_indicator world_bank_series youtube_transcript youtube_transcript_languages
    youtube_transcript_search
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

# The 38 tools that never cut what they return (plan, annex A): point 1, the window, does not
# apply. Every other tool cuts, and owes a window until a case proves it.
_WHOLE = """
    air_quality_current base64_decode base64_encode chembl_molecule chembl_target country_info
    date_add date_diff date_format datetime_now distance_between earthquake_count
    earthquake_event eonet_categories foodon_term gbif_species gbif_species_match get_forecast
    get_forecast_by_coords get_weather get_weather_by_coords ip_lookup json_extract math_eval
    openfda_food_recall pdb_chemical_component pdb_entry pdb_ligands python_repl
    reverse_geocode rxnorm_concept text_stats timezone_convert timezone_lookup unit_convert
    uniprot_sequence weather_units who_indicator
"""
WHOLE = frozenset(_WHOLE.split())

# Network modules whose errors no source documents, and why: point 2, the source's errors, does
# not apply to their tools. Every other network tool owes it until its source's answers prove it.
SOURCELESS: dict[str, str] = {
    "_web": "any URL the caller gives: no one source documents its errors",
    "_youtube": "youtube-transcript-api, not the HTTP door: it waits for a seam of its own (T09)",
}

# Integer parameters with no limit to declare, and why. Every other one needs a ``Range``.
UNBOUNDED: dict[tuple[str, str], str] = {}


# --- Point 1: the window ---------------------------------------------------------------------

_WIKI_LONG = wiki_pages.parse_answer("Long page", wiki_pages.long_page())
_OUTLINED = wiki_pages.parse_answer("Long page", wiki_pages.many_sections())
_ENTRY_LONG = wiki_pages.parse_answer(wiki_pages.ENTRY_TERM, wiki_pages.long_entry())

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
}


# --- Point 3: zero results ------------------------------------------------------------------

ZERO_CASES: dict[str, ZeroCase] = {
    "wiki_search": ZeroCase(
        args={"query": "zzqqxx"},
        answers=(Answer(body=wiki_pages.search_answer([], total=0)),),
        says="zzqqxx",
    ),
}
