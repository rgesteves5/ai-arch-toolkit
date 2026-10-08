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
- ``KINDS``, ``WHOLE``, ``ONCE`` and ``UNBOUNDED`` decide which points apply and how, so a change
  to them is reviewed like a change to the debt: a tool that stops cutting moves to ``WHOLE`` only
  when nothing it returns is cut any more.
- The ``youtube_*`` tools reach their source through ``youtube-transcript-api``, not ``_http``:
  their cases put a stand-in for the library at the module's one seam (``patches``,
  ``youtube_fakes.LOADER``; T09).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from tests.toolkit import biodata_answers as bio
from tests.toolkit import data_bodies, geo_answers, wiki_pages, youtube_fakes
from tests.toolkit import health_pages as health
from tests.toolkit import literature_answers as lit
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
    europe_pmc_citations eurostat_dataset eurostat_series
    foodon_term gbif_species gbif_species_match get_forecast get_weather internet_archive_item
    json_extract list_directory nvd_cve open_food_facts_compare
    open_food_facts_product open_library_isbn open_library_work openfda_food_recall
    pdb_chemical_component pdb_entry pdb_ligands pubmed_article read_file ror_organization
    rxnorm_concept rxnorm_ndcs rxnorm_related semantic_scholar_citations semantic_scholar_paper
    uniprot_crossrefs uniprot_entry uniprot_features uniprot_sequence who_indicator who_series
    wiki_outline wiki_read wikidata_entity wiktionary_entry world_bank_indicator
    world_bank_series youtube_transcript youtube_transcript_languages
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
    date_format datetime_now distance_between earthquake_count eonet_categories hacker_news
    http_get ip_lookup math_eval osm_reverse_geocode python_repl run_command scrape_text
    text_stats timezone_convert timezone_lookup unit_convert world_bank_countries
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
    get_weather ip_lookup json_extract math_eval openfda_food_recall osm_reverse_geocode
    pdb_chemical_component pdb_entry pdb_ligands rxnorm_concept tavily_search text_stats
    timezone_convert timezone_lookup unit_convert uniprot_sequence who_indicator
    open_food_facts_compare open_food_facts_product world_bank_indicator
"""
WHOLE = frozenset(_WHOLE.split())


@dataclass(frozen=True, slots=True)
class Once:
    """Why a tool's output exists only for its call, and the words its footer uses to say how to
    narrow the output, in place of a next call."""

    why: str
    narrow: str


# Tools whose output exists only for the call that made it: reading on would run it again, with
# its effects. Point 1 is a footer that says what was shown, the size and how to narrow the
# output (``Window.rest``), in the words declared here; their window case proves it.
ONCE: dict[str, Once] = {
    "run_command": Once(
        why="a shell command's output: reading on would run the command again",
        narrow="run the command again narrowed",
    ),
    "python_repl": Once(
        why="a program's output: reading on would run the program again",
        narrow="run the code again printing less",
    ),
}

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


def _answers(*bodies: Body) -> tuple[Answer, ...]:
    return tuple(Answer(body=body) for body in bodies)


def _world_bank_pages(**args: Any) -> Case:
    """Two pages of a World Bank list of 6, two items each."""
    first = data_bodies.world_bank_page(
        data_bodies.world_bank_records(2, start=1), page=1, pages=3, total=6
    )
    second = data_bodies.world_bank_page(
        data_bodies.world_bank_records(2, start=3), page=2, pages=3, total=6
    )
    return Case(args={"max_results": 2, **args}, answers=_answers(first, second))


# The data and news sources (T08a): each a list longer than its page, or a source that reads on.
_DATA_NEWS_WINDOWS: dict[str, Case] = {
    "eurostat_dataset_search": Case(
        args={"query": "population", "max_results": 2},
        answers=_answers(data_bodies.eurostat_catalogue(5)),
    ),
    "eurostat_dataset": Case(
        args={"dataset_id": "TPS00001", "dimension": "geo"},
        answers=_answers(data_bodies.eurostat_dataset(70, 1)),
    ),
    "eurostat_series": Case(
        args={"dataset_id": "TPS00001", "max_points": 3},
        answers=_answers(data_bodies.eurostat_dataset(2, 4)),
    ),
    "world_bank_topics": _world_bank_pages(),
    "world_bank_sources": _world_bank_pages(),
    "world_bank_countries": _world_bank_pages(),
    "world_bank_indicators": _world_bank_pages(),
    "world_bank_series": _world_bank_pages(),
    "who_indicators": Case(
        args={"max_results": 2},
        answers=_answers(data_bodies.who_values(3), data_bodies.who_values(3, start=3)),
    ),
    "who_series": Case(
        args={"max_results": 2},
        answers=_answers(data_bodies.who_values(3), data_bodies.who_values(3, start=3)),
    ),
    "gdelt_news_search": Case(
        args={"query": "climate", "max_results": 2},
        answers=_answers(data_bodies.gdelt_articles(3), data_bodies.gdelt_articles(5)),
    ),
    "gdelt_timeline": Case(
        args={"query": "climate"}, answers=_answers(data_bodies.gdelt_timeline(150))
    ),
    "wikidata_search": Case(
        args={"query": "item", "max_results": 2},
        answers=_answers(
            data_bodies.wikidata_search(2, start=1, more=2),
            data_bodies.wikidata_search(2, start=3, more=4),
        ),
    ),
    "wikidata_entity": Case(
        args={"qid": "Q1"},
        answers=_answers(
            data_bodies.wikidata_entity(50),
            data_bodies.wikidata_labels(),
            data_bodies.wikidata_entity(50),
            data_bodies.wikidata_labels(),
        ),
    ),
    "wikidata_sparql": Case(
        args={"query": "SELECT ?item WHERE { ?item ?p ?o } LIMIT 30", "max_results": 10},
        answers=_answers(data_bodies.sparql_rows(30)),
    ),
    "hacker_news": Case(
        args={"count": 2},
        answers=_answers(
            data_bodies.hn_ids(5),
            data_bodies.hn_story(1),
            data_bodies.hn_story(2),
            data_bodies.hn_ids(5),
            data_bodies.hn_story(3),
            data_bodies.hn_story(4),
        ),
    ),
}

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
    # Life sciences (T07b): UniProt, RCSB PDB, ChEMBL, GBIF.
    "uniprot_search": Case(
        args={"query": "insulin", "max_results": 2},
        answers=(
            Answer(
                body=bio.uniprot_hits("P01308", "P01315"), headers=bio.uniprot_headers(5, "c2")
            ),
            Answer(
                body=bio.uniprot_hits("P01317", "P01318"), headers=bio.uniprot_headers(5, "c3")
            ),
        ),
    ),
    "uniprot_entry": Case(args={"max_chars": 500}, answers=(Answer(body=bio.uniprot_entry()),)),
    "uniprot_features": Case(args={"max_results": 5}, answers=(Answer(body=bio.uniprot_entry()),)),
    "uniprot_crossrefs": Case(
        args={"max_results": 5}, answers=(Answer(body=bio.uniprot_entry()),)
    ),
    "pdb_search": Case(
        args={"query": "hemoglobin", "max_results": 2},
        answers=(
            Answer(body=bio.pdb_hits("4HHB", "1A3N", total=5)),
            Answer(body=bio.pdb_entries("4HHB", "1A3N")),
            Answer(body=bio.pdb_hits("2HHB", "3HHB", total=5)),
            Answer(body=bio.pdb_entries("2HHB", "3HHB")),
        ),
    ),
    **{
        name: Case(
            args={"max_results": 2} | ({"query": "aspirin"} if key != "activities" else {}),
            answers=(
                Answer(body=bio.chembl_page(key, count=2, offset=0, total=5)),
                Answer(body=bio.chembl_page(key, count=2, offset=2, total=5)),
            ),
        )
        for name, key in (
            ("chembl_molecule_search", "molecules"),
            ("chembl_target_search", "targets"),
            ("chembl_activity_search", "activities"),
        )
    },
    **{
        name: Case(
            args={"max_results": 2} | ({"query": "Puma"} if kind == "taxa" else {}),
            answers=(
                Answer(body=bio.gbif_page(kind, count=2, offset=0, total=5)),
                Answer(body=bio.gbif_page(kind, count=2, offset=2, total=5)),
            ),
        )
        for name, kind in (
            ("gbif_species_search", "taxa"),
            ("gbif_occurrence_search", "occurrences"),
        )
    },
    # Local tools: their files are written in the working directory first.
    "read_file": Case(
        args={"path": "long.txt", "max_lines": 40},
        files={"long.txt": "".join(f"line {number}\n" for number in range(200))},
    ),
    "list_directory": Case(
        args={"path": "many"},
        files={f"many/{number:04d}.txt": "x" for number in range(1001)},
    ),
    "search_files": Case(
        args={"pattern": "needle", "max_results": 3},
        files={"hay.txt": "".join(f"needle {number}\n" for number in range(10))},
    ),
    "csv_read": Case(
        args={"path": "rows.csv", "max_rows": 20},
        files={"rows.csv": "id,value\n" + "".join(f"{n},{n * 7}\n" for n in range(100))},
    ),
    "regex_search": Case(args={"text": "a" * 3000, "pattern": "a"}),
    # A fixed command, as in test_shell.py: the contract never builds arguments for it.
    "run_command": Case(args={"command": "seq 1 3000", "max_output": 1000}),
    "python_repl": Case(args={"code": "for number in range(6000):\n    print(number)"}),
    # Geo, weather and natural events (T08b). A source with no offset of its own is asked again:
    # Open-Meteo and Nominatim for all they give, EONET for one more each time
    # (``_first_results``).
    "geocode": Case(
        args={"city": "Springfield", "max_results": 2},
        answers=(Answer(body=geo_answers.geocoding(3)),),
    ),
    "osm_search_place": Case(
        args={"query": "Springfield", "max_results": 2},
        answers=(Answer(body=geo_answers.nominatim_places(3)),),
    ),
    "overpass_query": Case(
        args={"max_results": 2}, answers=(Answer(body=geo_answers.overpass_elements(5)),)
    ),
    "overpass_pois": Case(
        args={"max_results": 2}, answers=(Answer(body=geo_answers.overpass_elements(5)),)
    ),
    "air_quality_forecast": Case(
        args={"max_hours": 2}, answers=(Answer(body=geo_answers.air_quality_hours(5)),)
    ),
    "eonet_events": Case(
        args={"max_results": 2}, answers=(Answer(body=geo_answers.eonet_events(3)),)
    ),
    "eonet_event": Case(
        args={"event_id": "EONET_1", "max_points": 2},
        answers=(Answer(body=geo_answers.eonet_event(points=5)),),
    ),
    # USGS: the count of the search, then its page, for each call.
    "earthquake_search": Case(
        args={"max_results": 2},
        answers=(
            Answer(body="3"),
            Answer(body=geo_answers.usgs_features(1, 2)),
            Answer(body="3"),
            Answer(body=geo_answers.usgs_features(3)),
        ),
    ),
    **_DATA_NEWS_WINDOWS,
    # T06: literature and identifiers. A search's next call reads the source's next page; a
    # record's, the next characters of the same record.
    "arxiv_search": Case(
        args={"query": "agents", "max_results": 2},
        answers=(
            Answer(body=lit.arxiv_feed(lit.arxiv_entry("1"), lit.arxiv_entry("2"), total=5)),
            Answer(body=lit.arxiv_feed(lit.arxiv_entry("3"), lit.arxiv_entry("4"), total=5)),
        ),
    ),
    "arxiv_paper": Case(
        args={"max_chars": 500}, answers=(Answer(body=lit.arxiv_feed(lit.arxiv_long(), total=1)),)
    ),
    "crossref_search": Case(
        args={"query": "agents", "max_results": 2},
        answers=(
            Answer(
                body=lit.crossref_list(lit.crossref_item("10.1/a"), lit.crossref_item(), total=5)
            ),
            Answer(
                body=lit.crossref_list(lit.crossref_item("10.1/c"), lit.crossref_item(), total=5)
            ),
        ),
    ),
    "crossref_work": Case(
        args={"max_chars": 500},
        answers=(Answer(body=lit.crossref_work(lit.crossref_item(references=40))),),
    ),
    "datacite_search": Case(
        args={"query": "data", "max_results": 2},
        answers=(
            Answer(
                body=lit.datacite_list(lit.datacite_item("10.1/a"), lit.datacite_item(), total=5)
            ),
            Answer(
                body=lit.datacite_list(
                    lit.datacite_item("10.1/c"), lit.datacite_item(), total=5, page=2
                )
            ),
        ),
    ),
    "datacite_doi": Case(
        args={"max_chars": 500}, answers=(Answer(body=lit.datacite_record(lit.datacite_item())),)
    ),
    "europe_pmc_search": Case(
        args={"query": "learning", "max_results": 2},
        answers=(
            Answer(body=lit.epmc_search(lit.epmc_result("1"), lit.epmc_result("2"), hit_count=5)),
            Answer(
                body=lit.epmc_search(
                    lit.epmc_result("3"), lit.epmc_result("4"), hit_count=5, next_cursor="AoK"
                )
            ),
        ),
    ),
    "europe_pmc_article": Case(
        args={"max_chars": 500},
        answers=(Answer(body=lit.epmc_search(lit.epmc_result(authors=40), hit_count=1)),),
    ),
    "europe_pmc_citations": Case(
        args={"max_results": 2},
        answers=(
            Answer(
                body=lit.epmc_citations(
                    lit.epmc_citation("1"), lit.epmc_citation("2"), hit_count=5
                )
            ),
            Answer(
                body=lit.epmc_citations(
                    lit.epmc_citation("3"), lit.epmc_citation("4"), hit_count=5
                )
            ),
        ),
    ),
    # Each PubMed page is an ESearch, then an EFetch of its PMIDs.
    "pubmed_search": Case(
        args={"query": "learning", "max_results": 2},
        answers=(
            Answer(body=lit.esearch(["1", "2"], count=5)),
            Answer(body=lit.pubmed_set(lit.pubmed_article("1"), lit.pubmed_article("2"))),
            Answer(body=lit.esearch(["3", "4"], count=5, start=2)),
            Answer(body=lit.pubmed_set(lit.pubmed_article("3"), lit.pubmed_article("4"))),
        ),
    ),
    "pubmed_article": Case(
        args={"max_chars": 500},
        answers=(Answer(body=lit.pubmed_set(lit.pubmed_article(authors=40))),),
    ),
    "semantic_scholar_search": Case(
        args={"query": "learning", "max_results": 2},
        answers=(
            Answer(body=lit.s2_search(lit.s2_paper("a"), lit.s2_paper("b"), total=5)),
            Answer(body=lit.s2_search(lit.s2_paper("c"), lit.s2_paper("d"), total=5, offset=2)),
        ),
    ),
    "semantic_scholar_paper": Case(
        args={"max_chars": 500}, answers=(Answer(body=lit.s2_paper(authors=60)),)
    ),
    "semantic_scholar_citations": Case(
        args={"max_results": 2},
        answers=(
            Answer(body=lit.s2_citations(lit.s2_citation("a"), lit.s2_citation("b"), more=True)),
            Answer(body=lit.s2_citations(lit.s2_citation("c"), lit.s2_citation("d"), offset=2)),
        ),
    ),
    # ROR's pages hold 20 organizations.
    "ror_search": Case(
        args={"query": "university"},
        answers=tuple(
            Answer(body=lit.ror_page(*lit.ror_orgs(20, first=start), total=45))
            for start in (0, 20)
        ),
    ),
    "ror_organization": Case(args={"max_chars": 500}, answers=(Answer(body=lit.ror_org()),)),
    "nvd_cve_search": Case(
        args={"query": "log4j", "max_results": 2},
        answers=(
            Answer(body=lit.nvd_page(lit.nvd_item("CVE-2021-1"), lit.nvd_item(), total=5)),
            Answer(body=lit.nvd_page(lit.nvd_item("CVE-2021-3"), lit.nvd_item(), total=5)),
        ),
    ),
    "nvd_cve": Case(
        args={"max_chars": 500}, answers=(Answer(body=lit.nvd_page(lit.nvd_item(), total=1)),)
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
        "eurostat_series",
        "gbif_species",
        "open_library_isbn",
        "open_library_work",
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
    # GBIF's match service answers a name it cannot resolve with matchType NONE, in a 200
    # (https://github.com/gbif/matching-ws, MatchV1Controller).
    "gbif_species_match": Case(
        args={}, answers=(Answer(body={"confidence": 100, "matchType": "NONE"}),)
    ),
    # Open-Meteo's geocoding and Wikidata's search find nothing for the name (T08b).
    "country_info": Case(args={}, answers=(Answer(body={"search": []}),)),
    **{
        name: Case(
            args={"city": "Zzqqxx", "latitude": None, "longitude": None},
            answers=(Answer(body=geo_answers.NO_PLACES),),
        )
        for name in ("get_weather", "get_forecast")
    },
    # The WHO GHO API answers an unknown indicator code with an empty list; the World Bank, a
    # series of an unknown indicator or country with its error 120, in a 200 (2026-09-29).
    "who_indicator": Case(args={}, answers=_answers({"value": []})),
    "world_bank_series": Case(args={}, answers=_answers(data_bodies.WORLD_BANK_INVALID_VALUE)),
    # arXiv answers an unknown ID with an empty feed
    # (https://info.arxiv.org/help/api/user-manual.html).
    "arxiv_paper": Case(args={}, answers=(Answer(body=lit.arxiv_feed(total=0)),)),
    # Europe PMC answers an unknown identifier with no results; its citation list, with none,
    # and the record's search then finds nothing.
    "europe_pmc_article": Case(args={}, answers=(Answer(body=lit.epmc_search(hit_count=0)),)),
    "europe_pmc_citations": Case(
        args={},
        answers=(
            Answer(body=lit.epmc_citations(hit_count=0)),
            Answer(body=lit.epmc_search(hit_count=0)),
        ),
    ),
    # EFetch answers an unknown PMID with an empty article set.
    "pubmed_article": Case(args={}, answers=(Answer(body=lit.pubmed_set()),)),
    # NVD answers an unknown CVE ID with no vulnerabilities.
    "nvd_cve": Case(args={}, answers=(Answer(body=lit.nvd_page(total=0)),)),
    "read_file": Case(args={"path": "missing.txt"}),
    "csv_read": Case(args={"path": "missing.csv"}),
    "list_directory": Case(args={"path": "missing"}),
    # The Internet Archive's metadata API answers an unknown identifier with an empty array
    # (https://archive.org/developers/md-read.html).
    "internet_archive_item": Case(args={}, answers=(Answer(body=[]),)),
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
    # T07a, health: a 404 for a label or a product; OLS answers an unknown ID with no terms, and
    # openFDA an unknown recall number with its 404 NOT_FOUND
    # (https://github.com/FDA/openfda/blob/master/api/faers/api.js).
    **_missing_by_404("dailymed_label_text", "open_food_facts_product", "open_food_facts_compare"),
    "openfda_food_recall": Case(
        args={},
        answers=(
            Answer(
                status=404,
                body={"error": {"code": "NOT_FOUND", "message": "No matches found!"}},
            ),
        ),
    ),
    "foodon_term": Case(args={}, answers=(Answer(body=health.terms(0, total=0)),)),
    "json_extract": Case(args={"json_string": '{"a": [1, 2]}', "path": "missing"}),
}


# --- Point 3: zero results ------------------------------------------------------------------

# The data and news searches (T08a), each answering a query that matches nothing as its source
# does.
_DATA_NEWS_ZEROS: dict[str, ZeroCase] = {
    "eurostat_dataset_search": ZeroCase(
        args={"query": "zzqq"}, answers=_answers(data_bodies.eurostat_catalogue(3)), says="zzqq"
    ),
    "gdelt_news_search": ZeroCase(args={"query": "zzqq"}, answers=_answers({}), says="zzqq"),
    "gdelt_timeline": ZeroCase(args={"query": "zzqq"}, answers=_answers({}), says="zzqq"),
    "who_indicators": ZeroCase(
        args={"query": "zzqq"}, answers=_answers({"value": []}), says="zzqq"
    ),
    "wikidata_search": ZeroCase(
        args={"query": "zzqq"}, answers=_answers({"search": []}), says="zzqq"
    ),
    "wikidata_sparql": ZeroCase(
        args={"query": "SELECT ?zzqq WHERE { ?zzqq ?p ?o }"},
        answers=_answers(data_bodies.sparql_rows(0)),
        says="zzqq",
    ),
    "world_bank_indicators": ZeroCase(
        args={"query": "zzqq", "scan_pages": 1},
        answers=_answers(
            data_bodies.world_bank_page(
                data_bodies.world_bank_records(2, start=1), page=1, pages=1, total=2
            )
        ),
        says="zzqq",
    ),
}

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
    # Life sciences (T07b).
    "uniprot_search": ZeroCase(
        args={"query": "zzqqxx"},
        answers=(Answer(body={"results": []}, headers=bio.uniprot_headers(0)),),
        says="zzqqxx",
    ),
    # RCSB answers a search without hits 204 No Content (https://search.rcsb.org/).
    "pdb_search": ZeroCase(args={"query": "zzqqxx"}, answers=(Answer(status=204),), says="zzqqxx"),
    **{
        name: ZeroCase(
            args={"query": "zzqqxx"},
            answers=(Answer(body={"page_meta": {"total_count": 0, "next": None}, key: []}),),
            says="zzqqxx",
        )
        for name, key in (
            ("chembl_molecule_search", "molecules"),
            ("chembl_target_search", "targets"),
        )
    },
    "chembl_activity_search": ZeroCase(
        args={},
        answers=(Answer(body={"page_meta": {"total_count": 0, "next": None}, "activities": []}),),
        says="CHEMBL25",
    ),
    "gbif_species_search": ZeroCase(
        args={"query": "zzqqxx"},
        answers=(Answer(body={"count": 0, "endOfRecords": True, "results": []}),),
        says="zzqqxx",
    ),
    "gbif_occurrence_search": ZeroCase(
        args={},
        answers=(Answer(body={"count": 0, "endOfRecords": True, "results": []}),),
        says="2435099",
    ),
    "search_files": ZeroCase(args={"pattern": "zzqq"}, says="zzqq"),
    "regex_search": ZeroCase(args={"pattern": "zzqq"}, says="zzqq"),
    # Geo and natural events (T08b).
    "geocode": ZeroCase(
        args={"city": "Zzqqxx"}, answers=(Answer(body=geo_answers.NO_PLACES),), says="Zzqqxx"
    ),
    "osm_search_place": ZeroCase(
        args={"query": "zzqqxx"}, answers=(Answer(body=[]),), says="zzqqxx"
    ),
    "overpass_query": ZeroCase(args={}, answers=(Answer(body={"elements": []}),), says="node(1)"),
    "overpass_pois": ZeroCase(
        args={}, answers=(Answer(body={"elements": []}),), says="amenity=cafe"
    ),
    "eonet_events": ZeroCase(
        args={"category": "wildfires"},
        answers=(Answer(body={"events": []}),),
        says="category=wildfires",
    ),
    "earthquake_search": ZeroCase(args={}, answers=(Answer(body="0"),), says="2024-01-01"),
    **_DATA_NEWS_ZEROS,
    "arxiv_search": ZeroCase(
        args={"query": "zzqqxx"}, answers=(Answer(body=lit.arxiv_feed(total=0)),), says="zzqqxx"
    ),
    "crossref_search": ZeroCase(
        args={"query": "zzqqxx"}, answers=(Answer(body=lit.crossref_list(total=0)),), says="zzqqxx"
    ),
    "datacite_search": ZeroCase(
        args={"query": "zzqqxx"}, answers=(Answer(body=lit.datacite_list(total=0)),), says="zzqqxx"
    ),
    "europe_pmc_search": ZeroCase(
        args={"query": "zzqqxx"},
        answers=(Answer(body=lit.epmc_search(hit_count=0)),),
        says="zzqqxx",
    ),
    "pubmed_search": ZeroCase(
        args={"query": "zzqqxx"}, answers=(Answer(body=lit.ESEARCH_NOTHING),), says="zzqqxx"
    ),
    "semantic_scholar_search": ZeroCase(
        args={"query": "zzqqxx"}, answers=(Answer(body=lit.s2_search(total=0)),), says="zzqqxx"
    ),
    "ror_search": ZeroCase(
        args={"query": "zzqqxx"}, answers=(Answer(body=lit.ror_page(total=0)),), says="zzqqxx"
    ),
    "nvd_cve_search": ZeroCase(
        args={"query": "zzqqxx"}, answers=(Answer(body=lit.nvd_page(total=0)),), says="zzqqxx"
    ),
}
