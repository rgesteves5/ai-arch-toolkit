"""Pre-built tools — real, working tools for agents and examples."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.tools._air_quality import (
    air_quality_current,
    air_quality_forecast,
)
from ai_arch_toolkit.toolkit.tools._arxiv import arxiv_paper, arxiv_search
from ai_arch_toolkit.toolkit.tools._chembl import (
    chembl_activity_search,
    chembl_molecule,
    chembl_molecule_search,
    chembl_target,
    chembl_target_search,
)
from ai_arch_toolkit.toolkit.tools._clinical_trials import (
    clinical_trial_study,
    clinical_trials_search,
)
from ai_arch_toolkit.toolkit.tools._crossref import crossref_search, crossref_work
from ai_arch_toolkit.toolkit.tools._datacite import datacite_doi, datacite_search
from ai_arch_toolkit.toolkit.tools._datetime import (
    date_add,
    date_diff,
    date_format,
    datetime_now,
    timezone_convert,
)
from ai_arch_toolkit.toolkit.tools._earthquake import (
    earthquake_count,
    earthquake_event,
    earthquake_search,
)
from ai_arch_toolkit.toolkit.tools._eonet import eonet_categories, eonet_event, eonet_events
from ai_arch_toolkit.toolkit.tools._europe_pmc import (
    europe_pmc_article,
    europe_pmc_citations,
    europe_pmc_search,
)
from ai_arch_toolkit.toolkit.tools._eurostat import (
    eurostat_compare,
    eurostat_dataset,
    eurostat_dataset_search,
    eurostat_dimensions,
    eurostat_series,
)
from ai_arch_toolkit.toolkit.tools._foodon import foodon_search, foodon_term
from ai_arch_toolkit.toolkit.tools._gbif import (
    gbif_occurrence_search,
    gbif_species,
    gbif_species_match,
    gbif_species_search,
)
from ai_arch_toolkit.toolkit.tools._gdelt import gdelt_news_search, gdelt_timeline
from ai_arch_toolkit.toolkit.tools._geo import (
    country_info,
    distance_between,
    geocode,
    ip_lookup,
    timezone_lookup,
)
from ai_arch_toolkit.toolkit.tools._internet_archive import (
    internet_archive_item,
    internet_archive_search,
)
from ai_arch_toolkit.toolkit.tools._json import json_extract
from ai_arch_toolkit.toolkit.tools._math import math_eval, unit_convert
from ai_arch_toolkit.toolkit.tools._news import hacker_news
from ai_arch_toolkit.toolkit.tools._nvd import nvd_cve, nvd_cve_search
from ai_arch_toolkit.toolkit.tools._open_food_facts import (
    open_food_facts_compare,
    open_food_facts_product,
    open_food_facts_search,
)
from ai_arch_toolkit.toolkit.tools._open_library import (
    open_library_isbn,
    open_library_search,
    open_library_work,
)
from ai_arch_toolkit.toolkit.tools._openfda_food import (
    openfda_food_recall,
    openfda_food_recall_search,
)
from ai_arch_toolkit.toolkit.tools._osm import osm_reverse_geocode, osm_search_place
from ai_arch_toolkit.toolkit.tools._overpass import overpass_pois, overpass_query
from ai_arch_toolkit.toolkit.tools._pdb import (
    pdb_chemical_component,
    pdb_entry,
    pdb_ligands,
    pdb_search,
)
from ai_arch_toolkit.toolkit.tools._pubmed import pubmed_article, pubmed_search
from ai_arch_toolkit.toolkit.tools._ror import ror_organization, ror_search
from ai_arch_toolkit.toolkit.tools._rxnorm_dailymed import (
    dailymed_label,
    dailymed_label_search,
    dailymed_label_text,
    rxnorm_concept,
    rxnorm_drug_search,
    rxnorm_ndcs,
    rxnorm_related,
)
from ai_arch_toolkit.toolkit.tools._semantic_scholar import (
    semantic_scholar_citations,
    semantic_scholar_paper,
    semantic_scholar_search,
)
from ai_arch_toolkit.toolkit.tools._text import (
    base64_decode,
    base64_encode,
    regex_search,
    text_stats,
)
from ai_arch_toolkit.toolkit.tools._uniprot import (
    uniprot_crossrefs,
    uniprot_entry,
    uniprot_features,
    uniprot_search,
    uniprot_sequence,
)
from ai_arch_toolkit.toolkit.tools._weather import get_forecast, get_weather
from ai_arch_toolkit.toolkit.tools._web_search import brave_search, tavily_search
from ai_arch_toolkit.toolkit.tools._who_gho import who_indicator, who_indicators, who_series
from ai_arch_toolkit.toolkit.tools._wiki import (
    wiki_outline,
    wiki_read,
    wiki_search,
    wiktionary_entry,
)
from ai_arch_toolkit.toolkit.tools._wikidata import (
    wikidata_entity,
    wikidata_search,
    wikidata_sparql,
)
from ai_arch_toolkit.toolkit.tools._world_bank import (
    world_bank_compare,
    world_bank_countries,
    world_bank_indicator,
    world_bank_indicators,
    world_bank_series,
    world_bank_sources,
    world_bank_topics,
)
from ai_arch_toolkit.toolkit.tools._youtube import (
    youtube_transcript,
    youtube_transcript_languages,
    youtube_transcript_search,
)

__all__ = [
    "air_quality_current",
    "air_quality_forecast",
    "arxiv_paper",
    "arxiv_search",
    "base64_decode",
    "base64_encode",
    "brave_search",
    "chembl_activity_search",
    "chembl_molecule",
    "chembl_molecule_search",
    "chembl_target",
    "chembl_target_search",
    "clinical_trial_study",
    "clinical_trials_search",
    "country_info",
    "crossref_search",
    "crossref_work",
    "dailymed_label",
    "dailymed_label_search",
    "dailymed_label_text",
    "datacite_doi",
    "datacite_search",
    "date_add",
    "date_diff",
    "date_format",
    "datetime_now",
    "distance_between",
    "earthquake_count",
    "earthquake_event",
    "earthquake_search",
    "eonet_categories",
    "eonet_event",
    "eonet_events",
    "europe_pmc_article",
    "europe_pmc_citations",
    "europe_pmc_search",
    "eurostat_compare",
    "eurostat_dataset",
    "eurostat_dataset_search",
    "eurostat_dimensions",
    "eurostat_series",
    "foodon_search",
    "foodon_term",
    "gbif_occurrence_search",
    "gbif_species",
    "gbif_species_match",
    "gbif_species_search",
    "gdelt_news_search",
    "gdelt_timeline",
    "geocode",
    "get_forecast",
    "get_weather",
    "hacker_news",
    "internet_archive_item",
    "internet_archive_search",
    "ip_lookup",
    "json_extract",
    "math_eval",
    "nvd_cve",
    "nvd_cve_search",
    "open_food_facts_compare",
    "open_food_facts_product",
    "open_food_facts_search",
    "open_library_isbn",
    "open_library_search",
    "open_library_work",
    "openfda_food_recall",
    "openfda_food_recall_search",
    "osm_reverse_geocode",
    "osm_search_place",
    "overpass_pois",
    "overpass_query",
    "pdb_chemical_component",
    "pdb_entry",
    "pdb_ligands",
    "pdb_search",
    "pubmed_article",
    "pubmed_search",
    "regex_search",
    "ror_organization",
    "ror_search",
    "rxnorm_concept",
    "rxnorm_drug_search",
    "rxnorm_ndcs",
    "rxnorm_related",
    "semantic_scholar_citations",
    "semantic_scholar_paper",
    "semantic_scholar_search",
    "tavily_search",
    "text_stats",
    "timezone_convert",
    "timezone_lookup",
    "uniprot_crossrefs",
    "uniprot_entry",
    "uniprot_features",
    "uniprot_search",
    "uniprot_sequence",
    "unit_convert",
    "who_indicator",
    "who_indicators",
    "who_series",
    "wiki_outline",
    "wiki_read",
    "wiki_search",
    "wikidata_entity",
    "wikidata_search",
    "wikidata_sparql",
    "wiktionary_entry",
    "world_bank_compare",
    "world_bank_countries",
    "world_bank_indicator",
    "world_bank_indicators",
    "world_bank_series",
    "world_bank_sources",
    "world_bank_topics",
    "youtube_transcript",
    "youtube_transcript_languages",
    "youtube_transcript_search",
]
