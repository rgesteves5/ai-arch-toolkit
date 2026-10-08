# Tools Catalog

The complete list of pre-built tools, grouped by domain. All are built on the [`@tool`](tools.md) decorator, use the standard library only (zero extra pip dependencies; the three `youtube_*` tools need the `youtube` extra), and raise a typed `ToolFailure` when they cannot answer, which the executor returns as a failed result — so agents degrade gracefully. The wiki family declares its limits as [`Range`](tools.md#defining-tools) bounds, which the executor enforces; the other tools still move a numeric argument outside a limit given below to the nearest limit without a word.

Import any of them and drop them into a `ToolGroup`:

```python
from ai_arch_toolkit.toolkit.tools import get_weather, arxiv_search
from ai_arch_toolkit import ToolGroup

group = ToolGroup(get_weather, arxiv_search)
```

For the conceptual guide (`@tool`, `ToolGroup`, server tools), see [Tools](tools.md). For risk levels, approval, and the opt-in dangerous tools, see [Tool Governance & Safety](safety.md).

---

## General & utility

**Date & time** — `_datetime.py`

- `datetime_now` — Current date/time in a given timezone
- `timezone_convert` — Convert time between timezones
- `date_add` — Add days/hours/minutes to a date or datetime; each shift is bounded by the calendar's span (years 1–9999)
- `date_diff` — Difference between two dates/datetimes in a chosen unit
- `date_format` — Reformat a date/datetime using strftime syntax

**Math** — `_math.py`

- `math_eval` — Safely evaluate math expressions (functions, constants, operators); up to 1000 characters, and a result too large to print is refused before it is computed
- `unit_convert` — Convert between units (length, mass, volume, speed, area, time, temp) with the exact unit definitions; the answer is rounded to 6 significant digits and says so, never in scientific notation

**Text processing** — `_text.py`

- `regex_search` — Find regex matches with positions and groups, a page of up to 1000 with the total and the next `offset`; text up to 20 000 characters, pattern up to 500, and back-references or groups that repeat while holding a quantifier or an alternation are refused
- `text_stats` — Count words, characters, lines, sentences, paragraphs
- `base64_encode` — Encode text to base64
- `base64_decode` — Decode base64 to text

**Data** — `_json.py`

- `json_extract` — Extract values from JSON via dot-notation paths; a key or index the JSON lacks is `not_found`, naming the keys (or the length) there

---

## Weather, geo & places

**Weather** — `_weather.py`

- `get_weather` — Current weather at a city or a latitude/longitude pair, in metric or imperial units, from Open-Meteo; with a city it uses the first match and says when other places share the name
- `get_forecast` — Daily forecast (up to 16 days) at a city or a latitude/longitude pair, in metric or imperial units

**Air quality** — `_air_quality.py`

- `air_quality_current` — Current AQI and pollutant values for coordinates via Open-Meteo, at a UTC time
- `air_quality_forecast` — Hourly AQI and pollutant values for coordinates (up to 7 days ahead and 92 back), a page of hours at a time

**Geography** — `_geo.py`

- `geocode` — Places by name (coordinates, region, country, time zone, population) from Open-Meteo, a page at a time, up to the 100 it returns
- `timezone_lookup` — Time zone and current UTC offset of a point
- `distance_between` — Great-circle distance between coordinate pairs
- `ip_lookup` — Geographic location and ISP info for an explicit IP, from ipwho.is over HTTPS (free, 1000 requests a day per client IP)
- `country_info` — Country facts from Wikidata by name or ISO 3166-1 code (capital, population, area, languages, currencies, time zones)

**OpenStreetMap** — `_osm.py`, `_overpass.py`

- `osm_search_place` — Search places and addresses with OpenStreetMap Nominatim, a page at a time, up to the 40 it returns
- `osm_reverse_geocode` — The place at a point, with its full address, at the detail `zoom` asks (18 a building … 10 a city … 3 a country)
- `overpass_query` — Run an Overpass QL query and page through the elements it returns
- `overpass_pois` — OpenStreetMap nodes, ways and relations with a tag, in a box or around a point, a page at a time

---

## Reference & knowledge

**Wikis** — `_wiki.py`

English Wikipedia by default; any Wikimedia wiki by its host (`wiki="en.wikibooks.org"`,
`"pt.wikipedia.org"`). Pages are read as the HTML the wiki renders, converted to text: tables one
row per line with every cell, no navigation boxes, edit links or footnote markers.

- `wiki_search` — Search a wiki's pages (`intitle:`, `incategory:`, `morelike:Title` for related
  pages), numbered, with the total and the next offset
- `wiki_outline` — A page's sections: the number `wiki_read` takes, the heading and its size
- `wiki_read` — Read a page whole, one section, or the passages around a term (`find=`), window by
  window
- `wiktionary_entry` — The English Wiktionary's entry for a term in one language: senses by part
  of speech, pronunciation, etymology and examples

**Wikidata** — `_wikidata.py`

- `wikidata_search` — Search Wikidata entities by label or alias
- `wikidata_entity` — Get labels, aliases, claims, and Wikipedia links for a Wikidata QID
- `wikidata_sparql` — Run read-only Wikidata SPARQL SELECT/ASK queries

**News & events** — `_news.py`, `_gdelt.py`

- `hacker_news` — Top stories from Hacker News
- `gdelt_news_search` — Search global news coverage via GDELT DOC 2.0
- `gdelt_timeline` — Get a GDELT volume timeline for a news query

GDELT's free API takes few requests per minute from one address: after a 429 both tools wait 60 s, answering at once with when to try again.

**Video transcripts** — `_youtube.py`

- `youtube_transcript` — Fetch a public YouTube transcript as text, timestamped segments, JSON,
  SRT or VTT, window by window (`offset`; 1–50 000 characters a window)
- `youtube_transcript_languages` — A video's transcripts, and every language they translate to
- `youtube_transcript_search` — The timestamped passages of a transcript that mention a term, with
  their total, paged by `offset`

---

## Web search (your own key, billed)

**Web search** — `_web_search.py`

These two run on the toolkit's side, so they serve any model, local ones included, unlike the provider-hosted `web_search()` server tool ([the contrast](tools.md#web-search-hosted-or-local)). Each needs a key of your own in the environment, and says where to get one, without sending anything, when it is missing; each search is billed by the service, and the meter counts it at the price table's `[tools]` entry ([Pricing](pricing.md#paid-tools)). A search the service refuses costs nothing.

Both declare their limits as `Range` bounds, show each result whole, and leave out any result whose URL is not `http(s)` (a `javascript:` or `data:` link never reaches the model), saying how many they left out. They are network tools of low risk that run without approval, like the other network tools; their results are third-party text ([Safety](safety.md#web-search-tools)).

- `brave_search` — Search the web with Brave Search (`BRAVE_SEARCH_API_KEY`; 1–20 results a page, a country and an age limit). Brave serves ten pages: the footer names the next one (`next: offset=1, max_results=10`); $5 per 1,000 searches, with $5 of free credit a month
- `tavily_search` — Search the web with Tavily, which returns an excerpt of each page and, if asked, a short answer (`TAVILY_API_KEY`; 0–20 results, 0 for the answer alone, `general` or `news`, an age limit). Tavily serves one page, and a full page says so; a basic search is one credit, $0.008, with 1,000 free a month

## Scholarly & research

**Papers** — `_arxiv.py`, `_pubmed.py`, `_europe_pmc.py`

- `arxiv_search` — Search arXiv papers via the public arXiv API
- `arxiv_paper` — Get metadata for a specific arXiv paper by ID
- `pubmed_search` — Search PubMed articles via the public NCBI E-utilities API
- `pubmed_article` — Get metadata for a specific PubMed article by PMID
- `europe_pmc_search` — Search Europe PMC articles via the public REST API
- `europe_pmc_article` — Get Europe PMC metadata for a PMID, PMCID, DOI, or source ID
- `europe_pmc_citations` — Get articles that cite a Europe PMC record

**Academic graph & metadata** — `_semantic_scholar.py`, `_crossref.py`, `_ror.py`, `_datacite.py`

- `semantic_scholar_search` — Search Semantic Scholar papers via the public Academic Graph API (an optional free key in `SEMANTIC_SCHOLAR_API_KEY` gives 1 request per second; without it, callers share one limit that is often spent)
- `semantic_scholar_paper` — Get detailed metadata for a Semantic Scholar paper
- `semantic_scholar_citations` — Get papers that cite a Semantic Scholar paper
- `crossref_search` — Search Crossref works by title, DOI, topic, or citation fragment
- `crossref_work` — Get Crossref metadata for a specific DOI
- `ror_search` — Search ROR research organizations
- `ror_organization` — Get ROR organization metadata
- `datacite_search` — Search DataCite DOI metadata for datasets, software, text, and other research outputs
- `datacite_doi` — Get DataCite metadata for a specific DOI

**Books** — `_open_library.py`

Open Library takes one request a second from a caller that sends no email; a work or an edition
costs two, the second naming its authors.

- `open_library_search` — Search Open Library books and works, numbered, with the total and the
  next `start`; authors by name, with the key `author_key:OL…A` searches by
- `open_library_work` — A work, whole and window by window: its authors by name, description,
  subjects and links
- `open_library_isbn` — The edition with an ISBN, whole and window by window: its authors by name,
  publishers, date (ISO 8601), ISBNs and work

**Digital archives** — `_internet_archive.py`

- `internet_archive_search` — Search Internet Archive items with optional mediatype and collection
  filters, numbered, with the total and the next page, down to the 10 000th result (the deepest
  the archive pages)
- `internet_archive_item` — An item's metadata and every one of its files, whole and window by
  window, or the passages around a term (`find=`)

---

## Biomedical & chemistry

**Proteins & structures** — `_uniprot.py`, `_pdb.py`

- `uniprot_search` — Search UniProtKB proteins in UniProt's query syntax, with the total; the
  footer gives the next page's `cursor` (UniProt pages by cursor, not by offset)
- `uniprot_entry` — Read a UniProtKB entry: names, organism, sizes, dates and every annotation
  (function, catalytic activity, location, disease, interactions …) in full, window by window
- `uniprot_features` — List an entry's sequence features with positions and changes, by type,
  page by page
- `uniprot_sequence` — Get a UniProtKB protein sequence in FASTA form
- `uniprot_crossrefs` — List an entry's cross-references with all their properties, by database,
  page by page (PDB IDs read on with `pdb_entry`, ChEMBL IDs with `chembl_target`)
- `pdb_search` — Search RCSB PDB structures by free text: each entry with its title, method,
  resolution and release date (two requests a page)
- `pdb_entry` — Read an RCSB PDB entry: title, method, resolution, dates, entities, citation
- `pdb_ligands` — List the ligands of a PDB entry with their component IDs (two requests)
- `pdb_chemical_component` — Read an RCSB chemical component: formula, weight, SMILES, InChIKey

**Chemistry & bioactivity** — `_chembl.py`

- `chembl_molecule_search` — Search ChEMBL molecules by name or synonym, with the total
- `chembl_molecule` — Read a ChEMBL molecule: phase (with its label), properties with units,
  structure
- `chembl_target_search` — Search ChEMBL biological targets, with the total
- `chembl_target` — Read a ChEMBL target, with the UniProt accessions of its components
- `chembl_activity_search` — Search ChEMBL bioactivity measurements of a molecule, on a target,
  or both, with names, units and the assay

**Medication labels** — `_rxnorm_dailymed.py`, `_spl.py`

RxNorm's lists come whole from RxNav and are paged here, with the total; each term type comes with
its name (`SCD (Semantic Clinical Drug)`). A DailyMed label is its SPL document, read as text: lists
one item per line, tables one row per line.

- `rxnorm_drug_search` — Search RxNorm drug concepts by name, page by page (`offset`)
- `rxnorm_concept` — An RxNorm concept's name, term type and synonym by RxCUI
- `rxnorm_related` — The concepts related to one, by term types (`tty="IN BN"`) or all of them
- `rxnorm_ndcs` — The active NDCs of a drug concept (CMS 11-digit form), page by page
- `dailymed_label_search` — Search DailyMed labels by drug name, NDC or RxCUI, with the total and
  the next page
- `dailymed_label` — A label's title, version, effective date and labeler, and its sections,
  numbered, with their LOINC codes and sizes
- `dailymed_label_text` — Read a label's text: whole, one section (`section=N`, from
  `dailymed_label`), or the passages around a term (`find=`), window by window

**Clinical studies** — `_clinical_trials.py`

- `clinical_trials_search` — Search ClinicalTrials.gov studies, with the total; the footer gives
  the next page's token and position
- `clinical_trial_study` — Read a study's record: whole, one section (overview, summary,
  description, eligibility, arms, outcomes, locations, references), or the passages around a term
  (`find=`), window by window; eligibility criteria keep their lines

---

## Earth, life & public data

**Biodiversity** — `_gbif.py`

- `gbif_species_match` — Resolve a scientific name to its GBIF backbone taxon (exact, fuzzy, or
  only a higher rank, as the match says); common names go to `gbif_species_search`
- `gbif_species_search` — Search GBIF taxa by scientific or common name, rank, or parent taxon,
  with the total
- `gbif_species` — Read a GBIF taxon by its key: status, common name, classification, parent
- `gbif_occurrence_search` — Search GBIF occurrence records by taxon, country or year, with the
  total; GBIF's search reaches the first 100,000 records

**Food products** — `_open_food_facts.py`

- `open_food_facts_product` — A packaged food by barcode, whole: brand, Nutri-Score and NOVA
  group with what they mean, nutrients per 100 g with units, ingredients, and every allergen,
  trace, additive, category, label and country
- `open_food_facts_search` — Search packaged foods by name, brand, category, country or label,
  with the total and the next page
- `open_food_facts_compare` — Up to 5 products side by side, one line each: scores, the main
  nutrients per 100 g and every allergen

**Food safety & ontology** — `_openfda_food.py`, `_foodon.py`

- `openfda_food_recall_search` — Search FDA food enforcement recalls via openFDA, with the total
  and the next `skip` (openFDA reads up to `skip=25000`; narrow by date beyond)
- `openfda_food_recall` — Get a specific FDA food enforcement recall by recall number
- `foodon_search` — Search FoodOn ontology terms via EMBL-EBI OLS, with whole definitions, the
  total and the next `start`
- `foodon_term` — A FoodOn term by ID: label, definition and synonyms (imported terms keep their
  own prefix, e.g. `NCBITaxon:3750`)

**Natural events** — `_earthquake.py`, `_eonet.py`

- `earthquake_search` — Search USGS earthquake events by date, magnitude, depth, and location, with the total that match and times in UTC
- `earthquake_event` — Get a USGS earthquake event by ID (deleted events are reported as such)
- `earthquake_count` — Count USGS earthquake events for a date/magnitude query
- `eonet_categories` — List NASA EONET event categories
- `eonet_events` — Search NASA EONET natural events by category, status, source, box and period (up to 365 days, or dates)
- `eonet_event` — Get a NASA EONET event by ID, with its whole track a page at a time

**Official statistics** — `_world_bank.py`, `_who_gho.py`, `_eurostat.py`

- `world_bank_topics` — List World Bank indicator topics
- `world_bank_sources` — List World Bank data sources/databases
- `world_bank_countries` — List or search World Bank countries, economies, and aggregates
- `world_bank_indicators` — Browse or search World Bank indicators with topic/source filters
- `world_bank_indicator` — Get metadata for a specific World Bank indicator
- `world_bank_series` — Get a World Bank indicator time series for a country or aggregate
- `world_bank_compare` — Compare a World Bank indicator across multiple countries or aggregates
- `who_indicators` — Search WHO Global Health Observatory indicators
- `who_indicator` — Get WHO GHO indicator metadata by code
- `who_series` — Fetch WHO GHO observations for an indicator
- `eurostat_dataset_search` — Search Eurostat datasets by ID or title (on a Python whose CA file lacks Eurostat's root, such as uv's standalone builds on macOS, install the `truststore` extra)
- `eurostat_dataset` — Get Eurostat dataset metadata and dimension summary
- `eurostat_dimensions` — List Eurostat dimensions and sample category codes
- `eurostat_series` — Get Eurostat observations with generic dimension filters
- `eurostat_compare` — Compare a Eurostat dataset across geo codes

**Security** — `_nvd.py`

- `nvd_cve_search` — Search CVEs in the NVD 2.0 API by keyword, CVE, CPE, severity, or publication date
- `nvd_cve` — Get NVD metadata for a specific CVE ID

---

## Dangerous (opt-in — gate these)

These execute real side effects and live in the explicit `ai_arch_toolkit.toolkit.tools.dangerous` namespace. All of them require approval — without an `approval_handler`, governed execution returns `approval_denied` — and should also run behind sandboxing and permission checks. See [Tool Governance & Safety](safety.md#dangerous-tools).

**Filesystem** — `dangerous`

- `read_file` — Read a text file window by window: up to `max_lines` lines (1–10 000) and 100 000 characters a window, from a character `offset`; the footer gives the next offset, so following it reads the whole file, and the file's size once fewer than 100 million characters are left. The file is read a chunk at a time, never whole
- `list_directory` — List files/dirs with sizes and types, by name, a page of 1000 with the total and the next `offset`; a pattern that matches more than 100 000 entries is refused with a request to narrow it. Lists only entries inside the folder, so a pattern that climbs out (`../*`) or goes through a link (`link/*`) finds nothing there
- `search_files` — Recursively search the text files under a folder for the lines that contain a text (1–1000 a page, with the next `offset`), each as `path:line:offset: text`, where `read_file(path, offset=…)` reads from; whole files are searched, and a line over 300 characters shows the part around the match and says its length. A link that points out of the folder is not read
- `csv_read` — Read a CSV file as a table: the header, then a page of rows (1–10 000) from an `offset`, with the whole file's row count and the next offset

**Shell** — `dangerous`

- `run_command` — Execute shell commands and return output (timeout 1–600 s, output 1–100 000 characters, stdout and stderr together; `cwd` runs the command in another folder, without moving the process). An output over the limit shows its start, its size and how to narrow the command (`| grep`, `| tail -n`, `| sed -n 'A,Bp'`): it is not kept, so no call reads on. The exit code and stderr always show, and the output is read as it comes, so memory stays within the limit. Governed execution gives up at the tool's default 120 s `timeout_s` with a `timeout` failure, and the command runs on until it ends or its own timeout stops it

**Python** — `dangerous`

- `python_repl` — Execute Python-like code in the restricted toolkit evaluator; an output over 20 000 characters shows its start, its size and how to print less, and a long print never pushes out the last value or the error

**Web** — `dangerous`

Each call fetches the page again and asks for approval again; the footer names the call that
reads on.

- `http_get` — Fetch an http(s) URL and return the response text as it came, window by window
  (`offset`; 1–100 000 characters a window) or the passages around a term (`find=`); it reads at
  most the first 10 MB, and only as much as the window needs; redirects stay on the URL's host
- `scrape_text` — Fetch a web page and return its visible text (HTML stripped), window by window
  or the passages around a term; it reads the first 2 MB of HTML; redirects stay on the URL's host
