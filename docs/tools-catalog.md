# Tools Catalog

The complete list of pre-built tools, grouped by domain. All are built on the [`@tool`](tools.md) decorator, use the standard library only (zero extra pip dependencies; the three `youtube_*` tools need the `youtube` extra), and raise a typed `ToolFailure` when they cannot answer, which the executor returns as a failed result — so agents degrade gracefully. Every tool declares its limits as [`Range`](tools.md#defining-tools) bounds, which the executor enforces: a value outside a limit given below is refused with a `validation_error`. A long answer is a window whose last line gives the call that reads on.

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

- `regex_search` — Find regex matches with positions and groups, a page of up to 1000 with the total and the next `offset`; text up to 20 000 characters, pattern up to 500, and back-references or groups that repeat while holding a quantifier or an alternation are refused; the match runs in a child Python process given 5 s, so a pattern that backtracks (`a*a*b`) is refused when they run out and never freezes the program
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

- `wikidata_search` — Search Wikidata items by label or alias, numbered, with the next offset
- `wikidata_entity` — An item's or property's (`Q42`, `P31`) label, description, aliases,
  Wikipedia link and statements, 40 at a time: every code with its label
  (`instance of (P31): human (Q5)`), quantities with their unit, dates to their precision,
  coordinates as latitude and longitude
- `wikidata_sparql` — Run a read-only SPARQL SELECT/ASK query; every row it returns reads on by
  `offset` (a SELECT without a `LIMIT` gets `LIMIT 1000`, and the answer says when it reached it)

**News & events** — `_news.py`, `_gdelt.py`

- `hacker_news` — The Hacker News top stories (up to 500) in rank order, `count` at a time from
  `offset`
- `gdelt_news_search` — Search global news coverage via GDELT DOC 2.0: up to the 250 articles
  GDELT lists for a query, read on by `offset`
- `gdelt_timeline` — A query's share of all the coverage GDELT monitored (%), step by step, read
  on by `offset`

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

The literature and identifier tools declare their limits as `Range` bounds. A search numbers its
results and ends with the window's footer: the total the source gives, and the call for the next
page (`[results 1-5 of 4321 | next: start=5]`), or why the rest cannot be read here (each source
pages only so deep: arXiv 30,000 results, Crossref, DataCite, PubMed and ROR 10,000, Semantic
Scholar 1,000). A record tool (`arxiv_paper`, `crossref_work` …) gives the whole record, its
abstract and every author, reference or link, window by window (`offset`, `max_chars`); a search
shows a long list's first names and says which record tool lists them all.

**Papers** — `_arxiv.py`, `_pubmed.py`, `_europe_pmc.py`

- `arxiv_search` — Search arXiv papers, each with its whole summary (`start` to page; arXiv asks
  for 3 s between requests, which the tools keep)
- `arxiv_paper` — An arXiv paper's record: summary, every author with the affiliations,
  categories, journal reference, DOI and links
- `pubmed_search` — Search PubMed articles via NCBI E-utilities; zero results say which phrases
  PubMed did not find
- `pubmed_article` — A PubMed article's record by PMID: the abstract by section, every author with
  the affiliations, MeSH headings (major topics marked), keywords and publication types
- `europe_pmc_search` — Search Europe PMC (PubMed, PMC, preprints, patents …); pages by cursor, the
  footer giving the next `cursor_mark` and `offset`
- `europe_pmc_article` — An article's Europe PMC record by PMID, PMCID, DOI or `SOURCE/ID`
  (`MED/26017442`): abstract, authors, MeSH, keywords and every full-text link
- `europe_pmc_citations` — The articles that cite a Europe PMC record, page by page (`page`), with
  the total; a record Europe PMC does not have is `not_found`

**Academic graph & metadata** — `_semantic_scholar.py`, `_crossref.py`, `_ror.py`, `_datacite.py`

- `semantic_scholar_search` — Search Semantic Scholar papers by the words of their titles and
  abstracts (a DOI or arXiv ID goes to `semantic_scholar_paper`); an optional free key in
  `SEMANTIC_SCHOLAR_API_KEY` gives 1 request per second; without it, callers share one limit that
  is often spent
- `semantic_scholar_paper` — A paper's record by Semantic Scholar ID, DOI, arXiv ID, PMID or
  prefixed ID: abstract, every author, counts, and the identifiers as `semantic_scholar_paper` takes
  them (`DOI:…`, `ARXIV:…`, `PMID:…`, `CorpusId:…`)
- `semantic_scholar_citations` — The papers that cite a paper, each with every sentence that cites
  it, why, and whether the citation is influential; the footer says when more are available
- `crossref_search` — Search Crossref works by title, topic, author or a citation's words
- `crossref_work` — A work's Crossref record by DOI: abstract, every author with the affiliations
  and ORCID iD, licenses, full-text links and the references
- `ror_search` — Search ROR research organizations, ROR's pages of 20 (`page`), every location
- `ror_organization` — An organization's ROR record: every name (with type and language),
  location (with coordinates and GeoNames ID), link, external ID and relationship
- `datacite_search` — Search DataCite DOI records for datasets, software, texts and other research
  outputs; `resource_type` takes a resourceTypeGeneral as written (`JournalArticle`, `Dataset`)
- `datacite_doi` — A DOI's DataCite record: every title, creator with the affiliations,
  description (by type), subject, right and related identifier

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
  in an assay, or any of them together, with names, units and the assay

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

Every list reads on (`page`, `skip` or `offset`, as the footer says), values keep every digit the
source sent, and codes come with the labels the answer brings.

- `world_bank_topics` — List World Bank indicator topics, with their notes
- `world_bank_sources` — List World Bank data sources/databases
- `world_bank_countries` — List or search World Bank countries, economies, and aggregates, with
  the codes `world_bank_series` takes
- `world_bank_indicators` — Browse World Bank indicators (by topic or source), or search them,
  best matches first
- `world_bank_indicator` — An indicator's name, source, topics, whole definition and source
  organization
- `world_bank_series` — An indicator's values for one or several countries or aggregates
  (`"PRT,ESP,DEU"` compares them), page by page
- `who_indicators` — Search WHO Global Health Observatory indicators
- `who_indicator` — Get WHO GHO indicator metadata by code
- `who_series` — WHO GHO observations of an indicator: place, year, value with its uncertainty
  interval, and each code's dimension
- `eurostat_dataset_search` — Search Eurostat datasets by ID or title (on a Python whose CA file lacks Eurostat's root, such as uv's standalone builds on macOS, install the `truststore` extra)
- `eurostat_dataset` — A Eurostat dataset's title, update time, period and description, and each
  dimension's codes with their labels (`dimension="geo"` lists one)
- `eurostat_series` — Eurostat observations, one row each with its codes' labels; several codes
  of a dimension compare them (`filters="geo=PT+ES+FR,unit=NR"`)

**Security** — `_nvd.py`

- `nvd_cve_search` — Search CVEs in the NVD 2.0 API by keyword, CVE, CPE, CVSS v3 severity or
  publication date (a range of at most 120 days, NVD's limit), numbered, with the total; each with
  its whole description and one score per CVSS version (`CVSS 3.1: 10.0 CRITICAL`)
- `nvd_cve` — A CVE's NVD record: the description, every CVSS score with its version, scorer and
  vector, the weaknesses, every affected CPE with its version range, and every reference with its
  tags, window by window

---

## Dangerous (opt-in — gate these)

These execute real side effects and live in the explicit `ai_arch_toolkit.toolkit.tools.dangerous` namespace. All of them require approval — without an `approval_handler`, governed execution returns `approval_denied` — and should also run behind sandboxing and permission checks. See [Tool Governance & Safety](safety.md#dangerous-tools).

**Filesystem** — `dangerous`

The file tools to give an agent come from `filesystem_tools(policy)`: the three reads below, bound to a `FilesystemPolicy`'s `read_roots`, and the four writes, bound to its `write_roots` (POSIX only). Each one checks its paths with the policy right before it acts, and reaches them from the root down, never through a link; a `PathScopeGate` refuses a path outside before anyone is asked. See [Filesystem scope](safety.md#filesystem-scope). The module-level reads are the same three without a policy.

- `read_file` — Read a text file window by window: up to `max_lines` lines (1–10 000) and 100 000 characters a window, from a character `offset`; the footer gives the next offset, so following it reads the whole file, and the file's size once fewer than 100 million characters are left. The file is read a chunk at a time, never whole
- `list_directory` — List files/dirs with sizes and types, by name, a page of 1000 with the total and the next `offset`; a pattern that matches more than 100 000 entries is refused with a request to narrow it. Lists only entries inside the folder, so a pattern that climbs out (`../*`) or goes through a link (`link/*`) finds nothing there. Bound to a policy, it refuses a pattern with a `..` part and leaves out an entry whose target is outside the read roots
- `search_files` — Recursively search the text files under a folder for the lines that contain a text (1–1000 a page, with the next `offset`), each as `path:line:offset: text`, the path as `read_file` takes it (under `directory` as you passed it), where `read_file(path, offset=…)` reads from; whole files are searched, up to 500 million characters a call (then it says where it stopped), a file with a NUL in its first 8 KiB is skipped as binary, and a line over 300 characters shows the part around the match and says its length. A link that points out of the folder is not read; a folder or file it cannot read is named. Bound to a policy, it reads no file outside the read roots, and the paths it gives are canonical
- `write_file` — Write a text file whole, in UTF-8, exactly as given, in one step: a temporary file, synced to disk, takes its name, so a reader never sees half of it. Without `overwrite` an existing file stays as it is and the call fails (of two writes racing to one new path, one wins); with it, the file keeps its permissions. `create_parents` makes the missing folders. At most the policy's `max_write_bytes` (1 MiB by default) a call. The four writes run one at a time in a process. From `filesystem_tools` only
- `append_file` — Add UTF-8 text at the end of a regular file that exists and has no other hard link; not atomic. From `filesystem_tools` only
- `make_directory` — Make a folder (`parents` for the missing ones on the way); one that exists already is a success that says so. From `filesystem_tools` only
- `move_path` — Move or rename a file or a folder within one volume: it never copies, so another volume is refused. Without `overwrite` an existing destination stays as it is; with it, a file or an empty folder there is replaced. A root, or a folder that holds one, never moves; a file the agent cannot read never moves to a folder it reads; two hard links of one file are refused. From `filesystem_tools` only
- `csv_read` — Read a CSV file as a table: the header, then a page of rows (1–10 000, at most 100 000 characters, columns padded to 40) from an `offset`, with the file's row count (counted up to 50 million characters past the page, then "at least") and the next offset

**Shell** — `dangerous`

- `run_command` — Execute shell commands and return output (timeout 1–600 s, output 1–100 000 characters, stdout and stderr together; `cwd` runs the command in another folder, without moving the process). An output over the limit shows its start, its size and how to narrow the command (`| grep`, `| tail -n`, `| sed -n 'A,Bp'`): it is not kept, so no call reads on. The exit code and stderr always show, and the output is read as it comes, so memory stays within the limit. The command gets no input and runs in a process group of its own, killed when the call returns, so nothing it started in the background is left running (end the command with `wait` to let such a process finish); POSIX only. Governed execution gives up at the tool's default 120 s `timeout_s` with a `timeout` failure, and the command runs on until it ends or its own timeout stops it

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
