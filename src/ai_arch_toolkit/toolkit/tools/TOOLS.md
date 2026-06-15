Date & Time — _datetime.py

- datetime_now — Current date/time in a given timezone
- timezone_convert — Convert time between timezones
- date_add — Add days/hours/minutes to a date or datetime
- date_diff — Difference between two dates/datetimes in a chosen unit
- date_format — Reformat a date/datetime using strftime syntax

Weather — _weather.py

- get_weather — Current weather for a city (temp, humidity, wind, conditions)
- get_forecast — Multi-day weather forecast for a city
- get_weather_by_coords — Current weather for a latitude/longitude pair
- get_forecast_by_coords — Multi-day forecast for a latitude/longitude pair
- weather_units — Current weather for a city with Celsius/Fahrenheit output

Air Quality — _air_quality.py

- air_quality_current — Current AQI and pollutant values for coordinates via Open-Meteo
- air_quality_forecast — Hourly AQI and pollutant forecast for coordinates via Open-Meteo

Geography — _geo.py

- geocode — Coordinates and country for a city name
- reverse_geocode — Place name and region from coordinates
- timezone_lookup — Timezone and UTC offset from coordinates
- distance_between — Great-circle distance between coordinate pairs
- ip_lookup — Geographic location and ISP info for an IP
- country_info — Country details (capital, population, languages, etc.)

OpenStreetMap — _osm.py

- osm_search_place — Search places and addresses with OpenStreetMap Nominatim
- osm_reverse_geocode — Reverse geocode coordinates with OpenStreetMap Nominatim

Food Products — _open_food_facts.py

- open_food_facts_product — Get packaged food metadata by barcode from Open Food Facts
- open_food_facts_search — Search packaged foods with structured Open Food Facts filters
- open_food_facts_nutrition — Get a nutrition-focused Open Food Facts summary by barcode
- open_food_facts_compare — Compare nutrition signals for multiple Open Food Facts products

Food Safety — _openfda_food.py

- openfda_food_recall_search — Search FDA food enforcement recalls via openFDA
- openfda_food_recall — Get a specific FDA food enforcement recall by recall number

Food Ontology — _foodon.py

- foodon_search — Search FoodOn ontology terms via EMBL-EBI OLS
- foodon_term — Get a FoodOn ontology term by OBO ID

Biodiversity — _gbif.py

- gbif_species_match — Resolve scientific names to GBIF taxon keys
- gbif_species_search — Search GBIF taxa by name, rank, or parent taxon
- gbif_species — Get GBIF taxon metadata by taxon key
- gbif_occurrence_search — Search GBIF species occurrence records

Development Data — _world_bank.py

- world_bank_topics — List World Bank indicator topics
- world_bank_sources — List World Bank data sources/databases
- world_bank_countries — List or search World Bank countries, economies, and aggregates
- world_bank_indicators — Browse or search World Bank indicators with topic/source filters
- world_bank_indicator — Get metadata for a specific World Bank indicator
- world_bank_series — Get a World Bank indicator time series for a country or aggregate
- world_bank_compare — Compare a World Bank indicator across multiple countries or aggregates

Global Health Statistics — _who_gho.py

- who_indicators — Search WHO Global Health Observatory indicators
- who_indicator — Get WHO GHO indicator metadata by code
- who_series — Fetch WHO GHO observations for an indicator

European Statistics — _eurostat.py

- eurostat_dataset_search — Search Eurostat datasets/dataflows
- eurostat_dataset — Get Eurostat dataset metadata and dimension summary
- eurostat_dimensions — List Eurostat dimensions and sample category codes
- eurostat_series — Get Eurostat observations with generic dimension filters
- eurostat_compare — Compare a Eurostat dataset across geo codes

Earthquakes — _earthquake.py

- earthquake_search — Search USGS earthquake events by date, magnitude, depth, and location
- earthquake_event — Get a USGS earthquake event by ID
- earthquake_count — Count USGS earthquake events for a date/magnitude query

Natural Events — _eonet.py

- eonet_categories — List NASA EONET event categories
- eonet_events — Search NASA EONET natural events
- eonet_event — Get a NASA EONET event by ID

Wikipedia — _wikipedia.py

- wikipedia_search — Search Wikipedia, return titles with summaries
- wikipedia_article — Get a specific Wikipedia article summary
- wikipedia_related — Get related article titles from a Wikipedia page

Dictionary — _dictionary.py

- define_word — Dictionary definition via Free Dictionary API

News — _news.py

- hacker_news — Top stories from Hacker News

Video Transcripts — _youtube.py

- youtube_transcript — Fetch public YouTube transcript text or timestamped segments
- youtube_transcript_languages — List available transcript languages for a YouTube video
- youtube_transcript_search — Search within a YouTube transcript and return timestamped matches

Research — _arxiv.py

- arxiv_search — Search arXiv papers via the public arXiv API
- arxiv_paper — Get metadata for a specific arXiv paper by ID

Biomedical Research — _pubmed.py

- pubmed_search — Search PubMed articles via the public NCBI E-utilities API
- pubmed_article — Get metadata for a specific PubMed article by PMID

Biomedical Research — _europe_pmc.py

- europe_pmc_search — Search Europe PMC articles via the public REST API
- europe_pmc_article — Get Europe PMC metadata for a PMID, PMCID, DOI, or source ID
- europe_pmc_citations — Get articles that cite a Europe PMC record

Protein Knowledge — _uniprot.py

- uniprot_search — Search UniProtKB proteins
- uniprot_entry — Get UniProtKB entry metadata by accession
- uniprot_features — List UniProtKB sequence features
- uniprot_sequence — Get a UniProtKB protein sequence in FASTA form
- uniprot_crossrefs — List UniProtKB database cross-references

Biomolecular Structures — _pdb.py

- pdb_search — Search RCSB PDB structures by free text
- pdb_entry — Get RCSB PDB entry metadata
- pdb_ligands — List non-polymer ligands for a PDB entry
- pdb_chemical_component — Get RCSB chemical component metadata

Chemistry & Bioactivity — _chembl.py

- chembl_molecule_search — Search ChEMBL molecules by name or synonym
- chembl_molecule — Get ChEMBL molecule metadata
- chembl_target_search — Search ChEMBL biological targets
- chembl_target — Get ChEMBL target metadata
- chembl_activity_search — Search ChEMBL bioactivity measurements

Medication Labels — _rxnorm_dailymed.py

- rxnorm_drug_search — Search RxNorm drug concepts by name
- rxnorm_concept — Get RxNorm concept properties by RxCUI
- rxnorm_related — Get related RxNorm concepts
- rxnorm_ndcs — List NDC product codes for an RxNorm concept
- dailymed_label_search — Search DailyMed SPL drug labels
- dailymed_label — Get DailyMed SPL label metadata and section titles

Clinical Studies — _clinical_trials.py

- clinical_trials_search — Search ClinicalTrials.gov studies via the public API v2
- clinical_trial_study — Get detailed metadata for a specific ClinicalTrials.gov NCT ID

Academic Graph — _semantic_scholar.py

- semantic_scholar_search — Search Semantic Scholar papers via the public Academic Graph API
- semantic_scholar_paper — Get detailed metadata for a Semantic Scholar paper
- semantic_scholar_citations — Get papers that cite a Semantic Scholar paper

Books — _open_library.py

- open_library_search — Search Open Library books and works
- open_library_work — Get metadata for a specific Open Library work
- open_library_isbn — Get edition metadata for an ISBN

Scholarly Metadata — _crossref.py

- crossref_search — Search Crossref works by title, DOI, topic, or citation fragment
- crossref_work — Get Crossref metadata for a specific DOI

Knowledge Graph — _wikidata.py

- wikidata_search — Search Wikidata entities by label or alias
- wikidata_entity — Get labels, aliases, claims, and Wikipedia links for a Wikidata QID
- wikidata_sparql — Run read-only Wikidata SPARQL SELECT/ASK queries

Research Organizations — _ror.py

- ror_search — Search ROR research organizations
- ror_organization — Get ROR organization metadata

MediaWiki & Wiktionary — _mediawiki.py

- mediawiki_search — Search a public MediaWiki API
- mediawiki_page — Fetch and lightly clean a MediaWiki page's wikitext
- mediawiki_sections — List sections for a MediaWiki page
- wiktionary_entry — Fetch a Wiktionary entry focused on one language section

OpenStreetMap Queries — _overpass.py

- overpass_query — Run a bounded Overpass QL query
- overpass_pois — Search OpenStreetMap points/ways/relations by tag in a bbox or radius

News & Events — _gdelt.py

- gdelt_news_search — Search global news coverage via GDELT DOC 2.0
- gdelt_timeline — Get a GDELT volume timeline for a news query

Security — _nvd.py

- nvd_cve_search — Search CVEs in the NVD 2.0 API by keyword, CVE, CPE, severity, or publication date
- nvd_cve — Get NVD metadata for a specific CVE ID

Research Data — _datacite.py

- datacite_search — Search DataCite DOI metadata for datasets, software, text, and other research outputs
- datacite_doi — Get DataCite metadata for a specific DOI

Digital Archives — _internet_archive.py

- internet_archive_search — Search Internet Archive items with optional mediatype and collection filters
- internet_archive_item — Get Internet Archive item metadata and file listings

Math — _math.py

- math_eval — Safely evaluate math expressions (functions, constants, operators)
- unit_convert — Convert between units (length, mass, volume, speed, area, time, temp)

Text Processing — _text.py

- regex_search — Find all regex matches with positions
- text_stats — Count words, characters, lines, sentences, paragraphs
- base64_encode — Encode text to base64
- base64_decode — Decode base64 to text

Data — _json.py

- json_extract — Extract values from JSON via dot-notation paths
- csv_read — Read CSV files, return formatted table

Filesystem — _filesystem.py

- read_file — Read file contents with optional line limit
- list_directory — List files/dirs with sizes and types
- search_files — Recursively search for text in files

Shell — _shell.py

- run_command — Execute shell commands and return output

Web — _web.py

- http_get — Fetch a URL, return raw response text
- scrape_text — Fetch web page, extract visible text (strips HTML)
