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

Geography — _geo.py

- geocode — Coordinates and country for a city name
- reverse_geocode — Place name and region from coordinates
- timezone_lookup — Timezone and UTC offset from coordinates
- distance_between — Great-circle distance between coordinate pairs
- ip_lookup — Geographic location and ISP info for an IP
- country_info — Country details (capital, population, languages, etc.)

Wikipedia — _wikipedia.py

- wikipedia_search — Search Wikipedia, return titles with summaries
- wikipedia_article — Get a specific Wikipedia article summary
- wikipedia_related — Get related article titles from a Wikipedia page

Dictionary — _dictionary.py

- define_word — Dictionary definition via Free Dictionary API

News — _news.py

- hacker_news — Top stories from Hacker News

Research — _arxiv.py

- arxiv_search — Search arXiv papers via the public arXiv API
- arxiv_paper — Get metadata for a specific arXiv paper by ID

Biomedical Research — _pubmed.py

- pubmed_search — Search PubMed articles via the public NCBI E-utilities API
- pubmed_article — Get metadata for a specific PubMed article by PMID

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
