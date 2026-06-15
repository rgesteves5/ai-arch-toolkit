# Candidate Tools

Candidates to evaluate before implementation. Preference remains: public APIs, no account,
no API key, stable JSON/CSV/RSS endpoints, bounded outputs, and clear IDs that agents can reuse.

## 1. ECB FX and Reference Rates

- Area: Currencies / macro market data
- Source: European Central Bank Data Portal / SDW API
- Candidate tools: `ecb_fx_rates`, `ecb_fx_series`, `ecb_reference_rates`
- Inputs: `base_currency`, `quote_currency`, `start_date`, `end_date`, `frequency`
- Agent value: reliable EUR reference rates and central-bank time series.
- Risk: SDMX shape needs careful parsing.
- Reference: https://data-api.ecb.europa.eu/

## 2. Stooq Market Data

- Area: Companies / stock market / FX / indices
- Source: Stooq CSV endpoints
- Candidate tools: `stooq_quote`, `stooq_history`, `stooq_search_symbol`
- Inputs: `symbol`, `interval`, `start_date`, `end_date`
- Agent value: simple no-key quotes and historical prices for many symbols.
- Risk: unofficial-ish endpoint conventions; symbol discovery may need curation.
- Reference: https://stooq.com/db/h/

## 3. CoinGecko Crypto Markets

- Area: Crypto / market data
- Source: CoinGecko public API
- Candidate tools: `coingecko_coin_search`, `coingecko_coin`, `coingecko_market_chart`
- Inputs: `query`, `coin_id`, `vs_currency`, `days`
- Agent value: prices, market caps, volume, coin metadata.
- Risk: public/demo rate limits; must throttle.
- Reference: https://docs.coingecko.com/

## 4. Blockchain.com Chain Stats

- Area: Crypto / blockchain
- Source: Blockchain.com charts and stats APIs
- Candidate tools: `blockchain_stats`, `blockchain_chart`, `blockchain_block`
- Inputs: `chart`, `timespan`, `block_hash_or_height`
- Agent value: Bitcoin network statistics and chain-level signals.
- Risk: mostly Bitcoin-focused.
- Reference: https://www.blockchain.com/explorer/api/charts_api

## 5. Our World in Data Energy

- Area: Energy / macro data
- Source: Our World in Data Grapher CSV/JSON
- Candidate tools: `owid_energy_series`, `owid_energy_compare`, `owid_grapher_metadata`
- Inputs: `indicator_slug`, `countries`, `start_year`, `end_year`
- Agent value: energy mix, electricity, fossil fuels, renewables, emissions-adjacent data.
- Risk: dataset slugs need discovery/curation.
- Reference: https://ourworldindata.org/grapher

## 6. UK Carbon Intensity

- Area: Energy / electricity
- Source: National Grid ESO Carbon Intensity API
- Candidate tools: `uk_carbon_intensity_current`, `uk_carbon_intensity_forecast`,
  `uk_generation_mix`
- Inputs: `postcode`, `from_datetime`, `to_datetime`
- Agent value: real operational electricity carbon intensity and generation mix.
- Risk: UK-specific.
- Reference: https://carbon-intensity.github.io/api-definitions/

## 7. Package Registry Metadata

- Area: Package / software registries
- Sources: PyPI JSON API, npm registry, crates.io API, Maven Central Solr
- Candidate tools: `package_search`, `package_info`, `package_versions`,
  `package_dependencies`
- Inputs: `ecosystem`, `package_name`, `version`, `query`
- Agent value: versions, releases, licenses, maintainers, dependencies.
- Risk: each ecosystem has different schemas and rate expectations.
- References:
  - https://warehouse.pypa.io/api-reference/json.html
  - https://github.com/npm/registry/blob/master/docs/REGISTRY-API.md
  - https://crates.io/data-access
  - https://central.sonatype.org/search/rest-api-guide/

## 8. OSV Vulnerability Database

- Area: Security / package risk
- Source: OSV API
- Candidate tools: `osv_query_package`, `osv_vulnerability`, `osv_batch_query`
- Inputs: `ecosystem`, `package_name`, `version`, `vulnerability_id`
- Agent value: open-source package vulnerability lookup.
- Risk: package ecosystem naming must be normalized.
- Reference: https://google.github.io/osv.dev/api/

## 9. CISA KEV and EPSS Risk

- Area: Security / exploit prioritization
- Sources: CISA Known Exploited Vulnerabilities catalog, FIRST EPSS API
- Candidate tools: `cisa_kev_search`, `cisa_kev_cve`, `epss_score`
- Inputs: `cve_id`, `vendor`, `product`, `from_date`, `to_date`
- Agent value: prioritizes vulnerabilities by known exploitation and probability.
- Risk: EPSS and KEV are complementary, not complete vulnerability databases.
- References:
  - https://www.cisa.gov/known-exploited-vulnerabilities-catalog
  - https://www.first.org/epss/api

## 10. GLEIF LEI

- Area: Companies / legal entities
- Source: GLEIF LEI API
- Candidate tools: `lei_search`, `lei_entity`, `lei_relationships`
- Inputs: `query`, `lei`, `country`, `status`
- Agent value: legal entity identifiers, registration status, addresses, corporate links.
- Risk: not a full company registry; only LEI-covered entities.
- Reference: https://documenter.getpostman.com/view/7679680/SVYrrxuU

## 11. Nager.Date Public Holidays

- Area: Events / calendars
- Source: Nager.Date public API
- Candidate tools: `public_holidays`, `next_public_holidays`, `country_holiday_info`
- Inputs: `country_code`, `year`
- Agent value: holiday-aware planning, scheduling, localization.
- Risk: mostly public holidays, not events.
- Reference: https://date.nager.at/Api

## 12. RSS and Atom Feeds

- Area: News / events / web
- Source: generic RSS/Atom URLs
- Candidate tools: `feed_read`, `feed_search`, `feed_discover`
- Inputs: `url`, `query`, `max_results`
- Agent value: source-agnostic news/events ingestion without key.
- Risk: feed quality varies; XML parsing and date formats need care.
- Reference: https://www.rssboard.org/rss-specification

## 13. DuckDuckGo Instant Answer

- Area: Web / lightweight search
- Source: DuckDuckGo Instant Answer API
- Candidate tools: `duckduckgo_instant_answer`, `duckduckgo_related_topics`
- Inputs: `query`
- Agent value: no-key quick answers, disambiguation, related topics.
- Risk: not a general web search API; coverage is inconsistent.
- Reference: https://duckduckgo.com/api

## 14. WHO Disease Outbreak News

- Area: Public health alerts
- Source: WHO Disease Outbreak News pages/feed
- Candidate tools: `who_don_alerts`, `who_don_article`
- Inputs: `query`, `disease`, `country`, `from_date`, `to_date`
- Agent value: official outbreak alerts and public-health event context.
- Risk: may require feed/page parsing rather than clean JSON.
- Reference: https://www.who.int/emergencies/disease-outbreak-news

## 15. ReliefWeb

- Area: Public health alerts / humanitarian events / disasters
- Source: ReliefWeb API
- Candidate tools: `reliefweb_reports`, `reliefweb_disasters`, `reliefweb_report`
- Inputs: `query`, `country`, `disaster_type`, `from_date`, `to_date`
- Agent value: operational disaster, health, humanitarian and crisis reporting.
- Risk: broad taxonomy; filters should stay simple.
- Reference: https://apidoc.reliefweb.int/

## 16. NASA POWER Agroclimate

- Area: Agriculture / weather risk
- Source: NASA POWER API
- Candidate tools: `nasa_power_daily`, `nasa_power_climatology`, `agro_weather_risk`
- Inputs: `latitude`, `longitude`, `parameters`, `start_date`, `end_date`
- Agent value: solar radiation, temperature, precipitation and agri-relevant weather history.
- Risk: parameter catalog needs careful discovery and defaults.
- Reference: https://power.larc.nasa.gov/docs/services/api/

## 17. Math Statistics Toolkit

- Area: Math / data analysis
- Source: stdlib-only local implementation
- Candidate tools: `stats_summary`, `stats_correlation`, `linear_regression`,
  `percentiles`
- Inputs: numeric lists or CSV column values
- Agent value: quick quantitative reasoning without external services.
- Risk: input parsing must be strict to avoid confusing results.
- Reference: local tool only.

## 18. Library of Congress

- Area: Books / archives / cultural heritage
- Source: Library of Congress JSON API
- Candidate tools: `loc_search`, `loc_item`, `loc_collections`
- Inputs: `query`, `format`, `collection`, `item_id`
- Agent value: books, maps, images, manuscripts, public cultural metadata.
- Risk: heterogeneous records.
- Reference: https://www.loc.gov/apis/json-and-yaml/

## 19. Project Gutenberg / Gutendex

- Area: Books / archives
- Source: Gutendex API over Project Gutenberg metadata
- Candidate tools: `gutenberg_search`, `gutenberg_book`, `gutenberg_downloads`
- Inputs: `query`, `author`, `language`, `topic`, `book_id`
- Agent value: public-domain book discovery and downloadable formats.
- Risk: metadata can be sparse.
- Reference: https://gutendex.com/

## 20. iCalendar Parser

- Area: Events / calendars / web
- Source: generic `.ics` calendar URLs or local text
- Candidate tools: `ical_read`, `ical_events`, `ical_next_events`
- Inputs: `url_or_text`, `from_date`, `to_date`, `query`
- Agent value: calendar ingestion from public event feeds, project calendars, conferences.
- Risk: recurring events are tricky; implement a bounded subset first.
- Reference: https://datatracker.ietf.org/doc/html/rfc5545

## 21. YouTube Transcripts

- Status: implemented as `_youtube.py` with optional `youtube` extra.
- Area: Video / transcripts / web
- Source: public YouTube transcript endpoints via no-key client behavior
- Candidate tools: `youtube_transcript`, `youtube_transcript_languages`,
  `youtube_transcript_search`
- Inputs:
  - `youtube_transcript`: `video_url_or_id`, `languages`, `prefer_manual`,
    `allow_generated`, `translate_to`, `format`, `max_chars`
  - `youtube_transcript_languages`: `video_url_or_id`
  - `youtube_transcript_search`: `video_url_or_id`, `query`, `languages`,
    `prefer_manual`, `allow_generated`, `max_results`
- Agent value: retrieve public manual or auto-generated transcripts, inspect available
  languages, search within transcripts, and return timestamped segments for citation.
- Risk: no official public transcript API for arbitrary videos; relies on undocumented
  YouTube web behavior and may break or be rate-limited.
- Reference: https://github.com/jdepoix/youtube-transcript-api

## Keyed Web Search Providers

These candidates require API keys. They are kept separate from the no-key candidates because
they provide substantially stronger real web search, news search, SERP, and agent-grounding
capabilities.

### 22. Brave Search API

- Area: Web search / news / agent grounding
- Source: Brave Search API
- Candidate tools: `brave_web_search`, `brave_news_search`, `brave_llm_context`,
  `brave_image_search`, `brave_video_search`, `brave_local_search`
- Inputs:
  - `brave_web_search`: `query`, `country`, `search_lang`, `ui_lang`, `freshness`,
    `count`, `offset`, `safe_search`, `extra_snippets`, `goggles`
  - `brave_news_search`: `query`, `country`, `search_lang`, `freshness`, `count`,
    `offset`
  - `brave_llm_context`: `query`, `country`, `search_lang`, `freshness`, `count`,
    `max_urls`, `max_tokens`, `max_snippets`, `threshold_mode`, `goggles`
  - `brave_image_search`: `query`, `country`, `search_lang`, `count`, `safe_search`
  - `brave_video_search`: `query`, `country`, `search_lang`, `freshness`, `count`
  - `brave_local_search`: `query`, `lat`, `lon`, `city`, `state`, `country`, `count`
- Agent value: independent real search index, freshness filters, news/images/videos/local
  verticals, and LLM-ready grounding snippets via `brave_llm_context`.
- Risk: requires account and API key; pagination is intentionally bounded by Brave; local POI
  IDs can be ephemeral.
- Suggested priority: implement `brave_llm_context`, `brave_web_search`, then
  `brave_news_search`.
- References:
  - https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
  - https://api-dashboard.search.brave.com/documentation/services/llm-context
  - https://api-dashboard.search.brave.com/app/documentation/news-search/get-started

### 23. Tavily

- Area: Agent web search / extraction / crawl / news / finance
- Source: Tavily API
- Candidate tools: `tavily_search`, `tavily_extract`, `tavily_crawl`, `tavily_map`
- Inputs:
  - `tavily_search`: `query`, `topic`, `search_depth`, `max_results`, `time_range`,
    `start_date`, `end_date`, `include_domains`, `exclude_domains`, `include_answer`,
    `include_raw_content`
  - `tavily_extract`: `urls`, `include_images`, `extract_depth`, `format`
  - `tavily_crawl`: `url`, `max_depth`, `limit`, `include_paths`, `exclude_paths`
  - `tavily_map`: `url`, `max_depth`, `limit`
- Agent value: agent-first web search with cleaned content, optional raw page content, news and
  finance topics, domain filters, extraction, site crawling, and URL discovery.
- Risk: requires account and API key; advanced search/extraction consumes more credits; crawl
  tools need strict limits to avoid expensive broad traversals.
- Suggested priority: implement `tavily_search`, then `tavily_extract`.
- Reference: https://docs.tavily.com/documentation/api-reference/endpoint/search

### 24. Serper.dev

- Area: Google SERP / news / images / videos / places / scholar / patents
- Source: Serper.dev Google Search API
- Candidate tools: `serper_search`, `serper_news_search`, `serper_images_search`,
  `serper_videos_search`, `serper_places_search`, `serper_scholar_search`,
  `serper_patents_search`, `serper_autocomplete`
- Inputs:
  - `serper_search`: `query`, `country`, `language`, `location`, `page`, `num`,
    `autocorrect`
  - `serper_news_search`: `query`, `country`, `language`, `time_range`, `page`, `num`
  - `serper_images_search`: `query`, `country`, `language`, `page`, `num`
  - `serper_videos_search`: `query`, `country`, `language`, `page`, `num`
  - `serper_places_search`: `query`, `location`, `country`, `language`
  - `serper_scholar_search`: `query`, `language`, `page`, `num`
  - `serper_patents_search`: `query`, `country`, `language`, `page`, `num`
  - `serper_autocomplete`: `query`, `country`, `language`
- Agent value: low-cost Google-like SERP access across many verticals, including knowledge
  graph, organic results, people-also-ask, related searches, news, scholar, patents, and
  local places.
- Risk: requires account and API key; output mirrors SERP structure, so agents may need a
  normalization layer; commercial/legal risk profile differs from first-party APIs.
- Suggested priority: implement `serper_search`, then `serper_news_search`, then
  `serper_scholar_search`.
- Reference: https://serper.dev/
