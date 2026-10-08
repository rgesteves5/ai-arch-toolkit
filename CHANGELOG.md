# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Upgrade notes

Code written against the `main` before the hardening front (provider errors and metering, tools,
flows, manifests) needs these changes; each one is detailed below.

- **Python 3.13.4.** The package now requires Python 3.13.4 or later (D64): a path that does not
  exist yet is resolved with `os.path.realpath(strict=os.path.ALLOW_MISSING)`, which 3.13.4 added.
- **Provider errors.** Catch `ProviderError`, or one of `RequestError`, `TransportError`,
  `ProviderTimeout` and `ResponseError`: SDK and HTTP-library exceptions no longer leave the
  adapters (`APIError` handlers keep working). `fallback_on` defaults to `(ProviderError,)`;
  pass the old tuple to keep falling back on `ConnectionError`, `TimeoutError` or `OSError`.
- **Requests a model does not take.** A thinking effort, `thinking=True`, sampling parameter,
  forced `tool_choice`, server-tool config or message role that the model's documentation does not
  allow raises `RequestError` before anything is sent, instead of reaching the API or being
  dropped.
- **Prices.** A model id is priced by its own entry: register the variants you call
  (`pricing.register("o3-pro", ...)`), or keep a family with `match="prefix"`;
  `PricingRegistry.register()` takes `model` as its first parameter. Under a `MeterScope`, a model
  without a price raises `UnpricedModelError`: register one (`ModelPricing()` for a local model).
  `Response.cost` is `None`, not `0`, when the provider reported no usage.
- **Custom meters.** `MeterOperation.fail(...)` requires a delivery disposition, and
  `unknown_cost_count` counts only unbounded costs.
- **Streams.** A finished stream carries the same `Response` as `complete()`, and
  `stream_events()` emits the tool-call events after the text, in the response's order. Before
  them come the calls' pieces, as `tool_call_delta` events (G-37): code that matches every
  `StreamEvent.kind` meets a new one, and code that counts events sees more. A piece counts as
  delivered, as text does: in `stream_events()` and in an iterated flow, an answer of tool calls
  only that fails after its first piece is no longer retried or passed to the fallback.
- **Tools.**
  - A toolkit tool that cannot answer raises `ToolFailure` (T01, D42) instead of returning an
    error string: called through `run_tools()`, a `ToolGroup` or `execute_tool()` it gives
    `ToolResult(ok=False)` with `error.type` in `not_found`, `validation_error`, `upstream`,
    `rate_limited`. Code that calls a tool function directly catches `ToolFailure`; code that
    looked for "failed:" in a returned string reads `result.ok` and `result.error.type`.
  - `csv_read` is in `toolkit.tools.dangerous` and needs approval.
  - The eight wiki tools are four (T05, D41: one tool per job and per source, no aliases); the
    table below gives each old call's replacement. Their answers are text from the page the wiki
    renders, and a long one is a window: called directly, the tool returns a `ToolResult` whose
    `value` is the text (the executor hands the model that text, as before).
  - Every other toolkit tool now keeps the tools contract too (T06 to T09, C08), so the wiki
    family's rules hold for all of them: a tool whose answer can be long is a window and, called
    directly, returns a `ToolResult` whose `value` is the text; numeric limits are `Range` bounds,
    so a value outside is refused (`validation_error`) instead of moved to the nearest limit; and
    a request the source refuses for its arguments is a `validation_error`, where it was
    `upstream`. Eight tools are gone into others of their source, and `ror_search` no longer
    takes `max_results` (the table below).
  - Some arguments changed meaning: `uniprot_search` pages by `cursor` and `offset`, both from
    the footer; `world_bank_indicators(page=…)` pages through the matches, no longer the
    catalogue; `wikidata_sparql`'s `max_results` is the rows shown per call, no longer the
    `LIMIT` it appended. `search_files` lines read `path:line:offset: text`; `list_directory`
    refuses a pattern that matches more than 100,000 entries; `run_command` and `python_repl`
    show the start of an output that does not fit, with how to narrow it (no call reads on).
  - `search_files` gives each path under `directory` as you passed it (`sub/notes.txt`, or an
    absolute path), which `read_file` reads as it is; it was relative to `directory`.
    `run_command` stops every process the command starts when the call returns (one left in the
    background half a second after the command ends: end the command with `wait`), gives the
    command no input, and refuses to run on Windows (a typed `upstream` failure).
  - Tool names must be portable, `^[A-Za-z_][A-Za-z0-9_-]{0,63}$`, or `ValueError` (C02, D62);
    two different tools with one name in a `tools=[...]` list raise `ValueError` before anything
    is sent or runs; and `@tool(schema=...)` takes per-parameter overrides only (a complete
    schema raises `TypeError`: build that tool with `tool_from_schema`).
  - Every tool call has a 120-second deadline and a 200,000-character output cap: set
    `timeout_s`/`max_output_chars` (or `None`) on a tool that needs more.
  - On the sync path a synchronous tool runs in a thread of its own: open thread-bound resources
    inside the tool.
  - `ip_lookup` queries ipwho.is over HTTPS; the MediaWiki tools accept only HTTPS Wikimedia
    hosts.
  - A `ToolGroup` refuses another tool under a name it holds (`ValueError`), where it used to
    replace the first with a warning.
- **Flows and agents.**
  - An agent manifest, prompt manifest or JSON/TOML/YAML resource that nests deeper than 100
    levels, or whose YAML aliases expand past their bound, is refused (D59). So is a YAML alias
    inside its own anchor (`loop: &a [*a]`), which used to load as a list that contains itself,
    and an agent override whose dotted path and value together pass 100 levels.
  - A `FlowStep` names each dependency once: a step named twice, in `after` or across `after`,
    `after_any` and `after_optional`, raises `ValueError`. A DAG's skip reasons name every
    dependency that did not succeed (`"dependencies 'a' failed, 'b' was skipped"`), and "all
    dependencies skipped" is gone: read `StepTrace.blocked_by` instead of parsing them. Meter span
    ids are paths from the run's root (`run/3/7`), not `span-N`.
  - The steps of a parallel DAG wave read the wave's snapshot: return changes as artifacts, since
    a value mutated in place now changes the state in every mode. The trace records a wave's steps
    as they finish.
  - `ReasoningSpec.knobs` and `llm_kwargs` are read-only: build a new spec with
    `dataclasses.replace`. A spec no longer deep-copies or pickles.
  - An agent's answer is its flow's `"answer"` or, without one, its last step's value: a flow
    wrapped with `Agent.from_flow` writes `"answer"` (and `"response"`) to keep its text.
- **Manifests.** Shape errors changed wording (they name the field's path). These values are now
  refused:
  - a `strategy.system` that is not text;
  - `version: true`;
  - a boolean `order`;
  - a `metadata_attributes` that is not a list;
  - `select` or `serialize_as` on an inline template.

#### Renamed and removed tools

| Before | Now |
|---|---|
| `wikipedia_search(query, results=3)` | `wiki_search(query, max_results=3)` |
| `wikipedia_article(title, max_chars)` | `wiki_read(title, max_chars=…)`; `section=0` reads the introduction alone |
| `wikipedia_related(title, limit)` | `wiki_search(f"morelike:{title}", max_results=limit)`: the pages most like it, where `wikipedia_related` listed the page's own links |
| `mediawiki_search(query, api_url, max_results, offset)` | `wiki_search(query, wiki="en.wikibooks.org", max_results=…, offset=…)` |
| `mediawiki_page(title, api_url, max_chars)` | `wiki_read(title, wiki=…, max_chars=…)`, with `section=`, `find=` and `offset=` |
| `mediawiki_sections(title, api_url)` | `wiki_outline(title, wiki=…)` |
| `wiktionary_entry(term, language, max_chars)` | `wiktionary_entry(term, language, offset=…, max_chars=…)`; a language the entry lacks is `not_found` and names those it has, where the whole entry came back |
| `define_word(word)` | `wiktionary_entry(word)` (dictionaryapi.dev answered 522) |
| `ror_search(query, max_results=5, page=2)` | `ror_search(query, page=2)`: ROR's whole page of 20, numbered |
| `open_food_facts_nutrition(barcode)` | `open_food_facts_product(barcode)`: the same product, whole, with the Nutri-Score and NOVA group labelled |
| `dailymed_label(setid, max_sections)` | `dailymed_label(setid, offset=…)`, every section numbered; a section's text: `dailymed_label_text(setid, section=N)` |
| `eurostat_dimensions(dataset_id, max_values)` | `eurostat_dataset(dataset_id, dimension="geo")`: every code with its label, page by page |
| `eurostat_compare(dataset_id, geo_codes, filters, last_time_periods)` | `eurostat_series(dataset_id, filters="geo=PT+ES+FR,…", last_time_periods=…)` |
| `world_bank_compare(indicator, countries, year, start_year, end_year, max_points)` | `world_bank_series("PRT,ESP,DEU", indicator, start_year=…, end_year=…, max_results=…, page=…)`; one year is `start_year` alone |
| `reverse_geocode(lat, lon)` | `osm_reverse_geocode(latitude, longitude, zoom=10)` (zoom 10 is the city; 18, the default, a building) |
| `get_weather_by_coords(lat, lon)` | `get_weather(latitude=lat, longitude=lon)` |
| `weather_units(city, unit="f")` | `get_weather(city, units="imperial")`: °F, mph and inch, converted by Open-Meteo |
| `get_forecast_by_coords(lat, lon, days)` | `get_forecast(latitude=lat, longitude=lon, days=days)` |

The `api_url` of the old MediaWiki tools is now `wiki`, the wiki's host (`"en.wikibooks.org"`);
English Wikipedia is the default, where the MediaWiki tools defaulted to the English Wiktionary.
`max_chars` takes 500 to 20,000 and a value outside is refused (`validation_error`), where the old
tools moved it into their own limits without a word (1 to 100,000, or 200 to 4,000).

### Added
- **Typed filesystem writes bound to a `FilesystemPolicy`** (C07, D64):
  `filesystem_tools(policy)` in `ai_arch_toolkit.toolkit.tools.dangerous` gives `read_file`,
  `list_directory` and `search_files` bound to the policy's `read_roots` (same names, schemas and
  approval as the module-level ones), and `write_file`, `append_file`, `make_directory` and
  `move_path` bound to its `write_roots`. Each tool checks its paths with the policy right before
  it acts and reaches them from the root down, one folder at a time, never through a link. A
  write is atomic (a temporary file synced to disk takes the name in one step; without
  `overwrite` it never replaces a file, so of two writes racing to one new path one fails),
  `append_file` refuses a file with other hard links, `move_path` never copies, and no tool
  deletes. All need approval. The four writes take turns, one at a time in a process; one that
  waits over 30 s for another fails (`upstream`, retryable). `move_path` refuses to bring a file
  the agent cannot read into a folder it reads (`permission_denied`, in the gate and the tool),
  refuses a move between two hard links of one file (`validation_error`), and leaves the source's
  name in place when another process gave it to another file between the link and the unlink. A
  folder that cannot be synced once the change is made is a note in the answer, not a failure.
- `FilesystemPolicy(read_roots=, write_roots=, delete_roots=, cwd=, max_write_bytes=)`, whose
  `check(path, action)` returns the canonical path or raises `FilesystemPolicyError` (a
  `PermissionError`), and `PathScopeGate(policy, paths=...)`, a gate that refuses a path outside
  the policy's folders before anyone is asked (`permission_denied`, never run nor metered, with
  `audit["filesystem"]`), hands an allowed call on with its paths canonical, and blocks a
  `capability="filesystem"` tool it has no map for (C07). A path outside the roots gets one
  refusal, which names it as given, whatever lies there and wherever it leads: nothing outside is
  looked at before the roots are tested. Inside the roots, a path through a file or a loop of
  links is a `validation_error` in the tools. The gate refuses a `list_directory` pattern with a
  `..` part and a `move_path` that would widen what the agent reads, and its block of an unmapped
  filesystem tool names that tool's arguments.
- `"permission_denied"` in `ToolFailureType` and in `GovernanceOutcome`: the word a
  `FilesystemPolicy` refusal carries, in the gate and in the tool (D42 addendum, C07).
- **A tool's preview hook** (C07c, D64): `@tool(preview=fn)` and `tool_from_schema(...,
  preview=fn)` set `ToolDefinition.preview`, a plain function that receives a copy of the call's
  arguments (validated, and changed by the gates before the approval) and returns what the call
  will do, in words. `ApprovalRequest.preview` is its text, and `DryRunGate` records it under
  `audit["preview"]`. The hook runs in a daemon thread of its own on both paths, in a copy of
  the caller's context; a hook that raises, returns anything but text, or has not returned
  within 10 seconds (`PREVIEW_TIMEOUT_S`) is logged, and the preview falls back to the arguments
  as JSON, as for a tool without a hook. Either preview is cut at 16,000 characters with a note
  (200 KB of arguments made a preview of 200,063 characters); `ApprovalRequest.arguments` stays
  whole. An `async def` or a non-callable `preview` raises `TypeError`.
- The filesystem write tools preview their calls: `create /…/a.md (12 bytes)`, or `replace
  /…/a.md (1204 → 1311 bytes)` with a unified diff of the lines that change (cut at 80 lines or
  8 KB; none for a binary file, or one or new text over 256 KB), `append 6 bytes to …`, `make
  folder …`, `move file … to …`, or `will fail (<type>): <why>`. A preview checks the paths with
  the policy and reads from the root down without following a link, and never writes (C07c).
- **Dynamic tools** (C02, D62): `tool_from_schema(handler, name=, description=, input_schema=,
  policy=)` builds a governed tool from a complete JSON Schema, for tools that arrive as data (an
  MCP server's, for one). The handler receives the arguments in one `dict`, so keys need not be
  Python names; an `async def` handler runs on the loop, a synchronous one in a thread. The tool
  goes through the same executor as `@tool` (validation, gates, approval, `max_calls`,
  metering) and is accepted by `ToolGroup`, `prepare_tools`, `execute_tool` and `run_tools`. Its
  schema is copied and sent as it came (`title`, `$schema`, `x-*` keys included), with local
  `#/$defs/...` references inlined. A schema that is not JSON, whose root is not an object, that
  refers outside itself or to a definition that is not an object schema (`true`, a list), that
  nests deeper than 100 levels, or whose references would inline to more than about 100,000
  characters or 100 levels (each reference followed counts as one) raises `ValueError`, and
  nothing else does.
- `ToolGroup.add(fn, replace=True)` swaps the tool held under a name, and
  `ToolGroup.remove(name)` takes one out and returns its definition (`KeyError` if absent). A
  group changes copy-on-write: the next read (the next ReAct turn) sees the change, a call
  already running ends with the tool it started with, and `max_calls` keeps counting (C02).
- **A technical model catalog** (C06, D63): `model_catalog`, `ModelCatalog`, `ModelCapabilities`
  and `Provenance` (`ai_arch_toolkit.core`) say what each model takes through its adapter:
  context window, input and output limits as each provider publishes them, input and output
  modalities, tools and tool choices, structured output, JSON mode, streaming, reasoning mode
  and efforts, thinking budget, server tools. Each fact carries its source (`docs`, `probe`,
  `api`, `adapter`, `override`) and the day it was read, and is `None` when unknown, never "no".
  The shipped seed covers the current line (30 chat models still served) and the image models,
  read from the providers' pages on 2026-10-08; the rest comes from each adapter's own rules,
  through a pure `model_facts` classmethod, which also narrows a published modality list to what
  the adapter carries. Override with `load()` (a strict TOML loader) and `register()`, which
  checks its facts as `load()` does, field by field; both, and `unregister()`, name a model by its
  id, an alias or a dated snapshot, as `get()` does. `ModelCatalog(defaults=False)` starts empty,
  and a catalog is safe to share between threads. Nothing in the toolkit reads the catalog: the
  adapters keep their rules. See [Model catalog](docs/model-catalog.md).
- `scripts/probe_models.py` writes `<run_id>.catalog.toml` next to each report: what the run
  proved, as `kind = "probe"` catalog facts, for review or `model_catalog.load()` (C06d). A fact
  is false only where the adapter refused the call; a provider error that reads as unsupported is
  left in a comment, for review.
- `dailymed_label_text` (T07) reads a DailyMed label's text, whole, one section (by the number
  `dailymed_label` gives, subsections included) or the passages around a term (`find=`), window
  by window; lists come one item per line and tables one row per line. No tool returned a
  section's text before. `dailymed_label_search` takes `rxcui`, so the RxCUIs the `rxnorm_*`
  tools return find their labels, and `chembl_activity_search` takes `assay_chembl_id`, the assay
  each measurement names (T07b).
- Example 49 runs an agent on a local model that searches the web with `brave_search` or
  `tavily_search`, whichever has a key (C08).
- **Typed tool failures** (T01, D37, D42): `ToolFailure(type, message, *, retryable=False,
  details=None)` and `ToolFailureType`, public in `ai_arch_toolkit` and `ai_arch_toolkit.core`. A
  tool raises it when it cannot answer; the executor returns `ToolResult(ok=False)` with its type,
  its `retryable` and its redacted message, and meters the call as an unbilled failure.
  `tool_result()` takes `is_error=`, which `run_tools()`/`run_tools_sync()` and the ReAct flow set
  for a failed call: Anthropic receives it as the `tool_result` block's `is_error`, Gemini the
  result under the function response's `error` key.
- **A strict budget reserves an image by its model, quality and size** (G-40, D61): each image
  adapter gives the most image output tokens one image of the request costs, from its provider's
  published counts (`BaseProvider.image_token_bound`, carried to the meter as
  `OperationRequest.declared_image_tokens`). OpenAI's table covers gpt-image-1, 1.5 and
  `chatgpt-image-latest`, gpt-image-1-mini's per-image prices give its counts, and OpenAI's
  calculator covers gpt-image-2 and the 2.5 models (`low` 1024x1024 on Flare is 196 tokens, where
  16,000 were held); Gemini's counts go by image size, and its image models' thinking, billed at
  the text rate, is held up to their output limit (`image_text_token_bound`). A model billed per
  image (Grok Imagine, Muse Image) holds no image tokens. A quality or size left to the model
  holds the dearest it can pick.
- **Which models read images** (G-39): `docs/model-compatibility.md` lists, for every model of the
  probe inventory, whether it reads an image in a request, from its own page at its provider
  (read 2026-10-05), and the probe runner has a `vision` scenario (a small red square), listed
  for those models and not run yet.
- **Tool calls stream as the model writes them** (G-37, D60): `stream_events()` (and an
  iterated flow's `llm_event`s) emit `StreamEvent(kind="tool_call_delta")` events, each with a
  `ToolCallDelta`: the call's `index` among the answer's calls, its `id` and `name` (from the
  first piece, sent as soon as the name is known), and `input_json`, the next piece of its input
  as JSON text. Anthropic, the Responses API (OpenAI, Meta) and OpenAI-compatible servers send
  the pieces as the model writes them; Gemini and xAI send a call whole, so it arrives as one
  piece as soon as its chunk does. The finished `tool_call` events still follow the stream, and a
  stream left mid-call has no half-written call in its `Response`.
- **Bounded parsing of the manifests and resources the toolkit loads** (G-34, D59): agent
  manifests and the JSON, TOML and YAML resource codecs (prompt manifests and knowledge among
  their users) go through one module, `toolkit/_safe_data.py`. A YAML document's aliases may add
  at most 10,000 nodes and 1,000,000 characters, or as many as it holds if that is more, so an
  alias bomb is refused before it expands (412 bytes took 2.6 s and 170 MB, and each level
  multiplied by ten; an alias copying a long string, 139 KB took 3.3 s and 2 GB); anchors and
  merge keys keep working. An alias inside its own anchor is refused. A manifest that several
  others extend (`extends`) or include (`include`) is read and built once per load: ten agent
  manifests of 757 bytes, each extending the next four times, were read 349,525 times.
- **What each step spent** (G-32, D58): `StepTrace.metered` is what the meter measured in the
  step's own span (a `MeterSnapshot`): its LLM and tool calls, retries, fallback and nested flows,
  nothing of its siblings, and up to the cut for a step the run cut short; a flow a step runs
  itself has a span of its own, and its entry in the step's `children` carries its spend. It
  reaches the consumer in `step_end`'s `step_trace`, and `to_dict()`/`from_dict()` keep it
  (`MeterSnapshot` gained both, with amounts as exact USD text, and `Money.to_usd()`).
- **Spans are public** (G-32, D58): `open_span`, `current_meter`, `current_span_id` and
  `bind_meter` come from `ai_arch_toolkit.core`, to measure a block of code (a delegated
  subagent, a run nested in a turn) with `current_meter().for_span(span_id)`, and to carry the
  meter into a thread.
- **Weak dependencies in a DAG** (G-33, D58): `FlowStep(after_any=...)` runs once at least one of
  its dependencies succeeded, which joins the paths a `when` split, and
  `FlowStep(after_optional=...)` waits for steps that may fail or be skipped without being
  skipped itself. `StepTrace.blocked_by` names the dependencies that kept a skipped step from
  running, each `"failed"` or `"skipped"`; `FlowStep.dependencies` lists all three kinds.
- **A budget several runs share** (D57): `SharedBudget(policy, spent=...)` (on the core's
  `SharedMeter`), bound to each run with `RunConfig(shared=...)`. Runs in parallel are admitted
  and settled against it under one lock, each operation holding its worst case there, so
  together they never pass its `max_cost`; `spent` seeds it with what an app's ledger already
  holds. It shares `max_cost`, `max_llm_calls` and `max_tool_calls`.
- **Web search on the toolkit's side** (D55): `brave_search` (Brave Search API,
  `BRAVE_SEARCH_API_KEY`) and `tavily_search` (Tavily, `TAVILY_API_KEY`), for any model, local
  ones included. Without its key, each says where to get one and sends nothing.
- **Paid tools are priced** (D56): a `[tools]` section of the price table gives a tool's price
  per unit its service bills (`ToolPricing`, `pricing.register_tool`, `get_tool`, `list_tools`;
  dated like a model's). The meter holds one unit before the call and charges the units the
  service billed: a refused request costs nothing. Brave is $0.005 a search, Tavily $0.008 a
  credit. `Api(billed_as=..., bill_units=..., key_required=..., key_prefix=...)` declares this for
  any tool.
- The model routing is public (G-14): `resolve_provider_name`, the read-only tables
  `MODEL_PREFIXES` and `MODEL_IDS` (views of the ones `create_provider` routes by), and
  `is_local_url`, the loopback rule, from `ai_arch_toolkit.core`.
- `NetworkXBackend` is public (G-24): from `ai_arch_toolkit.toolkit.memory`,
  `ai_arch_toolkit.toolkit.memory.graph` and `ai_arch_toolkit.core.graph`, lazily (it needs the
  `graph` extra).
- **An iterated flow streams its LLM calls** (D54). Each `llm.complete` a step makes in a run
  being iterated (`flow.iter()`, `agent.iter()`, their sync forms) runs on the stream path, and
  its events arrive as `FlowEvent(type="llm_event")` with the `StreamEvent` (`llm_event`) and the
  call's id (`llm_call`): the ten strategies stream their model output without being rewritten,
  and nested loops stream under the step they run in. `run()` streams nothing. The mechanism is
  the core's `llm_events_to(sink)`, which streams any `complete` made where it is bound.
- `FlowEvent.step_trace`: every `step_end` and `step_skipped` carries the step's trace entry.
- **The `truststore` extra** (D51): with it installed, the tools verify TLS with the system's
  certificate store, as pip does. Without it, they keep OpenSSL's CA file, and a certificate error
  says how to switch. On uv's standalone Pythons for macOS, whose file has no root for Eurostat's
  certificate, the Eurostat tools need it.
- `SEMANTIC_SCHOLAR_API_KEY` (D52): the Semantic Scholar tools send a key found in the environment
  (`x-api-key`, 1 request per second for its holder); keyless callers share one limit, and their
  429 now says where to get a free key. `Api(key_env=..., key_header=..., key_url=...)` declares
  such a key for any tool.
- `run_command(cwd=...)` runs the command in another folder, which must exist; only the command's
  process changes folder, so the caller's runtime stays where it was.
- Prices for DeepSeek (`deepseek-flash` and its v4 aliases, `deepseek-v4-pro`; the peak-hour
  price), Mistral (`mistral-small-2603`, `codestral-2508`, with their `-latest` aliases) and
  Poolside's `poolside/laguna-s-2.1` on OpenRouter, models reached through an OpenAI-compatible
  `base_url`. The promotions in the table end on their day: `gpt-5.6-sol` after 2026-11-21 (then
  the price before the promotion, since OpenAI states none for after it), and `gemini-3.8-flash`,
  `-3.7-flash` and `-3.6-flash` after 2026-12-31 (then Google's 2027 price).
- **Prices with an end date** (D50). `ModelPricing` gains `until` (the last day, UTC) and `then`
  (the price from the day after); `pricing.get(model, on=date)` reads the price of a day, today by
  default, so the meter switches when a promotion ends. TOML entries take `until = 2026-11-21` and
  a `then` table.
- **Image generation: `LLM.generate_image()` and `generate_image_sync()`** (D46, D47). An image
  model draws from a prompt, or edits by a prompt and `images=[image(...)]`. The images come back
  in the new `Response.images` as `GeneratedImage(data, media_type, revised_prompt)`. The call runs
  on the `complete` path: middleware, retries, fallbacks, attempts and the meter.
  - The options are portable: `n`, `aspect_ratio` (`"16:9"`), `resolution` (`"512"`/`"1K"`/`"2K"`/
    `"4K"`), `quality` and `output_format`. Each model raises `RequestError` for what it does not
    take.
  - Models:
    - OpenAI's GPT Image models through the Images API (`gpt-image-2.5-sunburst`/`-flare`,
      `gpt-image-2`, `gpt-image-1.5`, `chatgpt-image-latest`, `gpt-image-1`, `gpt-image-1-mini`);
    - Gemini's image models (`gemini-3.1-flash-image`, `-flash-lite-image`,
      `gemini-3-pro-image`);
    - xAI's `grok-imagine-image-2.0`, `grok-imagine-image` and `-quality`;
    - Meta's `muse-image-1.0`, newly routed with the `muse-image-` prefix
      (`chatgpt-image-` is routed to OpenAI).
  - Prices: image models are in the default price table. `ModelPricing` gains `image_input`,
    `image_output`, their `batch_` variants, `per_image` and `batch_per_image`.
  - New types and fields:
    - `Usage` gains `image_input_tokens`, `image_output_tokens` and `image_count`, and adds up with
      `+`;
    - `ImageRequest` (also `Request.image`, for middleware) and the `ImageResolution`/`ImageFormat`
      aliases are exported;
    - a strict budget reserves each image asked for.
  - See `docs/images.md` and `examples/48_generate_image.py`.
- **OpenAI: `image_generation(model=...)`, a hosted tool for drawing inside a turn.**
  - The image comes back in `Response.images`. `Response.cost` and the meter include it, priced at
    the image model's rates from OpenAI's `tool_usage`.
  - The next turn edits it statelessly: `to_message()` sends it back as an input image.
  - Streamed, images arrive as `StreamEvent(kind="image")`, with `partial=True` for previews
    (OpenAI's partial images, Gemini's interim thought images).
- **Gemini and Meta image models answer `complete()` with their images**: Gemini's `inline_data`
  parts and Meta's `image_generation_call` items, which were dropped.
- **OpenAI: reasoning summaries, replayed reasoning, hosted web search and token counting.**
  `thinking=True` returns reasoning summaries as thinking blocks. A tool loop replays each turn's
  encrypted reasoning from `Response.to_message()` without server-side state (`store: false`),
  only to the same model family. `web_search()` without config runs as a hosted tool, and
  `LLM.count_tokens()` counts with `POST /v1/responses/input_tokens`.
- GPT-6.1 Sol (`gpt-6.1-sol`), from OpenAI's model page on 2026-10-01: prices (GPT-6 Sol's
  rates, with cached input at 5% of input; batch, long-context and fast rates included), request
  rules and a probe inventory entry; every probe passed live on 2026-10-02. Unlike GPT-6 Sol it
  takes no `"none"` (nor `"minimal"`) effort and always reasons, at `medium` by default, so it
  follows Astra's rules: no sampling parameters.
- `Range`, inclusive bounds for a numeric tool parameter: `Annotated[int, Range(1, 25)]` puts
  `minimum`/`maximum` in the schema the model reads, and the executor refuses a value outside them
  with a `validation_error` that names the range. One bound is enough, and a `Range` on a type with
  no numbers is a `ValueError` when the tool is decorated. Exported from `ai_arch_toolkit` and
  `ai_arch_toolkit.core`. The executor checks `minimum`/`maximum` at a parameter's top level
  whatever put them there, so bounds written in a `@tool(schema=...)` override, which used to reach
  only the model, are now enforced too.
- Claude Opus 5.5 (`claude-opus-5-5`), GPT-6 Sol and Luna (`gpt-6-sol`, `gpt-6-luna`), and Grok
  4.7 (`grok-4.7`), from the providers' pages on 2026-09-25: prices (with the cache, batch,
  long-context and fast rates each provider publishes), per-model request rules, and probe
  inventory entries. Sol and Luna passed every live probe on 2026-09-25; Opus 5.5 and Grok 4.7
  are not yet run live. Opus 5.5 refuses a forced `tool_choice`. Sol and Luna reason at
  `medium` unless sent `"none"`, and take sampling parameters only at `"none"`. There is no
  GPT-6 Terra: Terra is the GPT-5.6 tier `gpt-5.6-terra`.
- `UnpricedModelError`, and prices for `gpt-4o-2024-05-13` and `gpt-3.5-turbo-1106` (snapshots
  with a tariff of their own), `gpt-5.5-cyber`, and `gpt-5.1` (at `gpt-5`'s rates), from
  OpenAI's pricing page on 2026-09-18.
- Provider failures share `ProviderError` and a typed delivery disposition. New `RequestError`, `TransportError`, `ProviderTimeout`, and `ResponseError` preserve the existing builtin exception handlers.
- **Meta provider (Muse Spark).** `LLM("muse-spark-1.3")` routes to a new `MetaProvider` that
  drives the Meta Model API's Responses API through the `openai` SDK (Meta ships no SDK), with the
  key from `MODEL_API_KEY` and a new `meta` extra. Requests are stateless and replay the model's
  encrypted reasoning from `Response.to_message()`, so tool loops and agents keep their chain of
  thought across turns. Streaming, tool calls, structured output, JSON mode, reasoning summaries,
  web search, images, PDFs, and `count_tokens` are supported; pricing covers the standard and
  contributor tiers. Muse Spark always reasons: `thinking_effort` applies on its own and
  `thinking=True` requests summaries. Only `tool_choice="auto"` is available (`"none"` sends no
  tools; forced choices raise `ValueError`). `count_tokens_local` estimates Muse Spark with
  `o200k_base`. See [docs/model-compatibility.md](docs/model-compatibility.md#meta).
- The live probe inventory accepts a per-model `tool_choice` for providers that cannot force a call.
- **`Flow(timeout=...)`** bounds a whole run of a flow, in seconds. When it elapses, the steps in
  flight are cancelled, nothing else starts, and the trace ends with a `flow_timeout` step.
  `ReasoningSpec.timeout`, manifest `strategy.timeout`, and `limits.timeout_seconds` compile to it.
- **`Flow(trace_capture="keys" | "full" | "none")`** sets what each step's trace records. Also new:
  the `TraceCapture` type, `StepTrace.input_keys` / `output_keys`, `execute_step(capture=)`,
  `ReasoningSpec.trace_capture`, and manifest `strategy.trace_capture`. See
  [docs/flow-architecture.md](docs/flow-architecture.md#what-a-trace-captures).
- **Flow and agent executions.** `Flow.iter()` and `iter_flow()` return a `FlowExecution`,
  `Flow.iter_sync()` a `SyncFlowExecution`, and `Agent.iter()` an `AgentExecution`: iterate them as
  before, then read `.result`. `retry`, `timeout`, `fallback`, and `policy_decision` events stream
  while a step runs; `step_end` events carry the step's `error`; `FlowResult.meter_scope` exposes the
  run's scope; `StepTrace.children` holds the steps of flows run inside a step.
- `Agent.run()`, `run_sync()`, and `iter()` accept a per-run `config=RunConfig(...)`.
- **Tool-call arguments are validated and coerced against the tool's schema before any gate runs.**
  Numeric and boolean strings are coerced (`"3"` for an integer), `enum` and `anyOf` are checked,
  missing or unknown arguments are refused, and an approval handler sees the coerced values. See
  [docs/safety.md](docs/safety.md#argument-validation).
- `ToolGate`, `ExecutionContext`, `GateBlock`, `GateModify`, `GateDryRun`, and `GateResult` are
  exported from `ai_arch_toolkit` and `ai_arch_toolkit.core`, so custom gates no longer import
  private modules.
- GPT-6 Astra pricing (including cache, batch, long-context, and fast rates),
  Chat Completions parameter handling, and manual probe inventory. Unsupported
  reasoning efforts and tool calling fail clearly; Astra tools require Responses.
- **[docs/configuring-agents.md](docs/configuring-agents.md)** — the end-to-end agent
  configuration guide: the serializable-vs-runtime rule, code-first specs, knobs vs
  deps, per-phase configuration, prompt sourcing, budgets, manifests with
  `agent_from_manifest`, escape hatches, and a migration checklist for downstream
  projects. Linked from the README, docs nav, and `AGENTS.md`.
- **Per-phase configuration for `Agent`/`ReasoningSpec` and manifests.** Multi-phase
  strategies accept canonical per-phase overrides through the two existing buckets:
  runtime LLM/tools as deps (`planner_llm`, `executor_tools`, `reviewer_llm`, …) and
  prompts as knobs (`planner_system`, `evaluator_system`, …), validated per strategy —
  `FlowStrategy` gains `phases`, `allowed_deps`, `dep_validators`, and `validate_spec()`.
  Agent manifests gain a `strategy.phases` section: per-phase `system`/`system_file`
  prompts fold into the spec's canonical knobs via `reasoning_spec()` (verbatim text,
  governed by `allowed_roots`/override policy and re-verified against the load-time
  fingerprint — drift raises), and per-phase `model` configs are exposed via
  `ResolvedAgentManifest.phase_models()` and resolved by the application through the
  new `agent_from_manifest(…, llm_factory=…)` helper (spec and strategy validate
  before the factory runs). New CLI subcommands `ai-arch agent validate|inspect`
  (with `--allowed-root`) run registry-aware checks — strategy name, phase names,
  LLM-bindability of phases declaring models, knob values — for CI. Planner prompts (inline, knob, or
  `system_file`) may carry a `{tools}` token — the only substitution the framework
  performs — replaced at build time with the phase's resolved tool catalog. Also new:
  the `lats.exploration_weight` and `self_discovery.modules` knobs. See
  [docs/agents.md](docs/agents.md).
- Hermetic configured-agent system tests now exercise manifest inheritance and profiles,
  prompt rendering, governed tools, agent compilation, metering, and hard budgets as one
  end-to-end path. The separate `live_api` marker keeps paid provider smoke tests explicit.
- `ModelPricing` and `PricingRegistry` are public core/top-level exports, so custom pricing can be
  registered without importing a private module.
- **Public configurable-agent manifests.** `load_agent_manifest()` strictly loads
  versioned YAML/JSON/TOML agent definitions with multi-parent inheritance, embedded
  profiles, relative-path confinement, governed deterministic overrides, provenance,
  content-aware fingerprints, and direct `ReasoningSpec` / `BudgetPolicy` construction.
- **Recursive prompt subsections.** `PromptSection` accepts nested `sections=`, forming a tree: each section renders its own content and then its subsections, and every layout translates depth (Markdown deepens heading levels, XML nests elements; Text/JSON follow suit). Manifests, section spans, provenance, and the `ai-arch prompt` CLI all follow the hierarchy; see `examples/46_prompt_subsections.py`.
- **Complete prompt and resource system.** `toolkit.resources` now provides policy-controlled local/package loading, TXT/Markdown/JSON/YAML/TOML codecs, RFC 6901 and text selectors, deterministic serializers, fingerprints, and provenance. `toolkit.prompts` adds file-backed sections, typed `PromptTemplate` variables, explicit stdlib/Jinja engines, Text/Markdown/XML/JSON layouts with section spans, versioned YAML/JSON/TOML manifests with includes/extends, Knowledge sources, and `ai-arch prompt validate|inspect|render`. Nanope can append or replace its built-in prompt with a toolkit manifest.
- Prompt messages now compose resolved prompts, templates, literal text, and multimodal `Content` into deterministic system/user/assistant conversations. Resources support in-memory snapshots, resolver-scoped custom serializers, media-type policy allowlists, and direct `Prompt.from_resource()` / `PromptSection.from_resource()` conveniences. Knowledge adds deterministic lexical search and Nanope/CLI integrations.
- **Cost metering & budgets.** A neutral metering mechanism in `core` (`MeterScope` / `RunConfig`, opaque exact `Money`, `Cost`, `MeterSnapshot`, `UsageEvent` / `UsageSink`, the `AdmissionController` protocol) with an opinion layer on top in `toolkit.budget` (`BudgetPolicy`, `BudgetController`, `BudgetReport`, `budget_scope()`, `HeuristicEstimator`). Attach `budget_policy=BudgetPolicy(max_cost=…, max_llm_calls=…, max_total_tokens=…, max_wall_s=…)` to a `Flow`, at construction or per run (`flow.run(state, budget_policy=…)`), or to an `Agent` run (`agent.run(task, budget_policy=…)`); all nine agent flows and `Agent` honour it, and nested flows share one cumulative budget. Caps are enforced at the charge site (`BudgetExceeded`, surfaced as `policy_decision="budget_exceeded"`): call caps are hard and exact even under concurrency, so the call that would exceed one never runs; token and cost caps are soft by default (the call that crosses one runs, and the next is denied) and hard under `reserve="strict"` as far as its up-front hold is right. Two knobs shape a cost cap: `reserve="strict"` (reserve a worst-case hold up front) and `unpriced="fail_closed"` (default — deny once a call can't be priced, e.g. an unpriced model or a server tool). Read spend off the run's meter: `result.meter` (a `BudgetReport`), `result.total_cost`, `result.usage`, or `agent_result.report`. All metering types are re-exported at the top level; see [Tool Governance & Safety → Cumulative budgets](docs/safety.md).
- **Concurrency controls.** `inference_limit(n)` caps concurrent LLM calls globally — across every nested flow, agent, and fallback — to protect a shared resource (local GPU, rate-limited endpoint, connection pool); `Flow(max_parallelism=n)` bounds how many steps of one flow fan out at once. Both opt-in, independent, and composable; see [docs/concurrency.md](docs/concurrency.md).
- **Anthropic schema-in-prompt structured output.** `structured_output_mode="prompt"` (an `LLM(...)` constructor default or a per-call kwarg) makes the Anthropic adapter inject the JSON schema into the system prompt and parse the reply, instead of the native `output_config`. This handles large analysis/planning-style schemas that exceed Anthropic's native structured-output complexity limit (which otherwise returns a 400 "schema is too complex"). Defaults to `"native"` (unchanged behaviour); the OpenAI, Gemini, and xAI adapters accept and ignore the kwarg.
- **Agent & ReasoningSpec.** A declarative facade over the flow factories in `toolkit.agents`: `ReasoningSpec` is a frozen, declarative description of how an agent reasons (`strategy`, `system`, `max_iterations`, strategy-specific `knobs`, `policy`, `timeout`, `llm_kwargs`, `output_schema`; `from_mapping()` builds one from parsed JSON/YAML), and `Agent` binds it to an `LLM` + `ToolGroup`, compiles the `Flow` once, and exposes `run()` / `run_sync()` / `iter()`, `Agent.from_flow()`, and `as_step()`. `AgentResult` carries `text` / `response` / `flow_result` plus meter-derived `usage` / `cost` / `report`. The strategy registry (`register_strategy()` / `get_strategy()`) ships 10 built-ins — the nine flow factories plus `completion` (a single LLM call, no tool loop); `react`, `completion`, and `generate_review` support `output_schema`. See [docs/agents.md](docs/agents.md).
- `@deprecated` decorator helper for the pre-1.0 deprecation policy.
- **Tool execution governance.** One core pipeline governs every tool call: `@tool` carries `capability` / `risk_level` / `requires_approval` metadata, runtime gates (`DangerousToolGate`, `ApprovalGate` + `ApprovalHandler`, `DryRunGate`) run before execution, and results are structured `ToolResult` / `ToolError` values instead of bare strings. Side-effectful tools (shell, filesystem, Python eval, web fetch) moved behind the opt-in `toolkit.tools.dangerous` namespace, and traces/logs pass through secret redaction (`RedactionPolicy` / `Redactor`). See [Tool Governance & Safety](docs/safety.md).
- **Atomic, versioned graph persistence.** `Graph.save()` writes through a temp file + `os.replace` so a crash can't leave a half-written store, and saved payloads are versioned.
- **25+ new toolkit tools** across science, geo, health, and media APIs — arXiv, ClinicalTrials.gov, Crossref, DataCite, GDELT, Internet Archive, and more — bringing the pre-built, stdlib-only catalog to 132 tools ([docs/tools-catalog.md](docs/tools-catalog.md)).
- **Local OpenAI-compatible servers.** `LLM("gemma4:e4b", base_url="http://localhost:11434/v1")` routes arbitrary model tags to the OpenAI adapter for Ollama, LM Studio, and vLLM. A `provider=` kwarg on `LLM` / `create_provider()` forces a specific adapter regardless of the model name; an unknown model with `base_url=` set falls back to the OpenAI-compatible adapter automatically.
- **Optional API key on localhost.** When `base_url` points at a loopback host (`localhost` / `127.x` / `::1`), the API key is optional (a placeholder is used) and a cloud key from the environment is **not** forwarded to the local server. Remote endpoints (gateways, proxies) still require a key and fail fast when it is missing.
- **Real-time reasoning streaming.** Vendor reasoning deltas (`reasoning_content` / `reasoning`) from OpenAI-compatible servers surface as incremental `thinking` events in `stream_events()` and populate `Response.thinking` in `complete()`, streaming, and batch responses. The reasoning-so-far is preserved even when a stream is abandoned early or errors mid-way.
- **`StreamEvent.partial`** flag distinguishes incremental reasoning fragments (`True`, OpenAI-compatible servers) from complete thinking blocks (`False`, Anthropic). Concatenate consecutive partial `thinking` events for the full trace.
- nanope research_center, agent_swarm scaffold, and advanced configurable agent (work in progress, not part of the public toolkit API).
- `AGENTS.md` — the canonical instructions file for all coding agents, rebuilt lean (commands, layer rules, testing patterns, gotchas); `CLAUDE.md` is now an `@AGENTS.md` import stub so Claude Code reads the same file.
- `app` optional-dependency extra (`reflex>=0.7` + graph + yaml).
- `pyright>=1.1.390` and `pre-commit>=4.0` in the `dev` extra; `[tool.pyright]` configured in standard mode over `src/`.
- `.pre-commit-config.yaml` with ruff-check / ruff-format and standard hygiene hooks (trailing whitespace, EOF newline, yaml/toml validity, merge-conflict + large-file guards).
- `CONTRIBUTING.md` covering setup, conventions, and how to add a provider / tool / agent flow.
- `.env.example` documenting every provider API key; the sync-timeout configuration now validates its inputs.

- **Output and time limits for every tool.** `ToolRuntimePolicy.max_output_chars` (default
  200,000) and `timeout_s` (default 120 seconds), set per tool with
  `@tool(max_output_chars=..., timeout_s=...)` (`None` switches one off).
  `ToolGroup(max_output_chars=..., timeout_s=...)` sets a ceiling for all its tools: the stricter
  value applies. A longer result is cut and marked, in its text and in
  `ToolResult.metadata["truncated"]`; a call past its deadline returns a `timeout` failure. See
  [docs/safety.md](docs/safety.md#output-and-time-limits).
- `FlowOptions` (`ai_arch_toolkit.toolkit.agents.flows`): the four options every flow factory
  hands to its `Flow` (`timeout`, `trace_capture`, `policy`, `budget_policy`), declared once, so a
  wrapper can forward them with their types (`**options: Unpack[FlowOptions]`). See
  [docs/flow-architecture.md](docs/flow-architecture.md#flow-options).
- Agent manifests have a packaged JSON Schema,
  `ai_arch_toolkit/toolkit/agents/schemas/agent-manifest-v1.schema.json`, for editors and other
  tools. Like the prompt manifest's, it is generated from the declaration the loader enforces. See
  [docs/agents.md](docs/agents.md#file-backed-agent-manifests).

### Changed
- **Breaking:** the minimum Python is 3.13.4 (`requires-python = ">=3.13.4"`, D64).
- `ai_arch_toolkit.toolkit.tools.dangerous.__all__` also lists `FilesystemAction`,
  `FilesystemPolicy`, `FilesystemPolicyError`, `PathScopeGate` and `filesystem_tools`: code that
  takes every name there for a tool filters on `__tool_definition__`. Code that matches
  `ToolFailureType` or `GovernanceOutcome` exhaustively meets `permission_denied` (C07).
- The filesystem write tools are POSIX only: on Windows, `filesystem_tools` with `write_roots`
  raises `NotImplementedError` when built; the bound reads work there (C07, D64).
- The approval request of a `write_file`, `append_file`, `make_directory` or `move_path` call
  shows its preview instead of its arguments as JSON (`request.arguments` still holds them
  whole), and a dry run of one of them also records `audit["preview"]` (C07c).
- **Breaking:** every toolkit tool keeps the tools contract (T06 to T09, C08; D37 to D42): the
  contract's debt list is empty. A cut is a window whose footer says what was shown, the total
  when the source gives one and the exact call for the rest; a limit is a `Range` bound in the
  schema, refused outside it; a failure is typed with the source's words and the next step; zero
  results name the query; dates are ISO 8601 UTC, numbers carry no scientific notation, codes
  come with the labels the answer brings. The entries below give each family's changes.
- **Breaking:** tool names must be portable, `^[A-Za-z_][A-Za-z0-9_-]{0,63}$` (the names every
  provider and MCP accept), checked by `ToolSchema`: `@tool(name="a.b")`, `tool_schema()`, a
  `lambda` in a `ToolGroup` or in `prepare_tools`, and a tool dict with another name raise
  `ValueError` (C02, D62). Every tool the toolkit ships complies.
- **Breaking:** two different tools with one name, in a `tools=[...]` list or across the groups
  in it, raise `ValueError` in `prepare_tools` (so in `llm.complete`), `execute_tool`,
  `async_execute_tool` and `run_tools`, before anything is sent or runs (C02). The provider used
  to refuse the request with a 400, and execution silently ran the first match. A tool that
  appears twice is sent and counted once; server tools, dicts in wire form and groups never clash
  by their type's name, so a list `llm.complete` takes, `execute_tool` and `run_tools` take too.
- **Breaking:** `@tool(schema=...)` raises `TypeError` unless it maps parameter names to JSON
  Schema keywords (C02). A complete schema used to be merged in as parameters named `type` and
  `properties`; build such a tool with `tool_from_schema`.
- Argument validation (D7, extended by D62): `oneOf` and a `type` list (`["integer", "null"]`)
  coerce like `anyOf`, and a branch is read with its parent's keywords, so `{"type": "integer",
  "oneOf": [{"minimum": 1}, ...]}` and MCP's titled enums coerce `"5"` (past 1,000 alternatives a
  value passes as it came, and a refusal stays under about 1,000 characters);
  `additionalProperties: false` at the schema's root refuses an unknown
  argument even when the function takes `**kwargs`; an argument that arrives through `**kwargs`
  accepts `null` only where its schema admits it. Nested values are still left as they came.
  `prepare_tools` accepts a single plain callable, as it does in a list.
- **Literature and identifiers** (T06): `arxiv_*`, `crossref_*`, `datacite_*`, `europe_pmc_*`,
  `pubmed_*`, `semantic_scholar_*`, `ror_*` and `nvd_*`.
  - **Breaking:** `ror_search` no longer takes `max_results`: it shows ROR's whole page of 20,
    numbered, and pages by `page` (1 to 500). `max_results` cut each page on our side, so rows 6
    to 20 of every page were never shown.
  - A search numbers its results and ends with the source's total and the next call, or, past
    how deep the source pages (arXiv 30,000; Crossref, DataCite, PubMed and ROR 10,000; Semantic
    Scholar 1,000), says that the rest cannot be read here. A record tool gives the whole record,
    window by window (`offset`, `max_chars`): no abstract cut at 700, 900 or 1,200 characters,
    and every author, reference, link, MeSH heading, CPE and location listed. A search shows the
    first 8 names of a long list, how many more there are, and the call that lists them all.
  - `europe_pmc_citations` takes `page`, with the total; `europe_pmc_search`'s footer gives the
    next `cursor_mark` with an `offset` that numbers its page. Requests to arXiv go 3 s apart, as
    its manual asks.
  - `semantic_scholar_search` is a keyword search: a DOI or an arXiv ID is refused, pointing to
    `semantic_scholar_paper`; `semantic_scholar_citations` shows every citation context, whole.
  - `nvd_cve_search` refuses a publication range over NVD's 120 days before the request, saying
    how to split it; CVSS scores show their version (`CVSS 3.1: 10.0 CRITICAL`), and `nvd_cve`
    lists every score, CPE and reference.
  - `ror_search` takes any printable query ("Franklin & Marshall College"); `ror_organization`
    shows every location.
- **Health** (T07a): `clinical_trial*`, `rxnorm_*`, `dailymed_*`, `openfda_food_*`,
  `open_food_facts_*` and `foodon_*`.
  - **Breaking:** `open_food_facts_nutrition` is gone into `open_food_facts_product`, which reads
    the product whole: every allergen, trace, additive, category, label and country (once cut at
    10, 8 or 5), nutrients per 100 g in grams (energy in kcal), and the Nutri-Score and NOVA
    group with what they mean. `open_food_facts_compare` lists every allergen.
  - **Breaking:** `dailymed_label` drops `max_sections` and lists every section, numbered, with
    its LOINC code and size (50 a page, `offset=`); `dailymed_label_text` reads one.
  - `clinical_trial_study` reads the whole record (summary, eligibility criteria, every arm,
    outcome, site and reference, once cut at 900 and 1,200 characters and at 6 and 5 items) by
    section, offset or term, and the eligibility criteria keep their lines;
    `clinical_trials_search` gives the total and the next page's token with its position.
  - `rxnorm_drug_search`, `rxnorm_related` and `rxnorm_ndcs` page their whole lists with the
    total, where they kept the first 25; the searches of openFDA, Open Food Facts, DailyMed and
    FoodOn give the total and the next `skip`, `page` or `start`; FoodOn definitions are whole
    and `foodon_term` gives the synonyms.
  - **Breaking:** an openFDA `BAD_REQUEST`, a ClinicalTrials.gov 400 and an RxNav 400 are
    `validation_error`s in the source's words, where they were `upstream`.
- **Life sciences** (T07b): `uniprot_*`, `pdb_*`, `chembl_*` and `gbif_*`.
  - **Breaking:** `uniprot_search` pages by cursor, as UniProt does: `offset` alone was ignored
    and every page was the first. The footer gives `next: cursor="…", offset=N`; pass both. The
    total comes from UniProt's `x-total-results` header.
  - `uniprot_entry` reads every annotation in full (function, catalytic activity, location,
    disease, cofactors, isoforms, interactions, kinetics), window by window; the function text
    was cut at 500 characters. `uniprot_features` and `uniprot_crossrefs` page through every
    item and count them by type or database; they kept the first 25.
  - `pdb_search` lists each entry with its title, method, resolution and release date (one more
    request a page); `pdb_ligands` reads every ligand in one request (it made one per ligand);
    `pdb_entry` shows dates, the resolution in Å, entity counts and the citation's DOI and
    PubMed ID.
  - ChEMBL answers show the max phase with its label (`4 (approved)`), properties with units,
    and activities with the molecule's and the target's names; a search under 3 characters is
    refused before the request, as ChEMBL refuses it.
  - **Breaking:** `gbif_species_match` resolves scientific names only, as GBIF's match service
    does: a name it cannot resolve is `not_found`, pointing common names to
    `gbif_species_search`. `gbif_occurrence_search` stops at GBIF's 100,000-record depth with a
    note.
- **Data and news** (T08a): `eurostat_*`, `world_bank_*`, `who_*`, `gdelt_*`, `wikidata_*` and
  `hacker_news`.
  - **Breaking:** `eurostat_dataset` absorbs `eurostat_dimensions`, `eurostat_series` absorbs
    `eurostat_compare`, and `world_bank_series` absorbs `world_bank_compare` (D41; the table in
    the Upgrade notes). `eurostat_series` takes several codes of a dimension (`geo=PT+ES+FR`) in
    one request and names the series of each row; `world_bank_series` takes several countries
    (`"PRT,ESP,DEU"`) and pages.
  - `wikidata_entity` reads every statement, 40 at a time, each code with its label
    (`instance of (P31): human (Q5)`), quantities with their unit, dates to their precision,
    coordinates as latitude and longitude, ranks and every qualifier; it takes property IDs
    too. Labels are best effort: a label request that fails leaves its codes bare, and says why.
  - `wikidata_sparql` keeps every row the query asks for and reads them on by `offset`; a SELECT
    without a `LIMIT` gets `LIMIT 1000`, and the answer says when it reached it. A query the
    service cannot parse is a `validation_error`, one past its 60 s deadline an `upstream`
    failure that says how to narrow it. Comments, strings and IRIs are not read as the query's
    words: a trailing `VALUES` clause gets the `LIMIT` before it, and a query that starts with a
    comment or has a variable such as `?add` is accepted.
  - `gdelt_news_search` reads up to the 250 articles GDELT lists for a query, and
    `gdelt_timeline` every point, by `offset`; `hacker_news` reads the whole top-stories list (up
    to 500), a story that fails named in its place; `who_indicators` and `who_series` say when
    there is more, and `who_series` names each code's dimension and takes any place code its
    rows show (`SEAR`, `GLOBAL`, `WB_LMI`) as `country`.
  - Eurostat errors follow its guide: no data (error 100) is `not_found`, naming the query; a
    code the dataset does not have is a `validation_error`. World Bank errors follow its table:
    an unknown indicator or country is `not_found`, and so is error 175 (an indicator deleted or
    archived); error 105 is worth a retry; any other error says what to check.
  - `eurostat_series` gives each flag its label (`p: provisional`), upper-cases `geo` codes and
    refuses `format` and `lang` as filters; `eurostat_dataset` names the dimension a window of
    codes continues. A World Bank list's next call keeps its `max_results`.
- **Geo, weather and natural events** (T08b): `geocode`, `osm_*`, `overpass_*`, `get_weather`,
  `get_forecast`, `air_quality_*`, `eonet_*`, `earthquake_*` and the `_geo` tools.
  - **Breaking:** `get_weather` and `get_forecast` take a city or a point (`latitude`,
    `longitude`) and `units="metric"|"imperial"`, replacing `get_weather_by_coords`,
    `get_forecast_by_coords` and `weather_units`; with a city they still use Open-Meteo's first
    match, and say when other places share the name. `osm_reverse_geocode` takes the `zoom` that
    `reverse_geocode` fixed at 10 (the table in the Upgrade notes).
  - Every cut reads on: `geocode` lists up to 100 places (it showed 3), `osm_search_place` up to
    40 (it stopped at 10), each page cut from one answer, `overpass_*` and
    `air_quality_forecast` page through the whole answer,
    `eonet_event` reads an event's whole track (only its last point was shown), and
    `earthquake_search`'s total is USGS's `count` for the same filters (it was the page's).
  - **Breaking:** limits are `Range` bounds: `get_forecast` takes up to 16 days (it cut to 7),
    `air_quality_forecast` up to 92 past days, `eonet_events` up to 365 days. Coordinates are
    `Range` bounds too (latitude -90 to 90, longitude -180 to 180), as are `earthquake_search`'s
    radius and depths; `order_by`, EONET's `status` and `distance_between`'s `unit` are enums;
    `ip_lookup` requires its `ip`. `eonet_events` takes several categories or sources,
    comma-separated.
  - Times are ISO 8601 UTC (`earthquake_*` gave epoch milliseconds), coordinates are labelled,
    weather codes carry their WMO label, and units are the ones the source names.
- **Local tools** (T09a): `read_file`, `list_directory`, `search_files`, `csv_read`,
  `regex_search`, `run_command`, `python_repl`, `json_extract`, `unit_convert`, `date_add`.
  - `read_file` reads a file window by window, by characters from its start, a chunk at a time;
    following the footers rebuilds the file. `search_files` searches whole files (it read the
    first million characters of each) and shows the part of a long line around the match, with
    the line's length; `csv_read` repeats the header on every page and counts every row of the
    file; `list_directory` pages 1,000 entries by name with the total; `regex_search` pages 1,000
    matches with the total. A `csv_read` page holds at most 100,000 characters (a longer row
    comes alone) and pads a column to 40 at most; its rows are counted up to 50 million
    characters past the page, then the heading says "at least". `search_files` skips a file with
    a NUL in its first 8 KiB, reads at most 500 million characters a call (then it says where it
    stopped), and names the folders and files it cannot read; a folder `search_files` or
    `list_directory` cannot read is an `upstream` failure, not an empty answer. `regex_search`
    matches in a child Python process given 5 seconds: a pattern that backtracks polynomially
    (`a*a*b`) is refused when they run out, and the program never freezes.
  - **Breaking:** `run_command` and `python_repl` show the start of an output that does not fit,
    with its size and how to narrow the command or print less: the output is not kept, so no
    call reads on. `run_command` keeps stderr and the exit code under a long stdout, reads its
    streams as they come, so memory stays within the limit, and runs the command in a process
    group of its own, which it kills when the call returns: no process or reader is left behind
    (a process left in the background is stopped half a second after the command ends, with a
    note to end the command with `wait`). It keeps what was printed when the command times out,
    gives the command no input (`/dev/null`), and refuses to run on Windows. `python_repl` shows
    at most 20,000 characters, and a long print never pushes out the last value or the error.
  - `unit_convert` says it rounds to 6 significant digits, writes numbers without scientific
    notation, and converts with the exact unit definitions; `json_extract`'s `not_found` names
    the keys or the length where the path failed.
  - The gates and capabilities of the dangerous tools are unchanged.
- **Web, transcripts and archives** (T09b): `http_get`, `scrape_text`, `youtube_transcript*`,
  `internet_archive_*` and `open_library_*`.
  - `http_get` and `scrape_text` read a page window by window (`offset`, and `find=` for the
    passages around a term); `http_get` reads only as much of the page as the window needs, at
    most its first 10 MB, and reads further when a page's characters take more than four bytes
    (a byte-order mark, ISO-2022-JP), so every footer reads on. `scrape_text`, stopping at its
    2 MB of HTML, names the `http_get` offset that reads the rest. Every call, a continuation
    included, still asks for approval.
  - `youtube_transcript` reads a transcript window by window, and its footer names the next
    offset instead of "Increase max_chars", which it said at the ceiling too;
    `youtube_transcript_search` pages its matches with their total, and
    `youtube_transcript_languages` lists every translation target.
  - The Internet Archive and Open Library searches number their results and give the total and
    the next page; their records come whole (every file, subject, link and ISBN, the whole
    description), window by window. Open Library names authors (`J. R. R. Tolkien (OL26320A)`),
    not `/authors/OL…A` keys (when the author search fails, the record still reads, its authors
    by key and the reason given), and its requests go one a second, as it asks of callers that
    send no email.
- **Web search** (C08, D65): `brave_search` takes `offset` (0 to 9, in pages of `max_results`)
  and ends a page with the call for the next while Brave has more; `tavily_search` says when its
  one page is full. `max_results` is a `Range` (Brave 1 to 20, Tavily 0 to 20; 0 asks for the
  answer alone, with `include_answer=True`). Each title and snippet comes whole (it was cut at
  500 characters), and a result whose URL is not `http(s)` is left out, with a count. Brave's
  422 and Tavily's 400 and 422 are `validation_error`s in the service's words, and Brave's names
  the fields its `error.meta` lists; a `country` Brave does not list (`UK`: Brave's code is `GB`)
  is refused before any request. After a 429 without `Retry-After`, Brave's host rests 1 s and
  Tavily's 60 s (D53). Both stay
  network tools of low risk without approval; `docs/tools.md` compares them with the hosted
  `web_search()`.
- **Breaking:** the wiki family (T05, D39 to D41). `wiki_search`, `wiki_outline`, `wiki_read`
  and `wiktionary_entry` replace `wikipedia_search`, `wikipedia_article`, `wikipedia_related`,
  `mediawiki_search`, `mediawiki_page`, `mediawiki_sections`, `wiktionary_entry` and
  `define_word` (the table in the Upgrade notes). Any Wikimedia wiki is one argument away
  (`wiki="en.wikibooks.org"`). A page is read from the HTML the wiki renders, not cleaned
  wikitext: templates expanded, tables one row per line with every cell (a cell that spans rows
  repeats), no navigation boxes, edit links or footnote markers. `wiki_read` reads a page whole,
  one section (by the number `wiki_outline` gives, with each section's size) or the passages
  around a term (`find=`), and every long answer ends with the call that reads on: the rationale
  of the 1960 Nobel Prize in Physics, 13,888 characters into its list, is one `find="1960"` call
  away, where `mediawiki_page` stopped at 4,000. A missing page is `not_found` with the search
  to run, a title the wiki refuses a `validation_error`, zero results say so with the query (and
  the wiki's suggestion); the limits are `Range` bounds. `define_word` goes: its source answered
  522, and Wiktionary covers definitions.
- **Breaking:** every toolkit tool fails by raising `ToolFailure`, never by returning an error
  string (T01): an unknown page, record or identifier is `not_found`, a bad argument
  `validation_error`, a source that fails or explains an error `upstream`, a 429 `rate_limited`.
  The HTTP door's `HttpError` is a `ToolFailure` (retryable for a 429, a 5xx, a timeout or a
  network error). Zero results is still a successful answer that says so.
- **A network tool's failure says what its source said** (T02, D38). A 404 is `not_found` only
  where the tool asks for one resource (a page, an entry, a DOI, a taxon); anywhere else it reads
  "endpoint not found (HTTP 404); the API may have changed", where twelve sources used to answer
  "no matching records found." for any 404, a moved endpoint included. Where the source says what
  happened, the failure has that type: a missing MediaWiki page or an inactive UniProt entry is
  `not_found`, an invalid title or a malformed query `validation_error`, MediaWiki's `ratelimited`
  and `maxlag` and Open Food Facts' 503 `rate_limited`. An error status carries the source's own
  text (a JSON `message`, `error` or `detail`, a header such as NVD's `message`, or the start of a
  text body) in place of the status's reason, and a missing mandatory key is a
  `validation_error`. A source that says in its answer that it is rate limited rests its host as a
  429 does, and a call in that rest reads "X asked to slow down: try again in N s." (no "(HTTP
  429)", since no request went out). Internally, the HTTP door's `Api` takes one `error_reader=`
  in place of `body_error=` and `status_messages=`, and each call declares `missing=` or
  `empty_on_404=`.
- **An image model without published counts holds 24,000 image output tokens per image** in a
  strict budget and as a failure's worst case (D61), up from 16,000, which did not cover the
  dearest published image (23,719 tokens, gpt-image-2 at `high` and 2880x2880). So does a
  gpt-image-2 or 2.5 call that leaves both the quality and the size to the model (23,719 tokens):
  pass a quality to hold less.
- **No manifest or resource nests deeper than 100 levels** (D59) of mappings and lists, in
  JSON, TOML or YAML, nor does an agent override with its dotted path: a deeper one raises
  `AgentManifestError`, `AgentOverrideError` or `ResourceDecodeError` (and `PromptLoadError` for
  a prompt manifest), not a bare `RecursionError`.
- **A step runs in a meter span of its own** whenever a meter is bound (D58), not only under a
  `Policy.max_cost`, and so does a nested flow run. Span ids are paths from the run's root
  (`run/3/7`); an operation that starts after its span closed (a tool's thread a step left
  running) counts in the nearest span still open around it, instead of raising `ValueError`.
- **A dependency is named once** (D58): a `FlowStep` that names a step twice, in one field or
  across `after`, `after_any` and `after_optional`, raises `ValueError`. The skip reasons of a
  DAG name every dependency that did not succeed (`"dependencies 'a' failed, 'b' was skipped"`);
  "all dependencies skipped" is gone.
- **Every step that starts ends** (G-22). A step the flow's timeout cancels, a step whose call a
  budget denies, and a step still running when a bug makes the engine raise now report a
  `step_end`, before the run's `timeout`, `policy_decision` or exception, with a trace entry that
  gives the reason (`policy_decisions` `timeout`, `budget_exceeded` or `halt`) and the time it
  ran. The run's trace records them before its own `flow_timeout` or `budget_exceeded` entry,
  where they were missing, so `AgentResult.errors` lists them too.
- `iter()` delivers `llm_event`s, a new `FlowEvent.type`; inside an iterated run a call that fails
  after its first streamed event is not retried, as with `stream_events()`, and an
  OpenAI-compatible server that reports no usage in a stream leaves the call's cost unknown.
- `aclose()` right after `flow_start` now runs the run's cleanup (the meter scope's close).
- **A host that answered 429 rests** (D53): for its `Retry-After`, or the time the tool module
  declares (`Api(cooldown_s=...)`; GDELT 60 s), a call fails at once, without a request, and says
  when to try again, instead of prolonging the limit. An API's `min_interval_s` counts from the end
  of each request, so a slow answer no longer lets the next request out early.
- The tools' `User-Agent` names the package's real version and its repository
  (`ai-arch-toolkit/<version> (+https://github.com/rgesteves5/ai-arch-toolkit)`).
- **A `ToolGroup` holds one tool per name.** The model calls a tool by its name, so another tool
  under a name the group holds raises `ValueError` instead of replacing the first with a warning;
  adding a tool the group holds changes nothing.
- `DangerousToolGate`'s block is a sentence for the person the model repeats it to ("The tool
  'run_command' did not run: it is marked dangerous, and this run does not allow dangerous
  tools."), no longer a command-line flag.
- **Every failure that may have been billed has a ceiling, with or without a budget** (D49). A
  measure-only run used to leave a failed indeterminate call unknown, with no bound; it is now
  uncertain, at most the worst case of the request's facts at the run's prices, so a step's
  `Policy(max_cost=...)` passes after a failed attempt that its retry recovered. Only a
  provider-hosted tool's cost stays unbounded. The worst case moved to core
  (`core/_metering/_worst_case.py`): the `HeuristicEstimator` delegates to it, a soft budget no
  longer bounds failures with its own estimator, and the internal `FailureBoundController`
  protocol is gone.
- `StreamEvent.kind` can be `"image"`, with the new `StreamEvent.image`; a `match` over the kinds
  sees a new case.
- An OpenAI GPT Image model refuses `complete()`, and a chat model refuses `generate_image()`,
  with `RequestError`; xAI's `grok-imagine-image` models refuse `complete()` too.
- **Breaking: OpenAI's own host goes through the Responses API.** Without `base_url`, or with a
  `base_url` on `api.openai.com`, the OpenAI adapter drives the Responses API instead of Chat
  Completions; any other host is an OpenAI-compatible server and keeps Chat Completions, with no
  OpenAI model rule, and gets the same request as before. The public API is unchanged: the host
  decides. On OpenAI's host:
  - `stop`, `seed`, `frequency_penalty`, `presence_penalty` and a raw `response_format` raise
    `RequestError` before sending (the Responses API has no place for them; live, the two
    penalties got a 500 after about 90 s). Structured output is `output_schema=` or `json_mode=`.
  - Tool calls work at any reasoning effort: `thinking=True` with tools no longer raises from
    GPT-5.4 on, GPT-6 Astra and GPT-6.1 Sol call tools, and the GPT-6 models take `max` again.
  - GPT-6 Sol and Luna tool calls without `thinking=True` are no longer forced to the `none`
    effort: they run at the model's default (`medium`), like a request without tools, so their
    sampling parameters are dropped.
  - `Response.logprobs` holds the Responses API's token logprobs of the output text (a tuple of
    the SDK's `Logprob`), no longer Chat Completions' `ChoiceLogprobs`.
  - Function tools are sent with `strict: false`, so their optional parameters stay optional (left
    out, the Responses API rewrites the schema into strict mode).
  - Batches are submitted to `/v1/responses`. Results are read by each batch's endpoint, so a
    batch submitted to Chat Completions before still reads. A batch result's `raw` is the SDK's
    response object, no longer the parsed JSON (OpenAI-compatible servers too).
- A tool result longer than `max_output_chars` is cut on a line break when one lies in the second
  half of the kept text, and ends with `[chars 0-200000 of 5000000 | cut at the output limit; ask
  for less]` instead of `[Output truncated: kept 200000 of 5000000 characters.]`;
  `metadata["truncated"]["kept"]` is the length actually kept.
- **Breaking: prices match a model id exactly.** A model id is priced by its own entry or as a
  dated snapshot of one (`claude-haiku-4-5-20251001`, `gpt-4o-2024-08-06`, `grok-4-0709`,
  `-latest`). A variant no longer inherits the entry its name starts with: `o3-pro`,
  `gpt-4o-audio-preview`, `gpt-5-codex`, and `gemini-2.5-flash-image` used to be priced as `o3`,
  `gpt-4o`, `gpt-5`, and `gemini-2.5-flash`. `PricingRegistry.register()` is exact by default and
  its first parameter is now `model` (was `model_prefix`); `register(model, price, match="prefix")`,
  or `match = "prefix"` in a TOML entry, keeps prefix matching as an explicit choice for a family
  of local models. A TOML entry takes `aliases = [...]` for other ids billed at its rates, and the
  shipped table lists the documented ids that prefix matching used to cover
  (`grok-4-1-fast-reasoning`, `gemini-3.1-pro-preview`, `gpt-5.1`, …).
- **Breaking: a model without a price does not run under a meter.** Inside a `MeterScope`, with or
  without a budget, a call to a model the scope's pricer cannot price raises `UnpricedModelError`
  (a `RequestError`, so neither retried nor passed to a fallback) before anything is sent; the
  message names the `pricing.register(...)` call that fixes it. A local model registers an
  explicit zero: `pricing.register("llama3.2", ModelPricing())`. Without a scope nothing changes:
  the call runs and `Response.cost` is `None`. `BudgetPolicy(unpriced=...)` now governs only the
  other unknown costs (server tools, a pricer that fails when a call settles).
- **Every provider failure says whether the request was sent.** An adapter now knows when the
  SDK handed a request to its transport. A connection that is refused or times out while
  connecting, or a request the SDK itself refuses to build (a value that is not JSON, for
  example), is a failure that was never sent (`delivery="not_sent"`: no cost, and the next call is
  admitted); an HTTP 200 whose body cannot be read is a `ResponseError`, and so is an error event
  inside a stream; a transport failure while reading a stream is a `TransportError` or
  `ProviderTimeout`. None of these leave as the SDK's or the HTTP library's own exception any
  more.
- **xAI: requests follow each Grok model's documented rules, and gRPC errors keep their meaning.**
  `thinking_effort` is sent as `reasoning_effort` where the model documents it (`low` to `xhigh`
  on `grok-4.6`, `grok-4.5`, and any newer model; also `none` on `grok-4.3` and the retired ids it
  serves), and applies without `thinking=True`; it used to be dropped with a warning. A
  `thinking_effort` on a model that reasons without one (`grok-4.20-reasoning`, `grok-build-0.1`)
  raises `RequestError`, and so does `thinking=True` on a model that does not reason
  (`grok-4.20-non-reasoning`). The reasoning models refuse `stop`, `presence_penalty`, and
  `frequency_penalty` before sending, as xAI's API does. `tool_choice="<tool name>"` works (it
  failed inside the SDK); a server tool, or any tool on `grok-4.20-multi-agent`, raises
  `RequestError`, and so does a request the SDK cannot build, before anything is sent. gRPC
  `UNAVAILABLE` is a `TransportError` and `DEADLINE_EXCEEDED` a `ProviderTimeout` (they were
  `APIError` 503 and 504); every other code maps to its `google.rpc` HTTP status
  (`FAILED_PRECONDITION` is 400, `UNIMPLEMENTED` 501), and only `RESOURCE_EXHAUSTED` is unbilled.
- **Gemini: a turn's tool results go back together, and thinking follows each model's rules.**
  The results of a turn's tool calls are sent in one `user` content, as Gemini documents for
  parallel calls, each with its call's `id` when Gemini gave the call one (Gemini 3 maps results
  by id); they used to be split into one content each, without ids. `thinking_effort` is checked
  against the model's documented levels (`minimal` only where the model has it; `xhigh` or `max`
  now raise `RequestError` instead of reaching the API) and applies without `thinking=True`; on
  Gemini 2.5 it becomes a thinking budget, and a `thinking_budget` outside the model's range raises
  `RequestError`. A server tool with a config, or of a type the adapter does not send, raises
  `RequestError` instead of being dropped, and so does a message role Gemini does not have. The
  SDK always sends through `httpx` now: with `aiohttp` installed (the `xai` extra installs it), it
  used to re-send a request on its own after a connection error, unmetered. A failed request with
  HTTP 400 or 500 is unbilled, as Gemini's billing page says, and so is a 429; an error inside a
  stream is indeterminate.
- **Meta: failures keep their code's documented meaning and their usage.** A failure Meta reports
  inside a response or a stream (`response.failed`, an `error` event) gets the HTTP status Meta's
  error table gives its code; a code outside the table, or none (a 400 and a 500 share it), raises
  `ResponseError` instead of an invented 400 or 500. A failed response's usage is kept: the error
  carries it (`ProviderError.usage`), `Response.attempts` records it, and a meter settles the
  failure with its cost instead of an unknown one. `thinking_effort` is checked per model (`max`
  only on standard-tier `muse-spark-1.3`); `logprobs=True`, a message role the Responses API does
  not have, `code_execution`, and a server tool's config raise `RequestError` (the last two were
  dropped with a warning). `MetaProvider(timeout=...)` no longer needs `httpx`, which the `meta`
  extra does not install. A connection refused before sending is `delivery="not_sent"`.
- `MeterOperation.fail()` takes the `usage` and `cost` of a failure the provider reported usage
  for.
- **Anthropic: thinking, effort and tools follow each Claude model's documented rules.**
  `thinking=True` asks the models that think adaptively (the Claude 5 family, Opus 4.8, 4.7 and
  4.6, Sonnet 4.6) for `{type: "adaptive", display: "summarized"}`, so the summary shows on the
  models that hide it by default; `claude-sonnet-4-6` and `claude-opus-4-6` used to get a deprecated
  `budget_tokens`. `thinking_effort` goes in `output_config.effort` (merged with the structured
  output format) and applies without `thinking=True`; an effort the model does not take (`xhigh`
  on the 4.6 models, for example) raises `RequestError`. On the models that only take a budget
  (Haiku 4.5, Sonnet 4.5, Opus 4.5 and older), an effort or a budget turns thinking on, and the
  budget is added to `max_tokens` instead of being taken from it. The results of a turn's tool
  calls go back in one `user` message, and an assistant turn is replayed as Claude sent it,
  thinking signatures and server tool results included, while the message still matches it.
  Forced `tool_choice` on `claude-fable-5-1` and `claude-mythos-5-1`, a `top_p` or `top_k` on a
  model without sampling parameters, a message role the Messages API does not have, and a request
  without `max_tokens` raise `RequestError` before sending. Server tools carry their `name` (the
  API refused them without it), and `code_execution` moves from the legacy
  `code_execution_20250522` to `code_execution_20250825`; a server tool with a config, which used
  to be dropped, raises `RequestError`. A
  failed request is unbilled, as Anthropic documents; an error event inside a stream gets the
  status of its error type (it was an `APIError` with status 200), and a stream whose tool input is
  not valid JSON raises `ResponseError` (the input used to reach the tool as `{"_raw": ...}`).
  Batch bodies are built as a call's.
- **OpenAI: the output limit follows the host, and model rules follow the model's family.** On
  `api.openai.com` every model receives `max_completion_tokens` (`max_tokens` is deprecated and
  refused by o-series models); an OpenAI-compatible server (`base_url=`) receives `max_tokens`
  and no OpenAI model rule. `thinking=True` on a model that does not reason (`gpt-4o`,
  `gpt-4o-mini`, `gpt-4.1*`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`) raises `RequestError`
  before sending; so do a `thinking_effort` outside the SDK's values and a server tool (Chat
  Completions takes only function tools; server tools are planned separately). A model the
  adapter does not know gets the current generation's rules. `top_logprobs` is forwarded, a
  `developer` message is sent as one, and another unknown role raises `RequestError`. The batch
  JSONL body is built by the same code as a call.
- **A stream ends with the same `Response` as `complete()`.** The provider adapters now build
  one response from the SDK's final object on both paths (the SDKs accumulate Anthropic, OpenAI,
  xAI, and Meta streams; Gemini chunks are joined). A finished `stream()` or `stream_events()`
  therefore carries `parsed`, `citations`, `response_id`, and `logprobs`, and its text is the
  provider's, as `complete()` returns it; the Gemini stream's `raw` keeps every part of every
  chunk, so a replayed history keeps all its function calls. Tool-call events in
  `stream_events()` come after the text, from the finished response and in its order (Meta used
  to emit them in arrival order). A stream the caller abandons reports the text it consumed and
  the thinking and tool calls it saw, with unknown usage.
- **A response without usage has no cost.** When the provider reports no usage (an
  OpenAI-compatible server that sends no usage chunk, or an xAI stream whose chunks carry none),
  `Response.cost` is `None` and a meter records the call's cost as unknown instead of zero.
- **A request an adapter refuses is never metered.** Building the provider request happens
  before admission: an adapter's refusal (`RequestError`) opens no operation and counts no call. A stream raises it from `llm.stream(...)`
  itself, or on its first iteration when middleware rewrote the request (its reservation is
  released).
- Complete and both stream APIs share one physical-attempt pipeline. Stream lifecycle handles are explicit; public LLM call signatures are unchanged.
- The default `fallback_on` is `(ProviderError,)` instead of `(APIError, ConnectionError, TimeoutError, OSError)`. The built-in adapters raise `TransportError` or `ProviderTimeout` for network failures, so those still fall back; a raw `OSError` (a missing local file, for example) no longer does. Custom providers should raise the normalized errors.
- **Breaking:** `MeterOperation.fail` requires a delivery disposition. `unknown_cost_count` counts only unbounded costs; bounded uncertainty consumes caps separately from known spend. `BudgetReport.cost_at_most` reports the combined bound.
- MediaWiki tools accept only HTTPS Wikimedia domains and subdomains, without credentials or ports.
- **Breaking:** `csv_read` moves to `toolkit.tools.dangerous` and requires approval in governed execution.
- **Breaking: provider SDK majors and dependency floors.** The extras now require the current SDK
  majors, each capped at the next one: `anthropic>=1.0,<2`, `openai>=3.0,<4` (the `openai` and
  `meta` extras), `google-genai>=2.0,<3`, `xai-sdk>=1.18,<2` (was 1.7; 1.18 is the first release
  that takes the `xhigh` effort). `anthropic` 1.x and `openai` 3.x moved
  their HTTP transport to `httpx2`. `anthropic` 1.x also removed `temperature`, `top_p` and `top_k`
  from its signatures; the adapter now sends them in the request body, so they keep working on the
  Claude models that accept them (4.6 and earlier) and are still dropped where the API rejects
  them. Other floors rose to versions that install on Python 3.13:
  `pyyaml>=6.0.2` and `tiktoken>=0.11`. The old floors (`anthropic>=0.40`, `openai>=1.50`,
  `pyyaml>=6.0`) were never tested, and `pyyaml` 6.0 does not build on Python 3.13. A new CI job
  installs the lowest version of every direct dependency and runs the suite, so the declared
  floors are versions the tests have run against. Projects pinned to `anthropic` 0.x or `openai`
  1.x/2.x must upgrade those SDKs to take this release.
- **Breaking: keys from the environment only go to the provider's own API.** `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, and `MODEL_API_KEY` are sent to `api.openai.com`, `api.anthropic.com`, and
  `api.meta.ai`. A remote `base_url` on any other host — a gateway, a proxy, another vendor's
  OpenAI-compatible server — now needs `api_key=` and raises `ValueError` without it; it used to
  receive the environment key (the OpenAI key went to any OpenAI-compatible server).
- **`RetryConfig` retries network failures.** A request that gets no HTTP response (refused or
  dropped connection, timeout) is retried and falls back: the OpenAI, Meta, Anthropic, and Gemini
  adapters raise it as `ConnectionError` or `TimeoutError` instead of the SDK's own exception,
  which `LLM` neither retried nor treated as a provider error.
- The `dev` extra includes `pytest-timeout`, so `@pytest.mark.timeout` limits are enforced.
- **Breaking: step traces record key names, not values, by default.** `StepTrace.input_state` is
  empty and `output_result` drops the artifacts; `input_keys` and `output_keys` list what the step
  read and returned. A long agent loop's trace no longer grows with the square of its steps.
  `Flow(trace_capture="full")` records deep copies of the values instead.
- **Breaking: every tool in `toolkit.tools.dangerous` requires approval.** `read_file`,
  `list_directory`, and `search_files` declare `capability="filesystem"`; `http_get` and
  `scrape_text` declare `capability="network"`; all are `risk_level="high"`. Without an
  `approval_handler`, a call returns `approval_denied`.
- **Breaking: system prompts are merged, never replaced.** `system=` comes first, then the
  `system()` messages in order, separated by a blank line (Anthropic, Gemini, and xAI, batch
  included). The OpenAI adapter sends `system=` as a leading message and keeps each `system()`
  message at its position. `system=""` no longer hides `system()` messages.
- **Breaking: an argument that cannot be coerced to its type is a `validation_error` before the
  tool runs.** It used to reach the function (`"x"` for an `int`); `"3"` is now coerced to `3`.
- **Breaking: `run_tools()` / `run_tools_sync()` refuse `approval_handler=` together with a
  `ToolGroup`** (`ValueError`); set it on the group. A plain list of callables still takes it.
- **Tool parameters typed `Any` or `object` accept any JSON value.** Their schema is now empty
  instead of `{"type": "string"}`, so the model can send objects, lists, and numbers. Unannotated
  parameters and types the generator does not know are still described as strings.
- `Flow.iter()`, `iter_flow()`, and `Agent.iter()` return execution objects that still work with
  `async for`; `Flow.iter_sync()` returns a `SyncFlowExecution`. The run's `MeterScope` is no longer
  stored in `State.world["_meter_scope"]`.
- `Flow.as_step()` no longer copies the flow's policy onto the wrapping step; the nested run applies
  its own policy and timeout.
- A stream's metering operation is reserved when the stream is created and starts with its first
  attempt. A stream that is never iterated, or that middleware rejects, releases the reservation.
- `configure_sync_timeouts(stream_join_timeout=)` (`AI_ARCH_STREAM_JOIN_TIMEOUT`) also bounds the
  wait for the worker thread of an abandoned sync stream or a timed-out sync call, with a warning if
  it is still alive. Closing a sync stream early can block for up to that long (5 s by default).
- **Bundled model catalog and pricing refreshed from all four providers' official catalogs
  (2026-08-29).** OpenAI gains current GPT-5.6 prices and the base/Cyber/Daybreak/
  `chat-latest` aliases; Anthropic gains permanent Sonnet 5 pricing and published Opus 4.8
  fast rates; Gemini gains 3.7 Flash and current 3.6 promotional pricing; xAI gains current
  Grok aliases and correct Build/Batch prices. Retired Claude Opus 4.1 and Gemini 2.0 entries
  were removed. The pricing engine now combines batch/fast service modes with long-context
  and cache tariffs, including xAI's inclusive 200k boundary, instead of applying standard
  cache prices or ignoring the long-context tier in those combinations.
- Planner tool awareness is now an explicit `{tools}` token instead of a silent append:
  a prompt containing the token gets the phase's rendered tool catalog substituted at
  build time (`(none)` when empty), and a prompt without it is never modified. `rewoo`
  previously appended the catalog to custom `planner_system` prompts unconditionally;
  declare the token where you want the list — the built-in default planner prompts of
  `plan_execute`, `rewoo`, and `llm_compiler` carry it.
- Built-in agent strategies now validate `deps` the same way they validate knobs:
  unknown keys and wrongly-typed values raise `ValueError` at build time, so a typo
  like `deps={"evalutor": …}` can no longer be silently ignored. Custom strategies
  registered without `allowed_deps` keep the previous accept-anything behavior.
  `generate_review` accepts canonical `generator_llm`/`generator_tools` and
  `reviewer_llm`/`reviewer_tools` dep keys, with `review_llm`/`review_tools` kept as
  legacy aliases (passing both is an error).
- Built-in agent strategies now reject unknown strategy knobs and invalid knob values at
  compile time, before an agent can spend tokens.
- Reflexion and LATS evaluators are runtime dependencies only (`deps["evaluator"]` and
  `deps["evaluator_fn"]`); serializable strategy knobs no longer accept executable callables.
- `ReasoningSpec.output_schema` now accepts supported model classes in addition to `OutputSchema` instances and schema mappings, matching `LLM.complete()`.
- **Knowledge loading now delegates to Resources.** `KnowledgeRegistry.load()` and `.from_directory()` retain parsed data and source fingerprints; the original `load_text()` / `load_json()` / etc. signatures remain compatibility wrappers. Duplicate `register()` keys now require explicit `overwrite=True`.
- **The meter is the single source of truth for spend.** Agent flows no longer thread `cost` / `usage` into their step results by hand — `FlowResult.total_cost` / `.usage` / `.meter` and `AgentResult.cost` / `.usage` / `.report` all derive from the run's meter, so nested, parallel, and streaming spend can no longer drift from a parallel hand-rolled tally. A per-step `Policy.max_cost` is now enforced against that step's **metered span cost** (previously the step's manually declared `Result.cost`, which flows no longer set). `Trace.total_cost` / `total_usage` remain a raw view of any cost a custom step annotates via `Result(cost=…)`.
- **String fallback routing.** A string `fallback=` is now routed by its own model name: a recognizable model (e.g. `claude-…`) fails over to its own provider and connection, so a local-primary → cloud-fallback chain works; a bare local tag inherits the primary's `base_url` / `provider`, assuming it lives on the same server. Pass `LLM` instances as fallbacks for full per-fallback control.
- README rewritten with badges, copy-paste snippets (completion, streaming, tools, ReAct), a provider × feature matrix, and an agent-architecture table.
- CI hardened: `lint` (ruff check + format), `typecheck` (pyright — now blocking, driven to 0 errors), and `test` across ubuntu + macos × Python 3.13/3.14 with coverage; `uv lock --check` enforces lockfile consistency; an examples smoke test catches public-API drift. CI never calls paid provider APIs: the test job deselects `live_api` tests, and there is no live-API workflow; run `pytest -m live_api` locally.
- `[tool.ruff]` and `[tool.pyright]` both exclude `src/ai_arch_toolkit/nanope` (sub-projects have their own idioms).
- Pricing registry refreshed (2026-05-19): Claude 4.7 added; Claude 4.6/4.7 now ship with 1M context at standard rates (long-context tier removed).
- Examples 31–33 updated to the variadic `Flow(*steps)` API and async `flow.run(state)`.
- Docs restructured into per-subsystem pages with coverage gaps closed; added the code-style guide, the Agent & ReasoningSpec page (now surfaced as the recommended entry point across README and docs), and the prompt-system suite (templates, layouts, manifests, messages, migration, extensibility).
- Docs and example index aligned with the Flow-based architecture; legacy "pipelines" and "8 agent architectures" wording removed.
- Memory `Node` reconciled with `core.graph.Node[T]` (zero pyright ignores).
- Public API surface tightened: `__init__` re-exports audited so internals stop leaking.
- Sync timeouts in `core/_sync.py` no longer expose the dead `SYNC_TIMEOUT` / `STREAM_JOIN_TIMEOUT` aliases; use `configure_sync_timeouts()` instead.
- `uv lock --upgrade` brought every transitive dependency to its latest compatible version (pydantic 2.13, urllib3 2.7, requests 2.34, websockets 16, xai-sdk 1.12, ruff 0.15.13, …); resolved the four Dependabot alerts.

- **Breaking:** tools time out after 120 seconds and their results are cut at 200,000 characters
  by default. Declare `timeout_s=None` or `max_output_chars=None` on a tool that needs more.
- **Breaking:** on the sync path (`ToolGroup.execute`, `execute_tool`, `run_tools_sync`) a
  synchronous tool runs in a worker thread of its own, as on the async path, so its timeout holds.
  A tool that relies on the caller's thread (thread-local state, a SQLite connection opened there)
  must open those resources itself.
- **Breaking:** `ip_lookup` queries ipwho.is over HTTPS instead of ip-api.com over HTTP, whose free
  endpoint has no HTTPS and forbids commercial use. The output lines are the same.
- The network tools go only over HTTPS to their module's hosts, follow a redirect only on the same
  host (never down to HTTP), read at most 10 MB within their deadline, send one User-Agent
  (`ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)`), and report a 429 as "rate limited
  by <API> (HTTP 429). Try again later.". Where a tool built its query by hand, spaces are now sent
  as `+`.
- `http_get` and `scrape_text` refuse URLs with credentials, stop at a redirect to another host
  (the message names the target, so the model can ask for it under a new approval), and clamp
  `max_chars` to 1–100,000; `http_get` reads no more of the body than it can return.
- `regex_search` refuses back-references, groups that repeat while holding a quantifier or an
  alternation, patterns over 500 characters and texts over 20,000, and lists at most 1,000
  matches. `math_eval` refuses expressions over 1,000 characters, and results too large to print
  (`factorial(3000)`, `7**9000`) before computing them.
- The filesystem tools read at most 100,000 characters (`read_file`) or 1,000,000 (per file in
  `search_files`, and `csv_read`), and clamp their size arguments; `search_files` cuts matching
  lines at 300 characters and `list_directory` lists at most 1,000 entries. `run_command` clamps
  `timeout` to 1–600 seconds and `max_output` to 1–100,000.
- Every toolkit tool declares its `capability` (`network` or `compute`; the dangerous tools keep
  theirs).
- `reverse_geocode` shares Nominatim's one-request-per-second clock with the `osm_*` tools; it had
  none.

- **Breaking:** the steps of a parallel DAG wave read the snapshot taken when the wave starts,
  instead of a deep copy of the state each. Siblings still never see each other's artifacts. A step
  that mutates a value it read in place (instead of returning an artifact) now changes the state in
  every mode; in a parallel wave the change used to be lost. Long DAG runs no longer deep-copy the
  state for every parallel step.
- In a parallel wave, the trace records each step when it finishes, in the order of the
  `step_end` events, instead of in declaration order. The wave's artifacts are merged, in
  declaration order, before the last step's `step_end`.

- **Breaking:** `ReasoningSpec.knobs` and `llm_kwargs` are copied and frozen at construction
  (read-only mappings). Writing to them raises `TypeError`, a change to the dict passed in no longer
  reaches the spec, and a spec can no longer be deep-copied or pickled (use
  `dataclasses.replace`).
- `StateSnapshot` is documented as what it is: a read-only view whose layers are copied and whose
  values are shared with the state. A step returns changes as artifacts; one that mutates a value
  it read changes the state for every step after it. See
  [docs/flow-architecture.md](docs/flow-architecture.md#statesnapshot).
- The nine flow factories take `timeout`, `trace_capture`, `policy`, and `budget_policy` as
  `**options: Unpack[FlowOptions]`. Calls are unchanged, and type checkers still check each name
  and type; `inspect.signature` shows `**options` in place of the four parameters.
- **Breaking:** an agent's answer (`AgentResult.text`, `extract_text`) is the text its flow leaves
  under `"answer"`; a flow that leaves none answers with its last step's value, as a nested flow
  does. The answer no longer falls back to `"response"`, `"last_response"` or `"last_answer"`, and
  `AgentResult.response` comes from the same place: `"response"` next to an answer, otherwise the
  last step's value when it is a `Response` (was `None`). Every built-in strategy leaves both keys,
  also when it runs out of turns: ReAct writes `"answer"` each turn, Reflexion writes both after
  every evaluation, and Generate-Review writes `"response"` when it rejects a draft too. An empty
  answer is now the answer: before, a Reflexion that passed with an empty answer answered with its
  score, and a ReAct run that ended on a tool turn answered with the tool results. See
  [docs/agents.md](docs/agents.md#escape-hatch-agentfrom_flow).
- **Breaking:** agent and prompt manifests are checked against one declared shape each, and the
  packaged prompt JSON Schema is generated from it. It is now as strict as the loader: it used to
  leave the fields of `source`, `template`, `layout`, and the selectors open. Shape errors name
  the field's path, and an unknown field gets a suggestion (`strategy.max_iterations must be a
  positive integer`, `unknown fields: 'stratgey' (did you mean 'strategy'?)`), so their wording
  changed. A few values that used to pass are refused:
  - a `strategy.system` that is not text (it became its `str()`);
  - `version: true`;
  - a boolean as a section's `order`;
  - an XML layout's `metadata_attributes` that is not a list;
  - `select` or `serialize_as` on an inline template (they were ignored).

### Fixed
- Inlining a schema's `$ref` references (a Pydantic parameter's, or a `tool_from_schema`
  schema's) is bounded at about 100,000 characters and 100 levels, a reference followed counting
  as one: a schema whose references multiply (1.9 KB that became 17.5 MB in 1.4 s), or a chain of
  2,000 references alone, raises `ValueError` in milliseconds (C02).
- `datacite_search` sends `resource_type` in the kebab case DataCite's `resource-type-id` takes
  (`JournalArticle` → `journal-article`); lower case alone broke every type of more than one word
  (T06).
- Europe PMC's error answer (`errCode`/`errMsg`, sent with HTTP 200) is the tool's error, where it
  read as no results, and `europe_pmc_citations` of a record Europe PMC does not have is
  `not_found`, where it read as a record nobody cites (T06).
- `pubmed_article`: an error EFetch reports is that error, not a missing article; the last page
  within arXiv's, PubMed's and Semantic Scholar's reach asks for what is left of it, instead of a
  request the source refuses; an empty arXiv page inside the total is a retryable `upstream`
  failure, not the end of the results (T06).
- `rxnorm_related` sends its term types space-separated, as RxNav reads them (`tty="IN+BN"` went
  out as one literal `IN+BN`), and without `tty` asks for every related concept, where it called
  `related.json` without the parameter RxNav requires (T07a).
- `uniprot_search` quotes an organism name with a space (`organism_name:Homo sapiens` went out
  unquoted), and an `OR` in the query no longer takes the filters with it; an unreviewed (TrEMBL)
  entry shows its submission name, not "(no protein name)" (T07b).
- `gbif_occurrence_search(has_coordinate=False)` asked GBIF for records *without* coordinates; it
  applies no coordinate filter, as documented (T07b).
- `wikidata_sparql` returned 20 rows to a query with `LIMIT 100`, with no note; `wikidata_entity`
  showed 15 claims from the first 20 properties, 3 values each, as bare codes, without units, and
  coordinates as a Python dict (T08a).
- World Bank values were rounded to six digits without a word, and large ones written in
  scientific notation (`2.91849e+13`); `eurostat_series` sent several codes of a dimension as one
  value (`geo=PT+ES`), which the API does not define, and cut `last_time_periods` to 20 without a
  word; `eurostat_compare` kept the first points in index order, an arbitrary series when a
  dimension other than `geo` stayed open (T08a).
- `eonet_events` sends its `bbox` in EONET's order: the latitudes went out swapped (T08b).
- `earthquake_search` and `earthquake_event` read USGS's `204 No Content` as no results, where it
  read as "could not parse"; a deleted event (409) is `not_found`, and a query USGS rejects (400,
  413) a `validation_error` with its detail (T08b).
- `get_weather` with coordinates checks them, and Open-Meteo's reason for refusing a request
  reaches the agent as a `validation_error` in every Open-Meteo tool; `ip_lookup` reports a
  private or reserved address as a `validation_error` (T08b).
- `run_command` failed with "remove the null byte" on an output that is not UTF-8; it reads it
  with replacement characters. `csv_read`'s total counted the line breaks in the first million
  characters: wrong for a longer file or for fields over several lines (T09a).
- `internet_archive_item` asks the metadata API for its extended error codes: a deleted item is
  `not_found`, and an item it cannot read for now a retryable `upstream` failure (T09b).
- `brave_search` and `tavily_search` keep the service's words on a 429, where they lost them
  (C08).
- Open Food Facts nutrients per 100 g are in grams (energy in kcal): the unit a contributor
  entered (`mg` of sodium on a US label) was printed beside the gram value, 1000 times too low
  (T07a).
- An openFDA 404 reads as no recalls only when openFDA says `NOT_FOUND`; any other 404 is an
  `upstream` "endpoint not found" (T07a). The door's `empty_on_404=` takes a test over the 404's
  `Reply` for that.
- `dailymed_label_text` lays out table cells that span rows or columns (the wiki converter's grid,
  now shared), and reads the label's Highlights and the captions of lists and figures;
  `clinical_trial_study` keeps the nesting of the eligibility criteria without markdown's escapes;
  the ClinicalTrials.gov tools ask only for the parts they read, so a page of studies with posted
  results no longer passes the 10 MB limit; DailyMed dates read as ISO 8601 under any locale
  (T07a).
- `uniprot_search` refuses a `cursor` without its `offset`; `pdb_search` no longer fails on a
  record of an unexpected shape, nor says "of 0" past the hits; `gbif_species_search` reads on to
  offset 100,000, which GBIF serves; ChEMBL values show no scientific notation; an inactive UniProt
  accession redirected to plain HTTP is `not_found` naming its entry (the door's `HttpError` keeps
  a refused redirect's URL in `redirect`) (T07b).
- `earthquake_search` and `earthquake_count` include the last day of the period (USGS read the
  bare date as its first instant, so a one-day search found nothing); `overpass_query` refuses a
  query without `[out:json]` or an out statement; coordinates and other floats go to Overpass and
  USGS in decimal notation (`1e-05` was refused) (T08b).
- `run_command` left a process the command started in the background running, with two threads
  reading its output, after the call returned; `regex_search` froze the whole process (it held
  the GIL) on a pattern that backtracks polynomially, such as `a*a*b` on 20,000 characters (about
  18 minutes); `csv_read` built a page as wide as its widest cell on every row (1.3 GB for a
  200 KB file); `python_repl` showed `None` for an expression whose value is an empty string;
  `unit_convert` failed untyped on an integer beyond a float's range (T09a).
- `internet_archive_item` reads the empty array the metadata API answers for an unknown identifier
  as `not_found`, and an item that comes with a warning (code 106) as the item; the `youtube_*`
  tools report a page `youtube-transcript-api` cannot read as an `upstream` parse failure, not a
  retryable `runtime_error` (T09b).
- A window's footer names the next call on a page that came back empty, and says what is left
  where no call reads on although the source gives no total: a Brave page with no web results
  while Brave has more said "end" (C08).
- **A `Range` behind a PEP 695 alias reaches the schema** (T05). A bound written once as
  `type Chars = Annotated[int, Range(500, 20_000)]` and used as a parameter's type turned that
  parameter into a `string` with no bounds: the alias hid the `Annotated` from the schema. The
  schema now sees through the alias, its type and its `Range`, and the contract test fails an
  `int` parameter whose schema is not an integer.
- A tool call whose arguments are empty text (an OpenAI-compatible server's call to a tool that
  takes none) has an empty input, not `{"_raw": ""}`.
- The xAI adapter sends a user turn's images to Grok (G-39), JPEG or PNG, where it dropped them
  with a warning that xAI took no image input; an inline image of another type (by its bytes)
  raises `RequestError` before sending, and a web URL is left to xAI. Every Grok chat model in the inventory reads images
  (https://docs.x.ai/developers/model-capabilities/images/understanding).
- `MemoryMiddleware` finds memories for a request that brings an image (G-38): the query is the
  text of the message's parts (plain strings, `cache()` parts, `{"text": ...}` dicts), where a
  `user([text, image(...)])` gave an empty query, and the request went without memories.
- `GraphStore(NetworkXBackend())` type-checks: the core `NetworkXBackend` is generic in its node
  type, and the memory one holds memory `Node`s, so it satisfies `MemoryBackend` (pyright refused
  it).
- A `@tool` function wrapped with `functools.wraps` runs the wrapper. The wrapper carried a copy
  of the tool's definition, which ran the inner function and skipped the wrapper.
- Every tool call in a `Response` has an id of its own. An OpenAI-compatible server (Ollama,
  LM Studio, llama.cpp, vLLM) may send a call without an id, or with another call's: the tool ran,
  and `tool_result()` then refused the empty id. The base adapter now names such a call
  `call_<24 hex>`, as it names a Gemini call that came without one.
- An `LLM` with an xAI model builds without an event loop (in a worker thread, or before the loop
  runs), and its sync streams run from any thread: the gRPC client needed a running loop, and the
  request was bound to it while being prepared, in the caller's thread. Every adapter now builds
  its SDK client on first use, inside the call's loop, and xAI's request binds to the client only
  when it is sent.
- `close()` after sync calls no longer fails with "Event loop is closed" (`with LLM(...) as llm:`
  around `complete_sync()`): a client whose loop has closed is dropped instead of closed. The
  `OpenAIModerator` gets the same client handling, so its second `moderate_sync()` no longer fails
  on the first call's closed loop.
- Anthropic: a request the SDK refuses to send without streaming (one it expects to take over
  10 minutes, such as a `max_tokens` above about 21 000, a thinking budget included) goes by stream
  and returns the same `Response`, instead of raising `RequestError`. A `timeout=` of the caller's
  own still turns the SDK's refusal off.
- `search_files` no longer reads a file through a link that points out of the folder, and
  `list_directory` lists only entries inside the folder: a pattern such as `../*` or `link/*`
  used to list outside it.
- **Security: no endpoint is read from the environment** (D48). Without `base_url`, the OpenAI,
  Anthropic and Gemini SDKs read `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL` or
  `GOOGLE_GEMINI_BASE_URL`, and the toolkit sent its environment key to whatever host that
  variable named. Every adapter, and the `OpenAIModerator`, now passes its provider's own API to
  the SDK. The Gemini adapter stays on the Gemini Developer API even when
  `GOOGLE_GENAI_USE_VERTEXAI` is set. An OpenAI-compatible server no longer gets the
  `OpenAI-Organization`/`OpenAI-Project` headers the SDK reads from `OPENAI_ORG_ID` and
  `OPENAI_PROJECT_ID`. **Migration:** to reach a gateway or a proxy, pass `base_url=`, with
  `api_key=` unless it is on loopback.
- **Security: the `Redactor` masks xAI (`xai-…`), Groq (`gsk_…`) and Google (`AIza…`) keys**,
  which went through whole: only `sk-…` keys were recognized.
- **A request with images no longer reserves its bytes as text under a strict budget.** The
  request size counted an image part's bytes (or base64) as characters and the part itself as no
  media: a 1 MB image reserved about 735,000 input tokens. Images and documents now count as
  media parts, at the estimator's per-part allowance.
- Gemini: an image given as a `data:` URL went as a `file_uri`, which Gemini cannot fetch; it goes
  inline now.
- **Batch results are priced at the batch rates.** OpenAI (both endpoints, and OpenAI-compatible
  servers) and Anthropic batch results got `Response.cost` at the standard rates, twice the batch
  ones listed in the price table.
- **OpenAI: each model's efforts and default effort come from live measurements.** A model that
  reasons when no effort is sent (GPT-5, GPT-5 mini and nano, o3, GPT-5.5, GPT-5.6, GPT-6, and a
  model not listed) no longer gets the `temperature` the `LLM` always sends, which it refused with a
  400; GPT-5.1, 5.2 and 5.4, which run at `none` by default, keep it. Each model takes only the
  efforts it accepted live on 2026-10-02 (GPT-5 takes `minimal` but not `none`, GPT-5.5 no `max`,
  o3 only `low` to `high`, the pro models `medium` to `xhigh`).
- **OpenAI: `thinking_effort` applies without `thinking=True`.** It was dropped without a word, so
  `thinking_effort="none"` left a model that reasons by default reasoning at `medium`. As on the
  other providers, the effort now applies on its own (checked against the model), and
  `thinking=True` adds the reasoning summary.
- **Meta: a replayed turn keeps the API's field names, and only Muse Spark turns are replayed.**
  Replayed output items go out under their wire names (`async`, not the SDK's `async_`), and an
  assistant turn whose `_raw` came from another provider or model family is rebuilt from its
  fields, without the reasoning, instead of being replayed.
- **The documentation matches the code again.** Every page was checked against the source, and
  what had drifted now says what the code does. Among the corrections:
  - budgets: call caps are hard, while token and cost caps are soft under the default
    `reserve="none"`, and `reserve="strict"` reserves an estimate rather than a guarantee;
  - `@tool(schema=)` takes per-parameter schema fragments, merged into the inferred schema, not a
    whole JSON Schema;
  - only the `react` and `completion` strategies send a multimodal task to the model as content
    parts; the other strategies turn it into text;
  - how a `Flow` picks its execution mode (DAG, cyclic or sequential) and what stops each one, and
    each strategy's search and acceptance rules;
  - the `LLM` defaults (`temperature=0.0`, `max_tokens=4096`), `close()` and the context managers,
    what `batch_submit` bypasses (middleware, retry, fallbacks, the construction defaults and
    metering), and which providers count tokens;
  - import paths: `Agent`, `ReasoningSpec`, `AgentResult`, `agent_from_manifest` and
    `generate_review_flow` come from `ai_arch_toolkit.toolkit.agents`, and the API page now covers
    the agents, metering and budget surface and the whole `ProviderError` hierarchy;
  - installing from git, with commands that work for uv and for pip (the package is not on PyPI);
  - the model-compatibility page lists the 34 tracked model ids with each one's latest live result,
    where it has one;
  - the tools pages give the real counts (132 tools, 31 domains, 44 modules) and say that the
    pre-built tools clamp most out-of-range arguments and do not use the cut window, and that the
    `youtube_*` tools need the `youtube` extra;
  - examples that did not run (the `CostGuard` middleware, the graph-backed memory store,
    `document()` from a path, `result.usage.total_tokens`, a `ReasoningSpec(output_schema=...)`
    run on the old agent) now run, and the example index covers all 47 scripts.
- **Tools report the errors an API sends with a success status.** MediaWiki answers errors with
  HTTP 200 and an `error` object instead of the result, which the tools read as an empty result: on
  a page that does not exist, `mediawiki_page` returned only its header line, `wiktionary_entry`
  only `Wiktionary entry X (English):`, and `mediawiki_sections` "No MediaWiki sections found", so
  an agent never learned the page was missing. They now fail with the API's code and info
  (`MediaWiki page failed: missingtitle: The page you specified doesn't exist.`), and so do
  `mediawiki_search`, the `wikipedia_*` tools, `wikidata_search` and `country_info`. An
  `action=parse` answer without a `parse` object fails as an unexpected response, and a title the
  API flags `invalid` fails with its reason (`wikipedia_article` said "No extract available", and
  `wikipedia_related` searched for it). Other APIs that do the same are now read too, in their own
  words: a World Bank `message` (an unknown indicator or source read as an empty page, and
  `world_bank_indicator` said "not found"), the `ERROR` of a PubMed search ("No PubMed results"),
  an Internet Archive `error` (a search read as "No items found", an item lookup as "not found"),
  an Overpass "runtime error" remark (a query that timed out read as "No Overpass elements
  found."), and the line of text GDELT sends in place of the JSON (`Your query was too short or
  too long.` read as "could not parse API response"). The `Api` of each of these services
  declares how it reports such errors (`body_error`, which also reads a body that is not JSON),
  and `_http` checks that before any `parse` runs.
- **`gdelt_timeline` returns the timeline.** GDELT nests the points under the series
  (`{"timeline": [{"series": …, "data": [points]}]}`), and the tool read each series as a point, so
  every answer came back as "No GDELT timeline points found". It now lists the series' points, and
  when it shows only the first 20 it says of how many (the default 30-day timespan has 31).
- **An error status explained in the body reaches the agent.** `body_error` also reads the body of
  a 4xx or 5xx answer, and its text replaces the status's reason: an unknown Eurostat dataset says
  `HTTP error 404: ERR_NOT_FOUND_4: … is not available for dissemination.` (it said "no matching
  records found."), a request Eurostat would only serve later says so (`ASYNCHRONOUS_RESPONSE. …`,
  was "Request Entity Too Large"), and a query arXiv or UniProt cannot read says why
  (`Invalid query string: …`, `'x' is not a valid search field`; both were "Bad Request"). An
  arXiv error entry sent with a 200 fails too, instead of reading as a paper titled "Error". A
  request to a source that answers "nothing found" with `204 No Content` or an empty body says so
  (`allow_empty=True`), and reads that answer as an empty result: a `pdb_search` without hits says
  `No RCSB PDB entries found for '…'.` instead of "could not parse API response".
- **Tools say what became of a record that is gone, and what they left out.** An inactive UniProt
  accession fails with its fate (`P00001 is inactive: demerged into P99999, P99998`) in
  `uniprot_entry`, `uniprot_features` and `uniprot_crossrefs`, which showed a nameless entry or
  no features or cross-references. `open_library_work` and `open_library_isbn` follow a merged
  record to the one it went into and say a deleted one was deleted; both read as an untitled
  book. `wikidata_entity` reads a merged QID as the item it redirects to (it said "not found").
  `eonet_event` says EONET answers an unknown ID with HTTP 500, and fails on an answer without an
  event instead of showing a blank one. `hacker_news` numbers stories by rank and names the ones
  it could not load, which it dropped without a word, and `wikipedia_related` says why it
  searched instead: no such page, or a page without links.
- **Tree of Thoughts searches best first and always answers.** DFS expanded the worst-scored child
  first (`frontier.pop()` took the last of the children sorted best to worst); it now expands the
  most promising one. BFS kept every state of each level, so with the defaults it made 40 calls,
  never reached `max_depth` and left no `answer`; it now keeps the best `n_candidates` states of
  each level, as in Yao et al. 2023. When the iterations or the states to expand run out, both
  answer from the best state found (the best-scored of the deepest).
- `tot` and `lats` read an LLM evaluator's score as the last number in [0, 1] of its reply. They
  took the first number, so `Score (0.0-1.0): 0.8` scored 0.0 and `Step 2 looks strong: 0.9`
  scored 1.0.
- `generate_review` accepts a draft only when the review's first word is `ACCEPT`. Any first line
  containing "accept" did, so `RETRY: this is not acceptable yet` and `I cannot accept this draft`
  accepted it.
- `llm_compiler` replaces each `$N` in a subtask with the result of task N as a whole: `$1` was
  also replaced inside `$10`, so `Combine $1 and $10` became `Combine <R1> and <R1>0`.
- `country_info` reads Wikidata (its search API, then one SPARQL query), free and without a key:
  REST Countries took v1-v4 down, so every call failed, and its v5 needs a key. It takes a name or
  an ISO 3166-1 code, names the other countries the search matched, and gives the continent where
  it gave the UN subregion.
- `uniprot_search` calls `/uniprotkb/search`: the bare `/uniprotkb` path answered every search with
  a redirect to plain HTTP, which the tools refuse.
- `pdb_search` sends a `full_text` query: RCSB answers 400 to a `text` query without an attribute.
- `eurostat_dataset_search` reads the dataset stubs (IDs and titles, about 1.5 MB) instead of the
  whole catalogue (20 MB, over the 10 MB response cap) and matches the ID and title; its lines no
  longer carry the observation count and period, which `eurostat_dataset` gives.
- Local token counting (`count_tokens_local`) uses `o200k_base` for the `gpt-4.1` family, as
  tiktoken maps it; it used `cl100k_base`. GPT-6 keeps the default: neither OpenAI nor tiktoken
  names its encoding.
- Meta: `thinking_effort="none"` raises `RequestError` instead of reaching Meta, which answers it
  with a 400 (Muse Spark always reasons, https://dev.meta.ai/docs/reasoning).
- The bundled Reflex frontend now uses Reflex 0.9.11 and a security-audited dependency lock,
  removing current PostCSS, React Router, browser-tooling, Nano ID, and Socket.IO advisories.
- Filesystem tools preserve missing, denied, wrong-kind, and other OS error distinctions on Python
  3.14, whose `pathlib` status-query methods now suppress every `OSError`.
- Failed calls no longer poison enforcing scopes when their cost can be bounded. Rate-limit failures are unbilled; indeterminate failures consume a separate cap allowance, so retries, fallbacks and later steps can proceed within the remaining budget.
- Cancellation while waiting for an inference slot releases admission without counting or pricing a call. Streams release their slot before yielding to the consumer, and abandonment closes all delegated transport iterators immediately.
- Attempt history retains failed intermediate fallback models. Settlement still precedes async after hooks, and sync stream cleanup remains safe across threads.
- Strict budgets reserve custom-priced tools and deny unknown or raising prices before execution. Soft budgets admit server tools and fail closed on subsequent work if settlement is unpriced.
- Exact Fable/Mythos 5.1 prices use the published cache rate; Gemini 3.8 Flash gains its own promotional token and batch prices.
- xAI forwards the configured timeout to the SDK instead of ignoring it.
- Anthropic document blocks send their label as `title`, matching the SDK request schema.
- Default retry includes Anthropic overload responses (HTTP 529).
- Flow iteration emits `step_end` for completed steps before wall-budget denial in sequential and single-step DAG waves.
- `ip_lookup` rejects empty or invalid addresses before I/O instead of querying the host machine IP.
- **An empty `ToolGroup` is a value, not an absence.** `Agent(spec, llm, ToolGroup())` used to swap
  the empty group for a new one, so tools added to it later (`group.add(...)`) were unknown to the
  agent. An empty per-phase group (`executor_tools`, `solver_tools`, `rollout_tools`) used to fall
  back to the agent's main tools in `plan_execute`, `reflexion`, `llm_compiler`, `self_discovery` and
  `lats` — a phase declared without tools received all of them. Both now keep the group they were
  given.
- **`LLM(fallback=other)` no longer modifies `other`.** Building the parent emptied the nested
  LLM's own fallback chain and took ownership of the fallbacks it had created, so `other` stopped
  falling back when used on its own and could not be shared by two parents. The parent still walks
  one flat chain (a model reachable twice is tried once); each LLM closes only the fallbacks it
  created from strings.
- **`null` is refused for a tool parameter that does not admit `None`.** Argument validation let
  `None` through for every parameter, so `width: int` received `None`. It is now accepted only with a
  `None` default, an annotation that includes `None`, or no usable annotation; otherwise the call
  fails with `validation_error` before any gate or approval.
- **`ReasoningSpec.from_mapping` drops nothing silently.** A `policy` given as a mapping, a
  malformed `output_schema` and unknown keys were ignored. `policy` now builds a `Policy` from a
  mapping of its fields (`retry` from a mapping of `RetryConfig` fields); an unknown key or an
  unusable value raises `ValueError` naming it.
- **OpenAI structured output accepts Pydantic models again.** `output_schema=Model` was sent in
  strict mode as `model_json_schema()` produced it, and OpenAI answered 400 (`'additionalProperties'
  is required to be supplied and to be false`). Strict schemas are now normalized as the SDK's own
  `parse()` helpers do: every object is closed and lists all its properties as required,
  `default: null` is dropped, and a `$ref` with sibling keys is inlined.
- A tool parameter typed `X | None` without a default can be omitted by the model: the schema
  already made it optional, and the call now receives `None` instead of failing with
  `validation_error`.
- `ToolGroup` and `run_tools()` accept a `functools.partial`: it is named and described by the
  function it wraps and takes the arguments the partial leaves open. It used to raise
  `AttributeError`.
- `ApprovalDecision.approve(modified_args={})` runs the tool with no arguments; the empty dict was
  treated as "no change" and the model's arguments ran.
- **`cache()` parts reach OpenAI, Gemini, and xAI as their text.** Those adapters sent the part's
  Python repr (`CachePart(content='…', ttl='ephemeral')`) to the model. xAI also drops document
  parts with a warning instead of sending their repr.
- **A system message may hold text parts.** A list of strings and `cache()` parts is joined into
  the system prompt, and Anthropic keeps the cache marker, so a system prompt can be cached. Such a
  list used to raise `TypeError` in the Anthropic, Gemini, and xAI adapters and failed inside the
  OpenAI SDK. An image or document in a system message raises a `TypeError` that points to a user
  message. On Anthropic, native blocks passed as `system=` no longer drop the text of `system()`
  messages.
- Sync streams (`LLM.stream_sync()`, `LLM.stream_events_sync()`, `Flow.iter_sync()`) deliver an
  exception object that the source yields as a value, instead of raising it.
- nanope (work in progress): `--allow-dangerous-tools` approves the dangerous tools' calls, the
  BBEH solvers approve `python_repl`, and the research manager approves `http_get` and
  `scrape_text`. Every call to those tools was denied, because they require approval.
- **Anthropic streaming no longer double-counts usage.** `message_delta` usage is cumulative and now
  replaces the `message_start` counts instead of adding to them: input and cache tokens were metered
  twice, tripping token and cost budgets early, and a delta without `input_tokens` raised
  `TypeError`. `complete()` and batch results also record `null` token counts as 0.
- **`Flow(policy=...)` applies to every step without a policy of its own**, also when the flow runs
  directly. `ReasoningSpec.policy`, `ReasoningSpec.timeout`, and manifest `limits.timeout_seconds`
  were inert, and the flow factories dropped `timeout` when given a `policy`.
- **Async middleware runs on streams.** `abefore` / `aafter` run for `stream()`, `stream_events()`,
  and their sync wrappers, so moderation, memory, and rate limiting are no longer skipped when
  streaming. Fallback models receive the messages after middleware, in `complete()` and in streams.
- **`run_tools()` / `run_tools_sync()` apply a `ToolGroup`'s governance** (its gates, approval
  handler, and `max_calls`), which was silently dropped. They also check every tool name in the
  response before running any call, so an unknown name no longer leaves earlier side effects
  without results.
- Anthropic `batch_submit` no longer drops `system()` messages, and middleware that sets
  `request.system` (such as `MemoryMiddleware`) no longer erases them.
- `ToolGroup(web_search())` and `group.add(code_execution())` raise a `TypeError` explaining that
  server tools go next to the group (`tools=[group, web_search()]`), instead of an
  `AttributeError`; other non-callables raise `TypeError` too.
- `ToolGroup.execute()`, `execute_tool()`, and `run_tools_sync()` run `async def` tools to
  completion instead of returning an un-awaited coroutine as a success, and both execution paths
  await an awaitable returned by a sync function. Tools with positional-only parameters can be
  called.
- **Flow engine.** `Flow.iter()` runs DAG waves in parallel and isolates siblings like `run()`. A
  `when` condition or `Scope` callable that raises stops the flow with the error recorded on that
  step, instead of escaping `run()` or silently skipping it. A budget denial in a parallel wave keeps
  the finished siblings in the trace and state, and a step's `Policy(max_cost=...)` counts the spend
  of flows that step runs.
- `OperationRequest.provider` and `UsageEvent.provider` name the adapter that served the call,
  fallbacks included.
- Tool schemas: multi-type unions become a required `anyOf` (optional only with `None`), identical
  members collapse, `*args` / `**kwargs` never enter the schema, and PEP 695 type aliases produce
  their target type's schema.
- A `GateModify` from one gate reaches the next gates and the approval request, and a `TypeError`
  raised inside a tool's body is a `runtime_error` instead of a `validation_error`.
- `Scope.enrich` reads the snapshot the step will see, already filtered and transformed, so it can
  no longer read keys the scope excludes.
- Sync calls made while an event loop is running (`LLM.complete_sync()`, `Flow.run_sync()`,
  `Agent.run_sync()`, …) cancel their coroutine when the sync timeout expires, so the call no longer
  completes, and settles its meter, after the caller has moved on. Sync streams buffer at most 256
  items ahead of a slow consumer, and stopping early cancels the pending read.
- With `trace_capture="full"`, and in `Trace.initial_state`, a step that mutates a value in place no
  longer rewrites earlier records.
- The Gemini adapter no longer fails on tools with `tuple` parameters or schemas with `$defs` /
  `$ref`; they are sent as JSON Schema through `parameters_json_schema`.
- **Nested Pydantic models as tool parameters produce a valid schema.** The model's `$defs`
  table was embedded inside the parameter, so its `#/$defs/...` pointers dangled at the tool's
  root; Gemini rejected such tools with `400 reference to undefined schema`. Nested models are
  now inlined, and a recursive model's `$defs` table is hoisted to the root.
- **Flow timeouts:** a timeout during a parallel DAG wave keeps the siblings that already
  finished (results, trace and state), like a budget denial does; a steady stream of policy events
  can no longer postpone the deadline; and the steps in flight are cancelled before the `timeout`
  event is delivered.
- A cyclic flow with `max_iterations=0` runs no pass again, and the run's meter scope is closed
  even if a second cancellation interrupts the run's cleanup.
- `ReasoningSpec` rejects an unknown `trace_capture` when it is built, and retry backoff no longer
  raises `OverflowError` after about a thousand attempts. `SyncFlowExecution` is exported next to
  `FlowExecution` and works as a context manager.
- The docs no longer claim that `break` stops a flow or agent run: a held execution keeps its step
  running until `aclose()` or the end of an `async with` block.
- **A stream is admitted and priced on the request after async middleware.** Its reservation
  was built from the request before `abefore` ran, so a middleware that added a server tool,
  changed `max_tokens` or injected content left the budget admitting and pricing stale facts.
  The reservation is now replaced before the first attempt when those facts change.
- Fallbacks receive the tools and the kwargs that middleware changed, not only the messages and
  system prompt, in `complete()` and in streams; each fallback keeps its own defaults. A stream
  rejected by middleware records no `StreamAbandoned` attempt, since no provider was called.
- Argument validation no longer raises for an integer string longer than Python's conversion
  limit, a `null` schema branch accepts only `null`, and a custom gate that returns non-mapping
  arguments gets a `validation_error`. `Literal[True, False]` and boolean enums are described as
  booleans, so their values are accepted.
- The Gemini adapter returns an empty response, instead of raising `TypeError`, when a candidate
  cut off by `max_tokens` while thinking carries no content parts.
- Provider reasoning usage is now normalized without double-counting: xAI completion plus
  reasoning tokens and Gemini candidate plus thought tokens become inclusive billable
  `output_tokens`, while OpenAI and Anthropic keep their already-inclusive output totals. Gemini
  tool-use prompt tokens are also counted as input. xAI now prefers the exact per-request
  `cost_in_usd_ticks` charge for response and meter costs, falling back to the local pricing
  registry only when the provider omits it. OpenAI-compatible responses also recover separately
  reported generated tokens from `total_tokens` when it exceeds prompt plus completion.
- **`grok-4.6` is priced at its own rates.** It matched the `grok-4` entry and was estimated at
  $1.25/$2.50 per million input/output tokens instead of $2.00/$6.00 ($0.50 cached input; all
  three double at or above 200k prompt tokens).
- `generate_review` now forwards `ReasoningSpec.output_schema` to its generator while keeping the
  reviewer on its plain-text `ACCEPT` / `RETRY` protocol. Its strategy metadata and the Nanope
  configurable-agent validation now advertise the same support, and Nanope no longer injects
  ReAct-only knobs into other strategies.
- Examples and docs no longer reference retired Anthropic model ids: the retired
  `claude-sonnet-4-20250514` (404 since 2026-06-15) and the malformed
  `claude-opus-4-0-20250514` became `claude-sonnet-5` / `claude-opus-5`, and the dated
  `claude-haiku-4-5-20251001` was normalized to the recommended `claude-haiku-4-5`
  alias.
- **Bundled pricing table refreshed against each provider's live model list**
  (2026-07-26). Added: `claude-fable-5`/`claude-mythos-5`, `claude-opus-5` (with 2x
  fast-mode rates), `claude-sonnet-5`, and `claude-opus-4-8` — the latter previously
  fell through the `claude-opus-4` fallback prefix and was **billed at 3x its real
  price**; OpenAI `gpt-5.6-sol`/`-terra`/`-luna`; `gemini-3.6-flash`,
  `gemini-3.5-flash`, `gemini-3.5-flash-lite`; `grok-4.5` and `grok-build-0.1`, plus
  the 200k-prompt long-context tiers on all current Grok entries. Removed entries for
  models the providers no longer serve: Claude 3.x and 4.0, `o1-mini`/`o3-pro`/
  `o3-deep-research`, Gemini 1.5, and `grok-2`. Stale 6x fast-mode rates were dropped
  from Opus 4.6/4.7 (fast mode was removed on those models).
- A second sync call on the same `LLM` instance no longer fails with
  `APIConnectionError`. The sync wrappers run each call on a fresh `asyncio.run()`
  loop, but every adapter cached its async SDK client, whose connection pool stayed
  bound to the first (closed) loop. Providers now rebuild the client once the loop it
  served has closed (`LoopAwareClientCache`, all four adapters); directly assigned
  clients — e.g. test mocks — are never replaced. Repeated `complete_sync`/`run_sync`
  calls, notebook usage, and per-test event loops all recover; using one instance
  from two concurrently live loops remains unsupported.
- `uv sync --extra dev` now installs `jsonschema` and `jinja2`, so the prompt-template
  tests pass on a fresh dev environment (previously only the `prompts` extra pulled
  them in).
- The Anthropic adapter now drops the client-default `temperature` for every model
  family that rejects sampling parameters — Opus 4.7/4.8/5, Sonnet 5, and Fable/Mythos 5,
  matched by prefix. Previously only `claude-opus-4-7` was covered, so every call to a
  current Anthropic model failed with `400: temperature is deprecated for this model`.
- `ReasoningSpec.llm_kwargs` now reach every phase of all multi-phase strategies
  (`plan_execute`, `rewoo`, `reflexion`, `self_discovery`, `llm_compiler`, `tot`,
  `lats`, and `generate_review`'s reviewer — where `reviewer_kwargs` merges on top,
  winning per key); previously they were silently dropped.
- `llm_compiler_flow` supports an executor LLM override again (`exec_llm`); the inner
  ReAct always used the default LLM even though planner/joiner overrides existed.
- `plan_execute_flow`'s planner sees the executor's tool catalog again — and
  `llm_compiler_flow`'s planner gains it for the first time — via the `{tools}` token
  in their default planner prompts, so plans match what the execution phase can
  actually call.
- Provider adapters now disable hidden Anthropic, OpenAI, Gemini, and xAI SDK
  retry loops. `LLM.retry` is the single retry owner, so every attempt is
  metered and exposed through `Response.attempts`.
- Streaming now retries or enters the fallback chain when provider errors occur lazily
  before the first emitted item. Each physical attempt is metered and recorded; errors
  after observable output are surfaced without replay. Abandoned async streams close
  their provider iterators immediately; all streams remain outside `inference_limit` and
  preserve nested fallback chains configured on fallback `LLM` instances.
- Agent manifests now resolve relative paths inside embedded profiles against the
  declaring file and override-supplied paths against the entry manifest, validate
  inherited `*.agent.*` suffixes, and default a missing `strategy` section to ReAct.
  Their canonical JSON-like data boundary now guarantees deterministic fingerprints;
  secret scanning covers every source/profile; descendant deny rules cannot be bypassed
  through parent overrides; and multi-root source provenance cannot collide.
- `BudgetPolicy` and agent-manifest limits now reject invalid `reserve` / `unpriced`
  modes, malformed count caps, and non-finite numeric caps instead of silently
  weakening enforcement.
- The Gemini extra now requires `google-genai>=1.21.0`, the first supported SDK floor
  with `HttpOptions.retry_options`.
- Structured agent responses with an empty text field now retain their parsed payload in
  `AgentResult.response`.
- `AgentResult.errors` now retains failures from earlier iterations of cyclic flows instead of
  losing them when a later result with the same step name succeeds.
- **Cached input tokens are no longer double-counted.** OpenAI, Gemini, and xAI adapters (including OpenAI batch responses) now normalize provider-reported inclusive input totals into disjoint `Usage.input_tokens` and `cache_read_tokens`, so pricing and budget metering charge cached tokens only at the cache rate.
- **Anthropic structured output now validates into the Pydantic model.** When `output_schema` carries a `model_class`, the Anthropic adapter coerces the parsed JSON via `model_validate()` — at parity with the OpenAI, Gemini, and xAI adapters (it previously returned a raw `dict`). It also tolerates Markdown-fenced JSON in the reply.
- OpenAI-compatible streaming now flushes accumulated tool calls when a server ends the turn with `finish_reason="stop"` instead of `"tool_calls"` (some Ollama / LM Studio / vLLM builds), so tool-using agents no longer silently see zero tool calls.
- Structured output: parsed JSON is now validated against the Pydantic model before being returned.
- Gemini API key resolution: `GOOGLE_API_KEY` now takes precedence over `GEMINI_API_KEY` when both are set.
- Lint fixes: ternary form in `toolkit/tools/_datetime.py`; unused `pytest` import in `tests/test_python_eval.py`; misc `ruff format` across toolkit tools.

- The network tools no longer raise on a response of an unexpected shape (a JSON array or `null`
  where an object was expected, a `null` field, undecodable bytes): they return "…: could not parse
  API response: …". 93 of them raised before.
- Tools no longer raise on hostile arguments: very long paths in the filesystem tools, dates past
  the year 9999 in `date_add` and `timezone_convert`, deeply nested JSON in `json_extract`, an empty
  or absolute glob in `list_directory`, a null byte in `run_command`, network errors in the
  `youtube_*` tools.
- `math_eval("9**9**9")`, `factorial(10**7)`, `round(5, -10**9)` and `regex_search` with
  `(a+)+$` return an error at once instead of holding the process.
- An argument can no longer climb out of an API's path: `..` in `country_info`, `define_word`,
  `europe_pmc_citations` and every other path segment is refused.
- Rate-limit throttles are safe with parallel tool calls, which could skip the wait.
- A `null` field in an agent manifest means "not set" everywhere, as the loader already let it.
  Before:
  - `strategy: null` failed in `reasoning_spec()`;
  - `strategy.name: null` asked for a strategy named `"None"`;
  - `strategy.max_iterations: null` raised `TypeError`;
  - `strategy.system: null` became the prompt `"None"`;
  - `limits.reserve: null` failed in `budget_policy()`;
  - `override_policy: null` failed at load.
- A list or an object as an agent manifest's `strategy.trace_capture` fails with
  `AgentManifestError` instead of `TypeError`.
- `python_repl` refuses what it used to get wrong in silence:
  - `{**a}` (it evaluated to `{None: a}`);
  - `f(**d)` (`d` was dropped);
  - `del x.attr` (ignored).
- `python_repl`'s `and`/`or` stop at the operand that decides, as Python's do.
- `python_repl`'s `x **= n` has the same exponent limit as `x ** n`.
- **`lats` branches.** Each rollout added one child to a node that had none, so the search tree
  was a single chain of attempts and `n_candidates` was ignored. A node now takes up to
  `n_candidates` sibling attempts before the search goes below it, and UCT chooses among them, as
  in Zhou et al. 2024 (LATS). `max_rollouts` still caps the total number of ReAct attempts. The
  docstring now warns that rollouts re-run their tools, so tool side effects repeat.

### Removed
- The prices and per-model rules of the OpenAI models shut down by 2026-10-23
  (https://developers.openai.com/api/docs/deprecations): `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`,
  `gpt-4.1-nano`, `o1`, `o1-pro`, `o3-mini` and `o4-mini`, with their snapshots, and
  `gpt-3.5-turbo-1106` (2026-09-28). Under a `MeterScope` they now raise `UnpricedModelError`;
  `gpt-4o-2024-05-13` keeps its own tariff until its shutdown, so it is not priced as `gpt-4o`.
  The examples, docs, and live tests use `gpt-4.1-mini` where they used `gpt-4.1-nano`.
- **The legacy `core._budget` module** (`BudgetState`, its cooperative `BudgetPolicy`). Budgets now live in `toolkit.budget` and enforce at the charge site rather than by cooperative counter-checking as steps record usage. `BudgetPolicy.max_wall_time` is now `max_wall_s`; the `strict_cost` / `allow_unpriced` flags are now the `reserve` / `unpriced` knobs. `BudgetExceeded` keeps its `.limit` / `.maximum` / `.to_dict()` surface.

## Historical log

Chronological worklog of features and major changes prior to adopting Keep a Changelog.

| Date | Change |
|------------|--------|
| 2026-03-12 | Add nanope BBEH benchmark suite, expand toolkit tools (python eval, dictionary), and add generate-review flow |
| 2026-03-08 | Add content moderation system with Moderator protocol, OpenAI and LLM implementations |
| 2026-03-08 | Replace legacy agents and pipeline with Flow-based architecture |
| 2026-03-05 | Update pricing registry with latest model prices from all providers |
| 2026-03-05 | Add local token counting and extend pricing registry |
| 2026-03-05 | Extract general-purpose graph layer from memory system with new methods |
| 2026-03-04 | Add production readiness: timeouts, validation, rate limiting, observability |
| 2026-03-03 | Add pipeline system, knowledge registry, and LLM fallback chains with attempt tracking |
| 2026-03-02 | Add graph-backed memory system with views, middleware, and agent tools |
| 2026-02-28 | Add SelfDiscovery and LLMCompiler agents, per-phase customization (PhaseConfig) |
| 2026-02-28 | Add PlanExecute, ToT, and LATS agents, rich streaming events, stream fallback |
| 2026-02-28 | Add Reflexion and ReWOO agents, delete legacy layer |
| 2026-02-27 | Add agent architecture implementation with ReAct and multimodal capabilities |
| 2026-02-25 | Rewrite core/ layer, add Gemini and xAI providers, restructure project |
| 2026-02-24 | New standard paradigm, research knowledge, board system, file structure reorganization |
| 2026-02-23 | Add tools layer — schema inference, @tool decorator, execution, and ToolGroup |
| 2026-02-23 | Add client reuse, stream metadata, and OpenAI provider |
| 2026-02-22 | Rewrite LLM layer from first principles |
| 2026-02-21 | Add MkDocs documentation, build commands, API docs, and CI configuration |
| 2026-02-09 | Migrate from Poetry to uv, initial project structure with examples |
| 2026-02-08 | Add LLMs documentation: agents architecture and API guide |
| 2026-02-07 | Initial commit |
