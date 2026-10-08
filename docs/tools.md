# Tools

Tools let an LLM call your Python functions. The toolkit gives you three things:

1. The **`@tool`** decorator — turn any typed function into a tool (JSON Schema is generated for you); **`tool_from_schema`** builds one from a JSON Schema that arrives as data (an MCP server's, for one).
2. **`ToolGroup`** — a governed collection that validates, executes, and structures results.
3. A library of **123 pre-built tools** across 31 domains, ready to drop into a group.

Provider-hosted **server tools** (code execution, web search) are covered after `ToolGroup`. Safety controls — risk levels, approval gates, dangerous-tool blocking, trace redaction, and budgets — live on their own page: see [Tool Governance & Safety](safety.md).

---

## Defining tools

Decorate a typed function with a Google-style docstring. The schema (name, description, parameters) is inferred from the type hints and docstring — you don't write JSON Schema by hand.

A parameter typed `Any` or `object` accepts any JSON value, so its schema sets no type. A parameter with no annotation, or with a type the generator does not know (a `Path`, a custom class), is described as a string.

A Pydantic model parameter is described by its own JSON Schema, with nested models inlined, so the tool's schema stands on its own. A recursive model keeps a `$defs` table, placed at the root of the tool's schema.

```python
from ai_arch_toolkit import tool

@tool
def get_distance(origin: str, destination: str, unit: str = "km") -> str:
    """Compute the distance between two cities.

    Args:
        origin: Starting city.
        destination: Destination city.
        unit: Distance unit, "km" or "mi".
    """
    ...
```

Declare a numeric parameter's limits in its annotation with `Range`, instead of adjusting the value inside the tool. The model reads them in the schema (`minimum`/`maximum`), and the executor refuses a value outside them with a `validation_error` that names the range (`expected integer from 1 to 25, got int 40`). One bound is enough (`Range(maximum=1.0)`), and `Annotated[int | None, Range(1, 25)]` bounds an optional parameter. A `Range` on a type with no numbers is a `ValueError` when the tool is decorated. Bounds written in a `schema=` override are enforced the same way.

```python
from typing import Annotated

from ai_arch_toolkit import Range, tool

@tool
def search_papers(query: str, max_results: Annotated[int, Range(1, 25)] = 10) -> str:
    """Search papers by keyword.

    Args:
        query: Keywords.
        max_results: How many papers to return.
    """
    ...
```

The decorator also accepts governance metadata — `capability`, `risk_level`, `requires_approval`, `approval_reason` — and the bounds the executor holds each call to, `max_output_chars` and `timeout_s`:

```python
@tool(risk_level="high", requires_approval=True, approval_reason="Deletes data")
def delete_table(name: str) -> str:
    """Drop a database table."""
    ...
```

These attach a `ToolRuntimePolicy` to the tool that gates read at execution time — see [Tool Governance & Safety](safety.md). A result longer than `max_output_chars` is cut on a line break when one lies in the second half of the kept text, and ends with `[chars 0-200000 of 5000000 | cut at the output limit; ask for less]`; `result.metadata["truncated"]` holds the two sizes.

A tool built on the toolkit's window (`toolkit/tools/_window.py`, see `CONTRIBUTING.md`) that returns part of something longer ends its text with a footer in the same vocabulary, naming the call that reads on — `[chars 0-4000 of 34651 | next: offset=4000]`, `[results 21-40 of 1234 | next: offset=40]`, `[matches 1-3 of 5 for "1960" | next: find="1960", offset=16500]` — and puts the same facts in `result.metadata["window"]` (`unit`, `first`, `last`, `total`, `next_call`). None of the pre-built tools is built on it: those that mark a cut do so in their own words.

Full `@tool` signature:

```python
@tool(
    *,
    name: str | None = None,            # override the inferred tool name
    schema: dict[str, dict[str, object]] | None = None,  # per-parameter schema keywords
    capability: str | None = None,      # logical capability label
    risk_level: RiskLevel = "low",      # "low" | "medium" | "high" | "critical"
    requires_approval: bool = False,    # gate behind an approval handler
    approval_reason: str = "",          # shown to the approver
    max_output_chars: int | None = 200_000,  # longer results are cut and marked; None: no cut
    timeout_s: float | None = 120.0,    # then the call fails with "timeout"; None: no deadline
)
```

`schema=` does not replace the inferred input schema: it maps a parameter's name to JSON Schema keywords merged into what was inferred for that parameter — `@tool(schema={"unit": {"enum": ["km", "mi"]}})` on `get_distance` keeps `unit`'s type, description and default and adds the `enum`, which the executor then checks. Anything else raises `TypeError` when the decorator is applied: a complete schema (`{"type": "object", "properties": ...}`) would otherwise become parameters named `type` and `properties`. A tool described by a complete schema is built with [`tool_from_schema`](#tools-from-a-json-schema).

A tool name is 1 to 64 letters, digits, `_` or `-`, and starts with a letter or `_` (`^[A-Za-z_][A-Za-z0-9_-]{0,63}$`): the names Anthropic, OpenAI, Gemini, xAI, Meta and MCP all accept, since `LLM(fallback=...)` may move a run to another provider. Any other name (`a.b`, `1a`, 65 characters, a `lambda`'s `<lambda>`) raises `ValueError` where the tool is made: `@tool`, `tool_from_schema`, `ToolGroup(...)`, or `prepare_tools` for a plain callable or a tool dict.

Gemini function declarations take an OpenAPI subset of JSON Schema that has no `prefixItems` or `$defs`/`$ref`. A schema outside it (a fixed-length `tuple` parameter, or a `schema=` override with references) is sent through Gemini's `parameters_json_schema` field instead, unchanged.

### Tools from a JSON Schema

A tool whose definition arrives as data — from an MCP server, an OpenAPI document, or the app's own configuration — has a name, a description and a JSON Schema, but no Python signature. `tool_from_schema` builds it, and the result is a tool like any `@tool` function: a `ToolGroup`, `llm.complete(tools=...)`, `execute_tool` and `run_tools` take it as they are, and its calls go through the same governed executor (validation, gates, approval, `max_calls`, metering).

```python
from typing import Any

from ai_arch_toolkit import ToolGroup, ToolRuntimePolicy, tool_from_schema

async def search(arguments: dict[str, Any]) -> str:
    return await github.search_issues(arguments["q"], limit=arguments.get("limit", 10))

search_tool = tool_from_schema(
    search,
    name="github_search",
    description="Search issues.",
    input_schema={
        "type": "object",
        "properties": {"q": {"type": "string"}, "limit": {"type": ["integer", "null"]}},
        "required": ["q"],
        "additionalProperties": False,
    },
    policy=ToolRuntimePolicy(capability="github", risk_level="high", requires_approval=True),
)
group = ToolGroup(search_tool, approval_handler=ask)
```

- **The handler** receives the arguments in one `dict`, so a key need not be a Python name (`"x-api-version"`, `"from"`). An `async def` handler runs on the caller's loop, a synchronous one in a thread of its own (its `timeout_s` holds, as for `@tool`). It returns the tool's value, or a `ToolResult`, which passes intact. One that cannot answer raises `ToolFailure` with its type (`not_found`, `validation_error`, `upstream`, `rate_limited`); any other exception becomes a `runtime_error` with its message redacted.
- **Validation** is the one `@tool` gets, before any gate or approval (see [Argument validation](safety.md#argument-validation)): root arguments are coerced (`"3"` for an `integer`, through `anyOf`, `oneOf` or a `type` list too), `additionalProperties: false` at the root refuses an unknown key, and `null` passes only where the schema admits it. A branch is read with its parent's keywords, so MCP's titled enum, `{"type": "integer", "oneOf": [{"const": 1, "title": "Low"}, ...]}`, coerces `"1"` too. Nested values are left as they came, for the handler — or the server behind it — to check, and so is a value whose schema offers more than 1,000 alternatives once its branches are flattened. A refusal names the argument, what was expected and what came, in some 1,000 characters at most.
- **The schema** is copied and sent as it came, `title`, `$schema`, `default` and `x-*` keys included, with its local `#/$defs/...` references inlined (a recursive one stays a reference, with its `$defs` table at the root).
- **`ValueError`** when the tool is made, and no other exception for a schema that is JSON: a name outside the rule above, a schema that is not JSON (`NaN`, a set), a root that is not `"type": "object"`, a `$ref` outside the schema (`https://...`, another file — never fetched) or to a definition that is not an object schema (`true`, a list), or references whose inlining would make the schema larger than about 100,000 characters (some 25,000 tokens in every request) or deeper than 100 levels, each reference followed counting as one. A hostile schema fails in milliseconds.
- `policy` takes the same `ToolRuntimePolicy` `@tool` builds from its keywords; the default is a low-risk tool that needs no approval.

---

## ToolGroup

A `ToolGroup` bundles tools, exposes provider-safe definitions for the LLM, and runs the governed execution pipeline.

```python
from ai_arch_toolkit import ToolGroup

group = ToolGroup(get_distance, delete_table)

group.definitions          # provider-safe tool schemas to send to the LLM
group.tools                # the registered callables
group.add(another_tool)    # register one more
group.add(new_version, replace=True)   # swap the tool held under its name
group.remove("delete_table")           # -> its ToolDefinition; KeyError if absent
```

One name, one tool: the model calls a tool by its name, so another tool under a name the group already holds raises `ValueError` (give one of them another name with `@tool(name=...)`, or pass `replace=True` to swap it); adding a tool the group already holds changes nothing. A wrapper made with `functools.wraps` around a `@tool` function carries the tool's definition, its name and policy included, and the group runs the wrapper. The same rule holds across a list and the groups in it: `prepare_tools` (so `llm.complete(tools=[a, b])`), `execute_tool` and `run_tools` raise `ValueError` for two different tools with one name before anything is sent or runs, and send or count once a tool that appears twice. Server tools have no name of the app's, so a list that `llm.complete` takes, `execute_tool` and `run_tools` take too.

A group may change while it is in use, as tools arrive from an MCP server or leave with it: `add` and `remove` put a new table in place instead of editing the one being read, so the next read sees the change — the next ReAct turn offers the new set of tools — and a call already running ends with the tool it started with. `max_calls` keeps counting across changes. An empty `ToolGroup()` is a group like any other: an `Agent` given one keeps it, and tools added later reach it.

Constructor:

```python
ToolGroup(
    *fns,                          # the tool callables
    approval_handler=None,         # ApprovalHandler for tools requiring approval (see safety.md)
    gates=(),                      # extra pre-execution ToolGate instances (see safety.md)
    max_calls=None,                # cap executions, counted until group.reset()
    max_output_chars=None,         # ceiling on every tool's max_output_chars
    timeout_s=None,                # ceiling on every tool's timeout_s
)
```

The two ceilings only tighten: each call runs under the stricter of the group's value and the tool's own (see [Output and time limits](safety.md#output-and-time-limits)).

### Executing tool calls

Both `execute()` (sync) and `async_execute()` (async) take a `ToolCall` and return a structured **`ToolResult`** — they never raise on tool failure.

`execute()` also runs `async def` tools, completing them on a private event loop — or on a worker thread when called from inside a running loop, which blocks that loop until the tool returns — so prefer `async_execute()` in async code. A synchronous tool runs in a daemon thread of its own on both paths, so its `timeout_s` holds: past it the call returns a `timeout` failure and the executor stops waiting. An async tool that needs the caller's loop (a client or lock created on it) cannot finish there and fails — at the latest at its `timeout_s`, or at the sync wrapper's timeout (`AI_ARCH_SYNC_TIMEOUT`, 300 s by default) when that comes first. Both check the call's arguments against the tool's schema before any gate runs, coercing values such as `"3"` for an integer; see [Argument validation](safety.md#argument-validation).

```python
result = await group.async_execute(tool_call)   # tool_call: ToolCall from a Response

if result.ok:
    print(result.value)            # the function's return value
else:
    print(result.error.type, result.error.message)

text = result.to_model_text()      # LLM-safe string to feed back as a tool_result
```

`ToolResult` and `ToolError` (with `retryable` / `safe_to_show` flags) are detailed in [Tool Governance & Safety](safety.md#structured-results).

### run_tools helper

When you have a `Response` that contains tool calls, `run_tools()` executes all of them and returns ready-to-send `tool_result` messages — handy for a manual LLM loop.

```python
from ai_arch_toolkit import run_tools, run_tools_sync

response = llm.complete_sync("What's the distance from Lisbon to Porto?", tools=group)
results = run_tools_sync(response, group)        # list[dict] — tool_result messages
# feed `results` back into the next llm.complete(...) call
```

`run_tools()` accepts either a `ToolGroup` or a plain `list[Callable]`. A `ToolGroup` runs every call through its own governance — its gates, `approval_handler`, and `max_calls` budget, exactly as `group.execute()` does — so `approval_handler=` is only for a plain list: passing it together with a group raises `ValueError`. Every tool name in the response is checked before any call runs: an unknown name raises `KeyError`, and a list holding two different tools with one name raises `ValueError`, and nothing executes, so a response never half-runs.

---

## Server tools

Provider-hosted tools run on the provider's side (no local execution). Pass them in the `tools=` list next to your own tools or a `ToolGroup` (`tools=[group, web_search()]`), never inside the group: `ToolGroup(web_search())` raises `TypeError`, because a group only runs local callables.

```python
from ai_arch_toolkit import LLM, code_execution, web_search

llm = LLM("claude-sonnet-5")
response = llm.complete_sync(
    "Plot the first 10 primes and tell me their sum.",
    tools=[code_execution(), web_search()],
)
```

`code_execution(**config)` and `web_search(**config)` return a `ServerTool`. The Anthropic and Gemini adapters send both, and the OpenAI adapter (on OpenAI's own host, through the Responses API) and the Meta adapter send `web_search`; the xAI adapter and OpenAI-compatible servers (Chat Completions takes function tools only) raise `RequestError` for any server tool. A config (`web_search(max_uses=3)`, for example) raises `RequestError` on every provider. See [Model Compatibility](model-compatibility.md) for each provider.

`image_generation(model=..., quality=..., aspect_ratio=..., resolution=..., output_format=..., partial_images=...)` lets an OpenAI model draw in the middle of its turn, with the image model it names; it is the one server tool with a typed config, checked against that image model's rules. Only the OpenAI adapter on OpenAI's own host sends it. See [Image Generation](images.md).

### Web search: hosted or local

The toolkit has two ways to search the web. The server tool `web_search()` asks the model's provider to search inside its turn; the toolkit tools `brave_search` and `tavily_search` are function tools that the toolkit runs, with a search API key of your own ([catalog](tools-catalog.md#web-search-your-own-key-billed)).

| | `web_search()` | `brave_search`, `tavily_search` |
|---|---|---|
| Who runs it | The provider, inside the model's turn | The toolkit, through the governed executor (gates, limits, budget) |
| Who bills it | The provider, outside the token counts: the meter cannot price it, so the turn's cost is unknown ([Cost control](cost-control.md)) | Brave or Tavily, on your key; the meter charges each search the service accepted at the price table's `[tools]` entry ([Pricing](pricing.md#paid-tools)), and a refused search costs nothing |
| Which models | Anthropic, Gemini, OpenAI (its own host) and Meta; xAI and OpenAI-compatible servers raise `RequestError` | Any model that calls tools, local ones included (Ollama, LM Studio, vLLM) |
| Key | None beyond the provider's | `BRAVE_SEARCH_API_KEY` or `TAVILY_API_KEY`, read from the environment at each request |
| Options | None: a config raises `RequestError` | Brave: `max_results` (1–20 a page), `offset` (the page, 0–9), `country` (a code Brave lists: `GB`, not `UK`), `freshness`. Tavily: `max_results` (0–20, one page), `topic`, `time_range`, `include_answer` |
| What the model reads | The provider's citations and summary | A numbered list: title, URL (only `http(s)`), the whole snippet; Brave's pages end with the call for the next one |

```python
from ai_arch_toolkit import LLM, ModelPricing, ToolGroup, pricing
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec
from ai_arch_toolkit.toolkit.tools import brave_search

pricing.register("qwen3:8b", ModelPricing())  # free, but a run needs a price
llm = LLM("qwen3:8b", base_url="http://localhost:11434/v1")  # a local model, no provider key
agent = Agent(ReasoningSpec(strategy="react"), llm, ToolGroup(brave_search))
result = agent.run_sync("What changed in Python 3.14? Search the web.")
print(result.text)
```

The agent's loop runs each `brave_search` the model asks for and gives it the results; `llm.complete_sync(..., tools=...)` alone only returns the model's call, which [`run_tools_sync`](#run_tools-helper) runs in a loop of your own.

Both are network tools of low risk that run without approval, like the other network tools; [Safety](safety.md#web-search-tools) says what they send and what they bring back. [Example 49](examples.md) runs an agent on a local model with whichever key you have.

---

## Pre-built tools catalog

123 tools across 31 domains and 43 modules, all built on the `@tool` decorator and the standard library only (zero extra pip dependencies; the `youtube_*` tools need the `youtube` extra). One that cannot answer raises a typed `ToolFailure`, which the executor returns as a failed `ToolResult`, so agents degrade gracefully; each declares its `capability`. Every tool keeps the tools contract (T05 to T09): its limits are `Range` bounds in the signature, which the executor enforces (a value outside is a `validation_error`, never moved to the nearest limit), and its long answers go through the window (each text ends with the call that reads on, and `metadata["window"]` holds the same facts). The contract test holds every tool to it, and its debt list is empty. The wiki family (`wiki_search`, `wiki_outline`, `wiki_read`, `wiktionary_entry`) is the template for a new one.

Every network tool but the three `youtube_*` ones (which go through `youtube-transcript-api`) uses one module, `toolkit/tools/_http.py`: HTTPS to the host its module declares (path segments quoted one by one, so arguments cannot move a request elsewhere; the dangerous `http_get` and `scrape_text` fetch the http(s) URL they are given), redirects only on that host and never down to `http`, a body read bounded in bytes and time, the API's documented rate limit on a clock shared across threads (counted from the end of each request), and one error wording ("HTTP error 500: …", "rate limited by … (HTTP 429). Try again later.", "request timed out.", "could not parse API response: …"), each raised as a typed `ToolFailure` (`HttpError`). A response of an unexpected shape becomes that last error, an `upstream` failure, instead of an arbitrary exception. Each `Api` declares once how its source reports an error, an `error_reader=` that sees the status, the headers and the body of every JSON answer and of every error status: an error the API sends with a success status (MediaWiki's `error` object, a World Bank `message`, a PubMed search `ERROR`, an Internet Archive `error`, an Overpass "runtime error" remark, the line of text GDELT sends in place of the JSON) is the tool's error too, in the API's own words, never an empty result, and the same reader explains an error status (Eurostat's `label`, arXiv's error entry, NVD's `message` header). Where the source says what happened, the failure carries that type: a missing MediaWiki page or an inactive UniProt entry is `not_found`, MediaWiki's `ratelimited` is `rate_limited`. Without a reader, an error status carries the source's own error text (a JSON `message`, `error` or `detail`, or the start of a text body). A 404 is `not_found` only on a call that asks for one resource and says so (`missing="no ChEMBL molecule with ID …"`); any other 404 reads "endpoint not found (HTTP 404); the API may have changed", never "no results" (or, at a URL the caller gave, a `validation_error` that says nothing answers there). A missing mandatory key is a `validation_error` that says which variable to set. A source that answers "nothing found" with `204 No Content` or an empty body is declared on the request (`allow_empty=True`, as the RCSB search does; `empty_on_404=` where the answer is a 404: `True`, or a test over the 404's `Reply` when the source's 404 also means other things, as openFDA's `NOT_FOUND`), and that answer reads as an empty result; anywhere else it is a parse error. A redirect the door refuses keeps its URL on the failure (`HttpError.redirect`), so a tool can name where the source points.

A paid API declares the price its requests are billed at, `Api(billed_as="brave_search")`: each request it accepts records its units (one, or what the answer says with `bill_units=`), and the tool executor charges them at the price table's `[tools]` entry when the call ends ([Pricing](pricing.md#paid-tools)). With `key_required=True`, a request without the API's key fails before it is sent, saying where to get one; `key_prefix=` puts a scheme before the key (`"Bearer "`).

After a 429, or an answer in which the source says it is rate limited (MediaWiki's `maxlag`), a host rests for the time its `Retry-After` asks, or the time the module declares (GDELT: 60 s, since its free API shuts its gate a minute or more after one): a call in that time fails at once, without a request, and says when to try again ("GDELT asked to slow down: try again in 42 s."). A source that takes an optional key reads it from the environment at each request: `SEMANTIC_SCHOLAR_API_KEY` (free, sent in `x-api-key`) gives its holder 1 request per second, where keyless callers share one limit that is often spent; without the key, its 429 says where to get one. Every request names the toolkit, its version and its repository in the `User-Agent`.

TLS is verified against OpenSSL's CA file by default. On a Python whose file lacks recent roots (uv's standalone builds on macOS read `/etc/ssl/cert.pem`, which has no root for Eurostat's certificate), install the `truststore` extra: the tools then verify with the system's certificate store, as pip does, and the certificate error says so when it is missing.

```python
from ai_arch_toolkit.toolkit.tools import get_weather, arxiv_search, pubmed_search
from ai_arch_toolkit import ToolGroup

group = ToolGroup(get_weather, arxiv_search, pubmed_search)
```

The domains at a glance:

| Theme | Domains |
|-------|---------|
| General & utility | date & time, math, text processing, data (JSON) |
| Weather, geo & places | weather, air quality, geography, OpenStreetMap |
| Reference & knowledge | Wikipedia, Wikidata & MediaWiki, dictionary, news & events, video transcripts |
| Scholarly & research | papers (arXiv, PubMed, Europe PMC), academic graph & metadata (Semantic Scholar, Crossref, ROR, DataCite), books (Open Library), digital archives (Internet Archive) |
| Biomedical & chemistry | proteins & structures (UniProt, PDB), chemistry & bioactivity (ChEMBL), medication labels (RxNorm, DailyMed), clinical studies (ClinicalTrials.gov) |
| Earth, life & public data | biodiversity (GBIF), food products (Open Food Facts), food safety & ontology (openFDA, FoodOn), natural events (USGS, NASA EONET), official statistics (World Bank, WHO, Eurostat), security (NVD) |
| Dangerous (opt-in) | filesystem, shell, Python, web |

**→ Full per-tool list: [Tools Catalog](tools-catalog.md).** The filesystem/shell/Python/web tools execute real side effects and must be gated — see [Tool Governance & Safety](safety.md#dangerous-tools).

---

See also: [Tool Governance & Safety](safety.md) for risk levels, approvals, blocking, redaction, and budgets · [Flow Architecture](flow-architecture.md) for how tools plug into agent flows.
