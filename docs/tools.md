# Tools

Tools let an LLM call your Python functions. The toolkit gives you three things:

1. The **`@tool`** decorator — turn any typed function into a tool (JSON Schema is generated for you).
2. **`ToolGroup`** — a governed collection that validates, executes, and structures results.
3. A library of **132 pre-built tools** across 25 domains, ready to drop into a group.

Provider-hosted **server tools** (code execution, web search) are covered at the end. Safety controls — risk levels, approval gates, dangerous-tool blocking, trace redaction, and budgets — live on their own page: see [Tool Governance & Safety](safety.md).

---

## Defining tools

Decorate a typed function with a Google-style docstring. The schema (name, description, parameters) is inferred from the type hints and docstring — you don't write JSON Schema by hand.

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

The decorator also accepts governance metadata — `capability`, `risk_level`, `requires_approval`, `approval_reason`:

```python
@tool(risk_level="high", requires_approval=True, approval_reason="Deletes data")
def delete_table(name: str) -> str:
    """Drop a database table."""
    ...
```

These attach a `ToolRuntimePolicy` to the tool that gates read at execution time — see [Tool Governance & Safety](safety.md).

Full `@tool` signature:

```python
@tool(
    *,
    name: str | None = None,            # override the inferred tool name
    schema: dict | None = None,         # override the inferred JSON Schema
    capability: str | None = None,      # logical capability label
    risk_level: RiskLevel = "low",      # "low" | "medium" | "high" | "critical"
    requires_approval: bool = False,    # gate behind an approval handler
    approval_reason: str = "",          # shown to the approver
)
```

---

## ToolGroup

A `ToolGroup` bundles tools, exposes provider-safe definitions for the LLM, and runs the governed execution pipeline.

```python
from ai_arch_toolkit import ToolGroup

group = ToolGroup(get_distance, delete_table)

group.definitions          # provider-safe tool schemas to send to the LLM
group.tools                # the registered callables
group.add(another_tool)    # register one more
```

Constructor:

```python
ToolGroup(
    *fns,                          # the tool callables
    approval_handler=None,         # ApprovalHandler for tools requiring approval (see safety.md)
    gates=(),                      # extra pre-execution ToolGate instances (see safety.md)
    max_calls=None,                # cap total executions across the group's lifetime
)
```

### Executing tool calls

Both `execute()` (sync) and `async_execute()` (async) take a `ToolCall` and return a structured **`ToolResult`** — they never raise on tool failure.

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

When you have a `Response` that contains tool calls, `run_tools()` executes all of them and returns ready-to-send `tool_result` message parts — handy for a manual LLM loop.

```python
from ai_arch_toolkit import run_tools, run_tools_sync

response = llm.complete_sync("What's the distance from Lisbon to Porto?", tools=group)
results = run_tools_sync(response, group)        # list[dict] — tool_result parts
# feed `results` back into the next llm.complete(...) call
```

`run_tools()` accepts either a `ToolGroup` or a plain `list[Callable]`.

---

## Server tools

Provider-hosted tools run on the provider's side (no local execution). Pass them alongside your own tools.

```python
from ai_arch_toolkit import LLM, code_execution, web_search

llm = LLM("claude-sonnet-5")
response = llm.complete_sync(
    "Plot the first 10 primes and tell me their sum.",
    tools=[code_execution(), web_search()],
)
```

`code_execution(**config)` and `web_search(**config)` return a `ServerTool`. Availability depends on the provider/model.

---

## Pre-built tools catalog

132 tools across 25 domains, all built on the `@tool` decorator and the standard library only (zero extra pip dependencies). Each returns an error string rather than raising, so agents degrade gracefully.

```python
from ai_arch_toolkit.toolkit.tools import get_weather, arxiv_search, pubmed_search
from ai_arch_toolkit import ToolGroup

group = ToolGroup(get_weather, arxiv_search, pubmed_search)
```

The domains at a glance:

| Theme | Domains |
|-------|---------|
| General & utility | date/time, math, text, JSON/CSV |
| Weather, geo & places | weather, air quality, geography, OpenStreetMap |
| Reference & knowledge | Wikipedia, Wikidata/MediaWiki, dictionary, news, video transcripts |
| Scholarly & research | arXiv, PubMed, Europe PMC, Semantic Scholar, Crossref, ROR, DataCite, Open Library, Internet Archive |
| Biomedical & chemistry | UniProt, PDB, ChEMBL, RxNorm/DailyMed, ClinicalTrials |
| Earth, life & public data | GBIF, Open Food Facts, openFDA, FoodOn, USGS/EONET, World Bank, WHO, Eurostat, NVD |
| Dangerous (opt-in) | filesystem, shell, Python, web fetch |

**→ Full per-tool list: [Tools Catalog](tools-catalog.md).** The filesystem/shell/Python/web tools execute real side effects and must be gated — see [Tool Governance & Safety](safety.md#dangerous-tools).

---

See also: [Tool Governance & Safety](safety.md) for risk levels, approvals, blocking, redaction, and budgets · [Flow Architecture](flow-architecture.md) for how tools plug into agent flows.
